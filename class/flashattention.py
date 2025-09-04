import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F
@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, q_shape, k_shape, v_shape, o_shape, num_kv_groups, m_size, n_size, n_heads, batch_size, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, apply_causal_mask: tl.constexpr):
    '''
        q_shape:(q_batch_stride, q_heads_stride, q_seq_stride, q_dim_stride)
        k_shape:(k_batch_stride, k_heads_stride, k_seq_stride, k_dim_stride)
        v_shape:(v_batch_stride, v_heads_stride, v_seq_stride, v_dim_stride)
        o_shape:(o_batch_stride, o_heads_stride, o_seq_stride, o_dim_stride)
        num_kv_groups: n_heads // num_kv_heads
        m_size: q_seq_len
        n_size: kv_seq_len
        grid = cell_div(m_size, BLOCK_M) * bs * n_heads
    '''
    pid = tl.program_id(0)
    head_idx = pid % n_heads
    batch_idx = (pid // n_heads) % batch_size
    block_m_idx = pid // (n_heads * batch_size)
    kv_head_idx = head_idx // num_kv_groups

    m_offs = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    (q_batch_stride, q_heads_stride, q_seq_stride, q_dim_stride) = q_shape
    (k_batch_stride, k_heads_stride, k_seq_stride, k_dim_stride) = k_shape
    (v_batch_stride, v_heads_stride, v_seq_stride, v_dim_stride) = v_shape
    (o_batch_stride, o_heads_stride, o_seq_stride, o_dim_stride) = o_shape
    '''
        dim stride means basic "next unit" operation cost how much stride, usually 1
        take q for example:
        batch_idx * q_batch_stride: move pointer to batch_idx batch
        head_idx * q_heads_stride: move pointer to head_idx head
        m_offs[:, None] is a row vector, multiply q_seq_stride to get the address of each row
        tl.arange(0, BLOCK_DMODEL)[None, :] is a column vector, corresponding to accurate column address
    '''
    d_rng = tl.arange(0, BLOCK_DMODEL)[None, :]

    q_offs = q_ptr + (
        batch_idx * q_batch_stride +
        head_idx * q_heads_stride +
        m_offs[:, None] * q_seq_stride +
        tl.arange(0, BLOCK_DMODEL)[None, :] * q_dim_stride
    )

    k_offs =  v_ptr + (                          
        batch_idx * k_batch_stride +             
        kv_head_idx * k_heads_stride +           
        d_rng * k_dim_stride                
    )                

    v_offs = v_ptr + (
        batch_idx * v_batch_stride +             
        kv_head_idx * v_heads_stride +           
        d_rng * v_dim_stride      
    )
    
    o_offs = o_ptr + (
        batch_idx * o_batch_stride +
        head_idx * o_heads_stride +
        m_offs[:, None] * o_seq_stride +
        tl.arange(0, BLOCK_DMODEL)[None, :] * o_dim_stride
    )
    
    '''
        start calculate the softmax
    '''
    
    max_logits_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    softmax_denom_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_output_i = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    
    sm_scale = 1.0 / tl.sqrt(tl.full((), BLOCK_DMODEL, tl.float32))
    
    # a pre-loaded q block, reused for each k block
    q = tl.load(q_offs, mask=m_offs[:, None] < m_size, other=0.0)
    
    for block_n_idx in range(0, (n_size + BLOCK_N - 1) // BLOCK_N):
        n_offs = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        k = tl.load(k_offs + n_offs[:, None] * k_seq_stride, mask=n_offs[:, None] < n_size, other=0.0)                 
        v = tl.load(v_offs + n_offs[:, None] * v_seq_stride, mask=n_offs[:, None] < n_size, other=0.0)                 
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        row_mask = m_offs[:, None] < m_size          # [M, 1]
        col_mask = n_offs[None, :] < n_size          # [1, N]
        base_valid = row_mask & col_mask             # [M, N]

        if apply_causal_mask:                     
            causal = m_offs[:, None] >= n_offs[None, :]
            valid = base_valid & causal              # [M, N]
        else:
            valid = base_valid                       # [M, N]

        qk = tl.where(valid, qk, -float("inf"))
        # compute the max for the block
        # l_j: 行 max； numerators: e^{qk - l_j}; d_j: 行和
        l_j = tl.max(qk, axis=1)                                  # [M]
        numerators = tl.exp(qk - l_j[:, None])                    # [M, N]
        d_j = tl.sum(numerators, axis=1)                          # [M]
        # udpate
        l_new = tl.maximum(max_logits_i, l_j)               
        d_new = (tl.exp(max_logits_i - l_new) * softmax_denom_i + tl.exp(l_j - l_new) * d_j)
        
        #sigma: 旧输出缩放
        sigma = tl.where(d_new > 0, tl.exp(max_logits_i - l_new) * softmax_denom_i / d_new, 0.0)
        acc_output_i = acc_output_i * sigma[:, None]
        
        #p: new block weight, numerators * tl.exp(l_j - l_new) / d_new = e ^ (qk - l_new) / d_new
        p_scale = tl.where(d_new > 0, tl.exp(l_j - l_new) / d_new, 0.0)
        p = numerators * p_scale[:, None]           
        acc_output_i += tl.dot(p, v)
        
        max_logits_i = l_new
        softmax_denom_i = d_new
    # store the output
    tl.store(o_offs, acc_output_i, mask=m_offs[:, None] < m_size)
    
def flash_attention(q, k, v, num_kv_heads=None, causal=False):
    '''
        q: (batch_size, n_heads, q_seq_len, d_model)
        k: (batch_size, n_kv_heads, kv_seq_len, d_model)
        v: (batch_size, n_kv_heads, kv_seq_len, d_model)
        num_kv_heads: n_heads // num_kv_heads = num_kv_groups
    '''
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    batch_size, n_heads, q_seq_len, d_model = q.shape
    _, n_kv_heads, kv_seq_len, _ = k.shape
    assert k.shape == v.shape
    assert d_model % 16 == 0 and d_model <= 1024
    if num_kv_heads is None:
        num_kv_heads = n_heads
    assert n_heads % num_kv_heads == 0
    num_kv_groups = n_heads // num_kv_heads
    
    o = torch.empty_like(q)
    
    # calculate strides
    q_shape = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    k_shape = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
    v_shape = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
    o_shape = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = d_model
    
    grid_m = triton.cdiv(q_seq_len, BLOCK_M)
    grid = (grid_m * batch_size * n_heads,)
    
    flash_attention_kernel[grid](
        q, k, v, o,
        q_shape, k_shape, v_shape, o_shape,
        num_kv_groups,
        q_seq_len,
        kv_seq_len,
        n_heads,
        batch_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        apply_causal_mask=causal
    )
    
    return o
def standard_attention(Q, K, V, sm_scale, mask=None):
    """
    标准的 PyTorch 实现的自注意力机制。

    Args:
        Q (torch.Tensor): 查询张量，形状 (batch_size, num_heads, seq_length, head_dim)
        K (torch.Tensor): 键张量，形状 (batch_size, num_heads, seq_length, head_dim)
        V (torch.Tensor): 值张量，形状 (batch_size, num_heads, seq_length, head_dim)
        sm_scale (float): Softmax 缩放因子
        mask (torch.Tensor, optional): 遮罩张量，形状 (batch_size, num_heads, seq_length, seq_length)

    Returns:
        torch.Tensor: 注意力输出，形状与 Q 相同
    """
    # 计算 QK^T
    attn_scores = (
        torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    )  # (batch_size, num_heads, seq_length, seq_length)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

    # print("attn_scores", attn_scores)
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 计算注意力输出
    out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

    return out


def test_prefill_stage():
    # 设置测试参数
    batch_size = 2
    num_heads = 4
    seq_length = 32
    head_dim = 32
    BLOCK_M = 32
    BLOCK_N = 32

    # 生成固定的输入张量（使用固定随机种子以确保可重复性）
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )
    k = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )
    v = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )

    # 计算 Softmax 缩放因子
    sm_scale = 1.0 / math.sqrt(head_dim)  # 1 / sqrt(d_k) * 1/log(2)

    # 调用 Triton 内核
    out = flash_attention(q, k, v, causal=True)

    # 使用标准 PyTorch 实现计算注意力输出
    # 创建下三角矩阵
    mask = (
        torch.tril(torch.ones((seq_length, seq_length)))
        .unsqueeze(0)
        .unsqueeze(0)
        .type_as(q)
    )  # (1, 1, seq, seq)
    standard_o = standard_attention(q, k, v, sm_scale, mask)

    # 比较 Triton 内核输出与标准实现的输出
    if torch.allclose(out, standard_o, atol=1e-2):
        print(
            "Prefill Stage Test Passed: Triton output matches PyTorch standard implementation."
        )
    else:
        max_diff = (out - standard_o).abs().max()
        print(f"Prefill Stage Test Failed: Maximum difference {max_diff}")
        # 可选择打印更多信息进行调试


def test_decode_stage():
    # 设置测试参数
    batch_size = 1
    num_heads = 4
    initial_seq_length = 16
    generated_seq_length = 16
    head_dim = 64
    BLOCK_M = 16
    BLOCK_N = 16

    # 生成固定的初始输入张量
    torch.manual_seed(0)
    q_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    k_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    v_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    o_initial = torch.zeros_like(q_initial, device="cuda", dtype=torch.float32)
    new_token_q = torch.randn(
        batch_size, num_heads, 1, head_dim, device="cuda", dtype=torch.float32
    )

    triton_k_extended = k_initial
    triton_v_extended = v_initial
    torch_k_extended = k_initial
    torch_v_extended = v_initial
    torch_new_token_q = new_token_q
    triton_new_token_q = new_token_q
    # 模拟生成过程中逐步增加序列长度
    for step in range(1, generated_seq_length + 1):
        # 生成新的 token
        triton_k_extended = torch.cat([triton_k_extended, triton_new_token_q], dim=2)
        triton_v_extended = torch.cat([triton_v_extended, triton_new_token_q], dim=2)

        torch_k_extended = torch.cat([torch_k_extended, torch_new_token_q], dim=2)
        torch_v_extended = torch.cat([torch_v_extended, torch_new_token_q], dim=2)

        # 扩展 Q, K, V 和 Out
        # q_extended = torch.cat([q_initial, new_token_q], dim=2)

        # 计算 Softmax 缩放因子, sm_scale * 1.4426950408889634 精度可控制在 1e-2 内
        sm_scale_extended = 1.0 / math.sqrt(head_dim)

        # 计算 Triton 内核输出
        triton_new_token_q = flash_attention(
            new_token_q, triton_k_extended, triton_v_extended, causal=True
        )

        # 使用标准 PyTorch 实现计算扩展后的注意力输出
        torch_new_token_q = standard_attention(
            new_token_q, torch_k_extended, torch_v_extended, sm_scale_extended
        )

        # 比较 Triton 内核输出与标准实现的输出
        if torch.allclose(triton_new_token_q, torch_new_token_q, atol=1e-1):
            print(
                f"Decode Stage Step {step} Test Passed: Triton output matches PyTorch standard implementation."
            )
        else:
            max_diff = (triton_new_token_q - torch_new_token_q).abs().max()
            print(
                f"Decode Stage Step {step} Test Failed: Maximum difference {max_diff}"
            )
            # 可选择打印更多信息进行调试
            break  # 根据需要是否停止测试


if __name__ == "__main__":
    print("Running Prefill Stage Test...")
    test_prefill_stage()
    print("\nRunning Decode Stage Test...")
    test_decode_stage()