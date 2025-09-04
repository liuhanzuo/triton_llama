import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, q_shape, k_shape, v_shape, o_shape, num_kv_groups, m_size, n_size, n_heads, batch_size,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
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

    q_offs = (
        batch_idx * q_batch_stride +
        head_idx * q_heads_stride +
        m_offs[:, None] * q_seq_stride +
        tl.arange(0, BLOCK_DMODEL)[None, :] * q_dim_stride
    )

    k_offs = (
        batch_idx * k_batch_stride +
        kv_head_idx * k_heads_stride +
        tl.arange(0, BLOCK_N)[:, None] * k_seq_stride +
        tl.arange(0, BLOCK_DMODEL)[None, :] * k_dim_stride
    )

    v_offs = (
        batch_idx * v_batch_stride +
        kv_head_idx * v_heads_stride +
        tl.arange(0, BLOCK_N)[:, None] * v_seq_stride +
        tl.arange(0, BLOCK_DMODEL)[None, :] * v_dim_stride
    )
