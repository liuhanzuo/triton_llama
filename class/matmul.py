import triton
import triton.language as tl
import torch
import matplotlib.pyplot as plt
AUTOTUNE_CONFIGS_UNGROUPED = [
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=2),
]

AUTOTUNE_CONFIGS_GROUPED = [
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS_UNGROUPED, key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, 
                  BLOCK_SIZE_M: tl.constexpr = 128, 
                  BLOCK_SIZE_N: tl.constexpr = 128, 
                  BLOCK_SIZE_K: tl.constexpr = 64):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    tiled_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        # a_ptr = a_ptr + offs_m * K + offs_k 代表 a[m, k]
        mask_a = (offs_m < M) & (offs_k[None, :] < K)
        tiled_a = tl.load(a_ptr + offs_m * K + offs_k[None, :], mask=mask_a, other=0.0).to(tl.float32)
        
        mask_b = (offs_k[:, None] < K) & (offs_n < N)
        tiled_b = tl.load(b_ptr + offs_k[:, None] * N + offs_n, mask=mask_b, other=0.0).to(tl.float32)
        
        tiled_c += tl.dot(tiled_a, tiled_b)
    
    c_mask = (offs_m < M) & (offs_n < N)
    #c_ptr + offs_m * N + offs_n 代表 c[m, n]
    tl.store(c_ptr + offs_m * N + offs_n, tiled_c, mask=c_mask)

@triton.autotune(configs=AUTOTUNE_CONFIGS_GROUPED, key=['M', 'N', 'K'])
@triton.jit
def grouped_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, 
                          BLOCK_SIZE_M: tl.constexpr = 128, 
                          BLOCK_SIZE_N: tl.constexpr = 128, 
                          BLOCK_SIZE_K: tl.constexpr = 64,
                          GROUP_SIZE_M: tl.constexpr = 8):   # ← 改名对齐
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_each_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_each_group
    pid_in_group = pid % num_pid_each_group

    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_n = pid_in_group // group_size_m
    pid_m = first_pid_m + (pid_in_group % group_size_m)

    # 尾组不满时可能越界，这里直接返回
    if (pid_m >= num_pid_m) | (pid_n >= num_pid_n):
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_m * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n
        mask_a = (offs_m < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n < N)
        A = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)
        B = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)
        acc += tl.dot(A, B)

    c_ptrs = c_ptr + offs_m * N + offs_n
    c_mask = (offs_m < M) & (offs_n < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@torch.no_grad()
def matmul(a, b, BM=64, BN=64, BK=64):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    
    matmul_kernel[grid](a, b, c, M, N, K)
    return c

def matmul_grouped(a, b, BM=64, BN=64, BK=64, GROUP_SIZE_M=8):
    M, K = a.shape; K2, N = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(triton.cdiv(M, META['BLOCK_SIZE_M']), GROUP_SIZE_M)  # num_groups
        * GROUP_SIZE_M * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    grouped_matmul_kernel[grid](a, b, c, M, N, K)
    return c

import time

def benchmark(func, *args, warmup=10, iters=100):
    # 预热，避免首次启动的 lazy init 影响结果
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / iters

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    # 你可按显卡算力调整尺寸；先给一个中等规模，跑得动也能看出差异
    M, K, N = 4096, 2048, 1024
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    # 统一 block 配置（可按显卡与 SMEM 调整）
    BM, BN, BK = 64, 64, 64

    # 1) 结果一致性：未分组 vs torch
    c_ungrp = matmul(a, b, BM, BN, BK)
    c_torch = torch.matmul(a, b)
    ok_ungrp = torch.allclose(c_ungrp, c_torch, rtol=1e-2, atol=1e-2)
    print(f"[Check] ungrouped vs torch: {ok_ungrp}")

    # 2) 结果一致性：不同 GROUP_SIZE_M vs torch & 未分组
    group_list = [1, 2, 4, 8, 16, 32]  # 你也可加大到 64/128
    speed_torch = benchmark(torch.matmul, a, b)
    speeds = {"torch": speed_torch}

    for G in group_list:
        c_grp = matmul_grouped(a, b, BM, BN, BK, GROUP_SIZE_M=G)
        ok_grp_vs_torch = torch.allclose(c_grp, c_torch, rtol=1e-2, atol=1e-2)
        ok_grp_vs_ungrp = torch.allclose(c_grp, c_ungrp, rtol=1e-2, atol=1e-2)
        print(f"[Check] GROUP={G:<3} vs torch: {ok_grp_vs_torch} | vs ungrouped: {ok_grp_vs_ungrp}")

        t_grp = benchmark(matmul_grouped, a, b, BM, BN, BK, G)
        speeds[f"group={G}"] = t_grp
        print(f"[Speed ] GROUP={G:<3}: {t_grp*1000:.3f} ms")

    # 3) 画折线图：越低越好（单位：ms）
    names = ["torch"] + [f"group={G}" for G in group_list]
    times_ms = [speeds[n]*1000 for n in names]

    plt.figure()
    plt.plot(list(range(len(names))), times_ms, marker="o")
    plt.xticks(list(range(len(names))), names, rotation=30, ha="right")
    plt.ylabel("Avg time (ms)")
    plt.title(f"matmul speed vs GROUP (M={M}, K={K}, N={N}, BM={BM}, BN={BN}, BK={BK})")
    plt.tight_layout()
    plt.savefig("./speed_vs_group.png", dpi=160)
    print("Saved figure: speed_vs_group.png")