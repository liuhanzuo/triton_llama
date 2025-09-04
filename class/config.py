import triton
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

QUANTIZATION_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 64},   num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 128},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 512},  num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
]