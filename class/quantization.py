import torch
import triton
import triton.language as tl
from config import QUANTIZATION_CONFIGS

@triton.autotune(configs=QUANTIZATION_CONFIGS, key=['n_elements']) 
@triton.jit
def quantization_fp8_forward_kernel(input_ptr, output_ptr, scale_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inputs = tl.load(input_ptr + offs, mask = offs < n_elements, other = 0.0).to(tl.float32)
    scale = tl.max(tl.abs(inputs)) / 448.0
    scale = tl.where(scale > 0, scale, 1.0)
    outputs = (inputs / scale).to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + offs, outputs)
    tl.store(scale_ptr + pid, scale)
def _min_block_size_from_configs(configs):
    # 兼容性考虑
    def meta(cfg):
        return getattr(cfg, 'meta', getattr(cfg, 'kwargs', {}))
    return min(int(meta(cfg)['BLOCK_SIZE']) for cfg in configs)
@torch.no_grad()
def quantization_fp8_forward(x: torch.Tensor):
    assert x.is_contiguous()
    device = x.device

    flat = x.view(-1)
    n_elements = flat.numel()
    y = torch.empty_like(flat, dtype=torch.uint8)    

    min_bs   = _min_block_size_from_configs(QUANTIZATION_CONFIGS)
    n_blocks = triton.cdiv(n_elements, min_bs)
    scales   = torch.empty(n_blocks, device=device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)

    quantization_fp8_forward_kernel[grid](
        flat, y, scales, n_elements,
    )
    return y.view_as(x), scales
    