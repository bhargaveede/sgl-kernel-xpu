import itertools
from typing import Tuple

import torch
import triton
import triton.testing
from sgl_kernel import bmm_fp8

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_e4m3_type = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
fp8_e5m2_type = torch.float8_e5m2fnuz if _is_hip else torch.float8_e5m2


def to_float8(x, dtype=torch.float8_e4m3fn):
    """Convert tensor to float8 with scaling."""
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def torch_bmm_fp8(
    input: torch.Tensor,
    mat2: torch.Tensor,
    res_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Baseline implementation using torch.bmm."""
    return torch.bmm(input, mat2).to(res_dtype)


def sglang_bmm_fp8(
    input: torch.Tensor,
    mat2: torch.Tensor,
    input_dtype: torch.dtype = torch.float8_e4m3fn,
    mat2_dtype: torch.dtype = torch.float8_e4m3fn,
    res_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """SGL Kernel implementation of BMM FP8."""
    # Convert inputs to fp8
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)
    
    # Prepare output
    batch_size, m, k = input.shape
    _, _, n = mat2.shape
    res = torch.empty([batch_size, m, n], device=input.device, dtype=res_dtype)
    
    # Run kernel
    bmm_fp8(input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype, res)
    
    return res


def calculate_diff(batch_size: int, m: int, k: int, n: int):
    """Calculate difference between torch.bmm and SGL Kernel implementations."""
    device = torch.device("cuda")
    
    input = torch.randn([batch_size, m, k], dtype=torch.bfloat16, device=device)
    # mat2 in column-major format
    mat2 = torch.randn([batch_size, n, k], dtype=torch.bfloat16, device=device).transpose(-2, -1)
    
    torch_out = torch_bmm_fp8(input, mat2, res_dtype=torch.float16)
    sglang_out = sglang_bmm_fp8(
        input, 
        mat2, 
        input_dtype=fp8_e4m3_type,
        mat2_dtype=fp8_e4m3_type,
        res_dtype=torch.float16
    )
    
    output_diff = torch.abs(torch_out - sglang_out).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        torch_out.reshape(-1), sglang_out.reshape(-1), dim=0
    ).item()
    
    print(f"Mean absolute difference: {output_diff:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    
    if cos_sim > 0.99:
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1, 4, 8, 16]
m_range = [32, 64, 128]
k_range = [64, 128, 256]
n_range = [64, 128, 256]

# Create configurations (batch_size, m, k, n)
configs = list(itertools.product(batch_size_range, m_range, [128], [128]))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "m", "k", "n"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["Torch BMM", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="bmm-fp8-performance",
        args={},
    )
)
def benchmark_bmm_fp8(batch_size, m, k, n, provider):
    device = torch.device("cuda")
    res_dtype = torch.float16
    
    input = torch.randn([batch_size, m, k], device=device, dtype=torch.bfloat16)
    # mat2 in column-major format
    mat2 = torch.randn([batch_size, n, k], device=device, dtype=torch.bfloat16).transpose(-2, -1)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == "torch":
        fn = lambda: torch_bmm_fp8(input, mat2, res_dtype)
    elif provider == "sglang":
        fn = lambda: sglang_bmm_fp8(
            input, 
            mat2, 
            input_dtype=fp8_e4m3_type,
            mat2_dtype=fp8_e4m3_type,
            res_dtype=res_dtype
        )
    
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=16, m=48, k=64, n=80)
    benchmark_bmm_fp8.run(print_data=True)
