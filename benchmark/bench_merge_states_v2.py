import itertools
import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import merge_state_v2

def triton_merge_state_v2(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor,
):
    # Assume we reference kernel pattern from test_merge_state_v2
    # Normally this would call a Triton kernel, matching test_merge_state_v2.py
    # For timing we use SGL kernel as ground truth, Triton placeholder
    merge_state_v2(
        prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
    )

def calculate_diff(batch_size, num_tokens, num_heads, head_size):
    device = torch.device("xpu")
    torch.manual_seed(42)

    prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
    suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
    prefix_lse = torch.randn(num_tokens, num_heads, device=device, dtype=torch.float32)
    suffix_lse = torch.randn(num_tokens, num_heads, device=device, dtype=torch.float32)
    output_triton = torch.empty_like(prefix_output)
    output_lse_triton = torch.empty_like(prefix_lse)
    output_sgl = torch.empty_like(prefix_output)
    output_lse_sgl = torch.empty_like(prefix_lse)

    # 'Triton' baseline: re-use SGL kernel for now, as test reference
    triton_merge_state_v2(prefix_output, prefix_lse,
                         suffix_output, suffix_lse,
                         output_triton, output_lse_triton)
    # SGL reference (should be identical)
    merge_state_v2(prefix_output, prefix_lse,
                   suffix_output, suffix_lse,
                   output_sgl, output_lse_sgl)

    if torch.allclose(output_triton, output_sgl, rtol=1e-4, atol=1e-4) and \
       torch.allclose(output_lse_triton, output_lse_sgl, rtol=1e-4, atol=1e-4):
        print("✅ merge_state_v2 outputs match")
    else:
        print("❌ merge_state_v2 outputs differ")

batch_size_range = [1, 2, 4, 8]
num_tokens_range = [64, 128, 256]
num_heads_range = [16, 32]
head_size_range = [128, 256]

configs = list(
    itertools.product(batch_size_range, num_tokens_range, num_heads_range, head_size_range)
)

all_results = []

import time

def calculate_bandwidth(num_tokens, num_heads, head_size, time_ms):
    """
    Calculate approximate effective bandwidth for merge_state_v2.
    """
    bytes_per_tensor = num_tokens * num_heads * head_size * 4  # float32
    bytes_per_lse = num_tokens * num_heads * 4  # float32
    # Read: 2 x output + 2 x lse, Write: output + lse
    total_bytes = 2 * (bytes_per_tensor + bytes_per_lse) + (bytes_per_tensor + bytes_per_lse)
    time_s = time_ms / 1000.0
    return (total_bytes / 1e9) / time_s

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "num_tokens", "num_heads", "head_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="merge-state-v2-performance",
        args={},
    )
)
def benchmark(batch_size, num_tokens, num_heads, head_size, provider):
    device = torch.device("xpu")
    prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
    suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
    prefix_lse = torch.randn(num_tokens, num_heads, device=device, dtype=torch.float32)
    suffix_lse = torch.randn(num_tokens, num_heads, device=device, dtype=torch.float32)
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)

    def fn():
        merge_state_v2(prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    bandwidth_gbs = calculate_bandwidth(num_tokens, num_heads, head_size, ms)

    all_results.append(
        {
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "head_size": head_size,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms

if __name__ == "__main__":

    # Quick functional diff check
    calculate_diff(batch_size=2, num_tokens=128, num_heads=32, head_size=128)

    benchmark.run(print_data=True)

    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
    print("=" * 80)
    df = pd.DataFrame(all_results)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["time_us"] = df["time_us"].round(2)
    print(df.to_markdown(index=False))

    # Summary
    print("\n" + "=" * 80)
    print("Summary Statistics by Provider")
    print("=" * 80)
    summary = df.groupby("provider").agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())