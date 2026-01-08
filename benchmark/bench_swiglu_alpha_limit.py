import itertools
import pandas as pd
import torch
import time

# Replace with the correct import if needed
from sgl_kernel import swiglu_with_alpha_and_limit

def sglang_swiglu_with_alpha_and_limit(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    return swiglu_with_alpha_and_limit(x, alpha, limit)

def reference_swiglu(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    a, b = torch.chunk(x, 2, dim=-1)
    swiglu = a * torch.sigmoid(b)
    swiglu = alpha * swiglu
    swiglu = torch.clamp(swiglu, -limit, limit)
    return swiglu

def calculate_bandwidth_flops(batch_size, hidden_dim, time_ms):
    # x: [B, 2H], float32 (4 bytes each)
    num_elements = batch_size * hidden_dim * 2         # input
    input_bytes = num_elements * 4
    output_bytes = batch_size * hidden_dim * 4         # output [B, H], float32
    total_bytes = input_bytes + output_bytes
    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s if time_s > 0 else 0
    flops_per_elem = 5
    total_flops = (batch_size * hidden_dim) * flops_per_elem
    gflops = (total_flops / 1e9) / time_s if time_s > 0 else 0
    return dict(
        total_bytes=total_bytes,
        bandwidth_gbs=bandwidth_gbs,
        total_flops=total_flops,
        gflops=gflops,
    )

def correctness_check(x, y_kernel, y_ref):
    if torch.allclose(y_kernel, y_ref, rtol=1e-5, atol=1e-6):
        return True
    return False

batch_sizes = [1, 2, 4, 16, 32, 64]
hidden_dims = [512, 2048, 4096]
alpha_configs = [0.25, 1.0, 2.0]
limit_configs = [1.0, 6.0, 12.0]
test_configs = list(itertools.product(batch_sizes, hidden_dims, alpha_configs, limit_configs))

all_results = []

def benchmark_swiglu(batch_size, hidden_dim, alpha, limit, device='xpu', dtype=torch.float32, runs=100):
    x = torch.randn(batch_size, hidden_dim * 2, device=device, dtype=dtype)
    # Warmup both kernels
    y_kernel = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    y_ref = reference_swiglu(x, alpha, limit)
    # Time kernel
    start = time.time()
    for _ in range(runs):
        y_kernel = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    end = time.time()
    avg_us_kernel = (end - start) * 1e6 / runs

    # Time reference
    start_ref = time.time()
    for _ in range(runs):
        y_ref = reference_swiglu(x, alpha, limit)
    end_ref = time.time()
    avg_us_ref = (end_ref - start_ref) * 1e6 / runs

    # Bandwidth/flops
    bw_kernel = calculate_bandwidth_flops(batch_size, hidden_dim, avg_us_kernel/1000.)
    bw_ref = calculate_bandwidth_flops(batch_size, hidden_dim, avg_us_ref/1000.)

    # Correctness
    is_correct = correctness_check(x, y_kernel, y_ref)

    all_results.append(
        dict(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            alpha=alpha,
            limit=limit,
            dtype=str(dtype),
            device=device,
            time_us_kernel=avg_us_kernel,
            time_us_reference=avg_us_ref,
            bandwidth_gbs_kernel=round(bw_kernel["bandwidth_gbs"], 2),
            bandwidth_gbs_reference=round(bw_ref["bandwidth_gbs"], 2),
            total_bytes_mb=round(bw_kernel["total_bytes"] / 1e6, 2),
            gflops_kernel=round(bw_kernel["gflops"], 2),
            gflops_reference=round(bw_ref["gflops"], 2),
            correct="✅" if is_correct else "❌"
        )
    )
    print(
        f"B {batch_size} H {hidden_dim} α {alpha} lim {limit} | "
        f"Kernel {avg_us_kernel:.2f}us, Ref {avg_us_ref:.2f}us | "
        f"BW {bw_kernel['bandwidth_gbs']:.2f}/{bw_ref['bandwidth_gbs']:.2f} GB/s | "
        f"GFLOPS {bw_kernel['gflops']:.2f}/{bw_ref['gflops']:.2f} | {all_results[-1]['correct']}"
    )

if __name__ == "__main__":
    # Main benchmarking loop
    for (batch_size, hidden_dim, alpha, limit) in test_configs[:10]:  # adjust for more
        benchmark_swiglu(batch_size, hidden_dim, alpha, limit, device='xpu', dtype=torch.float32, runs=50)

    # Report results
    print("\n" + "=" * 80)
    print("Results Table")
    print("=" * 80)
    df = pd.DataFrame(all_results)
    print(df.to_markdown(index=False))

    print("\nSummary stats (Kernel):")
    kernel_summary = df.groupby("device").agg({
        "bandwidth_gbs_kernel": ["mean", "min", "max"],
        "time_us_kernel": ["mean", "min", "max"],
        "gflops_kernel": ["mean", "min", "max"],
    })
    print(kernel_summary.to_markdown())

    print("\nSummary stats (Reference):")
    ref_summary = df.groupby("device").agg({
        "bandwidth_gbs_reference": ["mean", "min", "max"],
        "time_us_reference": ["mean", "min", "max"],
        "gflops_reference": ["mean", "min", "max"],
    })
    print(ref_summary.to_markdown())