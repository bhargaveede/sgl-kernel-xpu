import itertools
import pandas as pd
import torch
import time

# Replace this import path as needed for your project
from sgl_kernel import swiglu_with_alpha_and_limit

def sglang_swiglu_with_alpha_and_limit(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    return swiglu_with_alpha_and_limit(x, alpha, limit)

def reference_swiglu(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    a, b = torch.chunk(x, 2, dim=-1)  # x shape [B, 2H] -> a,b [B, H]
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
    bandwidth_gbs = (total_bytes / 1e9) / time_s
    flops_per_elem = 5  # Estimate: chunk+sigmoid+mul+scale+clamp for each output element
    total_flops = (batch_size * hidden_dim) * flops_per_elem
    gflops = (total_flops / 1e9) / time_s
    return dict(
        total_bytes=total_bytes,
        bandwidth_gbs=bandwidth_gbs,
        total_flops=total_flops,
        gflops=gflops,
    )

def correctness_check(batch_size, hidden_dim, alpha, limit, device):
    x = torch.randn(batch_size, hidden_dim * 2, device=device, dtype=torch.float32)
    y_ref = reference_swiglu(x, alpha, limit)
    y_sglang = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    if torch.allclose(y_ref, y_sglang, rtol=1e-5, atol=1e-6):
        print(f"✅ Kernel output matches reference")
    else:
        print("❌ Kernel output differs from reference")

batch_sizes = [1, 2, 4, 16, 32, 64]
hidden_dims = [512, 2048, 4096]
alpha_configs = [0.25, 1.0, 2.0]
limit_configs = [1.0, 6.0, 12.0]
test_configs = list(itertools.product(batch_sizes, hidden_dims, alpha_configs, limit_configs))

all_results = []

def benchmark_swiglu(batch_size, hidden_dim, alpha, limit, device='cuda', dtype=torch.float32, runs=100):
    x = torch.randn(batch_size, hidden_dim * 2, device=device, dtype=dtype)
    # Warmup
    y = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        y = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    avg_us = (end - start) * 1e6 / runs

    bw_metrics = calculate_bandwidth_flops(batch_size, hidden_dim, avg_us/1000.)
    all_results.append(
        dict(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            alpha=alpha,
            limit=limit,
            dtype=str(dtype),
            device=device,
            time_us=avg_us,
            bandwidth_gbs=round(bw_metrics["bandwidth_gbs"], 2),
            total_bytes_mb=round(bw_metrics["total_bytes"] / 1e6, 2),
            gflops=round(bw_metrics["gflops"], 2),
        )
    )
    print(f"Batch {batch_size} Hidden {hidden_dim} alpha {alpha} limit {limit} -> {avg_us:.2f} us, BW {bw_metrics['bandwidth_gbs']:.2f} GB/s, GFLOPS {bw_metrics['gflops']:.2f}")

if __name__ == "__main__":
    # Correctness check
    correctness_check(4, 1024, 1.0, 6.0, device='cuda')

    # Main benchmarking loop
    for (batch_size, hidden_dim, alpha, limit) in test_configs[:10]:  # first 10 configs, adjust for more
        benchmark_swiglu(batch_size, hidden_dim, alpha, limit, device='cuda', dtype=torch.float32, runs=50)

    # Report results
    print("\n" + "=" * 80)
    print("Results Table")
    print("=" * 80)
    df = pd.DataFrame(all_results)
    print(df.to_markdown(index=False))

    print("\nSummary stats:")
    summary = df.groupby("device").agg({
        "bandwidth_gbs": ["mean", "min", "max"],
        "time_us": ["mean", "min", "max"],
        "gflops": ["mean", "min", "max"],
    })
    print(summary.to_markdown())