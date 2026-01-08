
import itertools
import pandas as pd
import torch
import time

# Make sure the import matches your Python interface
from sgl_kernel import swiglu_with_alpha_and_limit

def sglang_swiglu_with_alpha_and_limit(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    # Calls the kernel op you wish to benchmark
    return swiglu_with_alpha_and_limit(x, alpha, limit)

def reference_swiglu(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    # Reference implementation using PyTorch for correctness checking
    a, b = torch.chunk(x, 2, dim=-1)
    swiglu = a * torch.sigmoid(b)
    swiglu = alpha * swiglu
    # Clamp the output by limit, similar to limit in kernel
    swiglu = torch.clamp(swiglu, -limit, limit)
    return swiglu

def calculate_bandwidth_flops(batch_size, seq_len, hidden_dim, time_ms):
    num_elements = batch_size * seq_len * hidden_dim * 2  # x shape is (..., 2*hidden_dim)
    input_bytes = num_elements * 2  # Assuming bf16 input
    output_bytes = (batch_size * seq_len * hidden_dim) * 2  # Output bf16?
    total_bytes = input_bytes + output_bytes
    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s
    # FLOPs: 2x chunk + sigmoid + mul + clamp (+*alpha)
    flops_per_elem = 5
    total_flops = (batch_size * seq_len * hidden_dim) * flops_per_elem
    gflops = (total_flops / 1e9) / time_s
    return dict(
        total_bytes=total_bytes,
        bandwidth_gbs=bandwidth_gbs,
        total_flops=total_flops,
        gflops=gflops,
    )

def correctness_check(batch_size, seq_len, hidden_dim, alpha, limit, device):
    x = torch.randn(batch_size, seq_len, hidden_dim * 2, device=device, dtype=torch.bfloat16)
    y_ref = reference_swiglu(x, alpha, limit)
    y_sglang = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    if torch.allclose(y_ref, y_sglang, rtol=1e-1, atol=1e-1):
        print(f"✅ sglang kernel output matches reference")
    else:
        print("❌ Kernel output differs from reference")

shape_configs = list(itertools.product(
    [1, 2, 4, 8, 16],         # batch_size
    [32, 128, 512, 1024],     # seq_len
    [512, 2048, 4096],        # hidden_dim
))
alpha_configs = [0.25, 1.0, 2.0]
limit_configs = [1.0, 6.0, 12.0]
test_configs = list(itertools.product(shape_configs, alpha_configs, limit_configs))

all_results = []

def benchmark_swiglu(batch_size, seq_len, hidden_dim, alpha, limit, device='xpu', dtype=torch.bfloat16, runs=100):
    x = torch.randn(batch_size, seq_len, hidden_dim * 2, device=device, dtype=dtype)
    # Warmup
    y = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    if device == 'cuda' or device == 'xpu':
        torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(runs):
        y = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    if device == 'cuda' or device == 'xpu':
        torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()
    avg_us = (end - start) * 1e6 / runs

    bw_metrics = calculate_bandwidth_flops(batch_size, seq_len, hidden_dim, avg_us/1000.)
    all_results.append(
        dict(
            batch_size=batch_size,
            seq_len=seq_len,
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
    print(f"Batch {batch_size} SeqLen {seq_len} Hidden {hidden_dim} alpha {alpha} limit {limit} -> {avg_us:.2f} us, BW {bw_metrics['bandwidth_gbs']:.2f} GB/s, GFLOPS {bw_metrics['gflops']:.2f}")

if __name__ == "__main__":
    # Correctness check
    correctness_check(2, 16, 512, 1.0, 6.0, device='xpu')

    # Main benchmarking loop
    for ((batch_size, seq_len, hidden_dim), alpha, limit) in test_configs[:12]:  # example: limit tests to first 12 for speed
        benchmark_swiglu(batch_size, seq_len, hidden_dim, alpha, limit, device='xpu', dtype=torch.bfloat16, runs=50)

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