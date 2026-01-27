import itertools

import pandas as pd
import torch
import triton
from sgl_kernel import fused_qk_norm_rope


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Reference RMS normalization implementation."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool = True,
):
    """Reference rotary position embedding implementation."""
    if is_neox:
        # Neox style: split into two halves
        half_dim = x.shape[-1] // 2
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    else:
        # Interleaved style
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos[..., ::2]
        sin = sin[..., ::2]
        rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        ).flatten(-2)
        return rotated


def compute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device = None,
):
    """Compute RoPE frequency tensors."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def torch_fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    rotary_dim: int = None,
):
    """Reference PyTorch implementation of fused QK norm and RoPE."""
    num_tokens = qkv.shape[0]
    num_heads = num_heads_q + num_heads_k + num_heads_v

    if rotary_dim is None:
        rotary_dim = head_dim

    # Reshape to separate heads
    qkv = qkv.view(num_tokens, num_heads, head_dim)

    # Split Q, K, V
    q = qkv[:, :num_heads_q, :]
    k = qkv[:, num_heads_q : num_heads_q + num_heads_k, :]
    v = qkv[:, num_heads_q + num_heads_k :, :]

    # Apply RMS normalization
    q = torch_rms_norm(q, q_weight, eps)
    k = torch_rms_norm(k, k_weight, eps)

    # Compute RoPE
    max_pos = position_ids.max().item() + 1
    cos, sin = compute_rope_freqs(rotary_dim, max_pos, base, qkv.device)
    cos = cos[position_ids].unsqueeze(1)  # (num_tokens, 1, rotary_dim)
    sin = sin[position_ids].unsqueeze(1)

    # Apply RoPE to Q and K
    if rotary_dim < head_dim:
        q_rot = q[..., :rotary_dim]
        q_pass = q[..., rotary_dim:]
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]

        q_rot = apply_rotary_emb(q_rot, cos, sin, is_neox)
        k_rot = apply_rotary_emb(k_rot, cos, sin, is_neox)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
    else:
        q = apply_rotary_emb(q, cos, sin, is_neox)
        k = apply_rotary_emb(k, cos, sin, is_neox)

    # Concatenate back
    qkv_out = torch.cat([q, k, v], dim=1)
    return qkv_out.view(num_tokens, -1)


def calculate_diff(
    batch_size, seq_len, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox
):
    device = torch.device("xpu")
    num_tokens = batch_size * seq_len
    num_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(
        num_tokens, num_heads * head_dim, device=device, dtype=torch.bfloat16
    )
    q_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)

    eps = 1e-6
    base = 10000.0
    rotary_dim = head_dim

    # Clone for independent execution
    qkv_torch = qkv.clone()
    qkv_sglang = qkv.clone()

    # PyTorch reference
    qkv_out_torch = torch_fused_qk_norm_rope(
        qkv_torch,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        rotary_dim,
    )

    # SGL Kernel
    fused_qk_norm_rope(
        qkv_sglang,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor=1.0,
        low=1.0,
        high=1.0,
        attention_factor=1.0,
        rotary_dim=rotary_dim,
    )

    # Compare
    if torch.allclose(qkv_out_torch, qkv_sglang, rtol=1e-2, atol=1e-2):
        print(f"✅ is_neox={is_neox} implementations match")
    else:
        max_diff = (qkv_out_torch - qkv_sglang).abs().max().item()
        mean_diff = (qkv_out_torch - qkv_sglang).abs().mean().item()
        print(
            f"❌ Implementations differ - max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f}"
        )


# Benchmark configurations
batch_size_range = [1, 2, 4, 8, 16, 32]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
# DeepSeek-V3 config: 128 Q heads, 128 KV heads (MLA)
head_config_range = [
    (32, 8, 8, 128),  # Standard MQA config
    (32, 32, 32, 128),  # Standard MHA config
    (128, 128, 128, 128),  # DeepSeek-V3 style
]
is_neox_range = [True, False]

configs = []
for batch_size, seq_len, (nq, nk, nv, hd), is_neox in itertools.product(
    batch_size_range, seq_len_range, head_config_range, is_neox_range
):
    configs.append((batch_size, seq_len, nq, nk, nv, hd, is_neox))

all_results = []


def calculate_flops(
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    rotary_dim: int,
) -> int:
    """
    Calculate FLOPs for fused QK norm + RoPE kernel.

    RMS Norm per head: ~2*head_dim (square + normalize + multiply)
    RoPE per head: ~6*rotary_dim (sin, cos, multiply, add operations)
    """
    # RMS norm: 2 FLOPs per element (square, rsqrt, multiply, scale)
    flops_rmsnorm_q = num_tokens * num_heads_q * head_dim * 4
    flops_rmsnorm_k = num_tokens * num_heads_k * head_dim * 4

    # RoPE: ~6 FLOPs per rotary dimension (sin, cos, 4 mul/add)
    flops_rope_q = num_tokens * num_heads_q * rotary_dim * 6
    flops_rope_k = num_tokens * num_heads_k * rotary_dim * 6

    total_flops = flops_rmsnorm_q + flops_rmsnorm_k + flops_rope_q + flops_rope_k

    return total_flops


def calculate_effective_bandwidth(
    batch_size: int,
    seq_len: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    rotary_dim: int,
    time_ms: float,
) -> dict:
    """
    Calculate effective bandwidth and FLOPs for fused QK norm + RoPE kernel.

    Memory: read/write QKV tensor + read weights
    """
    num_tokens = batch_size * seq_len
    num_heads = num_heads_q + num_heads_k + num_heads_v

    # Input/output QKV tensor (bf16)
    qkv_bytes = num_tokens * num_heads * head_dim * 2

    # Weight tensors (bf16)
    weight_bytes = 2 * head_dim * 2  # q_weight + k_weight

    # Total bytes (read QKV + write QKV + read weights)
    total_bytes = 2 * qkv_bytes + weight_bytes

    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    total_flops = calculate_flops(
        num_tokens, num_heads_q, num_heads_k, head_dim, rotary_dim
    )
    gflops = (total_flops / 1e9) / time_s

    return {
        "num_tokens": num_tokens,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "batch_size",
            "seq_len",
            "num_heads_q",
            "num_heads_k",
            "num_heads_v",
            "head_dim",
            "is_neox",
        ],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fused-qk-norm-rope-performance",
        args={},
    )
)
def benchmark(
    batch_size,
    seq_len,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    is_neox,
    provider,
):
    device = torch.device("xpu")
    num_tokens = batch_size * seq_len
    num_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(
        num_tokens, num_heads * head_dim, device=device, dtype=torch.bfloat16
    )
    q_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)

    eps = 1e-6
    base = 10000.0
    rotary_dim = head_dim

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: torch_fused_qk_norm_rope(
            qkv.clone(),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            rotary_dim,
        )
    elif provider == "sglang":
        fn = lambda: fused_qk_norm_rope(
            qkv.clone(),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            factor=1.0,
            low=1.0,
            high=1.0,
            attention_factor=1.0,
            rotary_dim=rotary_dim,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Calculate effective bandwidth
    bw_metrics = calculate_effective_bandwidth(
        batch_size,
        seq_len,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        ms,
    )

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": bw_metrics["num_tokens"],
            "num_heads_q": num_heads_q,
            "num_heads_k": num_heads_k,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
            "is_neox": is_neox,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":

    # Run correctness checks
    print("Running correctness checks...")
    calculate_diff(
        batch_size=4,
        seq_len=128,
        num_heads_q=32,
        num_heads_k=8,
        num_heads_v=8,
        head_dim=128,
        is_neox=True,
    )
    calculate_diff(
        batch_size=2,
        seq_len=64,
        num_heads_q=32,
        num_heads_k=8,
        num_heads_v=8,
        head_dim=128,
        is_neox=False,
    )
    calculate_diff(
        batch_size=1,
        seq_len=32,
        num_heads_q=128,
        num_heads_k=128,
        num_heads_v=128,
        head_dim=128,
        is_neox=True,
    )

    print("\nRunning benchmarks...")
    benchmark.run(print_data=True)

    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes_mb"] = df["total_bytes_mb"].round(2)
    df["time_us"] = df["time_us"].round(2)
    df["total_flops_m"] = df["total_flops_m"].round(2)
    df["gflops"] = df["gflops"].round(2)

    print(df.to_markdown(index=False))

    # Print summary statistics per provider
    print("\n" + "=" * 80)
    print("Summary Statistics by Provider")
    print("=" * 80)
    summary = df.groupby("provider").agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
            "gflops": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())

    # Print best speedup
    print("\n" + "=" * 80)
    print("Speedup Analysis")
    print("=" * 80)

    # Pivot to compare providers
    pivot = df.pivot_table(
        index=[
            "batch_size",
            "seq_len",
            "num_heads_q",
            "num_heads_k",
            "num_heads_v",
            "head_dim",
            "is_neox",
        ],
        columns="provider",
        values="time_us",
    )

    if "torch" in pivot.columns and "sglang" in pivot.columns:
        pivot["speedup"] = pivot["torch"] / pivot["sglang"]
        print(f"\nAverage speedup: {pivot['speedup'].mean():.2f}x")
        print(f"Max speedup: {pivot['speedup'].max():.2f}x")
        print(f"Min speedup: {pivot['speedup'].min():.2f}x")
