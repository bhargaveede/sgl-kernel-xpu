import itertools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel import merge_state_v2


@triton.jit
def merge_state_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_merged
    output_lse,  # [NUM_TOKENS, NUM_HEADS] s_merged
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_a
    prefix_lse,  # [NUM_TOKENS, NUM_HEADS] s_a
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_b
    suffix_lse,  # [NUM_TOKENS, NUM_HEADS] s_b
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + token_idx * num_heads + head_idx)
    s_lse = tl.load(suffix_lse + token_idx * num_heads + head_idx)
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    out_se = tl.exp(p_lse) + tl.exp(s_lse)

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + token_idx * num_heads + head_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )

    p_scale = tl.exp(p_lse) / out_se
    s_scale = tl.exp(s_lse) / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        out,
        mask=head_mask,
    )


def triton_merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = prefix_output.shape[0]
    num_query_heads = prefix_output.shape[1]
    head_size = prefix_output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    
    # Avoid creating new tensors if they are already provided
    if output is None:
        output = torch.empty_like(prefix_output)
    if output_lse is None:
        output_lse = torch.empty_like(prefix_lse)

    merge_state_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        head_size,
        padded_head_size,
        output_lse is not None,
    )
    return output, output_lse


def sglang_merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Avoid creating new tensors if they are already provided
    if output is None:
        output = torch.empty_like(prefix_output)
    if output_lse is None:
        output_lse = torch.empty_like(prefix_lse)
    
    merge_state_v2(
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output,
        output_lse,
    )
    return output, output_lse


def calculate_diff(num_tokens, num_heads, head_size, dtype):
    device = torch.device("xpu")

    # Create test tensors with some inf values
    prefix_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)
    suffix_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)

    # Generate boolean masks to add some inf values
    mask_prefix = torch.rand(num_tokens, num_heads, device=device) < 0.1
    mask_suffix = torch.rand(num_tokens, num_heads, device=device) < 0.1
    # Ensure that the same position is not True at the same time
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float("inf")
    suffix_lse[mask_suffix] = float("inf")

    # Create output tensors
    prefix_output = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    suffix_output = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)

    output_triton = torch.empty_like(prefix_output)
    output_lse_triton = torch.empty_like(prefix_lse)
    
    output_sglang = torch.empty_like(prefix_output)
    output_lse_sglang = torch.empty_like(prefix_lse)

    output_triton, output_lse_triton = triton_merge_state(
        prefix_output.clone(),
        prefix_lse.clone(),
        suffix_output.clone(),
        suffix_lse.clone(),
        output_triton,
        output_lse_triton,
    )
    
    output_sglang, output_lse_sglang = sglang_merge_state(
        prefix_output.clone(),
        prefix_lse.clone(),
        suffix_output.clone(),
        suffix_lse.clone(),
        output_sglang,
        output_lse_sglang,
    )

    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    if torch.allclose(
        output_triton.to(torch.float32), output_sglang.to(torch.float32), rtol=rtol, atol=1e-3
    ) and torch.allclose(output_lse_triton, output_lse_sglang, rtol=rtol, atol=1e-3):
        print(f"✅ {dtype} implementations match")
    else:
        print(f"❌ {dtype} implementations differ")
        max_diff = torch.max(torch.abs(output_triton.float() - output_sglang.float()))
        max_lse_diff = torch.max(torch.abs(output_lse_triton - output_lse_sglang))
        print(f"   Max output diff: {max_diff:.6f}")
        print(f"   Max LSE diff: {max_lse_diff:.6f}")


num_tokens_range = [256, 512, 1024, 2048]
num_heads_range = [8, 16, 32]
head_size_range = [64, 128, 256]
dtype_range = [torch.float16, torch.bfloat16]

configs = list(
    itertools.product(
        num_tokens_range, num_heads_range, head_size_range, dtype_range
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_heads", "head_size", "dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "sglang"],
        line_names=["Triton", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="merge-state-v2-performance",
        args={},
    )
)
def benchmark(num_tokens, num_heads, head_size, dtype, provider):
    device = torch.device("xpu")

    # Create test tensors
    prefix_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)
    suffix_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)
    
    # Add some inf values
    mask_prefix = torch.rand(num_tokens, num_heads, device=device) < 0.1
    mask_suffix = torch.rand(num_tokens, num_heads, device=device) < 0.1
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)
    prefix_lse[mask_prefix] = float("inf")
    suffix_lse[mask_suffix] = float("inf")

    prefix_output = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    suffix_output = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_merge_state(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
    elif provider == "sglang":
        fn = lambda: sglang_merge_state(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Test correctness first
    print("Testing correctness...")
    calculate_diff(num_tokens=512, num_heads=16, head_size=128, dtype=torch.float16)
    calculate_diff(num_tokens=512, num_heads=16, head_size=128, dtype=torch.bfloat16)
    print()

    # Run benchmarks
    print("Running benchmarks...")
    benchmark.run(print_data=True)
