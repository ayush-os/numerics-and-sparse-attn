# Isolated correctness check for _dequantize_k_tile/_dequantize_v_tile,
# bypassing _load_kv_tile_quantized's regime composition and masking
# entirely. Written after test_quantized_attention_correctness.py failed
# and a dtype fix in the residual branch of _load_kv_tile_quantized had
# zero effect on the result -- narrows down whether the real bug lives in
# the dequant kernels themselves or in the composition logic layered on
# top. Same debugging-surface-isolation principle as staging dense-then-
# sparse: a wrong-output bug can only be in one of two now-separable
# systems, not an ambiguous mix of both.
#
# Tiny wrapper kernels call _dequantize_k_tile/_dequantize_v_tile directly
# (they're @triton.jit functions, not callable from plain Python) and
# store the result, no regime logic involved -- just packed data in,
# dequantized values out, compared against _reference_unpack_dequant
# (already independently verified via test_quantize_correctness.py).

import torch
import triton
import triton.language as tl

from phase3_kernel_scaffold import (
    D_HEAD, GROUP_SIZE, QUANT_BITS,
    quantize_k_cache, quantize_v_cache,
    _dequantize_k_tile, _dequantize_v_tile,
)
from test_quantize_correctness import _reference_unpack_dequant

DEVICE = "cuda"


@triton.jit
def _debug_dequant_k_kernel(
    Packed_ptr, Scale_ptr, Zero_ptr, Out_ptr,
    tok_start, num_packed, num_groups,
    D_HEAD: tl.constexpr, N_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    out = _dequantize_k_tile(
        Packed_ptr, Scale_ptr, Zero_ptr, tok_start, num_packed, num_groups,
        D_HEAD, N_KV_HEADS, pid_batch, pid_kv_head, GROUP_SIZE, BITS, BLOCK_N,
    )
    d_idx = tl.arange(0, D_HEAD)
    tok_idx = tl.arange(0, BLOCK_N)
    out_base = (pid_batch * N_KV_HEADS + pid_kv_head) * BLOCK_N * D_HEAD
    tl.store(Out_ptr + out_base + tok_idx[:, None] * D_HEAD + d_idx[None, :], out)


@triton.jit
def _debug_dequant_v_kernel(
    Packed_ptr, Scale_ptr, Zero_ptr, Out_ptr,
    tok_start, seq_len,
    D_HEAD: tl.constexpr, N_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    out = _dequantize_v_tile(
        Packed_ptr, Scale_ptr, Zero_ptr, tok_start, seq_len,
        D_HEAD, N_KV_HEADS, pid_batch, pid_kv_head, GROUP_SIZE, BITS, BLOCK_N,
    )
    d_idx = tl.arange(0, D_HEAD)
    tok_idx = tl.arange(0, BLOCK_N)
    out_base = (pid_batch * N_KV_HEADS + pid_kv_head) * BLOCK_N * D_HEAD
    tl.store(Out_ptr + out_base + tok_idx[:, None] * D_HEAD + d_idx[None, :], out)


def test_dequantize_k_tile_isolated():
    torch.manual_seed(4)
    kv_heads = 1
    BLOCK_N = 128
    seq_len = 256   # multiple of BLOCK_N and GROUP_SIZE -- no clamping triggered,
                     # isolates the core dequant math from the boundary-clamp logic

    k = torch.randn(1, kv_heads, seq_len, D_HEAD, dtype=torch.float32, device=DEVICE)
    packed, scale, zero_point = quantize_k_cache(k)
    num_packed = packed.shape[2]
    num_groups = scale.shape[2]

    out = torch.empty(1, kv_heads, BLOCK_N, D_HEAD, dtype=torch.float32, device=DEVICE)
    grid = (1, kv_heads)
    _debug_dequant_k_kernel[grid](
        packed, scale, zero_point, out,
        0, num_packed, num_groups,
        D_HEAD=D_HEAD, N_KV_HEADS=kv_heads,
        GROUP_SIZE=GROUP_SIZE, BITS=QUANT_BITS, BLOCK_N=BLOCK_N,
    )

    ref = _reference_unpack_dequant(packed, scale, zero_point, seq_len, pack_axis=2)[:, :, :BLOCK_N, :]

    max_diff = (out - ref).abs().max().item()
    passed = max_diff < 1e-2
    print(f"K dequant tile isolated  max_abs_diff={max_diff:.6f}  {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError("_dequantize_k_tile diverged from reference in isolation")


def test_dequantize_v_tile_isolated():
    torch.manual_seed(5)
    kv_heads = 1
    BLOCK_N = 128
    seq_len = 256

    v = torch.randn(1, kv_heads, seq_len, D_HEAD, dtype=torch.float32, device=DEVICE)
    packed, scale, zero_point = quantize_v_cache(v)

    out = torch.empty(1, kv_heads, BLOCK_N, D_HEAD, dtype=torch.float32, device=DEVICE)
    grid = (1, kv_heads)
    _debug_dequant_v_kernel[grid](
        packed, scale, zero_point, out,
        0, seq_len,
        D_HEAD=D_HEAD, N_KV_HEADS=kv_heads,
        GROUP_SIZE=GROUP_SIZE, BITS=QUANT_BITS, BLOCK_N=BLOCK_N,
    )

    ref = _reference_unpack_dequant(packed, scale, zero_point, D_HEAD, pack_axis=3)[:, :, :BLOCK_N, :]

    max_diff = (out - ref).abs().max().item()
    passed = max_diff < 1e-2
    print(f"V dequant tile isolated  max_abs_diff={max_diff:.6f}  {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError("_dequantize_v_tile diverged from reference in isolation")


if __name__ == "__main__":
    test_dequantize_k_tile_isolated()
    test_dequantize_v_tile_isolated()
