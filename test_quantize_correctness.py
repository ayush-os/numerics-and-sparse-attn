# Correctness check for the KIVI quantize functions (quantize_k_cache,
# quantize_v_cache) -- comparison target: an independent reference
# unpack+dequant, written entirely in plain PyTorch. Doesn't reuse
# _dequantize_tile (doesn't exist yet, and shouldn't be reused even once it
# does -- an independent check written by the same source as the thing being
# checked isn't independent, same reasoning as test_sparse_correctness.py's
# mask sourcing).
#
# Requires a real GPU (quantize_k_cache/quantize_v_cache launch actual
# Triton kernels) -- can't run until you're on hardware, written now so it's
# ready the moment you are. This also exercises _quantize_k_kernel's and
# _quantize_v_kernel's own flagged-unverified spots (axis reductions, uint8
# cast, compile-time tile indexing) for the first time.

import torch

from phase3_kernel_scaffold import (
    D_HEAD, SINK_SIZE, GROUP_SIZE, QUANT_BITS,
    quantize_k_cache, quantize_v_cache,
)

DEVICE = "cuda"


def _reference_unpack_dequant(packed, scale, zero_point, full_axis_len, pack_axis,
                              group_size=GROUP_SIZE, bits=QUANT_BITS):
    """Independent reference: unpack (shift+mask) + affine dequant, plain
    PyTorch, no packing cleverness. `pack_axis` is the tensor axis that was
    packed/grouped -- 2 (seq_len) for K's layout, 3 (D_HEAD) for V's --
    everything else is generic between the two.
    """
    pack_factor = 8 // bits
    qmax = (1 << bits) - 1

    shape = list(packed.shape)
    shape[pack_axis] = full_axis_len
    codes = torch.zeros(shape, dtype=torch.float32, device=packed.device)
    packed_int = packed.to(torch.int32)

    for j in range(pack_factor):
        code_j = (packed_int >> (j * bits)) & qmax
        idx = [slice(None)] * len(shape)
        idx[pack_axis] = slice(j, None, pack_factor)
        codes[tuple(idx)] = code_j.float()

    group_idx = torch.arange(full_axis_len, device=packed.device) // group_size
    idx = [slice(None)] * len(shape)
    idx[pack_axis] = group_idx
    scale_expanded = scale[tuple(idx)].float()
    zero_expanded = zero_point[tuple(idx)].float()

    return scale_expanded * (codes - zero_expanded)


def _tolerance(x, group_size, group_dim):
    """Loose, data-dependent tolerance: roughly one quantization step for
    2-bit (range/3), with slack for the zero-point-rounding error the
    worked example in conversation already showed is real and expected."""
    shape = list(x.shape)
    n_groups = shape[group_dim] // group_size
    new_shape = shape[:group_dim] + [n_groups, group_size] + shape[group_dim + 1:]
    grouped = x.reshape(new_shape)
    group_range = (grouped.amax(dim=group_dim + 1) - grouped.amin(dim=group_dim + 1)).max().item()
    return group_range / 2 + 1e-3


def test_quantize_k_roundtrip():
    torch.manual_seed(0)
    kv_heads = 1

    all_passed = True
    for seq_len in [32, 256, 8192]:
        k = torch.randn(1, kv_heads, seq_len, D_HEAD, dtype=torch.float32, device=DEVICE)
        packed, scale, zero_point = quantize_k_cache(k)

        recon = _reference_unpack_dequant(packed, scale, zero_point, seq_len, pack_axis=2)
        max_diff = (recon - k).abs().max().item()
        tol = _tolerance(k, GROUP_SIZE, group_dim=2)
        passed = max_diff < tol
        all_passed &= passed
        print(f"K roundtrip seq_len={seq_len:6d}  max_abs_diff={max_diff:.5f}  tol={tol:.5f}  "
             f"{'PASS' if passed else 'FAIL'}")

    if not all_passed:
        raise AssertionError("quantize_k_cache round-trip diverged from reference -- see above")


def test_quantize_v_roundtrip():
    torch.manual_seed(1)
    kv_heads = 1

    all_passed = True
    for seq_len in [32, 256, 8192]:
        v = torch.randn(1, kv_heads, seq_len, D_HEAD, dtype=torch.float32, device=DEVICE)
        packed, scale, zero_point = quantize_v_cache(v)

        recon = _reference_unpack_dequant(packed, scale, zero_point, D_HEAD, pack_axis=3)
        max_diff = (recon - v).abs().max().item()
        tol = _tolerance(v, GROUP_SIZE, group_dim=3)
        passed = max_diff < tol
        all_passed &= passed
        print(f"V roundtrip seq_len={seq_len:6d}  max_abs_diff={max_diff:.5f}  tol={tol:.5f}  "
             f"{'PASS' if passed else 'FAIL'}")

    if not all_passed:
        raise AssertionError("quantize_v_cache round-trip diverged from reference -- see above")


def test_sink_padding_boundary():
    """The boundary flagged in conversation: SINK_SIZE=4 real tokens padded
    up to one full GROUP_SIZE=32 group (by repeating the first real token,
    per the padding recommendation), since quantize_k_cache assumes seq_len
    is a clean multiple of group_size. Checks the padding doesn't corrupt
    reconstruction of the 4 real values -- the actual risk, not a shape
    smoke test."""
    torch.manual_seed(2)
    kv_heads = 1

    sink = torch.randn(1, kv_heads, SINK_SIZE, D_HEAD, dtype=torch.float32, device=DEVICE)
    pad = sink[:, :, :1, :].expand(-1, -1, GROUP_SIZE - SINK_SIZE, -1)
    padded_sink = torch.cat([sink, pad], dim=2)   # [1, kv_heads, GROUP_SIZE, D_HEAD]

    packed, scale, zero_point = quantize_k_cache(padded_sink)
    recon = _reference_unpack_dequant(packed, scale, zero_point, GROUP_SIZE, pack_axis=2)

    real_recon = recon[:, :, :SINK_SIZE, :]
    max_diff = (real_recon - sink).abs().max().item()
    tol = _tolerance(sink, SINK_SIZE, group_dim=2)   # tolerance from the real values' own range only
    passed = max_diff < tol
    print(f"Sink padding boundary  max_abs_diff={max_diff:.5f}  tol={tol:.5f}  "
         f"{'PASS' if passed else 'FAIL'}")

    if not passed:
        raise AssertionError("sink padding corrupted real-value reconstruction -- see above")


if __name__ == "__main__":
    test_quantize_k_roundtrip()
    test_quantize_v_roundtrip()
    test_sink_padding_boundary()
