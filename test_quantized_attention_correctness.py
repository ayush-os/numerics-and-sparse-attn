# Correctness check for the full quantized sparse decode attention path --
# _dequantize_k_tile, _dequantize_v_tile, _load_kv_tile_quantized, and
# _sparse_decode_attn_kernel_quantized together (not just the isolated
# quantize round-trip test_quantize_correctness.py already covers).
#
# Comparison target: an independent reference built by COMPOSING two
# already-independently-verified pieces, not re-deriving from scratch --
# _reference_unpack_dequant (already checks quantize_k_cache/quantize_v_cache
# independently) for the dequant step, and the same masked-softmax attention
# math test_sparse_correctness.py/test_dense_correctness.py already use
# (broadcast-based GQA, not repeat_interleave -- same ~1GiB-vs-~128GiB
# memory reasoning as test_dense_correctness.py, relevant here too at the
# larger window sizes).
#
# Tolerance is tight (~1e-2, matching the other two test files), not a loose
# quantization-noise bound: both reference and kernel operate on the SAME
# already-quantized packed/scale/zero data (built once via
# build_quantized_kv_cache, shared -- given infrastructure here, not what's
# under test), so this is checking whether the Triton dequant+attention math
# matches an independent reimplementation of that same math, not
# characterizing quantization's overall distortion.

import torch

from phase3_kernel_scaffold import (
    BATCH, N_HEADS, N_KV_HEADS, GQA_GROUP, D_HEAD, SINK_SIZE, RESIDUAL_SIZE, GROUP_SIZE,
    build_quantized_kv_cache, sparse_decode_attention_quantized,
)
from test_quantize_correctness import _reference_unpack_dequant

DEVICE = "cuda"


def _reference_quantized_attention(q, win_old_len, cache):
    """Reconstructs dequantized K/V (via _reference_unpack_dequant, the same
    independent reference already checking quantize_k_cache/quantize_v_cache
    elsewhere), reassembles sink/window-old/residual into one
    [BATCH, N_KV_HEADS, SINK+WINDOW, D_HEAD] array, then runs plain
    masked-softmax attention -- GQA via broadcast/einsum (not
    repeat_interleave), same memory reasoning as test_dense_correctness.py.
    """
    k_sink = _reference_unpack_dequant(cache["k_sink_packed"], cache["k_sink_scale"], cache["k_sink_zero"],
                                       GROUP_SIZE, pack_axis=2)[:, :, :SINK_SIZE, :]
    v_sink = _reference_unpack_dequant(cache["v_sink_packed"], cache["v_sink_scale"], cache["v_sink_zero"],
                                       D_HEAD, pack_axis=3)[:, :, :SINK_SIZE, :]

    k_win = _reference_unpack_dequant(cache["k_win_packed"], cache["k_win_scale"], cache["k_win_zero"],
                                      win_old_len, pack_axis=2)
    v_win = _reference_unpack_dequant(cache["v_win_packed"], cache["v_win_scale"], cache["v_win_zero"],
                                      D_HEAD, pack_axis=3)

    k_recon = torch.cat([k_sink, k_win, cache["k_residual"].float()], dim=2)   # [B, KV_H, total, D]
    v_recon = torch.cat([v_sink, v_win, cache["v_residual"].float()], dim=2)

    B, H, D = q.shape
    _, KV_H, N, _ = k_recon.shape
    group = H // KV_H
    q_grouped = q.float().view(B, KV_H, group, D)
    sm_scale = D_HEAD ** -0.5
    qk = torch.einsum("bkgd,bknd->bkgn", q_grouped, k_recon) * sm_scale
    p = torch.softmax(qk, dim=-1)
    o = torch.einsum("bkgn,bknd->bkgd", p, v_recon)
    return o.reshape(B, H, D)


def test_quantized_attention_correctness():
    torch.manual_seed(3)

    # Steady-state only, matching build_quantized_kv_cache's own stated
    # assumption (WINDOW >= RESIDUAL_SIZE). Ramp-up x quantization
    # interaction and the WINDOW < RESIDUAL_SIZE case both stay
    # deliberately unhandled here -- same standing scope calls as before,
    # not silently dropped.
    windows = [256, 512, 8192]

    all_passed = True
    for window in windows:
        seq_len = SINK_SIZE + window + 1000   # comfortably past ramp-up

        q = torch.randn(BATCH, N_HEADS, D_HEAD, dtype=torch.float16, device=DEVICE)
        k_compacted = torch.randn(BATCH, N_KV_HEADS, SINK_SIZE + window, D_HEAD,
                                  dtype=torch.float16, device=DEVICE)
        v_compacted = torch.randn(BATCH, N_KV_HEADS, SINK_SIZE + window, D_HEAD,
                                  dtype=torch.float16, device=DEVICE)

        cache = build_quantized_kv_cache(k_compacted, v_compacted)

        o_kernel = sparse_decode_attention_quantized(
            q, cache["k_residual"], cache["v_residual"],
            cache["k_sink_packed"], cache["k_sink_scale"], cache["k_sink_zero"],
            cache["v_sink_packed"], cache["v_sink_scale"], cache["v_sink_zero"],
            cache["k_win_packed"], cache["k_win_scale"], cache["k_win_zero"],
            cache["v_win_packed"], cache["v_win_scale"], cache["v_win_zero"],
            seq_len=seq_len, window=window,
        ).float()

        o_ref = _reference_quantized_attention(q, window - RESIDUAL_SIZE, cache)

        max_diff = (o_kernel - o_ref).abs().max().item()
        passed = max_diff < 1e-2   # tight -- same-input comparison, not a quantization-noise bound
        all_passed &= passed
        print(f"window={window:6d}  max_abs_diff={max_diff:.6f}  {'PASS' if passed else 'FAIL'}")

    if not all_passed:
        raise AssertionError("quantized attention diverged from reference -- see above")


if __name__ == "__main__":
    test_quantized_attention_correctness()
