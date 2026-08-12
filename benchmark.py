# Phase 3 benchmark harness — real wall-clock latency across the same
# context-length x window-size sweep as Phase 1's hand-derivation, compared
# against Phase 1/2's predictions. Two separate sweeps, kept independent
# (not merged into one function) so a bug in one code path can't be
# confused for the other: benchmark() compares dense vs. native sparse;
# benchmark_quantized() adds the quantized path on top, once its kernels
# were correctness-verified (test_quantize_correctness.py,
# test_dequantize_tile_isolated.py, test_quantized_attention_correctness.py).
#
# Results saved to results/benchmark_results.csv and
# results/benchmark_results_quantized.csv.

import csv

import torch
import triton

from constants import (
    BATCH, N_HEADS, N_KV_HEADS, D_HEAD, SINK_SIZE, RESIDUAL_SIZE,
    CONTEXT_LENGTHS, WINDOW_SIZES,
)
from dense_kernel import dense_decode_attention
from sparse_kernel import sparse_decode_attention
from quant_kernel import sparse_decode_attention_quantized, build_quantized_kv_cache

# Predicted bytes at fp16 (Q/O and K/V), matching what these kernels
# actually run — B_FIXED = Q+O bytes (fixed, doesn't depend on W or L),
# C = per-token K+V bytes across the whole batch. See notes.md's
# fp16-baseline addendum for the derivation.
B_FIXED_FP16 = 1_048_576
C_FP16 = 131_072


def benchmark():
    """(a) vs. (b): native sparse vs. dense, both fp16, across the full sweep."""
    sm_scale = D_HEAD**-0.5
    device = "cuda"

    results = []
    for L in CONTEXT_LENGTHS:
        for W in WINDOW_SIZES:
            q = torch.randn(BATCH, N_HEADS, D_HEAD, dtype=torch.float16, device=device)
            k_cache = torch.randn(BATCH, N_KV_HEADS, SINK_SIZE + W, D_HEAD, dtype=torch.float16, device=device)
            v_cache = torch.randn(BATCH, N_KV_HEADS, SINK_SIZE + W, D_HEAD, dtype=torch.float16, device=device)

            ms_sparse = triton.testing.do_bench(
                lambda: sparse_decode_attention(q, k_cache, v_cache, seq_len=L, window=W)
            )

            k_dense = torch.randn(BATCH, N_KV_HEADS, L, D_HEAD, dtype=torch.float16, device=device)
            v_dense = torch.randn(BATCH, N_KV_HEADS, L, D_HEAD, dtype=torch.float16, device=device)
            ms_dense = triton.testing.do_bench(
                lambda: dense_decode_attention(q, k_dense, v_dense, seq_len=L, sm_scale=sm_scale)
            )

            bytes_sparse_pred = B_FIXED_FP16 + C_FP16 * (W + SINK_SIZE)
            bytes_dense_pred = B_FIXED_FP16 + C_FP16 * L

            # Memory-bound regime (Phase 1: AI never clears the ridge), so
            # bytes ratio — not FLOPs ratio — is the right predicted proxy.
            predicted_ratio = bytes_dense_pred / bytes_sparse_pred
            measured_ratio = ms_dense / ms_sparse
            gap_pct = 100 * (measured_ratio - predicted_ratio) / predicted_ratio

            results.append({
                "L": L, "W": W,
                "ms_sparse": ms_sparse, "ms_dense": ms_dense,
                "measured_ratio": measured_ratio,
                "predicted_ratio": predicted_ratio,
                "gap_pct": gap_pct,
            })

    return results


def benchmark_quantized():
    """(a) vs. (b): quantized-sparse vs. native sparse (the marginal
    multiplier from adding quantization on top of sparsity) — dense
    re-timed here too for a self-contained, directly comparable set of
    measurements. WINDOW < RESIDUAL_SIZE skipped, matching
    build_quantized_kv_cache's own standing assumption."""
    sm_scale = D_HEAD**-0.5
    device = "cuda"

    results = []
    windows = [w for w in WINDOW_SIZES if w >= RESIDUAL_SIZE]
    for L in CONTEXT_LENGTHS:
        for W in windows:
            q = torch.randn(BATCH, N_HEADS, D_HEAD, dtype=torch.float16, device=device)

            k_compacted = torch.randn(BATCH, N_KV_HEADS, SINK_SIZE + W, D_HEAD, dtype=torch.float16, device=device)
            v_compacted = torch.randn(BATCH, N_KV_HEADS, SINK_SIZE + W, D_HEAD, dtype=torch.float16, device=device)

            ms_sparse = triton.testing.do_bench(
                lambda: sparse_decode_attention(q, k_compacted, v_compacted, seq_len=L, window=W)
            )

            k_dense = torch.randn(BATCH, N_KV_HEADS, L, D_HEAD, dtype=torch.float16, device=device)
            v_dense = torch.randn(BATCH, N_KV_HEADS, L, D_HEAD, dtype=torch.float16, device=device)
            ms_dense = triton.testing.do_bench(
                lambda: dense_decode_attention(q, k_dense, v_dense, seq_len=L, sm_scale=sm_scale)
            )

            # Quantize once outside the timed region (setup-time-only) --
            # only sparse_decode_attention_quantized itself is timed.
            cache = build_quantized_kv_cache(k_compacted, v_compacted)
            ms_quantized = triton.testing.do_bench(
                lambda: sparse_decode_attention_quantized(
                    q, cache["k_residual"], cache["v_residual"],
                    cache["k_sink_packed"], cache["k_sink_scale"], cache["k_sink_zero"],
                    cache["v_sink_packed"], cache["v_sink_scale"], cache["v_sink_zero"],
                    cache["k_win_packed"], cache["k_win_scale"], cache["k_win_zero"],
                    cache["v_win_packed"], cache["v_win_scale"], cache["v_win_zero"],
                    seq_len=L, window=W,
                )
            )

            # Predicted bytes from real allocated tensor sizes, not a
            # re-derived formula -- more direct/accurate than hand-deriving
            # the metadata-overhead accounting from scratch.
            quant_tensors = [
                cache["k_residual"], cache["v_residual"],
                cache["k_sink_packed"], cache["k_sink_scale"], cache["k_sink_zero"],
                cache["v_sink_packed"], cache["v_sink_scale"], cache["v_sink_zero"],
                cache["k_win_packed"], cache["k_win_scale"], cache["k_win_zero"],
                cache["v_win_packed"], cache["v_win_scale"], cache["v_win_zero"],
            ]
            bytes_quantized_pred = sum(t.numel() * t.element_size() for t in quant_tensors) \
                + 2 * q.numel() * q.element_size()   # + Q and O, same fixed term as B_FIXED_FP16

            bytes_sparse_pred = B_FIXED_FP16 + C_FP16 * (W + SINK_SIZE)

            # Marginal multiplier from adding quantization on top of
            # sparsity (sparse_native_bytes / quantized_bytes) -- the
            # ~1.76x fp16-baseline prediction from notes.md.
            predicted_marginal_multiplier = bytes_sparse_pred / bytes_quantized_pred
            measured_marginal_multiplier = ms_sparse / ms_quantized
            gap_pct = 100 * (measured_marginal_multiplier - predicted_marginal_multiplier) / predicted_marginal_multiplier

            results.append({
                "L": L, "W": W,
                "ms_dense": ms_dense, "ms_sparse": ms_sparse, "ms_quantized": ms_quantized,
                "bytes_sparse_pred": bytes_sparse_pred, "bytes_quantized_pred": bytes_quantized_pred,
                "predicted_marginal_multiplier": predicted_marginal_multiplier,
                "measured_marginal_multiplier": measured_marginal_multiplier,
                "gap_pct": gap_pct,
            })

    return results


def _print_and_save(results, path):
    header = list(results[0].keys())
    print(" ".join(f"{h:>12}" for h in header))
    for r in results:
        print(" ".join(f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12}" for v in r.values()))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {path}")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    _print_and_save(benchmark(), "results/benchmark_results.csv")
    _print_and_save(benchmark_quantized(), "results/benchmark_results_quantized.csv")
