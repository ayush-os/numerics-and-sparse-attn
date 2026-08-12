# Correctness check for the sparse sliding-window+sink decode kernel
# (comparison target (a) in benchmark.py). Run before benchmark.py.
#
# Reference mask sourced from mit-han-lab/streaming-llm's own kv_cache.py
# (StartRecentKVCache.__call__), not derived from the same reasoning as
# _kv_indices — an independent check written by the same source as the
# thing being checked isn't independent.

import torch

from constants import BATCH, N_HEADS, N_KV_HEADS, GQA_GROUP, D_HEAD, SINK_SIZE
from sparse_kernel import sparse_decode_attention

DEVICE = "cuda"
GARBAGE_VALUE = 1e4  # deliberately large, not random — a masking bug that
                      # lets a should-be-invalid slot leak through should
                      # produce an obviously wrong output, not one that
                      # coincidentally looks fine because garbage happened
                      # to be drawn from the same distribution as real data.


def _reference_sparse_attention(q, k_full, v_full, seq_len, window, sink):
    """q: [BATCH, N_HEADS, D_HEAD]. k_full/v_full: [BATCH, N_KV_HEADS, seq_len,
    D_HEAD] — the full, un-evicted context (reference doesn't need to
    simulate the FIFO cache, just mask directly over every real position).

    Mask derived from mit-han-lab/streaming-llm's own reference
    (streaming_llm/kv_cache.py, StartRecentKVCache.__call__) rather than
    ported from _kv_indices — a real third-party source, not either of
    our own derivations checking themselves. Its actual behavior: no
    eviction at all (full dense attention) until seq_len exceeds
    sink+window; only past that point does it switch to the disjoint
    sink-slice + recent-slice mask this project derived independently.
    """
    positions = torch.arange(seq_len, device=q.device)
    if seq_len <= sink + window:
        mask = torch.ones(seq_len, dtype=torch.bool, device=q.device)
    else:
        mask = (positions < sink) | (positions >= seq_len - window)

    k_full = k_full.repeat_interleave(GQA_GROUP, dim=1)  # [BATCH, N_HEADS, seq_len, D_HEAD]
    v_full = v_full.repeat_interleave(GQA_GROUP, dim=1)

    sm_scale = D_HEAD ** -0.5
    qk = torch.einsum("bhd,bhnd->bhn", q, k_full) * sm_scale
    qk = qk.masked_fill(~mask[None, None, :], float("-inf"))
    p = torch.softmax(qk, dim=-1)
    return torch.einsum("bhn,bhnd->bhd", p, v_full)


def _build_compacted_cache(k_full, v_full, seq_len, window, sink):
    """Build the [BATCH, N_KV_HEADS, sink+window, D_HEAD] cache the real
    kernel expects, per _load_kv_tile's layout: slot i<sink -> position i;
    slot i>=sink -> position (seq_len-window)+(i-sink). Slots whose logical
    position doesn't exist yet (ramp-up) get GARBAGE_VALUE instead of real
    data, so the correctness check actually exercises the masking path."""
    k_cache = torch.full((BATCH, N_KV_HEADS, sink + window, D_HEAD), GARBAGE_VALUE,
                         dtype=k_full.dtype, device=k_full.device)
    v_cache = torch.full_like(k_cache, GARBAGE_VALUE)

    n_sink_valid = min(sink, seq_len)
    k_cache[:, :, :n_sink_valid] = k_full[:, :, :n_sink_valid]
    v_cache[:, :, :n_sink_valid] = v_full[:, :, :n_sink_valid]

    window_start = seq_len - window  # may be negative during ramp-up
    for i in range(window):
        pos = window_start + i
        if 0 <= pos < seq_len:
            k_cache[:, :, sink + i] = k_full[:, :, pos]
            v_cache[:, :, sink + i] = v_full[:, :, pos]

    return k_cache, v_cache


def test_sparse_attention_correctness():
    torch.manual_seed(0)

    # Covers pre-sink, mid-ramp-up (cache not yet full, nothing evicted),
    # and steady-state (sink/window disjoint) — the three regimes _kv_indices
    # has to handle differently. Plus W=0 explicitly (SINK+WINDOW=4, the
    # tightest possible cache — never covered before this, and the kind of
    # boundary that's bitten every other part of this kernel so far).
    test_cases = [
        (2, 256),
        (100, 256),
        (5_000, 256),
        (2, 0),
        (10_000, 0),
    ]

    all_passed = True
    for seq_len, window in test_cases:
        q = torch.randn(BATCH, N_HEADS, D_HEAD, dtype=torch.float32, device=DEVICE)
        k_full = torch.randn(BATCH, N_KV_HEADS, seq_len, D_HEAD, dtype=torch.float32, device=DEVICE)
        v_full = torch.randn(BATCH, N_KV_HEADS, seq_len, D_HEAD, dtype=torch.float32, device=DEVICE)

        k_cache, v_cache = _build_compacted_cache(k_full, v_full, seq_len, window, SINK_SIZE)

        o_kernel = sparse_decode_attention(
            q.half(), k_cache.half(), v_cache.half(), seq_len=seq_len, window=window
        ).float()
        o_ref = _reference_sparse_attention(q, k_full, v_full, seq_len, window, SINK_SIZE)

        max_diff = (o_kernel - o_ref).abs().max().item()
        passed = max_diff < 1e-2  # loose tolerance — kernel runs fp16, reference fp32
        all_passed &= passed
        print(f"seq_len={seq_len:6d} window={window:5d}  max_abs_diff={max_diff:.6f}  "
             f"{'PASS' if passed else 'FAIL'}")

    if not all_passed:
        raise AssertionError("sparse_decode_attention diverged from the reference — see above")


if __name__ == "__main__":
    test_sparse_attention_correctness()
