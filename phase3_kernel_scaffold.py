# Phase 3 — sparse decode attention, Triton kernel scaffold
#
# Staged per plan: get sliding-window + sink working end-to-end (correctness +
# benchmarked) at native precision first, then layer Phase 2's quantized cache
# (KIVI 2-bit main + int8 residual, RESIDUAL_SIZE=128) on top of a kernel
# that's already known-good. Quant is deliberately not in this file yet.
#
# Given: grid/launch mechanics, online-softmax bookkeeping, benchmark harness —
# standard flash-attention infra, not the object of study here.
# Yours: the stubs below (Decision 1's window+sink mechanism). Your own
# notes.md already has the derivation (ramp-up transient, non-contiguity) —
# work from that, not from this file.
#
# Dense baseline (comparison target (b), spec.md Phase 3) is
# dense_decode_reference.py — a compact, purpose-built dense-causal-GQA
# kernel, not the Triton docs tutorial (that kernel doesn't support GQA;
# see dense_decode_reference.py's header for why it was dropped).

import torch
import triton
import triton.language as tl

BATCH = 32
N_HEADS = 64
N_KV_HEADS = 8
GQA_GROUP = N_HEADS // N_KV_HEADS
D_HEAD = 128
SINK_SIZE = 4
BLOCK_N = 128

CONTEXT_LENGTHS = [8_192, 16_384, 32_768, 65_536, 131_072, 163_840]
WINDOW_SIZES = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


@triton.jit
def _kv_indices(seq_len, slot_start, WINDOW: tl.constexpr, SINK: tl.constexpr, BLOCK_N: tl.constexpr):
    """Physical cache slot numbers for one BLOCK_N-sized chunk starting at
    slot_start (0, BLOCK_N, 2*BLOCK_N, ...), plus a validity mask.

    Returns PHYSICAL slot numbers into the compacted [SINK+WINDOW] cache —
    NOT logical sequence positions. The compacted cache is already laid out
    in slot order (slot i holds whatever token is currently kept there), so
    _load_kv_tile can use this directly as an offset. An earlier version of
    this function returned the logical position instead (e.g. ~8000 for a
    late window slot at seq_len=8192) and _load_kv_tile multiplied that
    directly into the offset math — reaching far past the actual buffer and
    causing a real illegal-memory-access on hardware. Logical position is
    still needed, but only internally, to decide whether a physical slot's
    content is valid yet — never to compute an address.

    BLOCK_N is fixed (a real constant, always a power of 2) rather than
    SINK+WINDOW itself, which usually isn't one (e.g. 4+256=260) and would
    violate tl.arange's power-of-2 requirement directly. Chunking by a fixed
    BLOCK_N sidesteps that entirely, and also avoids materializing the whole
    (SINK+WINDOW, D_HEAD) tile at once, which hits Triton's max single-tensor
    size at large WINDOW (spec.md's SRAM-footprint-should-be-tile-bounded
    concern, deferred from the single-block version — this is that follow-up).
    The last chunk naturally runs past SINK+WINDOW when it doesn't divide
    evenly by BLOCK_N; in_bounds masks that off the same way the ramp-up
    transient is masked below, not a separate special case.

    Real (in-bounds) slots: ramp-up (seq_len <= SINK+WINDOW, nothing evicted
    yet) is handled by masking:
      - sink slots invalid once >= seq_len (not generated yet)
      - window slots invalid if < SINK (already covered by the sink block —
        avoids double-counting before the window's grown past it)
    Steady state (seq_len > SINK+WINDOW) makes everything valid.
    """
    slot = slot_start + tl.arange(0, BLOCK_N)
    in_bounds = slot < (SINK + WINDOW)
    is_sink = slot < SINK

    # Logical position this physical slot currently holds — used only to
    # decide validity below, never returned as an address.
    window_logical_pos = (seq_len - WINDOW) + (slot - SINK)

    sink_valid = slot < seq_len
    window_valid = window_logical_pos >= SINK
    valid = in_bounds & tl.where(is_sink, sink_valid, window_valid)

    addr_slot = tl.where(in_bounds, slot, 0)

    return addr_slot, valid


@triton.jit
def _load_kv_tile(K_cache_ptr, V_cache_ptr, slot_idx, valid, D_HEAD: tl.constexpr, idx_in_batch, kv_head_idx,
                  window: tl.constexpr, sink: tl.constexpr, N_KV_HEADS: tl.constexpr):
    """Load a KV tile at these cache slots, from a [BATCH, N_KV_HEADS,
    SINK+WINDOW, D_HEAD] cache. `slot_idx` is a physical cache offset (see
    _kv_indices), always a valid address — but `valid` still has to gate the
    load itself (not just the softmax weights), since during the ramp-up
    transient some physical slots haven't been populated with a real token
    yet; their address is safe to compute, their content isn't safe to use."""
    d_idx = tl.arange(0, D_HEAD)
    seq_len_kv = window + sink
    offset = (idx_in_batch * N_KV_HEADS * seq_len_kv * D_HEAD) + (kv_head_idx * seq_len_kv * D_HEAD + slot_idx * D_HEAD)

    k = tl.load(K_cache_ptr + offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0)

    # slot_idx: (BLOCK_N,) -> (BLOCK_N, D_HEAD), one address per
    # (slot, head-dim element) pair, matching the tile shape v needs to be.
    v = tl.load(V_cache_ptr + offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0)

    return k, v


@triton.jit
def _sparse_decode_attn_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
    seq_len, sm_scale,
    WINDOW: tl.constexpr, SINK: tl.constexpr,
    D_HEAD: tl.constexpr, GQA_GROUP: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program per (batch, query head) — no query-tiling, decode's
    # seq_len_q=1 gives nothing to tile over.
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    kv_head = pid_head // GQA_GROUP

    q_offset = pid_batch * N_HEADS * D_HEAD + pid_head * D_HEAD
    d_idx = tl.arange(0, D_HEAD)
    q = tl.load(Q_ptr + q_offset + d_idx)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D_HEAD], dtype=tl.float32)

    # Chunked over BLOCK_N-sized slices of the SINK+WINDOW cache, online-
    # softmax accumulation — see _kv_indices for why (tile-bounded SRAM
    # footprint, per spec.md's Phase 3 ask, and the max-single-tensor-size
    # limit this replaced the single-block version to fix).
    cache_size = SINK + WINDOW
    for slot_start in tl.range(0, cache_size, BLOCK_N):
        slot_idx, valid = _kv_indices(seq_len, slot_start, WINDOW, SINK, BLOCK_N)
        k, v = _load_kv_tile(K_cache_ptr, V_cache_ptr, slot_idx, valid, D_HEAD, pid_batch, kv_head, WINDOW, SINK, N_KV_HEADS)

        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(valid, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.math.exp(qk - m_ij)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_ij

    o = acc / l_i
    tl.store(O_ptr + q_offset + d_idx, o)


def sparse_decode_attention(q, k_cache, v_cache, seq_len, window, sink=SINK_SIZE, sm_scale=None):
    """Given: allocation + launch. k_cache/v_cache must be
    [BATCH, N_KV_HEADS, SINK+WINDOW, D_HEAD] to match _load_kv_tile."""
    o = torch.empty_like(q)
    if sm_scale is None:
        sm_scale = D_HEAD**-0.5
    grid = (BATCH, N_HEADS)
    _sparse_decode_attn_kernel[grid](
        q, k_cache, v_cache, o,
        seq_len, sm_scale,
        WINDOW=window, SINK=sink,
        D_HEAD=D_HEAD, GQA_GROUP=GQA_GROUP,
        N_HEADS=N_HEADS, N_KV_HEADS=N_KV_HEADS, BLOCK_N=BLOCK_N,
    )
    return o


def benchmark():
    # (b) dense baseline — dense_decode_reference.py, real GQA, no broadcast hack needed.
    from dense_decode_reference import dense_decode_attention
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

            # Predicted bytes at this stage's actual precision (fp16 Q/O + fp16
            # K/V, not notes.md's original int8 assumption — see B_FIXED_FP16/
            # C_FP16 derivation in the conversation, not re-derived here).
            # Dense substitutes L directly for (W+S), per Phase 1's own dense
            # sanity check (W+4=L recovers the dense case exactly).
            B_FIXED_FP16 = 1_048_576
            C_FP16 = 131_072
            bytes_sparse_pred = B_FIXED_FP16 + C_FP16 * (W + SINK_SIZE)
            bytes_dense_pred = B_FIXED_FP16 + C_FP16 * L

            # Memory-bound regime (Phase 1 Finding #3: AI never clears ridge),
            # so bytes ratio — not FLOPs ratio — is the right proxy for the
            # wall-clock ratio, matching Phase 2's own "sparsity-alone"
            # methodology (bytes ratios, not FLOPs ratios).
            predicted_ratio = bytes_dense_pred / bytes_sparse_pred
            measured_ratio = ms_dense / ms_sparse
            gap_pct = 100 * (measured_ratio - predicted_ratio) / predicted_ratio

            # Gap-hunting *why* gap_pct is nonzero is the actual Phase 3/4
            # content (kernel launch overhead, BLOCK_N tiling granularity,
            # boundary masking, etc.) — not resolved here.
            results.append({
                "L": L, "W": W,
                "ms_sparse": ms_sparse, "ms_dense": ms_dense,
                "measured_ratio": measured_ratio,
                "predicted_ratio": predicted_ratio,
                "gap_pct": gap_pct,
            })

    return results


if __name__ == "__main__":
    benchmark()
