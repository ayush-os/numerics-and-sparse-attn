# Sparse sliding-window + attention-sink decode attention (Decision 1,
# notes.md Phase 0) — native fp16 precision, no quantization. Comparison
# target (a)/(b) baseline for the quant layer in quant_kernel.py, which
# reuses _kv_indices unchanged (validity logic doesn't care about precision).
# Correctness-verified against a reference sourced from mit-han-lab/
# streaming-llm's own kv_cache.py (test_sparse_correctness.py).

import torch
import triton
import triton.language as tl

from constants import BATCH, N_HEADS, N_KV_HEADS, GQA_GROUP, D_HEAD, SINK_SIZE, BLOCK_N


@triton.jit
def _kv_indices(seq_len, slot_start, WINDOW: tl.constexpr, SINK: tl.constexpr, BLOCK_N: tl.constexpr):
    """Physical cache slot numbers for one BLOCK_N-sized chunk starting at
    slot_start, plus a validity mask.

    Returns PHYSICAL slot numbers into the compacted [SINK+WINDOW] cache —
    NOT logical sequence positions (an earlier version conflated the two,
    causing a real illegal-memory-access on hardware — logical position is
    still used internally to decide validity, never to compute an address).

    Chunked by a fixed BLOCK_N (always a power of 2) rather than SINK+WINDOW
    itself (usually isn't one, e.g. 4+256=260) since tl.arange requires a
    power-of-2 bound; this also keeps SRAM footprint tile-bounded rather
    than context-bounded (spec.md's Phase 3 ask).

    Ramp-up (seq_len <= SINK+WINDOW, nothing evicted yet) is handled by
    masking: sink slots invalid once >= seq_len; window slots invalid if
    < SINK (avoids double-counting before the window's grown past it).
    """
    slot = slot_start + tl.arange(0, BLOCK_N)
    in_bounds = slot < (SINK + WINDOW)
    is_sink = slot < SINK

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
    SINK+WINDOW, D_HEAD] cache. `slot_idx` is always a valid address, but
    `valid` still gates the load itself: during ramp-up, some physical
    slots haven't been populated with a real token yet — their address is
    safe, their content isn't."""
    d_idx = tl.arange(0, D_HEAD)
    seq_len_kv = window + sink
    offset = (idx_in_batch * N_KV_HEADS * seq_len_kv * D_HEAD) + (kv_head_idx * seq_len_kv * D_HEAD + slot_idx * D_HEAD)

    k = tl.load(K_cache_ptr + offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0)
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
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    kv_head = pid_head // GQA_GROUP

    q_offset = pid_batch * N_HEADS * D_HEAD + pid_head * D_HEAD
    d_idx = tl.arange(0, D_HEAD)
    q = tl.load(Q_ptr + q_offset + d_idx)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D_HEAD], dtype=tl.float32)

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
    """k_cache/v_cache must be [BATCH, N_KV_HEADS, SINK+WINDOW, D_HEAD]."""
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
