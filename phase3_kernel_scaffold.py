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

CONTEXT_LENGTHS = [8_192, 16_384, 32_768, 65_536, 131_072, 163_840]
WINDOW_SIZES = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


@triton.jit
def _kv_indices(seq_len, WINDOW: tl.constexpr, SINK: tl.constexpr):
    """Cache-slot indices this query attends to, plus a validity mask.

    Fixed size (SINK+WINDOW) since Triton needs a static shape — ramp-up
    (seq_len <= SINK+WINDOW, nothing evicted yet) is handled by masking:
      - sink slots invalid once >= seq_len (not generated yet)
      - window slots invalid if < SINK (already covered by the sink block —
        avoids double-counting before the window's grown past it)
    Steady state (seq_len > SINK+WINDOW) makes everything valid — your
    original tl.cat(arange, arange) line, unchanged.
    """
    sink_idx = tl.arange(0, SINK)
    window_idx = (seq_len - WINDOW) + tl.arange(0, WINDOW)
    idx = tl.cat(sink_idx, window_idx)

    sink_valid = sink_idx < seq_len
    window_valid = window_idx >= SINK
    valid = tl.cat(sink_valid, window_valid)

    return idx, valid


@triton.jit
def _load_kv_tile(K_cache_ptr, V_cache_ptr, slot_idx, valid, D_HEAD: tl.constexpr, idx_in_batch, kv_head_idx,
                  window: tl.constexpr, sink: tl.constexpr):
    """Load a KV tile at these cache slots, from a [BATCH, N_KV_HEADS,
    SINK+WINDOW, D_HEAD] cache. `valid` marks which slots are real (see
    _kv_indices) — masked at load time (not just in the softmax weights),
    since invalid window slots can be negative addresses during the ramp-up
    transient and must never actually be read."""
    d_idx = tl.arange(0, D_HEAD)
    seq_len_kv = window + sink
    offset = (idx_in_batch * N_KV_HEADS * seq_len_kv * D_HEAD) + (kv_head_idx * seq_len_kv * D_HEAD + slot_idx * D_HEAD)

    k = tl.load(K_cache_ptr + offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0)

    # slot_idx: (SINK+WINDOW,) -> (SINK+WINDOW, D_HEAD), one address per
    # (slot, head-dim element) pair, matching the tile shape v needs to be.
    v = tl.load(V_cache_ptr + offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0)

    return k, v


@triton.jit
def _sparse_decode_attn_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
    seq_len, sm_scale,
    WINDOW: tl.constexpr, SINK: tl.constexpr,
    D_HEAD: tl.constexpr, GQA_GROUP: tl.constexpr,
):
    # One program per (batch, query head) — no query-tiling, decode's
    # seq_len_q=1 gives nothing to tile over.
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    kv_head = pid_head // GQA_GROUP

    q_offset = pid_batch * N_HEADS * D_HEAD + pid_head * D_HEAD
    d_idx = tl.arange(0, D_HEAD)
    q = tl.load(Q_ptr + q_offset + d_idx)

    # Fixed-size (SINK+WINDOW,) index + validity arrays — see _kv_indices.
    # NOTE: this loads the whole sink+window range in one shot, no BLOCK_N
    # tiling. Fine for getting things correct first, but spec.md's Phase 3
    # explicitly wants the kernel's SRAM footprint checked against real
    # FlashAttention's tile-bounded (not context-bounded) behavior — for
    # large WINDOW that likely means this needs to become a tl.range loop
    # over BLOCK_N-sized chunks of (idx, valid), with the running m_i/l_i
    # accumulation the dense tutorial kernel uses, rather than one big block.
    # Worth coming back to once this version is verified correct.
    slot_idx, valid = _kv_indices(seq_len, WINDOW, SINK)
    k, v = _load_kv_tile(K_cache_ptr, V_cache_ptr, slot_idx, valid, D_HEAD, pid_batch, kv_head, WINDOW, SINK)

    qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
    qk = tl.where(valid, qk, float("-inf"))
    m_i = tl.max(qk, axis=0)
    p = tl.math.exp(qk - m_i)
    p = tl.where(valid, p, 0.0)
    l_i = tl.sum(p, axis=0)
    acc = tl.sum(p[:, None] * v, axis=0)

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
    )
    return o
