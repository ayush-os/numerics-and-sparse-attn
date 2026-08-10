# Phase 3 — dense decode attention reference, comparison target (b).
#
# Not a fork of a published kernel — after checking (see notes.md/handoff
# discussion), both real candidates were a bad fit for a lightweight,
# verifiable reference: the Triton docs tutorial kernel doesn't support GQA
# (it requires Q/K/V to share the same head count, and this project's "dense"
# baseline is dense-with-GQA per decode_notes.md, so feeding it plain MHA
# shapes would silently benchmark a heavier, wrong baseline), and vLLM's
# triton_unified_attention.py — while GQA-native and decode-shaped — carries
# ~1,600 lines across itself and its helper module (alibi, softcap, qq_bias,
# paging, quantization, a hardware-specific tensor-descriptor load path,
# split-KV 2D/3D toggle), none of which this project needs, and none of which
# could be verified here without a GPU. Simplifying that blind risked a
# silently-wrong "real" reference, which defeats the point of using one.
#
# So: a compact, purpose-built dense-causal-GQA decode kernel instead —
# informed by the GQA head-group indexing and online-softmax patterns in both
# real kernels above, not invented from nothing, but small enough to actually
# reason about correctness without hardware. Still needs the same treatment
# as the sparse kernel before trusting any benchmark numbers off it: diff
# against a PyTorch reference.
#
# Scope, deliberately: causal decode attention (seq_len_q=1) over the full
# context, real GQA (num_kv_heads groups), nothing else — no paging, no
# quantization, no alibi/softcap, no split-KV (same batch*heads=2048
# occupancy argument as the sparse kernel — see phase3_kernel_scaffold.py).
# Chunked BLOCK_N loop over KV since, unlike the sparse kernel's SINK+WINDOW,
# the full context (up to 163,840) can't be loaded in one block.

import torch
import triton
import triton.language as tl

BATCH = 32
N_HEADS = 64
N_KV_HEADS = 8
GQA_GROUP = N_HEADS // N_KV_HEADS
D_HEAD = 128
BLOCK_N = 128


@triton.jit
def _dense_decode_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_len, sm_scale,
    D_HEAD: tl.constexpr, GQA_GROUP: tl.constexpr,
    N_KV_HEADS: tl.constexpr, N_HEADS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program per (batch, query head) — same convention as the sparse
    # kernel: seq_len_q=1 gives nothing to query-tile over.
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    kv_head = pid_head // GQA_GROUP

    q_offset = pid_batch * N_HEADS * D_HEAD + pid_head * D_HEAD
    d_idx = tl.arange(0, D_HEAD)
    q = tl.load(Q_ptr + q_offset + d_idx)

    # K/V layout: [BATCH, N_KV_HEADS, seq_len, D_HEAD], same convention as
    # the sparse kernel's cache (see _load_kv_tile in phase3_kernel_scaffold.py).
    kv_base = pid_batch * N_KV_HEADS * seq_len * D_HEAD + kv_head * seq_len * D_HEAD

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D_HEAD], dtype=tl.float32)

    for start_n in tl.range(0, seq_len, BLOCK_N):
        n_idx = start_n + tl.arange(0, BLOCK_N)
        valid = n_idx < seq_len  # boundary mask for the last (possibly partial) tile

        kv_offset = kv_base + n_idx[:, None] * D_HEAD + d_idx[None, :]
        k = tl.load(K_ptr + kv_offset, mask=valid[:, None], other=0.0)
        v = tl.load(V_ptr + kv_offset, mask=valid[:, None], other=0.0)

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


def dense_decode_attention(q, k, v, seq_len, sm_scale=None):
    """q: [BATCH, N_HEADS, D_HEAD]. k/v: [BATCH, N_KV_HEADS, seq_len, D_HEAD]."""
    o = torch.empty_like(q)
    if sm_scale is None:
        sm_scale = D_HEAD**-0.5
    grid = (BATCH, N_HEADS)
    _dense_decode_attn_kernel[grid](
        q, k, v, o,
        seq_len, sm_scale,
        D_HEAD=D_HEAD, GQA_GROUP=GQA_GROUP,
        N_KV_HEADS=N_KV_HEADS, N_HEADS=N_HEADS, BLOCK_N=BLOCK_N,
    )
    return o
