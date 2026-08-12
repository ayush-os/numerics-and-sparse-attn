# Dense causal-GQA decode attention — comparison target (b) for Phase 3
# (spec.md). Not forked from a published kernel: the Triton docs tutorial
# doesn't support GQA, and vLLM's triton_unified_attention.py, while
# GQA-native, carries ~1,600 lines of unrelated functionality (paging,
# quantization, alibi/softcap) that couldn't be verified without a GPU —
# simplifying that blind risked a silently-wrong "real" reference, which
# defeats the point of using one. Built compact and purpose-built instead,
# informed by both kernels' GQA-indexing and online-softmax patterns.
#
# Scope: causal decode attention (seq_len_q=1) over the full context, real
# GQA, nothing else — no paging, no quantization, no alibi/softcap.
# Correctness-verified against a PyTorch reference (test_dense_correctness.py).

import torch
import triton
import triton.language as tl

from constants import BATCH, N_HEADS, N_KV_HEADS, GQA_GROUP, D_HEAD, BLOCK_N


@triton.jit
def _dense_decode_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_len: tl.int64, sm_scale,
    D_HEAD: tl.constexpr, GQA_GROUP: tl.constexpr,
    N_KV_HEADS: tl.constexpr, N_HEADS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program per (batch, query head) — decode's seq_len_q=1 gives
    # nothing to query-tile over. seq_len is int64: at seq_len>~67,653 the
    # offset math below overflows Triton's default int32 arithmetic and
    # silently wraps into a garbage address (a real bug found on hardware).
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    kv_head = pid_head // GQA_GROUP

    q_offset = pid_batch * N_HEADS * D_HEAD + pid_head * D_HEAD
    d_idx = tl.arange(0, D_HEAD)
    q = tl.load(Q_ptr + q_offset + d_idx)

    kv_base = pid_batch * N_KV_HEADS * seq_len * D_HEAD + kv_head * seq_len * D_HEAD

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D_HEAD], dtype=tl.float32)

    # BLOCK_N-tiled online softmax: the full context (up to 163,840) can't
    # be loaded in one block.
    for start_n in tl.range(0, seq_len, BLOCK_N):
        n_idx = start_n + tl.arange(0, BLOCK_N)
        valid = n_idx < seq_len  # boundary mask for the last, possibly partial, tile

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
