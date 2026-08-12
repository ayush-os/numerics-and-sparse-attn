# Phase 3 — sparse decode attention, Triton kernel scaffold
#
# Staged per plan: get sliding-window + sink working end-to-end (correctness +
# benchmarked) at native precision first, then layer Phase 2's quantized cache
# (KIVI 2-bit main + FP16 residual, RESIDUAL_SIZE=128 — corrected from an
# earlier int8-residual call, see notes.md's fp16-baseline addendum) on top of
# a kernel that's already known-good. The native-precision kernel below this
# comment block is that known-good kernel, untouched. The quant layer lives in
# its own clearly-marked section further down.
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


# ---------------------------------------------------------------------------
# KIVI quant layer — 2-bit main cache + FP16 residual, on top of the kernel
# above. Per the scope call worked out in conversation: quantize-on-write is
# setup-time-only (plain PyTorch, not part of the timed per-decode-step
# benchmark — same treatment as k_cache/v_cache construction above); the
# read-side unpack+dequant has to be real, fast, in-kernel Triton, since
# that's what runs every decode step and is what Phase 2's headline finding
# (precision's marginal multiplier collapsing to ~1.76x once layered on
# sparsity, fp16-baseline-corrected) is actually about.
#
# Given below: the new kernel/launcher, mirroring the grid/online-softmax
# structure already verified above exactly — nothing new to reason about
# there, it's the same loop calling a different tile-loader.
# Yours: quantize_k_cache/quantize_v_cache (the per-channel(K)/per-token(V) +
# group-32 quant math), _dequantize_tile (bit-unpacking + the affine dequant
# formula — the new Triton skill), and _load_kv_tile_quantized (composing
# per-slot regime: residual passthrough vs. compressed dequant, given
# `slot_idx`/`valid` from the unchanged _kv_indices above — validity logic
# doesn't care about precision, reuse it as-is).
# ---------------------------------------------------------------------------

RESIDUAL_SIZE = 128   # KIVI's real default (notes.md) — most recent tokens stay fp16
GROUP_SIZE = 32       # KIVI's real group size (notes.md), both K and V
QUANT_BITS = 2        # KIVI's 2-bit main cache


@triton.jit
def _quantize_k_kernel(
    K_ptr, Packed_ptr, Scale_ptr, Zero_ptr,
    seq_len,
    D_HEAD: tl.constexpr, N_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr,
):
    # UNVERIFIED — no GPU access to confirm this actually compiles/runs.
    # One program per (batch, kv_head); loops over groups of GROUP_SIZE
    # consecutive tokens, vectorized across all D_HEAD channels at once
    # per group (per-channel: each channel gets its own min/max/scale/
    # zero-point, computed from that group's GROUP_SIZE tokens).
    pid_batch = tl.program_id(0)
    pid_kv_head = tl.program_id(1)

    QMAX: tl.constexpr = (1 << BITS) - 1          # 3 for 2-bit
    PACK_FACTOR: tl.constexpr = 8 // BITS          # 4 codes per byte, 2-bit
    BYTES_PER_GROUP: tl.constexpr = GROUP_SIZE // PACK_FACTOR

    num_groups = seq_len // GROUP_SIZE
    num_packed = seq_len // PACK_FACTOR

    d_idx = tl.arange(0, D_HEAD)
    tok_in_group = tl.arange(0, GROUP_SIZE)

    k_base = (pid_batch * N_KV_HEADS + pid_kv_head) * seq_len * D_HEAD
    packed_base = (pid_batch * N_KV_HEADS + pid_kv_head) * num_packed * D_HEAD
    meta_base = (pid_batch * N_KV_HEADS + pid_kv_head) * num_groups * D_HEAD

    for group in tl.range(0, num_groups):
        tok_idx = group * GROUP_SIZE + tok_in_group                  # (GROUP_SIZE,)
        offs = k_base + tok_idx[:, None] * D_HEAD + d_idx[None, :]   # (GROUP_SIZE, D_HEAD)
        x = tl.load(K_ptr + offs)

        # CHECK #1: tl.min/tl.max with axis=0 — reducing over the GROUP_SIZE
        # rows, leaving one value per channel (D_HEAD,). Confirm this axis
        # convention matches your Triton version before trusting it.
        x_min = tl.min(x, axis=0)
        x_max = tl.max(x, axis=0)
        scale = (x_max - x_min) / QMAX
        scale = tl.where(scale == 0, 1.0, scale)   # guard a constant (all-equal) group
        # tl.math.round doesn't exist in this Triton version (found on real
        # hardware -- tl.math.exp does, so it's specifically round that's
        # missing, not the whole module). floor(x+0.5) is a portable
        # round-half-up substitute; exact .5-boundary tie-breaking differs
        # slightly from Python's round-half-to-even, but that's a rare,
        # sub-LSB-scale difference, not worth chasing here.
        zero_point = tl.floor(-x_min / scale + 0.5)

        meta_off = meta_base + group * D_HEAD + d_idx
        tl.store(Scale_ptr + meta_off, scale)
        tl.store(Zero_ptr + meta_off, zero_point)

        # REAL BUG FOUND ON HARDWARE (was CHECK #3): indexing a row out of an
        # already-computed tensor with a non-Python-int index isn't
        # supported on this Triton version ("unsupported tensor index:
        # int32[]"). Fixed by never computing one big (GROUP_SIZE, D_HEAD)
        # quantized tensor and slicing it -- instead compute PACK_FACTOR
        # separate (BYTES_PER_GROUP, D_HEAD) tensors directly via fresh
        # strided loads from K_ptr, then combine with plain elementwise ops.
        # Same broadcast-only style already used in the dequant functions.
        # Slightly redundant re-loading of K from HBM; acceptable since this
        # kernel is setup-time-only, not perf-critical.
        byte_in_group = tl.arange(0, BYTES_PER_GROUP)
        # int32 accumulator, not uint8: Triton's type promotion on |=/<<
        # silently widens uint8 to int32, and a loop-carried variable's
        # type has to stay fixed across iterations -- found on hardware
        # ("Loop-carried variable packed has initial type uint8 but is
        # re-assigned to int32"). Cast down to uint8 once, at the store.
        packed = tl.zeros([BYTES_PER_GROUP, D_HEAD], dtype=tl.int32)
        for j in range(PACK_FACTOR):   # constexpr -> unrolled
            tok_idx_j = group * GROUP_SIZE + j + PACK_FACTOR * byte_in_group   # (BYTES_PER_GROUP,)
            offs_j = k_base + tok_idx_j[:, None] * D_HEAD + d_idx[None, :]     # (BYTES_PER_GROUP, D_HEAD)
            x_j = tl.load(K_ptr + offs_j)
            q_j = tl.floor(x_j / scale[None, :] + 0.5) + zero_point[None, :]
            q_j = tl.minimum(tl.maximum(q_j, 0.0), float(QMAX)).to(tl.int32)
            packed |= (q_j << (j * BITS))

        byte_idx = group * BYTES_PER_GROUP + byte_in_group   # (BYTES_PER_GROUP,)
        tl.store(Packed_ptr + packed_base + byte_idx[:, None] * D_HEAD + d_idx[None, :], packed.to(tl.uint8))


def quantize_k_cache(k, group_size=GROUP_SIZE, bits=QUANT_BITS):
    """Launch wrapper for _quantize_k_kernel. k: [BATCH, N_KV_HEADS, seq_len,
    D_HEAD]. Returns (packed_codes, scale, zero_point). Given (mechanical),
    but see _quantize_k_kernel's own header — the kernel it launches is the
    unverified part.
    """
    batch, n_kv_heads, seq_len, d_head = k.shape
    assert seq_len % group_size == 0, "this simple version assumes seq_len is a multiple of group_size"
    pack_factor = 8 // bits
    num_groups = seq_len // group_size
    num_packed = seq_len // pack_factor

    packed = torch.empty((batch, n_kv_heads, num_packed, d_head), dtype=torch.uint8, device=k.device)
    scale = torch.empty((batch, n_kv_heads, num_groups, d_head), dtype=torch.float16, device=k.device)
    zero_point = torch.empty((batch, n_kv_heads, num_groups, d_head), dtype=torch.float16, device=k.device)

    grid = (batch, n_kv_heads)
    _quantize_k_kernel[grid](
        k, packed, scale, zero_point,
        seq_len,
        D_HEAD=d_head, N_KV_HEADS=n_kv_heads,
        GROUP_SIZE=group_size, BITS=bits,
    )
    return packed, scale, zero_point


@triton.jit
def _quantize_v_kernel(
    V_ptr, Packed_ptr, Scale_ptr, Zero_ptr,
    seq_len,
    D_HEAD: tl.constexpr, N_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # UNVERIFIED, same caveats as _quantize_k_kernel. Mirror image: per-token
    # (vectorize over tokens, loop over channel-groups) instead of
    # per-channel (vectorize over channels, loop over token-groups).
    #
    # Real structural difference from _quantize_k_kernel, not a pure
    # axis-swap: D_HEAD (128) is small enough to fully vectorize in one
    # tl.arange, but seq_len is a *runtime* value, up to 163,840 -- can't
    # tl.arange(0, seq_len) directly (same constraint that forced
    # BLOCK_N-tiling in _sparse_decode_attn_kernel). So the token axis still
    # needs its own BLOCK_N-tiled loop here, nested inside the (small,
    # constexpr, 4-iteration) channel-group loop.
    pid_batch = tl.program_id(0)
    pid_kv_head = tl.program_id(1)

    QMAX: tl.constexpr = (1 << BITS) - 1
    PACK_FACTOR: tl.constexpr = 8 // BITS
    NUM_CHAN_GROUPS: tl.constexpr = D_HEAD // GROUP_SIZE        # 4, constexpr
    BYTES_PER_GROUP: tl.constexpr = GROUP_SIZE // PACK_FACTOR    # 8, constexpr
    D_PACKED: tl.constexpr = D_HEAD // PACK_FACTOR               # 32 -- packed-channel axis size

    tok_in_block = tl.arange(0, BLOCK_N)

    v_base = (pid_batch * N_KV_HEADS + pid_kv_head) * seq_len * D_HEAD
    packed_base = (pid_batch * N_KV_HEADS + pid_kv_head) * seq_len * D_PACKED
    meta_base = (pid_batch * N_KV_HEADS + pid_kv_head) * seq_len * NUM_CHAN_GROUPS

    for group in range(NUM_CHAN_GROUPS):    # constexpr (4) -> unrolled, not tl.range
        chan_idx = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)   # this group's 32 channels

        for seq_start in tl.range(0, seq_len, BLOCK_N):    # runtime -> must be tiled
            tok_idx = seq_start + tok_in_block
            valid = tok_idx < seq_len   # last tile may run past seq_len

            offs = v_base + tok_idx[:, None] * D_HEAD + chan_idx[None, :]   # (BLOCK_N, GROUP_SIZE)
            x = tl.load(V_ptr + offs, mask=valid[:, None], other=0.0)

            # CHECK #1: axis=1 reduction (collapsing the GROUP_SIZE=32
            # channel axis, keeping BLOCK_N tokens) -- mirror of
            # _quantize_k_kernel's axis=0 reduction. Verify against your
            # Triton version before trusting it.
            x_min = tl.min(x, axis=1)   # (BLOCK_N,) -- one min per token
            x_max = tl.max(x, axis=1)
            scale = (x_max - x_min) / QMAX
            scale = tl.where(scale == 0, 1.0, scale)   # guard a constant group
            # tl.math.round doesn't exist in this Triton version -- see
            # _quantize_k_kernel's identical fix for why floor(x+0.5) is
            # used instead.
            zero_point = tl.floor(-x_min / scale + 0.5)

            meta_off = meta_base + tok_idx * NUM_CHAN_GROUPS + group
            tl.store(Scale_ptr + meta_off, scale, mask=valid)
            tl.store(Zero_ptr + meta_off, zero_point, mask=valid)

            # REAL BUG FOUND ON HARDWARE (was CHECK #3, mirror of the same
            # fix in _quantize_k_kernel): indexing a column out of an
            # already-computed tensor with a non-Python-int index isn't
            # supported on this Triton version. Fixed the same way -- fresh
            # strided loads per j (along the channel axis this time, not
            # the token axis) instead of slicing one big computed tensor.
            # int32 accumulator, not uint8 -- same loop-carried-type fix as
            # _quantize_k_kernel, see that comment for why.
            byte_in_group = tl.arange(0, BYTES_PER_GROUP)
            packed = tl.zeros([BLOCK_N, BYTES_PER_GROUP], dtype=tl.int32)
            for j in range(PACK_FACTOR):    # constexpr -> unrolled
                chan_idx_j = group * GROUP_SIZE + j + PACK_FACTOR * byte_in_group   # (BYTES_PER_GROUP,)
                offs_j = v_base + tok_idx[:, None] * D_HEAD + chan_idx_j[None, :]   # (BLOCK_N, BYTES_PER_GROUP)
                x_j = tl.load(V_ptr + offs_j, mask=valid[:, None], other=0.0)
                q_j = tl.floor(x_j / scale[:, None] + 0.5) + zero_point[:, None]
                q_j = tl.minimum(tl.maximum(q_j, 0.0), float(QMAX)).to(tl.int32)
                packed |= (q_j << (j * BITS))

            packed_chan_idx = group * BYTES_PER_GROUP + byte_in_group   # (BYTES_PER_GROUP,)
            packed_off = packed_base + tok_idx[:, None] * D_PACKED + packed_chan_idx[None, :]
            tl.store(Packed_ptr + packed_off, packed.to(tl.uint8), mask=valid[:, None])


def quantize_v_cache(v, group_size=GROUP_SIZE, bits=QUANT_BITS):
    """Launch wrapper for _quantize_v_kernel. v: [BATCH, N_KV_HEADS, seq_len,
    D_HEAD]. Returns (packed_codes, scale, zero_point) -- note the shapes
    mirror-flip vs. quantize_k_cache: seq_len stays full-size here, D_HEAD
    shrinks (opposite of K, where D_HEAD stayed full and seq_len shrank).
    """
    batch, n_kv_heads, seq_len, d_head = v.shape
    assert d_head % group_size == 0, "this simple version assumes D_HEAD is a multiple of group_size"
    pack_factor = 8 // bits
    num_chan_groups = d_head // group_size
    d_packed = d_head // pack_factor

    packed = torch.empty((batch, n_kv_heads, seq_len, d_packed), dtype=torch.uint8, device=v.device)
    scale = torch.empty((batch, n_kv_heads, seq_len, num_chan_groups), dtype=torch.float16, device=v.device)
    zero_point = torch.empty((batch, n_kv_heads, seq_len, num_chan_groups), dtype=torch.float16, device=v.device)

    grid = (batch, n_kv_heads)
    _quantize_v_kernel[grid](
        v, packed, scale, zero_point,
        seq_len,
        D_HEAD=d_head, N_KV_HEADS=n_kv_heads,
        GROUP_SIZE=group_size, BITS=bits, BLOCK_N=BLOCK_N,
    )
    return packed, scale, zero_point


@triton.jit
def _dequantize_k_tile(
    Packed_ptr, Scale_ptr, Zero_ptr,
    tok_start, num_packed, num_groups,
    D_HEAD: tl.constexpr, N_KV_HEADS: tl.constexpr, idx_in_batch, kv_head_idx,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Dequantize one BLOCK_N-token tile of K's per-channel-quantized main
    cache, starting at local token position `tok_start` within whichever
    quantized sub-block (sink or window) this call is for — that mapping is
    _load_kv_tile_quantized's job, not this function's; this just consumes
    an already-computed local offset, same as _load_kv_tile takes slot_idx.
    `num_packed`/`num_groups` are that sub-block's own packed-codes/scale
    axis lengths, needed for correct (batch, kv_head) striding (see
    quantize_k_cache's layout — sink's and window's tensors have different
    sizes, so this can't be a fixed constant).

    UNVERIFIED, but leans on _load_kv_tile's already-hardware-verified
    broadcasting pattern rather than the row-indexing tricks in
    _quantize_k_kernel — higher confidence than that one, not zero risk.
    """
    PACK_FACTOR: tl.constexpr = 8 // BITS
    QMAX: tl.constexpr = (1 << BITS) - 1
    BYTES_PER_GROUP: tl.constexpr = GROUP_SIZE // PACK_FACTOR

    d_idx = tl.arange(0, D_HEAD)
    tok_in_tile = tl.arange(0, BLOCK_N)              # this tile's BLOCK_N local token positions

    # CHECK #1: tok_start is assumed a multiple of PACK_FACTOR (4) — true as
    # long as _load_kv_tile_quantized only ever starts tiles on group/byte
    # boundaries, which it should given BLOCK_N and GROUP_SIZE are both
    # powers of 2 — but worth confirming explicitly when you write that
    # caller, not assuming it silently holds.
    byte_idx = tok_start // PACK_FACTOR + tok_in_tile // PACK_FACTOR   # (BLOCK_N,)
    sub_pos = tok_in_tile % PACK_FACTOR                                 # (BLOCK_N,) — position within its byte

    # SAFETY FIX (added once _load_kv_tile_quantized started composing this
    # with other regimes): a physical BLOCK_N tile can span multiple regimes
    # (sink/window-old/residual), so this function may get called with a
    # tok_start where most of the BLOCK_N range doesn't actually belong to
    # this sub-block. Without clamping, byte_idx/group_idx would run past
    # num_packed/num_groups and load out of bounds — a real illegal-memory-
    # access risk, not just a "computes garbage that gets discarded" one.
    # The caller's tl.where discards these results anyway; clamping just
    # makes computing them safe in the first place.
    byte_idx = tl.minimum(byte_idx, num_packed - 1)

    packed_base = (idx_in_batch * N_KV_HEADS + kv_head_idx) * num_packed * D_HEAD
    meta_base = (idx_in_batch * N_KV_HEADS + kv_head_idx) * num_groups * D_HEAD

    byte_off = packed_base + byte_idx[:, None] * D_HEAD + d_idx[None, :]   # (BLOCK_N, D_HEAD)
    # CHECK #2: 4 tokens per byte means this load repeats the same byte up
    # to 4x across consecutive tok_in_tile values — correct, just not
    # bandwidth-optimal; fine for a correctness-first pass, same "get it
    # right, then fast" order your other kernels already followed.
    packed_byte = tl.load(Packed_ptr + byte_off).to(tl.int32)              # (BLOCK_N, D_HEAD)

    code = (packed_byte >> (sub_pos[:, None] * BITS)) & QMAX                # (BLOCK_N, D_HEAD)

    group_idx = tl.minimum(byte_idx // BYTES_PER_GROUP, num_groups - 1)     # (BLOCK_N,), also clamped
    meta_off = meta_base + group_idx[:, None] * D_HEAD + d_idx[None, :]     # (BLOCK_N, D_HEAD)
    # CHECK #3: scale/zero_point are stored fp16 (per the earlier
    # recommendation) — confirm tl.load here upcasts cleanly to fp32 for the
    # dequant arithmetic rather than staying fp16 and losing precision
    # mid-computation.
    scale = tl.load(Scale_ptr + meta_off).to(tl.float32)
    zero_point = tl.load(Zero_ptr + meta_off).to(tl.float32)

    return scale * (code.to(tl.float32) - zero_point)                      # (BLOCK_N, D_HEAD)


@triton.jit
def _dequantize_v_tile(
    Packed_ptr, Scale_ptr, Zero_ptr,
    tok_start, seq_len,
    D_HEAD: tl.constexpr, N_KV_HEADS: tl.constexpr, idx_in_batch, kv_head_idx,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Mirror of _dequantize_k_tile: V is per-token, grouped along D_HEAD.
    NOT a pure axis-swap — see the two flagged differences below and the
    chat for why. `seq_len` here is the relevant sub-block's own length
    (sink's or window's), used as a stride, not an axis-size — unlike K's
    version, no num_packed/num_groups params needed, see chat for why.
    UNVERIFIED, same confidence note as _dequantize_k_tile.
    """
    PACK_FACTOR: tl.constexpr = 8 // BITS
    QMAX: tl.constexpr = (1 << BITS) - 1
    D_PACKED: tl.constexpr = D_HEAD // PACK_FACTOR

    # D_HEAD is compile-time-known and small, so — unlike K's packed axis
    # (seq_len, runtime, tiled) — V's packed axis needs no tiling at all,
    # ever, regardless of which regime/sub-block this call is for.
    d_idx = tl.arange(0, D_HEAD)
    tok_in_tile = tl.arange(0, BLOCK_N)
    # tok_start used DIRECTLY here, no dividing by PACK_FACTOR — V's packed
    # tensor keeps full token resolution (only D_HEAD shrinks), opposite of
    # K's version where tok_start had to convert to a byte index.
    # SAFETY FIX, same reasoning as _dequantize_k_tile: clamp so an
    # out-of-regime call (this sub-block spans less than the full tile)
    # can't compute an out-of-bounds address. `seq_len` here is this
    # sub-block's own length, doubling as the clamp bound.
    tok_idx = tl.minimum(tok_start + tok_in_tile, seq_len - 1)

    packed_chan_idx = d_idx // PACK_FACTOR      # (D_HEAD,) -- which of the D_PACKED=32 packed slots
    sub_pos = d_idx % PACK_FACTOR                # (D_HEAD,)
    group_idx = d_idx // GROUP_SIZE              # (D_HEAD,) -- which of the 4 channel-groups

    packed_base = (idx_in_batch * N_KV_HEADS + kv_head_idx) * seq_len * D_PACKED
    meta_base = (idx_in_batch * N_KV_HEADS + kv_head_idx) * seq_len * (D_HEAD // GROUP_SIZE)

    byte_off = packed_base + tok_idx[:, None] * D_PACKED + packed_chan_idx[None, :]   # (BLOCK_N, D_HEAD)
    packed_byte = tl.load(Packed_ptr + byte_off).to(tl.int32)

    code = (packed_byte >> (sub_pos[None, :] * BITS)) & QMAX                          # (BLOCK_N, D_HEAD)

    meta_off = meta_base + tok_idx[:, None] * (D_HEAD // GROUP_SIZE) + group_idx[None, :]
    scale = tl.load(Scale_ptr + meta_off).to(tl.float32)
    zero_point = tl.load(Zero_ptr + meta_off).to(tl.float32)

    return scale * (code.to(tl.float32) - zero_point)                                 # (BLOCK_N, D_HEAD)


@triton.jit
def _load_kv_tile_quantized(
    K_cache_ptr, V_cache_ptr,                                     # residual (fp16), own small cache
    K_sink_packed_ptr, K_sink_scale_ptr, K_sink_zero_ptr,         # sink's own quantized K
    V_sink_packed_ptr, V_sink_scale_ptr, V_sink_zero_ptr,         # sink's own quantized V
    K_win_packed_ptr, K_win_scale_ptr, K_win_zero_ptr,            # window-old's own quantized K
    V_win_packed_ptr, V_win_scale_ptr, V_win_zero_ptr,            # window-old's own quantized V
    slot_start, slot_idx, valid,
    D_HEAD: tl.constexpr, idx_in_batch, kv_head_idx,
    window: tl.constexpr, sink: tl.constexpr, N_KV_HEADS: tl.constexpr,
    RESIDUAL_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Composes the three regimes (sink / window-old / window-new=residual)
    into one (BLOCK_N, D_HEAD) tile per K and V: compute a candidate value
    from all three sources for the whole tile (safe, thanks to the clamps
    just added to _dequantize_k_tile/_dequantize_v_tile), then select per
    slot with tl.where. slot_start is threaded through separately from
    slot_idx/valid -- needed to compute each regime's own local offset,
    which _kv_indices doesn't give you directly.

    Known limitation, matching your own standing scope call (not fixed
    here): WIN_OLD_LEN can be <=0 when window < RESIDUAL_SIZE (several
    WINDOW_SIZES sweep points), which breaks WIN_NUM_PACKED/WIN_NUM_GROUPS
    below (num_groups-1 goes negative, defeats the clamp). Same deprioritized
    edge case as before -- not handled, flagged instead.

    UNVERIFIED, composing two already-flagged-uncertain pieces.
    """
    PACK_FACTOR: tl.constexpr = 8 // BITS
    WIN_OLD_LEN: tl.constexpr = window - RESIDUAL_SIZE          # window-old's own seq_len
    WIN_OLD_END: tl.constexpr = sink + WIN_OLD_LEN               # physical slot: window-old ends, residual begins

    # Sink is always padded to exactly one GROUP_SIZE group (per the sink-
    # padding decision), so its shape is a fixed compile-time constant --
    # unlike window's, it never needs to be threaded through as a runtime
    # parameter at all.
    SINK_SEQ_LEN: tl.constexpr = GROUP_SIZE
    SINK_NUM_PACKED: tl.constexpr = GROUP_SIZE // PACK_FACTOR
    SINK_NUM_GROUPS: tl.constexpr = 1

    # Window-old's own shape IS runtime-varying across the sweep (different
    # W per benchmark point) -- but since `window` is itself tl.constexpr
    # (fixed per kernel compilation, per _sparse_decode_attn_kernel_quantized's
    # own launch convention), these are still compile-time constants *within*
    # one compiled kernel variant, just not shared across variants. Confirmed
    # exact (no remainder) per the earlier dominance-ratio math: WINDOW and
    # RESIDUAL_SIZE are both multiples of GROUP_SIZE.
    WIN_NUM_PACKED: tl.constexpr = WIN_OLD_LEN // PACK_FACTOR
    WIN_NUM_GROUPS: tl.constexpr = WIN_OLD_LEN // GROUP_SIZE

    d_idx = tl.arange(0, D_HEAD)

    is_sink = slot_idx < sink
    is_old = (slot_idx >= sink) & (slot_idx < WIN_OLD_END)
    # is_residual is implicit -- the else-branch of the final tl.where below.

    # Local offsets, one per regime, clamped non-negative (upper-bound
    # clamping happens inside the dequant functions themselves now).
    sink_tok_start = slot_start                       # sink's own numbering == physical slot directly
    win_tok_start = tl.maximum(slot_start - sink, 0)

    k_sink = _dequantize_k_tile(
        K_sink_packed_ptr, K_sink_scale_ptr, K_sink_zero_ptr,
        sink_tok_start, SINK_NUM_PACKED, SINK_NUM_GROUPS,
        D_HEAD, N_KV_HEADS, idx_in_batch, kv_head_idx,
        GROUP_SIZE, BITS, BLOCK_N,
    )
    v_sink = _dequantize_v_tile(
        V_sink_packed_ptr, V_sink_scale_ptr, V_sink_zero_ptr,
        sink_tok_start, SINK_SEQ_LEN,
        D_HEAD, N_KV_HEADS, idx_in_batch, kv_head_idx,
        GROUP_SIZE, BITS, BLOCK_N,
    )

    k_old = _dequantize_k_tile(
        K_win_packed_ptr, K_win_scale_ptr, K_win_zero_ptr,
        win_tok_start, WIN_NUM_PACKED, WIN_NUM_GROUPS,
        D_HEAD, N_KV_HEADS, idx_in_batch, kv_head_idx,
        GROUP_SIZE, BITS, BLOCK_N,
    )
    v_old = _dequantize_v_tile(
        V_win_packed_ptr, V_win_scale_ptr, V_win_zero_ptr,
        win_tok_start, WIN_OLD_LEN,
        D_HEAD, N_KV_HEADS, idx_in_batch, kv_head_idx,
        GROUP_SIZE, BITS, BLOCK_N,
    )

    # Residual: no packing, direct fp16 load -- own small [.., RESIDUAL_SIZE,
    # D_HEAD] cache, same load shape as _load_kv_tile, just against a
    # smaller, separately-allocated tensor with its own local numbering.
    res_local_slot = tl.maximum(slot_idx - WIN_OLD_END, 0)
    res_offset = (idx_in_batch * N_KV_HEADS * RESIDUAL_SIZE * D_HEAD) + \
                 (kv_head_idx * RESIDUAL_SIZE * D_HEAD + res_local_slot * D_HEAD)
    k_res = tl.load(K_cache_ptr + res_offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0)
    v_res = tl.load(V_cache_ptr + res_offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0)

    # Compute-broadly, select-narrowly: all three candidates were computed
    # safely for the whole tile above; this is the only place that decides
    # which one is real per slot.
    k = tl.where(is_sink[:, None], k_sink, tl.where(is_old[:, None], k_old, k_res))
    v = tl.where(is_sink[:, None], v_sink, tl.where(is_old[:, None], v_old, v_res))

    return k, v


@triton.jit
def _sparse_decode_attn_kernel_quantized(
    Q_ptr, K_cache_ptr, V_cache_ptr,
    K_sink_packed_ptr, K_sink_scale_ptr, K_sink_zero_ptr,
    V_sink_packed_ptr, V_sink_scale_ptr, V_sink_zero_ptr,
    K_win_packed_ptr, K_win_scale_ptr, K_win_zero_ptr,
    V_win_packed_ptr, V_win_scale_ptr, V_win_zero_ptr,
    O_ptr,
    seq_len, sm_scale,
    WINDOW: tl.constexpr, SINK: tl.constexpr,
    D_HEAD: tl.constexpr, GQA_GROUP: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr, BLOCK_N: tl.constexpr,
    RESIDUAL_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr, BITS: tl.constexpr,
):
    # Identical structure to _sparse_decode_attn_kernel above — same grid
    # convention, same online-softmax loop — just calling the quantized tile
    # loader instead. Given, nothing new here. Pointer list grew a lot since
    # this was first sketched — 14 cache-related pointers now instead of 8,
    # direct consequence of sink/window needing separate quantized tensors
    # (see chat) — real complexity, not accidental bloat.
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
        k, v = _load_kv_tile_quantized(
            K_cache_ptr, V_cache_ptr,
            K_sink_packed_ptr, K_sink_scale_ptr, K_sink_zero_ptr,
            V_sink_packed_ptr, V_sink_scale_ptr, V_sink_zero_ptr,
            K_win_packed_ptr, K_win_scale_ptr, K_win_zero_ptr,
            V_win_packed_ptr, V_win_scale_ptr, V_win_zero_ptr,
            slot_start, slot_idx, valid, D_HEAD, pid_batch, kv_head,
            WINDOW, SINK, N_KV_HEADS, RESIDUAL_SIZE, GROUP_SIZE, BITS, BLOCK_N,
        )

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


def sparse_decode_attention_quantized(
    q, k_cache, v_cache,
    k_sink_packed, k_sink_scale, k_sink_zero,
    v_sink_packed, v_sink_scale, v_sink_zero,
    k_win_packed, k_win_scale, k_win_zero,
    v_win_packed, v_win_scale, v_win_zero,
    seq_len, window, sink=SINK_SIZE, residual_size=RESIDUAL_SIZE,
    group_size=GROUP_SIZE, bits=QUANT_BITS, sm_scale=None,
):
    """Given: allocation + launch, identical pattern to sparse_decode_attention
    above. k_cache/v_cache here is the RESIDUAL cache specifically (own
    [.., RESIDUAL_SIZE, D_HEAD] shape) — not the full compacted cache like
    the native-precision version took."""
    o = torch.empty_like(q)
    if sm_scale is None:
        sm_scale = D_HEAD**-0.5
    grid = (BATCH, N_HEADS)
    _sparse_decode_attn_kernel_quantized[grid](
        q, k_cache, v_cache,
        k_sink_packed, k_sink_scale, k_sink_zero,
        v_sink_packed, v_sink_scale, v_sink_zero,
        k_win_packed, k_win_scale, k_win_zero,
        v_win_packed, v_win_scale, v_win_zero,
        o,
        seq_len, sm_scale,
        WINDOW=window, SINK=sink,
        D_HEAD=D_HEAD, GQA_GROUP=GQA_GROUP,
        N_HEADS=N_HEADS, N_KV_HEADS=N_KV_HEADS, BLOCK_N=BLOCK_N,
        RESIDUAL_SIZE=residual_size, GROUP_SIZE=group_size, BITS=bits,
    )
    return o


def build_quantized_kv_cache(k_compacted, v_compacted, residual_size=RESIDUAL_SIZE, sink=SINK_SIZE):
    """Given: slices a compacted [BATCH, N_KV_HEADS, SINK+WINDOW, D_HEAD] K/V
    cache into residual/sink/window-old and quantizes sink+window-old
    separately, matching _load_kv_tile_quantized's expected tensor layout.
    Every slicing/padding decision here was already derived in conversation
    -- pure plumbing, nothing new to decide.

    Assumes WINDOW >= RESIDUAL_SIZE (matches the standing, deliberately
    unhandled W<RESIDUAL_SIZE scope call) -- window_old's slice would be
    empty or invalid otherwise, not guarded against here.
    """
    k_residual = k_compacted[:, :, -residual_size:, :].contiguous()
    v_residual = v_compacted[:, :, -residual_size:, :].contiguous()

    # Sink padded to exactly GROUP_SIZE by repeating its own first real
    # token (keeps the padded group's min/max identical to the real 4
    # values' range, per the padding-value reasoning from earlier -- avoids
    # wasting precision on the real values to accommodate fake ones).
    k_sink_real = k_compacted[:, :, :sink, :]
    v_sink_real = v_compacted[:, :, :sink, :]
    pad = GROUP_SIZE - sink
    k_sink_padded = torch.cat([k_sink_real, k_sink_real[:, :, :1, :].expand(-1, -1, pad, -1)], dim=2).float().contiguous()
    v_sink_padded = torch.cat([v_sink_real, v_sink_real[:, :, :1, :].expand(-1, -1, pad, -1)], dim=2).float().contiguous()

    # .float() here: quantize_k_cache/quantize_v_cache were only ever
    # validated with fp32 input (test_quantize_correctness.py) -- this
    # cache is fp16 like everything else in Phase 3, so cast explicitly
    # rather than silently running the quantize kernels on untested-dtype
    # input. Found while working through the attention correctness test.
    k_window_old = k_compacted[:, :, sink:-residual_size, :].float().contiguous()
    v_window_old = v_compacted[:, :, sink:-residual_size, :].float().contiguous()

    k_sink_packed, k_sink_scale, k_sink_zero = quantize_k_cache(k_sink_padded)
    v_sink_packed, v_sink_scale, v_sink_zero = quantize_v_cache(v_sink_padded)
    k_win_packed, k_win_scale, k_win_zero = quantize_k_cache(k_window_old)
    v_win_packed, v_win_scale, v_win_zero = quantize_v_cache(v_window_old)

    return {
        "k_residual": k_residual, "v_residual": v_residual,
        "k_sink_packed": k_sink_packed, "k_sink_scale": k_sink_scale, "k_sink_zero": k_sink_zero,
        "v_sink_packed": v_sink_packed, "v_sink_scale": v_sink_scale, "v_sink_zero": v_sink_zero,
        "k_win_packed": k_win_packed, "k_win_scale": k_win_scale, "k_win_zero": k_win_zero,
        "v_win_packed": v_win_packed, "v_win_scale": v_win_scale, "v_win_zero": v_win_zero,
    }


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


def _print_and_save(results, path="benchmark_results.csv"):
    import csv

    header = ["L", "W", "ms_sparse", "ms_dense", "measured_ratio", "predicted_ratio", "gap_pct"]
    print(f"{'L':>7} {'W':>7} {'ms_sparse':>10} {'ms_dense':>10} "
         f"{'meas_ratio':>11} {'pred_ratio':>11} {'gap_%':>8}")
    for r in results:
        print(f"{r['L']:>7} {r['W']:>7} {r['ms_sparse']:>10.4f} {r['ms_dense']:>10.4f} "
             f"{r['measured_ratio']:>11.3f} {r['predicted_ratio']:>11.3f} {r['gap_pct']:>8.2f}")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {path}")


if __name__ == "__main__":
    _print_and_save(benchmark())
