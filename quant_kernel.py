# KIVI quantized KV cache on top of sparse_kernel.py's sliding-window+sink
# attention: 2-bit main cache (K per-channel, V per-token, group size 32)
# + fp16 residual for the most recent RESIDUAL_SIZE tokens (notes.md Phase 2).
#
# Three regimes make up the compacted [SINK+WINDOW] cache: sink (always
# quantized), window-old (quantized), and the most recent window tokens
# (residual, never quantized). Quantize-on-write (quantize_k_cache/
# quantize_v_cache) is setup-time-only, plain-PyTorch-callable; the
# read-side unpack+dequant (_dequantize_k_tile/_dequantize_v_tile) is fused
# into the attention kernel itself, since that's what runs every decode step.
#
# Real bugs found only on hardware while building this (full narrative and
# the two-round optimization story in notes.md's Phase 3 section):
#  - tl.math.round doesn't exist on this Triton version; floor(x+0.5) instead.
#  - Indexing a specific row/column out of an already-computed tensor with a
#    non-Python-int index isn't supported; fixed via fresh strided loads per
#    pack position instead of slicing one materialized tensor.
#  - Triton's |=/<< silently widens a uint8 loop-carried accumulator to
#    int32, which then conflicts with the loop's declared type; accumulate
#    in int32 throughout, cast to uint8 once at the store.
#  - A tile-level clamp (meant to keep addressing safe) corrupted real
#    in-range window-old positions in the first tile; clamping needs to
#    happen per-position, inside the dequant functions, not on the shared
#    tile-start offset.
#  - A runtime `if` meant to skip inapplicable regime computation per tile
#    measured ~20x SLOWER on hardware, not faster (likely predicated
#    execution and/or broken loop pipelining) — reverted in favor of
#    splitting the outer loop into compile-time-bounded segments instead
#    (see _sparse_decode_attn_kernel_quantized), which worked.

import torch
import triton
import triton.language as tl

from constants import (
    BATCH, N_HEADS, N_KV_HEADS, GQA_GROUP, D_HEAD, SINK_SIZE, BLOCK_N,
    RESIDUAL_SIZE, GROUP_SIZE, QUANT_BITS,
)
from sparse_kernel import _kv_indices


# --- Quantize-on-write (setup-time, one-shot per cache build) --------------

@triton.jit
def _quantize_k_kernel(
    K_ptr, Packed_ptr, Scale_ptr, Zero_ptr,
    seq_len,
    D_HEAD: tl.constexpr, N_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr,
):
    """Per-channel quantization: one program per (batch, kv_head), looping
    over groups of GROUP_SIZE consecutive tokens, vectorized across all
    D_HEAD channels per group (each channel gets its own min/max/scale/
    zero-point from that group's GROUP_SIZE tokens)."""
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

        x_min = tl.min(x, axis=0)
        x_max = tl.max(x, axis=0)
        scale = (x_max - x_min) / QMAX
        scale = tl.where(scale == 0, 1.0, scale)   # guard a constant (all-equal) group
        zero_point = tl.floor(-x_min / scale + 0.5)   # tl.math.round doesn't exist here

        meta_off = meta_base + group * D_HEAD + d_idx
        tl.store(Scale_ptr + meta_off, scale)
        tl.store(Zero_ptr + meta_off, zero_point)

        # Pack PACK_FACTOR codes/byte via fresh strided loads (not slicing a
        # materialized (GROUP_SIZE, D_HEAD) tensor — see module header).
        # int32 accumulator: Triton's |=/<< widens uint8 to int32 mid-loop,
        # which conflicts with a fixed loop-carried type; cast down once,
        # at the store.
        byte_in_group = tl.arange(0, BYTES_PER_GROUP)
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
    """k: [BATCH, N_KV_HEADS, seq_len, D_HEAD], fp32. Returns (packed_codes,
    scale, zero_point). Assumes seq_len is a multiple of group_size."""
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
    """Per-token quantization — mirror of _quantize_k_kernel, not a pure
    axis-swap: D_HEAD is small and compile-time-known, so it vectorizes in
    one shot, but seq_len is a runtime value up to 163,840 and still needs
    its own BLOCK_N-tiled loop, nested inside the small (4-iteration)
    channel-group loop."""
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

    for group in range(NUM_CHAN_GROUPS):    # constexpr -> unrolled, not tl.range
        chan_idx = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)   # this group's 32 channels

        for seq_start in tl.range(0, seq_len, BLOCK_N):    # runtime -> must be tiled
            tok_idx = seq_start + tok_in_block
            valid = tok_idx < seq_len   # last tile may run past seq_len

            offs = v_base + tok_idx[:, None] * D_HEAD + chan_idx[None, :]   # (BLOCK_N, GROUP_SIZE)
            x = tl.load(V_ptr + offs, mask=valid[:, None], other=0.0)

            x_min = tl.min(x, axis=1)   # (BLOCK_N,) -- one min per token
            x_max = tl.max(x, axis=1)
            scale = (x_max - x_min) / QMAX
            scale = tl.where(scale == 0, 1.0, scale)
            zero_point = tl.floor(-x_min / scale + 0.5)

            meta_off = meta_base + tok_idx * NUM_CHAN_GROUPS + group
            tl.store(Scale_ptr + meta_off, scale, mask=valid)
            tl.store(Zero_ptr + meta_off, zero_point, mask=valid)

            # Same fresh-strided-load packing pattern as _quantize_k_kernel,
            # along the channel axis this time instead of the token axis.
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
    """v: [BATCH, N_KV_HEADS, seq_len, D_HEAD], fp32. Returns (packed_codes,
    scale, zero_point) — shapes mirror-flip vs. quantize_k_cache: seq_len
    stays full-size, D_HEAD shrinks (opposite of K)."""
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


# --- Dequantize-on-read (fused into the attention kernel, every step) ------

@triton.jit
def _dequantize_k_tile(
    Packed_ptr, Scale_ptr, Zero_ptr,
    tok_start, num_packed, num_groups,
    D_HEAD: tl.constexpr, N_KV_HEADS: tl.constexpr, idx_in_batch, kv_head_idx,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Dequantize one BLOCK_N-token tile of K's per-channel-quantized cache,
    starting at local token position `tok_start` within whichever quantized
    sub-block (sink or window-old) this call is for — that mapping is the
    caller's job, not this function's. `num_packed`/`num_groups` are that
    sub-block's own axis lengths, needed for correct striding (sink's and
    window-old's tensors are different sizes).

    Fully vectorized via [:, None]/[None, :] broadcasting, no per-row
    indexing into a materialized tensor (see module header).
    """
    PACK_FACTOR: tl.constexpr = 8 // BITS
    QMAX: tl.constexpr = (1 << BITS) - 1
    BYTES_PER_GROUP: tl.constexpr = GROUP_SIZE // PACK_FACTOR

    d_idx = tl.arange(0, D_HEAD)
    tok_in_tile = tl.arange(0, BLOCK_N)

    byte_idx = tok_start // PACK_FACTOR + tok_in_tile // PACK_FACTOR   # (BLOCK_N,)
    sub_pos = tok_in_tile % PACK_FACTOR                                 # (BLOCK_N,) -- position within its byte

    # Clamp both directions: a tile can span multiple regimes upstream, so
    # tok_start may put some positions outside this sub-block's real range
    # (in either direction — tok_start can be negative too). Clamping keeps
    # every computed address safe; the caller discards whichever positions
    # don't actually belong to this regime.
    byte_idx = tl.maximum(tl.minimum(byte_idx, num_packed - 1), 0)

    packed_base = (idx_in_batch * N_KV_HEADS + kv_head_idx) * num_packed * D_HEAD
    meta_base = (idx_in_batch * N_KV_HEADS + kv_head_idx) * num_groups * D_HEAD

    byte_off = packed_base + byte_idx[:, None] * D_HEAD + d_idx[None, :]   # (BLOCK_N, D_HEAD)
    packed_byte = tl.load(Packed_ptr + byte_off).to(tl.int32)              # (BLOCK_N, D_HEAD)

    code = (packed_byte >> (sub_pos[:, None] * BITS)) & QMAX                # (BLOCK_N, D_HEAD)

    group_idx = tl.minimum(byte_idx // BYTES_PER_GROUP, num_groups - 1)     # (BLOCK_N,), also clamped
    meta_off = meta_base + group_idx[:, None] * D_HEAD + d_idx[None, :]     # (BLOCK_N, D_HEAD)
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
    NOT a pure axis-swap: `seq_len` here is the sub-block's own length,
    used as a stride (not an axis-size), and unlike K, no num_packed/
    num_groups params are needed at all — V's channel-derived axes are the
    same size regardless of which regime this call is for."""
    PACK_FACTOR: tl.constexpr = 8 // BITS
    QMAX: tl.constexpr = (1 << BITS) - 1
    D_PACKED: tl.constexpr = D_HEAD // PACK_FACTOR

    # D_HEAD is compile-time-known and small, so — unlike K's packed axis
    # (seq_len, runtime, tiled) — V's packed axis needs no tiling at all.
    d_idx = tl.arange(0, D_HEAD)
    tok_in_tile = tl.arange(0, BLOCK_N)
    # tok_start used directly, no dividing by PACK_FACTOR: V's packed
    # tensor keeps full token resolution (only D_HEAD shrinks) — opposite
    # of K, where tok_start converts to a byte index. Clamped both
    # directions, same reasoning as _dequantize_k_tile.
    tok_idx = tl.maximum(tl.minimum(tok_start + tok_in_tile, seq_len - 1), 0)

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


# --- Composing the three regimes into one attention kernel ------------------

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
    """General-purpose regime composer: computes a candidate value from all
    three sources (sink / window-old / residual) for the whole tile, then
    selects per slot with tl.where ("compute broadly, select narrowly").
    Used only for the boundary tiles that can genuinely mix regimes (see
    _sparse_decode_attn_kernel_quantized's 3-segment loop) — calling this
    unconditionally for every tile was tried first and measured at ~3x
    real memory traffic versus necessary; a subsequent runtime-`if` attempt
    to skip inapplicable branches measured ~20x *slower* instead (see
    module header and notes.md for the full story). The loop-segmentation
    approach that replaced both is what actually worked.
    """
    PACK_FACTOR: tl.constexpr = 8 // BITS
    WIN_OLD_LEN: tl.constexpr = window - RESIDUAL_SIZE          # window-old's own seq_len
    WIN_OLD_END: tl.constexpr = sink + WIN_OLD_LEN               # physical slot: window-old ends, residual begins

    # Sink is always padded to exactly one GROUP_SIZE group, so its shape
    # is a fixed compile-time constant, never threaded through as a runtime
    # parameter.
    SINK_SEQ_LEN: tl.constexpr = GROUP_SIZE
    SINK_NUM_PACKED: tl.constexpr = GROUP_SIZE // PACK_FACTOR
    SINK_NUM_GROUPS: tl.constexpr = 1

    # Window-old's shape varies across the benchmark sweep, but `window` is
    # itself tl.constexpr (fixed per kernel compilation), so these are still
    # compile-time constants within one compiled variant.
    WIN_NUM_PACKED: tl.constexpr = WIN_OLD_LEN // PACK_FACTOR
    WIN_NUM_GROUPS: tl.constexpr = WIN_OLD_LEN // GROUP_SIZE

    d_idx = tl.arange(0, D_HEAD)

    is_sink = slot_idx < sink
    is_old = (slot_idx >= sink) & (slot_idx < WIN_OLD_END)
    # is_residual is implicit -- the else-branch of the final tl.where below.

    sink_tok_start = slot_start                       # sink's own numbering == physical slot directly
    win_tok_start = slot_start - sink                  # can go negative; clamped inside the dequant functions

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

    # Residual: no packing, direct fp16 load against its own small
    # [.., RESIDUAL_SIZE, D_HEAD] cache. Upcast to fp32 to match k_sink/
    # k_old/v_sink/v_old's dtype before the tl.where combination below.
    res_local_slot = tl.maximum(slot_idx - WIN_OLD_END, 0)
    res_offset = (idx_in_batch * N_KV_HEADS * RESIDUAL_SIZE * D_HEAD) + \
                 (kv_head_idx * RESIDUAL_SIZE * D_HEAD + res_local_slot * D_HEAD)
    k_res = tl.load(K_cache_ptr + res_offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    v_res = tl.load(V_cache_ptr + res_offset[:, None] + d_idx[None, :], mask=valid[:, None], other=0.0).to(tl.float32)

    k = tl.where(is_sink[:, None], k_sink, tl.where(is_old[:, None], k_old, k_res))
    v = tl.where(is_sink[:, None], v_sink, tl.where(is_old[:, None], v_old, v_res))

    return k, v


@triton.jit
def _load_kv_tile_window_old(
    K_win_packed_ptr, K_win_scale_ptr, K_win_zero_ptr,
    V_win_packed_ptr, V_win_scale_ptr, V_win_zero_ptr,
    win_tok_start, num_packed, num_groups, win_seq_len,
    D_HEAD: tl.constexpr, idx_in_batch, kv_head_idx, N_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Pure window-old tile load — no sink or residual candidates computed
    at all (not skipped via a runtime branch, which measured slower; just
    never present in this function's body). Used only for the branch-free
    middle segment of the loop in _sparse_decode_attn_kernel_quantized that
    provably can't be anything but window-old — this is where the real
    WINDOW-scaling cost lives, so keeping it branch-free is what actually
    delivered the optimization win."""
    k = _dequantize_k_tile(
        K_win_packed_ptr, K_win_scale_ptr, K_win_zero_ptr,
        win_tok_start, num_packed, num_groups,
        D_HEAD, N_KV_HEADS, idx_in_batch, kv_head_idx,
        GROUP_SIZE, BITS, BLOCK_N,
    )
    v = _dequantize_v_tile(
        V_win_packed_ptr, V_win_scale_ptr, V_win_zero_ptr,
        win_tok_start, win_seq_len,
        D_HEAD, N_KV_HEADS, idx_in_batch, kv_head_idx,
        GROUP_SIZE, BITS, BLOCK_N,
    )
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
    """Same grid/online-softmax structure as _sparse_decode_attn_kernel, but
    the KV loop is split into 3 compile-time-bounded segments instead of one
    unified loop, since SINK and the window-old/residual boundary are both
    small, fixed constants that don't scale with WINDOW — so segments 1 and
    3 stay ~1-2 tiles always, and only segment 2 (the one that scales with
    WINDOW) needs to be branch-free. This structure is what recovered most
    of the ~3x overhead a naive unconditional-every-tile version measured;
    see quant_kernel.py's module header for the full optimization story.
    """
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

    PACK_FACTOR: tl.constexpr = 8 // BITS
    WIN_OLD_LEN: tl.constexpr = WINDOW - RESIDUAL_SIZE
    WIN_OLD_END: tl.constexpr = SINK + WIN_OLD_LEN
    WIN_OLD_END_ALIGNED: tl.constexpr = (WIN_OLD_END // BLOCK_N) * BLOCK_N
    SEG3_START: tl.constexpr = max(BLOCK_N, WIN_OLD_END_ALIGNED)
    WIN_NUM_PACKED: tl.constexpr = WIN_OLD_LEN // PACK_FACTOR
    WIN_NUM_GROUPS: tl.constexpr = WIN_OLD_LEN // GROUP_SIZE

    # Segment 1: tile 0, always present unconditionally — the only tile
    # that can ever contain a sink position, and (for WINDOW near
    # RESIDUAL_SIZE) possibly residual too.
    slot_start = 0
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

    # Segment 2: pure window-old, branch-free — where the real
    # WINDOW-scaling cost lives.
    for slot_start in tl.range(BLOCK_N, SEG3_START, BLOCK_N):
        slot_idx, valid = _kv_indices(seq_len, slot_start, WINDOW, SINK, BLOCK_N)
        win_tok_start = slot_start - SINK
        k, v = _load_kv_tile_window_old(
            K_win_packed_ptr, K_win_scale_ptr, K_win_zero_ptr,
            V_win_packed_ptr, V_win_scale_ptr, V_win_zero_ptr,
            win_tok_start, WIN_NUM_PACKED, WIN_NUM_GROUPS, WIN_OLD_LEN,
            D_HEAD, pid_batch, kv_head, N_KV_HEADS,
            GROUP_SIZE, BITS, BLOCK_N,
        )
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(valid, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.math.exp(qk - m_ij)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_ij

    # Segment 3: boundary (window-old meets residual) + pure residual —
    # small and fixed-size regardless of WINDOW (~1-2 tiles).
    for slot_start in tl.range(SEG3_START, cache_size, BLOCK_N):
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
    """k_cache/v_cache here is the RESIDUAL cache specifically (own
    [.., RESIDUAL_SIZE, D_HEAD] shape) — not the full compacted cache like
    sparse_decode_attention takes. Assumes WINDOW >= RESIDUAL_SIZE
    (see build_quantized_kv_cache)."""
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
    """Slices a compacted [BATCH, N_KV_HEADS, SINK+WINDOW, D_HEAD] K/V cache
    into residual/sink/window-old and quantizes sink+window-old separately,
    producing the tensors sparse_decode_attention_quantized expects.

    Assumes WINDOW >= RESIDUAL_SIZE — window-old's slice would be empty or
    invalid otherwise, not guarded against here.
    """
    k_residual = k_compacted[:, :, -residual_size:, :].contiguous()
    v_residual = v_compacted[:, :, -residual_size:, :].contiguous()

    # Sink padded to exactly GROUP_SIZE by repeating its own first real
    # token — keeps the padded group's min/max identical to the real 4
    # values' range, so the fake values don't dilute precision on the real
    # ones. .float(): quantize_k_cache/quantize_v_cache are only validated
    # against fp32 input (test_quantize_correctness.py); this cache is fp16
    # like everything else here, so cast explicitly rather than silently
    # running the quantize kernels on an untested dtype.
    k_sink_real = k_compacted[:, :, :sink, :]
    v_sink_real = v_compacted[:, :, :sink, :]
    pad = GROUP_SIZE - sink
    k_sink_padded = torch.cat([k_sink_real, k_sink_real[:, :, :1, :].expand(-1, -1, pad, -1)], dim=2).float().contiguous()
    v_sink_padded = torch.cat([v_sink_real, v_sink_real[:, :, :1, :].expand(-1, -1, pad, -1)], dim=2).float().contiguous()

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
