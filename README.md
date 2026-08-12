# Sparse + Quantized Decode Attention: Roofline to Real Kernels

Hand-derived roofline analysis of StreamingLLM-style sparse decode attention
layered with KIVI 2-bit KV-cache quantization, validated end-to-end with real
Triton kernels on an A100 — the headline result is that the two levers are
**not independent**: quantization's savings collapse once sparsity has
already shrunk the cache.

**Stack:** Triton · PyTorch · CUDA (A100) · Python

- **Up to 9.5× measured speedup** vs. a dense causal-GQA baseline at long
  context (163,840 tokens), correctness-verified against an independent
  third-party reference (`mit-han-lab/streaming-llm`) with adversarial
  garbage-value tests.
- **Precision's marginal savings collapse from ~7.17× to ~1.76×** once
  quantization is layered on top of an already-sparsified cache — derived by
  hand, then confirmed as a real (not just algebraic) effect on real
  hardware.
- **14 real hardware-only bugs found and fixed** across two kernels
  (sliding-window+sink attention, 2-bit KIVI quantization) — none visible by
  static inspection, including a tile-level clamp that silently corrupted
  real cache data, and a kernel "optimization" that measured **20× slower**
  than predicted before a working fix was found.

## The question

Sparsity (attend to fewer KV positions) and numerics (store each position in
fewer bytes) both reduce decode's memory traffic. Are they independent,
multiplicative levers on the same regime — or does one eat into the other's
headroom?

## Method

1. **Roofline derivation** — FLOPs/bytes/arithmetic-intensity for
   sliding-window + attention-sink (StreamingLLM-style) sparse decode
   attention, as a function of window size *and* context length, swept from
   8,192 (Llama-3's native cap) to 163,840 (DeepSeek-V2's real deployed cap).
2. **Numerics layered on top** — KIVI-style 2-bit KV-cache quantization (K
   per-channel, V per-token, group size 32) composed with the sparse cache,
   including its structural fp16 residual buffer for the most recent ~128
   tokens.
3. **Real kernels, real hardware** — both mechanisms implemented as fused
   Triton kernels, correctness-verified against dense/sparse/quantized
   references, benchmarked on a real A100 across the full context-length ×
   window-size sweep.

## The derivation

**Roofline (Phase 1)** — sink size `S=4`, window `W`, batch=32, GQA (64
query heads / 8 KV heads / 128 head-dim), int8:

```
FLOPs(W) = 4 · batch · n_heads · d_head · (W + S)  = 1,048,576 × (W + 4)
c        = 2 · d_head · n_kv_heads · precision      = 2,048 bytes/position/layer
Bytes(W) = 786,432 + 65,536 × W                      [fixed Q+O floor + variable K+V]

AI(W)    = FLOPs(W) / Bytes(W) = 16(W+4) / (W+12)
```

Sanity check: `W+4 = 8192` (window = full dense context) recovers the dense
GQA roofline numbers exactly. `AI(W)` is strictly increasing in `W` but
**capped at exactly 16** as `W→∞` — the same fixed Q+O byte term that caps it
is what breaks clean separability from precision, below.

**Non-independence, made concrete (Phase 2)** — solving `AI(W,p) = ridge`
for the crossover precision `p(W)`:

```
p(W) = A∞ − K/(W+S),   A∞ = 32/961 ≈ 0.0333,  K = 8

naive (product) hypothesis:   p_naive(W)  = A∞ · AI(W)/16
actual (difference) result:   p_actual(W) = A∞ − 8/(W+4)
```

These are not the same function — different shape, different denominator.
At `W=256`: `p_naive ≈ 0.0323` vs. `p_actual ≈ 0.0025`, naive overstates by
**~12.8×**. At `W=32`, the divergence turns qualitative, not just
quantitative: `p_naive ≈ 0.0273` (still positive, still "achievable" in
principle) vs. `p_actual ≈ −0.189` (negative — **no precision, not even a
hypothetical zero-byte one, reaches the ridge**). The naive model can't
produce that phase change at all.

**Where the two levers actually collide** — KIVI's fp16 residual buffer
(~128 uncompressed tokens, structural, not optional) is a *fixed* absolute
token count. Once sparsity has already shrunk the cache to `W+S` tokens, the
residual stops being a small tail and starts being most of the cache:

```
precision-alone (dense cache, realistic)        ≈ 7.17×
naive product (sparsity-alone × precision-alone)  ≈ 219×
actual combined (sparse cache, realistic)        ≈ 53.8×   (naive overstates by ~4×)

precision's marginal multiplier once layered on sparsity = 53.8 / 30.6 ≈ 1.76×
```

Mechanism: the residual's fp16 tokens cost 8× more bytes each than the
main cache's 2-bit tokens, so at `W=256` the 128-token residual — only
~49% of the cache *by count* — claims **~88.6% of its bytes**. Precision's
own savings are real, just mostly already spent by the time sparsity gets
to them.

One more finding the formulas above don't show directly: `AI(W)` has **no
dependence on context length `L`** at all once `L ≥ W+S` — confirmed
numerically identical (`AI≈15.52`) at both `L=8,192` and `L=163,840` with
`W=256` fixed. But the *absolute* dense/sparse FLOPs ratio still grows
linearly with `L` (≈31.5× → ≈630× across that same range), since dense cost
scales with `L` while sparse cost stays pinned at `W+S` — sparsity's value
is entirely in that growing gap, not in AI itself moving.

## Real hardware, real bugs

Two kernels, 14 bugs total, all invisible by inspection:

- **Sparse kernel**: the headline bug — `_kv_indices` returned *logical*
  sequence positions instead of *physical* cache slots, silently correct for
  small contexts and a real CUDA illegal-memory-access at scale. Also hit
  Triton's `tl.arange` power-of-2/compile-time-constant constraints,
  `tl.cat` reordering ambiguity, and an int32 offset-overflow at long
  context (fixed by typing offsets `int64`, matching vLLM's own real
  convention).
- **Quantized kernel**: a tile-level clamp silently corrupted real in-range
  window data — root-caused via an isolated dequantization test that
  bypassed the buggy composition logic entirely. A first optimization
  attempt (runtime `if`/`else` to skip inapplicable regime work) measured
  **~20× slower**, not faster; a structurally different fix (splitting the
  loop into compile-time-bounded segments) worked, roughly
  doubling-to-tripling the measured speedup.

## Results

| | dense | sparse (fp16) | sparse + 2-bit quant |
|---|---|---|---|
| 163,840 ctx, W=16,384 | ~90.5 ms | fastest of the three | ~9.5 ms (**~9.5× vs. dense**) |

Quantization never beats native sparse in raw kernel time (redundant
scale/zero-point loads add real overhead the byte-count-only prediction
didn't capture) — but both comfortably beat dense at long context, and lose
to it once the window approaches the full context length (no sparsity left
to amortize the quantization tax).

## Repo layout

```
constants.py     shared workload constants
dense_kernel.py  dense causal-GQA baseline
sparse_kernel.py sliding-window + sink decode kernel
quant_kernel.py  KIVI 2-bit quantization on top of the sparse kernel
benchmark.py     sweep + CSV output
test_*.py        5 correctness suites (dense, sparse, quantize round-trip,
                  isolated dequant, full quantized-attention path)
results/         raw benchmark CSVs
```
