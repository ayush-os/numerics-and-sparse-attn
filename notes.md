# numerics-and-sparse-attn — Working Notes

Live log, kept during derivation — mirrors how `prefill_notes.md`/`decode_notes.md`/
`disagg_and_placement_notes.md` each started as a working notes/log file before being
polished into a final writeup. This is that stage for this project. Reused numbers cite
their source doc + section per this repo's own "reused number needs re-verification for
the new context" discipline.

---

## Phase 0 — Setup

### Reading (real mechanisms, not headline numbers — full findings below)

**Sparse/structural attention:**
- **StreamingLLM** (arXiv 2309.17453): cache = `start_size` sink tokens (default **4**,
  diminishing returns past that) + `recent_size` sliding window. Sink tokens never
  evicted, never re-scored. Window portion is pure FIFO (slice-and-cat). RoPE positions
  reassigned to *cache-relative* indices post-eviction — no data-dependent branching
  anywhere. Per-step cost is a flat constant in `(S+W)`, independent of how long the
  conversation has run. Mechanism: models learn to dump "leftover" softmax mass onto
  early tokens (visible to everything during training), so evicting them collapses
  quality even though they're not semantically special.
- **H2O** (arXiv 2306.14048): tracks a running cumulative attention-score sum per cached
  slot; evicts lowest-scoring slot(s) when over budget. Real added cost: O(k)
  compare/argmin (or O(k log k) sorted) every step, over a **data-dependent** index set
  (gather/scatter, not a clean slice). Independently confirmed as non-trivial overhead
  in practice by a 2026 survey (arXiv 2512.12008), not just asymptotically nonzero.
- **Longformer/BigBird**: ruled out — bidirectional full-sequence encoders (classification/
  summarization), no rolling KV-cache-eviction analog exists for causal decode. Global-
  token count `g` is linear/closed-form but the framing doesn't map onto decode-step
  cache-bytes derivation.

**Quantized attention:**
- **KIVI** (arXiv 2402.02750): K quantized **per-channel** (K has consistent outlier
  channels), V quantized **per-token** (no such structure). Group size **32**. Residual
  buffer of **up to 128 most-recent tokens kept full FP16**, never quantized — structural,
  not optional. Compute is **decoupled** (dequant fused into the matmul kernel) — matches
  the "realistic" model `decode_notes.md` §4.1 already used. Key ablation: **smaller
  groups are *more* accurate** (32/64 similar, 128 degrades) — counter to a naive
  "more data averages error better" intuition.
- **Implication flagged for Phase 2**: if sparsity shrinks the retained cache down near
  KIVI's residual-buffer size (~128 tokens), a large fraction of what survives has to sit
  in the uncompressed residual — sparsity may not be numerically *hostile* to
  quantization (small groups are fine per KIVI's own ablation) but may be structurally
  hostile to the *achievable compression ratio*, since the residual overhead doesn't
  shrink proportionally with a small sparse window. Worth deriving explicitly in Phase 2.
- **SmoothQuant** (arXiv 2211.10438): W8A8 for weights/activations (incl. attention BMMs,
  coupled INT8), not a KV-cache method — low direct relevance beyond confirming coupled
  INT8 attention compute is viable.
- **Fresh kernels**: FlashAttention-4 (arXiv 2603.05451) stays BF16-only, defers to the
  **SageAttention** line (INT8 → SageAttention2 INT4-QK/FP8-PV → SageAttention3 FP4
  microscaling) for quantization. **OSCAR** (arXiv 2605.17757) — post-KIVI, spectral-
  rotation 2-bit KV quant, targets KIVI's small-group accuracy problem — possibly a
  better Phase 3 precision baseline than KIVI itself.

**Sparsity × latent-compression composability (confirms it's shipping, not speculative):**
- **DeepSeek-V3.2 DSA**: a separate **lightning indexer** (own small projections computed
  from the *raw* hidden state `h_t`, NOT from the compressed MLA latent `c_t`; own
  separate cache) scores and selects top-k=2048 **individual tokens** (not blocks),
  every step. *After* selection, attention runs over the **MLA-compressed latents** of
  just the selected tokens (MQA mode). So compression governs storage; a parallel,
  purpose-built, uncompressed mechanism governs selection — they're not the same
  computation.
- **GLM-5.2**: same DSA+MLA pairing, plus **IndexShare** (reuses one layer's indexer
  output across 3 subsequent layers, ~2.9× indexer-FLOP cut at 1M context — justified by
  70–100% selection overlap between adjacent layers).
- **ASA** (arXiv 2511.00819): the reported ~50% additional KV-cache reduction (vs.
  sparse-only) comes from applying latent compression to **all three** NSA branches —
  sliding-window branch gets MLA, but the compression/selective branches needed a new
  **GLA** (Grouped-head Latent Attention) instead of plain MLA, because plain MLA/MQA
  collapses the per-group distinguishability that block-selection needs. GLA still stores
  only `d_c` bytes/position (so the byte formula stays a clean substitution), but the
  *selection mechanism's architecture* had to change.
- **Verdict for Decision 3**: `c`-substitution (swap GQA's/MLA's cache-entry-size formula)
  is defensible for the **bytes** term — both real systems keep per-position storage down
  to just the compressed latent regardless of sparsity pattern. It is **not** defensible
  as "selection logic left otherwise unchanged" — real systems needed a genuinely
  separate (if small) mechanism: an additive indexer term (DeepSeek/GLM) or an
  architectural change to the latent projection (ASA/GLA).

### Decision 1 — Sparsity mechanism: **sliding window + attention sink (StreamingLLM-style)**

Chosen over H2O on exactly the axis the spec named: static/closed-form vs.
dynamic/data-dependent. StreamingLLM gives a fixed cache size, FIFO eviction, no runtime
scoring — provably closed-form for Phase 1's FLOPs/bytes-as-function-of-(W, context)
formula. H2O's eviction-scoring overhead is real (confirmed by an independent survey),
data-dependent, and would force either an approximated added term or abandoning
closed-form entirely. Longformer/BigBird ruled out as not decode-shaped at all.

### Decision 2 — Kernel environment: **cloud GPU instance** (provider/instance TBD closer to Phase 3)

### Decision 3 — MLA carry-forward parameterization

Per spec: parameterize the K/V bytes term as `c` bytes/position/layer rather than
inlining GQA's `2×d_head×n_kv_heads` formula. Confirmed viable for bytes accounting
(research above). **Refinement over the spec's original framing**: Phase 4's MLA
substitution should also carry an explicit (small) additive indexer/selection-overhead
term, following DeepSeek's own published numbers, rather than treating the substitution
as a pure no-op swap — flagged now so Phase 4 isn't surprised by it later. If it turns
out not to be cheap, the spec's own escape hatch applies (flag as open thread, don't
expand scope).

**Scope confirmation**: this project's derivation is **SDPA-only** (QK^T → softmax → ·V),
matching `decode_notes.md`'s explicit scope call — correct here specifically (not just
inherited by convention) because window/sink size only changes which K/V positions get
touched by QK^T/·V; QKVO projections and FFN operate on the current token's hidden state
regardless of sparsity pattern. Flag for later: disagg's own synthesis found SDPA-only
silently incomplete once reused for a *throughput* question (had to add QKVO back in,
21.4% of FFN's magnitude, to get the authoritative 830.59 req/s/chip number) — if Phase 4
wants to compare against that specific number, QKVO will need the same correction.

**Checkpoint reached**: sparsity mechanism chosen and defended, GPU environment
identified, MLA-carry-forward plan set.

---

## Phase 1 — Hand-derive the sparsity lever (in isolation)

### Setup

Reused workload constants (`decode_notes.md` §0, unchanged): batch=32, n_heads=64,
n_kv_heads=8, d_head=128, precision=int8 (1 byte). Sink size S=4 (StreamingLLM's real
default). `seq_len_q = 1` (decode's defining feature — fixed, sparsity does not touch it,
since sparsity prunes which keys/values one query attends to, not how many queries exist).
`seq_len_kv = min(L, W+S)` in general; for the sweep range (L from 8,192 to 163,840, any
sane window size), `L ≫ W+S` for essentially the entire sweep except the first `S+W`
decode steps of any conversation (a fixed, self-resolving ramp-up transient, same
category as `decode_notes.md` §2.2's ramp-up note) — so `seq_len_kv = S+W` is used
directly.

Terminology note (self-corrected mid-derivation): Q/K/V/O are **activations**, not
weights — SDPA has no weight matrices; QKVO *projection* weights are exactly what's out
of scope. Total bytes moved = Q(load) + K(load) + V(load) + O(write) — four tensors, not
three; O is structurally identical to Q (same shape, `seq_len_q`-sized, unaffected by
window size).

### FLOPs

```
FLOPs(W) = 4 × batch × n_heads × d_head × (W + 4)
         = 1,048,576 × (W + 4)
```
(QK^T FLOPs = PV FLOPs, general structural identity per `decode_notes.md` §1.1 — total is
2× the per-matmul term.)

### Bytes

```
c = 2 × d_head × n_kv_heads × precision = 2,048 bytes/position/layer   [Decision-3 param]

Bytes(W) = [2 × batch × n_heads × d_head × precision]   (Q+O, fixed, = 524,288)
         + [batch × c] × (W + 4)                          (K+V, = 65,536 × (W+4))
         = 786,432 + 65,536 × W
```

**Sanity check**: setting `W+4 = 8192` (window = full dense context) recovers
`decode_notes.md`'s own dense GQA numbers *exactly* — FLOPs → 2³³, bytes → 537,395,200 B
(0.5005 GiB). Confirms the algebra is consistent with the formulas being reused.

### Arithmetic Intensity

```
AI(W) = FLOPs(W) / Bytes(W) = 16(W+4) / (W+12)
```

### Key Findings — Phase 1

1. **Sparsity is a FLOPs lever too, not just a bytes lever (unlike GQA)** — but FLOPs and
   the *variable* part of bytes both scale linearly with `seq_len_kv` at the same rate,
   while bytes carries a fixed Q+O floor (524,288) that FLOPs has no analog of. That
   asymmetry is the entire mechanism behind finding #2.
2. **AI(W) is strictly increasing in W, bounded above by exactly 16** — dense GQA's own
   AI ceiling (`decode_notes.md`'s 15.98 at L=8,192 already sits at 99.9% of this
   ceiling). Sparsity can only *approach* dense GQA's AI (as W grows toward full
   context), never exceed it. At W=256: AI≈15.52. At W=32: AI≈13.09. At W=0 (sink only):
   AI≈5.33.
3. **Sparse attention's AI can never clear the ridge point (480.5), for any window size,
   at any context length** — the ceiling (16) already sits ~30× below ridge. Same
   structural flavor as `decode_notes.md` Phase 3's quantization-proof floor, but for
   sparsity instead of precision. This fully answers Phase 1's "does sparse AI clear the
   ridge" question: no, structurally, by construction.
4. **The context-length sweep (8,192 → 163,840) is a non-event for AI** — the formula has
   no `L` dependence at all once `L ≥ S+W`. Confirmed numerically at both endpoints with
   W=256 fixed: AI≈15.52 at both L=8,192 and L=163,840, identical.
5. **But the *absolute savings ratio* (dense/sparse) grows linearly with L** — confirmed
   numerically: dense/sparse FLOPs ratio ≈31.5× at L=8,192 vs. ≈630× at L=163,840 (a 20×
   growth, tracking the L ratio itself almost exactly, since dense cost scales linearly
   with L while sparse cost is pinned flat at S+W). This is the real content of the
   sweep, and directly confirms the spec's own opening premise — "sparsity's value
   proposition is context-length-dependent" — with an actual derived number behind it.
6. **The compute-bound-asymptote check (spec's "does something else become the
   bottleneck first") is moot for Phase 1/sparsity alone** — since AI never clears the
   ridge under sparsity by itself, there's no "far enough above ridge" case to check.
   Becomes a live question only in Phase 2, once numerics stacks on top and might
   actually push combined AI across the ridge.
7. **The disagg §1.4 HBM-capacity-crossover sub-question ("does the crossover context
   length match where dense KV cache dominates HBM budget") — judged moot, skipped.**
   Its premise (a discrete crossover context length) doesn't survive finding #4 (no
   crossover exists — AI is L-invariant, savings grow smoothly/continuously, not via a
   switch-flip point). Also mechanistically circular: sparsity's absolute savings *are*
   driven by the same growing dense-bytes quantity that eats HBM capacity — not two
   independently-derived numbers that happen to coincide.

**Phase 1 checkpoint: reached.** Formula derived (FLOPs/bytes as function of W and
context length), stated finding on FLOPs-vs-bytes-lever (both, but bounded), sweep done,
compute-bound-asymptote and capacity-crossover sub-questions resolved as moot with
reasoning kept on record (not silently dropped).

---

## Phase 2 — Layer numerics on top, quantify the interaction

### Setup — generalized bytes formula

Generalized `Bytes(W)` from Phase 1 to make cache precision `p` (bytes/element) an
explicit variable, separate from the fixed activation precision (`precision_act`=int8=1
byte, unchanged, applies only to Q/O):

```
F(W)        = 4·batch·n_heads·d_head·(W+S)                [FLOPs, unchanged by p — decoupled
                                                             model, dequant fused into matmul,
                                                             per decode_notes.md §4.1]
B_fixed     = 2·batch·n_heads·d_head·precision_act = 524,288 B   [Q+O, fixed, p-independent]
c(p)        = 2·d_head·n_kv_heads·p                        [KV bytes/position/layer — p replaces
                                                             the fixed int8 term Phase 1 used]
Bytes(W,p)  = B_fixed + batch·c(p)·(W+S)
AI(W,p)     = F(W) / Bytes(W,p)
```

Sanity check: `c(1) = 2,048` recovers Phase 1's `c` exactly (int8, p=1).

### Crossover solve — reusing decode_notes.md §4.1/§4.2's method, new starting AI

Solved `AI(W,p) = R` (ridge point, R=480.5, TPU v5e/int8, per decode_notes.md) for `p`:

```
p(W) = [2·n_heads/(R·n_kv_heads)]  −  [B_fixed/(2·batch·d_head·n_kv_heads)] / (W+S)
     = A∞ − K/(W+S)                    where A∞ = 32/961 ≈ 0.03330, K = 8

Plugged in:  p(W) = 32/961 − 8/(W+4)
```

**Sanity check (passed):** as `W→∞`, `p(W) → 32/961 ≈ 0.0333` — exactly recovers
`decode_notes.md` Phase 3's own dense-GQA crossover floor (~0.033 bytes/element). This
is the fully-dense edge case of the sparse formula (W = full context, no sparsity), not
a coincidence — confirms the algebra is consistent with the reused number.

**Key findings — crossover solve:**
1. `p(W)` is strictly increasing in W, with `32/961` as a strict supremum, never attained
   at finite W — sparse crossover is *always* more extreme than dense's already-
   unrealizable floor, confirming the tension flagged in Phase 1's carried-forward notes.
2. `p(W)` goes **negative** below a critical window size: `W* = K/A∞ − S = 236.25`. Below
   `W≈236`, no precision — not even a hypothetical zero-byte one — gets sparse AI to
   ridge. Above it, `p(W)` is positive but still far below any realizable quantization
   (e.g. `p(256) ≈ 0.00253` bytes/element, ~0.02 bits/element — KIVI's own smallest
   published group is 2-bit ≈ 0.25 bytes/element, ~100× larger).
3. **Confirmed: precision cannot be a compute-bound lever here, at any window size.**
   Same structural flavor as Phase 1's sparsity-alone finding, now shown for the combined
   case too.

### Multiplicative-vs-non-independent check

Tested the naive hypothesis that combining levers just multiplies AI gains:
`p_naive(W) = A∞ · AI(W)/16 = (32/961)·(W+4)/(W+12)` — a pure product form.

Compared against the actual derived `p(W) = A∞ − 8/(W+4)` — a difference form, different
denominator shape (`W+12` vs `W+4`) entirely; not the same function.

Numerically, at W=256: `p_naive ≈ 0.03231` vs `p_actual ≈ 0.00253` — naive overstates by
**~12.8×**. At W=32: `p_naive ≈ 0.02725` (still positive) vs `p_actual ≈ −0.18892`
(negative/unreachable) — a *qualitative* divergence the naive model can't produce at all,
since it never predicts a critical-window phenomenon.

**Verdict: non-independent, confirmed.** Mechanism: same root cause as Phase 1 Finding
#1 — the fixed `B_fixed` (Q+O) additive term in `Bytes(W,p)` breaks clean multiplicative
separability between the W-lever and the p-lever, the same way it capped `AI(W)` at 16
instead of letting it scale freely. One structural fact (the fixed Q+O floor), two
separate manifestations (bounded AI in Phase 1, non-multiplicative crossover here).

**Consequence for Phase 4:** the MLA `c`-substitution (Decision 3) cannot shortcut
through an AI-ratio multiplier — it needs the full `Bytes(W,p)` solve redone with MLA's
own cache-entry size, not a ratio trick, since the additive `B_fixed` term breaks that
shortcut here and will break it there too.

### Dominance check — does one lever swamp the other (disagg §2.5 echo)?

Reused decode's own workload point; ran a 2×2 factorial (dense/sparse × normal/low
precision) at W=256, `p_low=0.25` bytes/element (KIVI 2-bit), first idealized (whole
window compressible), at two context lengths:

**Idealized (no residual-buffer constraint), L=8,192:**
```
(a) dense,  p=1     = 537,395,200 B    [= decode_notes.md's own dense number, sanity check passed]
(b) sparse, p=1     =  17,563,648 B
(c) dense,  p=0.25  = 134,742,016 B
(d) sparse, p=0.25  =   4,784,128 B

sparsity-alone  = a/b = 30.597×
precision-alone = a/c =  3.988×
combined        = a/d = 112.33×
naive product (30.597×3.988) = 122.03×  →  naive overstates by ~8.6%
dominance ratio (sparsity/precision) = 7.67×
```

**Idealized, L=163,840** (b,d unchanged — Phase 1 Finding #4, AI/bytes-ratio formulas
are L-invariant once L≥W+S):
```
(a) = 10,737,942,528 B     (c) = 2,684,878,848 B
sparsity-alone  = 611.37×    precision-alone = 3.999× (L-invariant, as expected —
                                                        B_fixed is negligible either way)
combined        = 2,244.49×
naive product (611.37×3.999) = 2,445.13×  →  naive overstates by ~8.9% (stable vs L=8,192)
dominance ratio (sparsity/precision) = 152.9×
```

**Finding: dominance ratio grows linearly with L**, tracking the context-length ratio
almost exactly (163,840/8,192 = 20×; dominance ratio grew 152.9/7.67 = 19.94×). Direct
mechanism: precision-alone is L-invariant (pure function of p, B_fixed negligible either
way); sparsity-alone scales with L (Phase 1 Finding #5). So the growing gap is entirely
sparsity's own L-scaling, not a new effect.

**Compute-bound-asymptote check (Phase 1 Finding #6, deferred here):** at the combined
operating point, `AI(256, 0.25) = F(256)/Bytes(256,0.25) ≈ 272,629,760/4,784,128 ≈ 56.98`
— still ~8.4× below ridge (480.5). Even with both realistic levers combined, decode stays
solidly memory-bound. **Resolved: still moot**, not newly live.

**Answer to the disagg-echo question: no, not FFN-style dominance.** Disagg's FFN
dominance made attention "a rounding error" — negligible. Here precision's ~4× is real
and substantial, just smaller in magnitude than sparsity's. Two comparably-important-but-
unequal levers, not one swamping the other into irrelevance — the opposite shape from
disagg's result, worth stating explicitly rather than folded into "sparsity wins."

### Residual-buffer correction — KIVI's structural floor collides with a small window

Phase 0 flagged this and deferred it: KIVI keeps the most recent ~128 tokens
uncompressed (structural, not optional). At W=256, the total sparse cache is only
`W+S=260` tokens — the 128-token residual is **~49% of the entire cache**, not a small
tail as it would be in a normal (thousands-of-tokens) dense cache.

**Precision choice for the uncompressed residual — a real judgment call, stated
explicitly:** KIVI's own paper uses FP16 (2 bytes) as its native/residual precision, but
*this project's* own baseline precision (established in Phase 1's reused workload
constants) is int8 (1 byte). Used **int8** for the residual, consistent with this
project's own operating point rather than importing KIVI's paper-native number
unverified — the exact "reused number needs re-verification for the new context" trap
the spec names.

**Recomputed (c) and (d) with the mixed-precision split** (residual tokens at int8=1B,
remainder at p_low=0.25B), L=8,192, W=256:
```
c_realistic: 128 tok × 2,048 B/tok(int8) + 8,064 tok × 512 B/tok(p_low)  → total 141,033,472 B
             (+4.7% vs idealized 134,742,016 — small; residual is only ~1.6% of L=8,192)
d_realistic: 128 tok × 2,048 B/tok(int8) + 132 tok × 512 B/tok(p_low)   → total  11,075,584 B
             (+131.5% vs idealized 4,784,128 — large; residual is ~49% of W+S=260)
```

**Updated ratios:**
```
sparsity-alone            = a/b            = 30.597×  (unchanged — no quantization involved)
precision-alone realistic = a/c_realistic  =  3.810×  (−4.5% vs idealized — dense case barely affected)
combined realistic        = a/d_realistic  = 48.521×  (−56.8% vs idealized 112.33× — cut by more than half)

naive product (realistic) = 30.597×3.810 = 116.59×
actual combined realistic                = 48.52×
naive overstates by ~140% (predicts >2.4× the real savings)
```

**Key finding:** sparsity's benefit is completely untouched by this (it never involves
sub-int8 quantization). What collapses is *precision's marginal contribution once layered
on top of sparsity*: precision's own effective multiplier drops from ~3.81× (applied to
the full dense cache) to only **~1.59×** (`48.52/30.597`) once sparsity has already
shrunk the cache to near the residual buffer's own size. Mechanism: this isn't a 50/50
dilution — uncompressed tokens cost 4× more bytes per position than compressed ones
(int8 vs 0.25B), so the ~49%-by-count residual claims a hugely disproportionate **share
of bytes** (residual accounts for ~80% of `d_realistic`'s KV bytes despite being only
half the token count). This is the concrete, quantified version of the "sparsity may be
structurally hostile to the achievable compression ratio" concern flagged in Phase 0 —
confirmed, not assumed.

### Phase 2 checkpoint: reached

**Stated, derived answer:** sparsity and numerics are **not independent, multiplicative
levers** — confirmed two ways: (1) the crossover-precision solve is a difference form,
not a product form, tracing to the same fixed Q+O bytes term that bounded Phase 1's AI;
(2) once KIVI's residual-buffer structure is included (not just the idealized crossover
math), precision's *marginal* savings on top of an already-sparsified cache collapse from
~3.81× to ~1.59× — headroom genuinely eaten, not just algebraically non-clean. Neither
lever dominates the other FFN-style (disagg §2.5 echo does *not* recur here) — both
contribute real, comparable-order-of-magnitude savings, with sparsity's advantage over
precision growing linearly with context length (20× dominance-ratio growth for a 20×
context-length increase).

**For Phase 3:** the realistic, non-floor-hitting KV-cache precision to actually
implement is **KIVI's 2-bit (p=0.25 bytes/element) with an int8 residual for the most
recent ~128 tokens** — not the crossover-derived `p(W)` (which is unrealizable/negative
for any practical W), and not an idealized uniform-precision cache (which Phase 2 just
showed overstates savings by >2× once the residual is real).

## Open Threads Carried Forward

- Phase 4's MLA substitution needs an additive indexer-overhead term (DeepSeek's
  published numbers), not a pure `c` swap — flagged in Decision 3, not yet quantified.
  **Sharpened by Phase 2**: also can't shortcut via an AI-ratio multiplier (confirmed
  non-independent) — needs the full `Bytes(W,p)` solve with MLA's `c`, from scratch.
- If Phase 4 wants to compare against disagg's authoritative 830.59 req/s/chip / 5.82:1
  numbers, QKVO needs to be added back in the same way disagg did for dense decode.
- Phase 3's kernel should implement KIVI-style quantization exactly as scoped above
  (2-bit main + int8 residual, ~128 tokens) — real hardware numbers will show whether
  this hand-derived residual-dilution effect is the whole story or whether real kernel
  overhead (dequant cost, non-contiguous residual/main-buffer access) adds more on top.
  That's a live "real vs. predicted gap" candidate for Phase 4 Q3.
