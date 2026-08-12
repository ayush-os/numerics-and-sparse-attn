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

**[Superseded below — the "int8 residual" call reasoned from an int8-baseline assumption
that turned out not to match what Phase 3 actually built. See the fp16 addendum
immediately following this section for the corrected precision to implement.]**

### Addendum — fp16 baseline correction (discovered after Phase 3 kernel build)

**The mismatch, discovered late:** everything above reused `decode_notes.md`'s workload
constants, which set `precision_act = int8 (1 byte)` for Q/O — carried through every
formula above (`B_fixed = 524,288`, the int8 residual choice). But Phase 3's actual
kernels (`dense_decode_reference.py`, `phase3_kernel_scaffold.py`) were built and
benchmarked entirely in **fp16** (2 bytes) — Q, K, V, O all fp16 — never reconciled back
against this section's int8 assumption. Exactly the "reused number needs
re-verification for the new context" trap this project's own spec names — caught here,
not silently carried forward.

**Scope of the fix, a deliberate choice:** only the numbers that depend on
`precision_act`/the residual-precision decision are recalculated below — straight
substitution (`precision_act: 1→2`), nothing new to derive. The AI-ceiling/crossover/
dominance sections above are left as-is, un-rewritten, flagged rather than silently
replaced — same discipline as `spec.md`'s in-place (c)-descoping annotation. Fully
reconciling every downstream Phase 1/2 number is deferred to Phase 4 on purpose, per this
project's own scope discipline — only what's actually blocking Phase 3's next build step
(the residual precision decision) gets resolved now.

**General formula, `precision_act` as an explicit variable `p₀`** (Q/O and the
"normal"/uncompressed KV precision share this value in the baseline case):

```
AI(W, p₀) = 16(W+4) / [p₀·(W+12)]        [reduces to the original formula at p₀=1]
```

AI ceiling (W→∞) = `16/p₀` — **8 at fp16** (p₀=2), not 16. Still ~60× below ridge
(480.5) — conclusion unchanged (sparse AI never clears ridge), just a lower ceiling.

**Crossover solve, corrected:**
```
p(W) = 32/961 − 16/(W+4)          [A∞ = 32/961 is unchanged — it cancels out of the
                                     variable-bytes/FLOPs ratio and doesn't depend on p₀
                                     at all; only K doubles: K = 8·p₀ = 16 at fp16]
```
Critical window `W* = 476.5` (**double** the original 236.25) — sparse crossover now
requires an even larger window before *any* precision reaches ridge. At `W=256`
(previously just above critical), `p(256) ≈ −0.0282` — **now negative**: that window
moved from "barely above critical" to "below critical" purely from the baseline
correction, not a new derivation.

**Idealized 2×2 factorial, recomputed (L=8,192, W=256):**
```
(a) dense,  p=2(fp16) = 1,074,790,400 B   [exactly 2× original — both terms scale w/ p₀]
(b) sparse, p=2(fp16) =    35,127,296 B   [exactly 2× original, same reason]
(c) dense,  p=0.25    =   135,266,304 B   [barely moved — only B_fixed changed; the
                                            2-bit KV term doesn't depend on p₀ at all]
(d) sparse, p=0.25    =     5,308,416 B   [same reason, barely moved]

sparsity-alone   = a/b = 30.597×   [UNCHANGED — algebraically independent of p₀]
precision-alone  = a/c ≈  7.946×   [~2× the original 3.988× — (c) barely moved while
                                     (a) doubled, so this ratio nearly doubles]
combined         = a/d ≈ 202.47×
naive product (30.597×7.946) ≈ 243.1×  → naive overstates by ~20.1% (vs ~8.6% at int8 —
                                          non-independence is a bigger effect now, not
                                          just a differently-sized one)
dominance ratio (sparsity/precision) ≈ 3.851×   [down from 7.67× — precision became
                                                   relatively more valuable]
```
At L=163,840: precision-alone → 7.997× (asymptotically approaches `p₀/p_low = 2/0.25=8`
exactly — the clean large-L limit, matching how the original approached `1/0.25=4`).
combined ≈ 4,045.6×, dominance ratio ≈ 76.45× (grows ~19.85× from L=8,192→163,840 — same
~20× L-scaling pattern as before, mechanism unchanged, just the anchor values differ).

**Residual-buffer correction, corrected — the load-bearing result:**

Per the corrected reasoning (residual precision should match *this project's real
operating point*, and that's now established as fp16, not int8), the residual buffer is
**fp16 (2 bytes/element)**, not int8. Recomputed with 128 residual tokens at fp16
(4,096 B/tok) + remainder at 2-bit (512 B/tok), L=8,192, W=256:

```
c_realistic (dense, mixed)  = 149,946,368 B  (+10.9% vs idealized 135,266,304 — bigger
                                               residual tax than int8's +4.7%)
d_realistic (sparse, mixed) =  19,988,480 B  (+276.5% vs idealized 5,308,416 — vs
                                               int8's +131.5%)

precision-alone realistic = a/c_realistic ≈ 7.168×
combined realistic        = a/d_realistic ≈ 53.77×

naive product (realistic) = 30.597×7.168 ≈ 219.3×
actual combined realistic                 ≈  53.77×
naive overstates by ~308% (predicts >4× the real savings — worse than int8's ~140%)

precision's marginal multiplier once layered on sparsity = 53.77/30.597 ≈ 1.757×
   (down from precision-alone-realistic's own 7.168× — a 75.5% relative collapse,
    vs int8's 3.81×→1.59× = 58.2% collapse — same effect, now sharper)
```

**Why it's sharper, mechanistically:** fp16 residual tokens cost 4,096 B vs. the 2-bit
main cache's 512 B — an **8× per-token tax** (vs. int8's 4×). The 128-token residual is
still ~49% of the W=256 cache *by count*, same as before, but now claims **~88.6%** of
the cache's total bytes (up from ~80% under int8) — the residual dominates the sparse
cache's byte budget even more completely. Same root mechanism as the original finding
(uncompressed tokens cost disproportionately more bytes per position), amplified by the
baseline correction.

**Resolved for Phase 3:** the KV-cache precision to implement is **KIVI 2-bit
(p=0.25 bytes/element) with an FP16 residual (2 bytes) for the most recent ~128
tokens** — superseding this section's original "int8 residual" call, which reasoned from
an assumption (int8 baseline) that didn't match what was actually built.

---

## Phase 3 — real Triton kernel

### Decision 2, resolved: cloud A100

User had "whatever GPU, willing to pay" — recommended A100 (80GB preferred, 40GB
workable) over H100: decode is memory-bandwidth-bound end to end (Phase 1's own
finding), so frontier compute isn't the bottleneck; A100's ~2TB/s HBM bandwidth is
plenty, and it's the cheaper, more standard, better-documented choice for iterative
kernel dev. Sizing check: dense baseline's K/V tensors alone at the sweep's largest
context (L=163,840) are ≈20 GiB (`BATCH·N_KV_HEADS·L·D_HEAD·2·2`), so anything under
~32GB VRAM was a real OOM risk, not theoretical. Ran on a Paperspace A100 instance.

### Reference-kernel decision — not a straight fork, and why

Per spec, Phase 3 should start from a published kernel and modify, not write from
scratch. Two real candidates checked, both rejected as *modification targets* for
different reasons:

- **Triton docs tutorial (`06-fused-attention.py`)**: real, vendored into the repo
  (`triton_fused_attention_tutorial.py`), but prefill-shaped (tiles over `BLOCK_M` query
  rows, serial K/V loop) — decode's `seq_len_q=1` collapses the query-tile dimension to
  nothing, and it has no GQA support at all (requires Q/K/V to share head count). Kept in
  the repo but not used as the active dense baseline.
- **vLLM's `triton_unified_attention.py`**: GQA-native and decode-shaped (has
  `SLIDING_WINDOW`/`USE_SINKS` built in, split-KV via `IS_3D`), which would have
  defeated the point of writing the sparsity mechanism ourselves. Also, on closer
  inspection, carries ~1,600 lines across itself and a helper module (alibi, softcap,
  qq_bias, paging, quantization, a hardware-specific tensor-descriptor load path, 2D/3D
  split-KV toggle) — extracting/simplifying that blind, with no GPU access to verify the
  simplification, risked a silently-wrong "real" reference, which defeats the actual
  point of starting from a real system.

**Resolution**: user gave explicit permission to deviate from "must be a real fetched
kernel" — spec is a starting point, not gospel. Built two purpose-built files instead,
each informed by patterns in the real kernels above (GQA head-group indexing,
online-softmax accumulation) but small enough to reason about correctness without
hardware:
- `phase3_kernel_scaffold.py` — the sparse sliding-window+sink kernel, written
  collaboratively (see next section for the given/theirs split).
- `dense_decode_reference.py` — the dense-causal-GQA baseline (comparison target (b)),
  written directly (no sparsity-selection content to preserve as the user's own work).

### Scaffold design — given vs. theirs

Per this project's own methodology (assistant as sounding-board/checker for 🧠 work, not
derivation engine), the scaffold split boilerplate from content explicitly:
- **Given** (filled in directly): grid/launch mechanics, online-softmax bookkeeping,
  benchmark harness plumbing (tensor construction once shapes were fully determined,
  `do_bench` calls), the dense reference kernel in full.
- **Theirs** (left as stubs, user-derived): `_kv_indices` — which cache slots a query
  attends to given `(seq_len, WINDOW, SINK)`, including the ramp-up transient. First
  attempt caught and corrected (missing ramp-up handling, per notes.md's own Phase 1
  ramp-up framing) before being handed back to fill in fully.
- **Explicitly declined to fill in even when asked**: the comparison-against-predictions
  logic (deciding what predicted quantity to check measured wall-clock against, and
  interpreting any gap) — this is the actual point of Phase 3/4, not incidental to it.
  Also declined to write `test_sparse_correctness.py`'s reference mask myself even after
  being asked directly — an independent check written by the same source as the thing
  being checked isn't independent; the mask was instead sourced from a real third party
  (`mit-han-lab/streaming-llm`'s own `kv_cache.py`), confirming the ramp-up assumption
  exactly (no eviction, full dense attention, until `seq_len` exceeds `sink+window`) and
  giving the reference an actual chance to catch a shared mistake, not just restate one.

### Staged build order

Explicit user call, mirroring the project's own Phase 1→Phase 2 structure: get
sliding-window+sink correct and benchmarked at native fp16 precision first, defer KIVI
quant (2-bit main + int8 residual) to a second pass on top of a known-good kernel.
Reasoning given at the time: isolates debugging surface (a wrong-output bug can only be
in one of two now-separable systems, not an ambiguous mix of both), and produces a real,
measured Phase-1-only checkpoint before quant complicates the picture. **Quant has not
been started** — this is still the next real chunk of Phase 3 work.

### Real bugs found via iterative hardware testing — full list, root causes

Every one of these was invisible by inspection and only surfaced once actually run on
the A100. Listed in the order found, since several build on each other:

1. **Missing `kv_head` offset + missing head-dim broadcast** in `_load_kv_tile`'s first
   draft — every query head in a GQA group besides head 0 would've silently read the
   wrong KV head's cache; the load was also computing one address per slot instead of
   one per `(slot, head-dim-element)` pair.
2. **`tl.arange(seq_len - WINDOW, seq_len)` — Triton requires `tl.arange` bounds to be
   compile-time constants**, but `seq_len` is a runtime kernel argument. Fixed via the
   standard idiom: a compile-time-sized `tl.arange` plus a runtime-offset add, not a
   runtime-bounded range.
3. **Bare Python globals (`N_HEADS`, `N_KV_HEADS`) referenced inside `@triton.jit`
   functions** — this specific Triton version rejects that outright (`NameError`);
   needed explicit `tl.constexpr` parameters threaded through every call site instead.
4. **`tl.cat` requires `can_reorder=True`, and even then doesn't guarantee the same
   reordering across two separate calls** — a real risk of `idx[i]`/`valid[i]` silently
   decoupling if the `idx` and `valid` concatenations reordered differently. Fixed by
   building both from one shared `tl.arange`-derived index via `tl.where`, removing the
   ambiguity rather than just silencing the compiler.
5. **`tl.arange`'s range must be a power of 2** — `SINK+WINDOW` (e.g. 4+256=260) usually
   isn't one. First fix: pad to `next_pow2(SINK+WINDOW)` and mask the padding invalid
   (same mechanism as ramp-up masking). Superseded by fix #7 below.
6. **The real headline bug: `_kv_indices` returned logical sequence positions, not
   physical cache slots.** Present since the very first draft (`tl.arange(seq_len-WINDOW,
   seq_len)` was always a logical position). `_load_kv_tile` multiplied that directly
   into its offset math as if it were a physical array index — at `seq_len=8192,
   window=256`, window slots got `idx` values like 7936–8191, used as offsets into a
   260-slot buffer. Caused a real CUDA illegal-memory-access, confirmed via
   `CUDA_LAUNCH_BLOCKING=1`. Every intermediate fix (masking, padding) preserved this
   same underlying confusion instead of catching it — it took an actual crash to surface
   it. Fix: `_kv_indices` returns the physical slot number directly (the compacted cache
   is already laid out in slot order — `slot i` holds whatever's currently kept there);
   logical position is still computed internally, but only to decide *validity*, never
   to compute an address.
7. **Triton's max single-tensor size (1,048,576 elements) exceeded at large `WINDOW`**
   — the single-block design materialized the whole `(PADDED_KV, D_HEAD)` tile at once;
   at `W=8192`, `PADDED_KV=16384 × D_HEAD=128 = 2,097,152`, over the limit. This was the
   deferred SRAM-footprint concern flagged in the kernel's own comments from the start
   (spec.md's "check tile-bounded, not context-bounded" ask) — correctness was verified
   first, then this became the real follow-up. Fixed by restructuring to a `BLOCK_N=128`
   tiled loop with running online-softmax accumulation (`m_i`/`l_i`/`acc`), mirroring
   `dense_decode_reference.py`'s own structure. This also **eliminated** the `PADDED_KV`
   workaround from fix #5 entirely — a fixed power-of-2 `BLOCK_N` satisfies `tl.arange`'s
   constraint on its own, regardless of `SINK+WINDOW`'s actual value.
8. **int32 overflow in the dense kernel's offset arithmetic at large context lengths.**
   Triton's default integer arithmetic is 32-bit. `kv_base = pid_batch * N_KV_HEADS *
   seq_len * D_HEAD + ...` overflows int32 (~2.1B) once `seq_len` gets large enough — at
   `L=131,072` the dominant term alone is ≈4.16B. Silently wraps into a garbage address
   rather than erroring cleanly. Crossover for this workload's sizes: `L≈67,653`
   (between the sweep's `65,536` and `131,072`). This is why an isolated smoke test at
   `L=8,192` passed clean while the full `CONTEXT_LENGTHS` sweep didn't — confirmed via
   `CUDA_LAUNCH_BLOCKING=1` pinning the fault precisely to `_dense_decode_attn_kernel`'s
   own launch. Fixed by typing `seq_len: tl.int64` in the kernel signature (matches
   vLLM's own real convention of typing every stride parameter `tl.int64` rather than
   leaving them default-width — noticed in passing while reading that kernel earlier,
   didn't carry the lesson into our own kernels until this bug forced it). **The sparse
   kernel is structurally immune to this one for the current sweep** — its offset math
   is bounded by `SINK+WINDOW` (≤16,388), never the full context length, so it never
   approaches int32 range regardless of `L`. Not a guarantee if `WINDOW_SIZES` ever grows
   much larger, just not a live issue at current values — the cache-compaction that's
   the whole point of the sparsity mechanism is what buys this safety margin.
9. **`sparse_decode_attention` reused the plain `_load_kv_tile`/`_kv_indices` names
   across a signature change** (parameters added for `N_KV_HEADS`, `BLOCK_N`, etc.) —
   several rounds of "argument missing" errors from call sites not being updated in
   lockstep with signature changes. Mechanical, not conceptual, but contributed real
   iteration count.

### Correctness verification

**`test_sparse_correctness.py`**: harness (random inputs, compacted-cache construction,
comparison) filled in directly; the reference's attention mask sourced from
`mit-han-lab/streaming-llm`'s real `kv_cache.py` (`StartRecentKVCache.__call__`), not
derived by either the user or the assistant, for genuine independence. Adversarial
`GARBAGE_VALUE` (not random data) planted in slots that should be masked out, so a
masking bug produces an obviously-wrong output rather than a coincidental pass. Test
cases cover pre-sink, mid-ramp-up, steady-state, and (added after the fact, closing a
flagged coverage gap) `W=0` — the tightest possible cache (`SINK+WINDOW=4`), never
covered until explicitly checked. **All pass** (max abs diff 0.001–0.004, consistent
with fp16 rounding against an fp32 reference, well under the 1e-2 tolerance).

**`test_dense_correctness.py`**: written entirely by the assistant — no sparsity-
selection content in dense causal+GQA attention, so no reason to hold it back. GQA
handled via reshape+einsum broadcast rather than `repeat_interleave`, to avoid
materializing an 8×-larger K/V copy at the large-`seq_len` case (`repeat_interleave`
would need ~128GiB at `L=131,072`; the broadcast approach needs ~1GiB). Includes
`seq_len=131,072` specifically — past the int32 overflow crossover — to confirm the fix
produces the *correct* result at that scale, not just that it no longer crashes. **All
pass**, and notably the diff *shrinks* at scale (0.00003 at `L=131,072` vs. 0.0038 at
`L=2`), reinforcing that the fix is real, not a coincidental non-crash.

### Benchmark sweep — results and gap-hunt (in progress)

Full `CONTEXT_LENGTHS × WINDOW_SIZES` sweep (66 points) ran clean after all bugs above
were fixed. Results in `benchmark_results.csv` **on the remote GPU box only — not yet
copied into the repo.** Predicted-bytes formula re-derived for this stage's actual
precision (fp16 Q/O + fp16 K/V, not notes.md's original int8 assumption — the reused-
number-needs-reverification trap, caught before it became a real error): `B_FIXED_FP16 =
1,048,576`, `C_FP16 = 131,072` (both exactly 2× Phase 1's int8 constants, as expected).
Compared via bytes ratio (not FLOPs ratio) as the predicted proxy for measured wall-clock
ratio, since Phase 1 Finding #3 already established this regime never clears the
roofline ridge — matches Phase 2's own "sparsity-alone" ratio methodology.

**Findings so far:**

1. **Dense-recovery sanity check passes clean on real hardware.** `L=8192,W=8192` and
   `L=16384,W=16384` (sparse window covers the whole context, should collapse to dense)
   give `gap_%` of 0.51% and 1.29% — the same sanity check Phase 1 used on the hand-
   derived formulas, now confirmed on measured wall-clock time, not just algebra. Good
   evidence the harness itself (tensor construction, timing methodology, the bytes-ratio
   prediction) is sound, independent of whether individual kernels have bugs.

2. **`gap_%` has a large, systematic pattern**: strongly negative at small `W` (as
   extreme as -82% at `W=0`) — the hand-derived formula *overpredicts* the real speedup
   — swinging positive at large `W` (up to +46%) — the formula *underpredicts*. The
   crossover `W` (where `gap_%≈0`) shifts to smaller absolute values as `L` grows: ~900
   at `L=8,192`, under 64 by `L=163,840`. Not yet fully resolved — see below.

3. **A specific, confirmed sub-mechanism: `BLOCK_N=128` tiling creates a sawtooth in
   `gap_%`, superimposed on the broader trend.** `gap_%` dips sharply at `W=128`
   relative to *both* neighbors (`W=64`, `W=256`), on *every single `L`* in the sweep —
   e.g. at `L=163,840`: `W=64→+7.98`, `W=128→-19.04`, `W=256→+5.98`. Root cause,
   confirmed empirically (not just hypothesized) via a targeted `W=124/125/128/256`
   sweep at `L=8,192`:
   ```
   W=64  → ms=0.0390
   W=124 → ms=0.0569   (still 1 tile: SINK+WINDOW=128 exactly)
   W=125 → ms=0.0947   (now needs 2 tiles: SINK+WINDOW=129)
   W=128 → ms=0.0945   (same 2-tile band as W=125 — nearly identical time)
   W=256 → ms=0.1400   (3rd tile boundary)
   ```
   A **+66% wall-clock jump from `W=124` to `W=125`** — a 1-unit change in nominal window
   size — is definitively not explained by one extra token of real work; it's the tile
   count (`ceil((SINK+WINDOW)/BLOCK_N)`) stepping from 1→2. `W=125` and `W=128` being
   nearly identical (`0.0947` vs `0.0945`) confirms the flip side: cost is flat *within*
   a tile-count band, only jumping at boundaries. Because `SINK=4` is small relative to
   `BLOCK_N=128`, every power-of-2 `W≥128` in the sweep happens to land just 4 past a
   fresh tile boundary — so every sampled point in that range is catching a boundary
   penalty, and the *size* of that penalty shrinks with each successive boundary (`1→2`
   tiles is a 100% relative jump, `2→3` is 50%, `3→4` is 33%, ... — ordinary `1/N`
   arithmetic on a roughly-fixed per-tile overhead). The kernel's cost is a step function
   of tile count; the hand-derived prediction assumes smooth linear-in-`W` cost; `gap_%`
   is picking up exactly that mismatch.
   **Not yet resolved**: whether this tiling artifact is the *whole* story behind
   finding #2's broader trend, or whether there's an additional smooth component (e.g.
   real launch-overhead amortization, achieved-bandwidth differences between the two
   kernels) once the sawtooth is controlled for. `ms_sparse` at `W=0` sitting at ≈0.027ms
   regardless of `L` (consistent with a pure launch-overhead floor, since real work at
   `W=0` is negligible) is a flagged candidate for part of the story, not confirmed.

4. **Investigated further, then deliberately deprioritized (not resolved, not
   abandoned-by-oversight — a stated call)**: computed achieved bandwidth for the dense
   kernel (`bytes_dense_pred(L) / ms_dense`, averaged across the 11 `W`-rows per `L`
   since dense doesn't depend on `W`):
   ```
   L         achieved GB/s
   8,192      323.7
   16,384     288.7
   32,768     258.5
   65,536     241.1
   131,072    232.5
   163,840    228.5
   ```
   Two real observations: (i) achieved bandwidth is only ~15-20% of an A100's peak HBM
   bandwidth (~1.5-2 TB/s) even at its best; (ii) it declines monotonically, ~29% from
   smallest to largest `L`. Two candidate explanations put forward (not verified with
   profiling):
   - **Low absolute ceiling**: `dense_decode_reference.py` was explicitly built for
     correctness, not performance — no autotuning, no explicit `num_warps`/`num_stages`,
     unlike the real kernels read earlier (the Triton tutorial's autotuned configs,
     vLLM's tuning machinery). This is high-confidence — a direct, expected consequence
     of an already-stated design choice, not a new mystery.
   - **Declining-with-L trend**: hypothesized as GQA cache-reuse fading out. Each
     `(batch, kv_head)` group's K/V is read redundantly by `GQA_GROUP=8` sibling
     query-head programs; at small `L` that working set (~4MB at L=8,192) fits
     comfortably in the A100's 40MB L2 cache, letting sibling reads hit cache instead of
     HBM and inflating measured "achieved GB/s" above what real HBM traffic would give;
     by `L=163,840` a single `(batch,kv_head)`'s K tensor alone is ~40MB (the *entire*
     L2 cache), so the caching benefit disappears and the numbers converge toward the
     kernel's real, still-untuned ceiling. Mechanistically plausible (the 40MB crossover
     landing inside the sweep's own range is suggestive) but genuinely unconfirmed — no
     L2 hit-rate profiling was done. **User judged this not worth pursuing further** —
     explicit scope call, not an oversight — in favor of moving to the KIVI quant layer.

## Scope decision — comparison target (c) descoped

Per spec.md Phase 3, "compare against three things": (a) hand-derived predictions —
done, gap-hunted above. (b) dense baseline — done, gap-hunted above. (c) a published
reference system (vLLM sliding window, StreamingLLM's own numbers, or ASA's ~50%
figure) — **explicitly decided against**, not attempted.

Reasoning: the user asked directly whether (c) and the KIVI quant layer were worth the
remaining time in terms of learning, given the project doesn't have to follow spec.md
verbatim. Assessment given and accepted:
- **KIVI quant layer**: two concrete, distinct payoffs — 2-bit packing/unpacking is a
  genuinely new skill (Triton has no native sub-byte dtype), and it's the only way to
  get real hardware evidence for Phase 2's headline finding (precision's marginal
  multiplier collapsing from ~3.81× to ~1.59× once layered on sparsity). Worth doing.
- **Comparison target (c)**: two possible versions, neither compelling. Standing up a
  real system (e.g. actually running vLLM's sliding-window path) is mostly
  integration/systems work, not conceptual learning — the same reasoning that already
  ruled out forking vLLM's kernel earlier in Phase 3 (too much unfamiliar surface area
  to verify without deep investment). Checking against published numbers from a paper
  is cheap but shallow — a box-check, not a learning experience. Given the project
  explicitly doesn't need spec-completeness for its own sake, not worth the time either
  way.

**Consequence**: Phase 3's comparison is (a) and (b) only, both completed and
gap-hunted on real hardware. `spec.md` annotated in place (not rewritten) to record this
as a stated scope decision, matching this project's own established pattern (Decision 3
in Phase 0, the MLA-scope calls) of keeping rejected/descoped paths on record rather
than silently dropping them.

## Phase 3 continued — KIVI quant layer (2-bit main + fp16 residual)

Picked up from the open thread below: sparse-only kernel was correct and benchmarked,
quant layer was the deferred next step. Full build, real-hardware debugging, two rounds
of optimization, and a final benchmark, all in one continuous push. Code now lives in
`quant_kernel.py` (was `phase3_kernel_scaffold.py`'s second half) — see that file's own
module-level header for a condensed version of this bug list.

### Design, worked out before any code

- **K is per-channel** (each of the 128 channels gets its own scale/zero-point,
  computed from that channel's own values), **grouped in chunks of 32 along seq_len**
  (a channel's token history is chunked into 32-token groups, not one scale for the
  whole history) — matches KIVI's real scheme (Phase 0 reading) and the empirical
  reason it exists: K has channels that are consistently large across every token
  (an outlier-channel pattern), and sharing a scale across channels would let one
  outlier channel's magnitude dictate the step size for 127 well-behaved channels,
  crushing their precision. Per-channel isolates that.
- **V is per-token** (each token gets its own scale, from that token's own 128
  channels), **grouped in chunks of 32 along D_HEAD** (a token's 128-dim vector splits
  into 4 groups of 32 channels) — the mirror image, since V doesn't have K's
  outlier-channel structure; what varies is token-to-token, not channel-to-channel.
- **Packing**: 4 codes/byte (2-bit × 4 = 8 bits), via shift+OR
  (`byte = v0 | (v1<<2) | (v2<<4) | (v3<<6)`). Packed-codes tensor shapes mirror-flip
  between K and V: K keeps `D_HEAD` full-size and shrinks `seq_len` (packed axis =
  `seq_len/4`, scale axis = `seq_len/32`); V keeps `seq_len` full-size and shrinks
  `D_HEAD` (packed axis = `D_HEAD/4`, scale axis = `D_HEAD/32`) — a real structural
  asymmetry, not a coincidence of which axis got divided.
- **Affine quantization formula**: `scale = (max-min)/QMAX`, `zero_point =
  round(qmin - min/scale)`, quantize `q = clamp(round(x/scale)+zero_point, 0, QMAX)`,
  dequantize `x_hat = scale*(q-zero_point)`. Zero-point doesn't need to fall inside
  `[0, QMAX]` itself — it's just a shift constant, usually far outside that range
  since K/V data is rarely centered near zero.
- **Residual/sink/window-old split**: the compacted `[SINK+WINDOW]` cache splits into
  three regimes by physical slot — sink (`[0, SINK)`, always quantized), window-old
  (the older part of the window, quantized), and residual (the most recent
  `RESIDUAL_SIZE=128` tokens, i.e. the *last* slots of the window — never quantized,
  stays fp16). Getting "most recent" right required using slot-index direction, not
  ambiguous language: window slots are laid out oldest-to-newest as slot index
  increases (per `_kv_indices`'s own convention), so residual is the *highest* slot
  indices, not intuitively "the front" or "the back" of anything.
- **Sink padding**: `quantize_k_cache`/`quantize_v_cache` assume `seq_len` is a clean
  multiple of `GROUP_SIZE=32`. `SINK+WINDOW-RESIDUAL_SIZE` (the "main"/quantized
  region's natural size) is *never* a multiple of 32 for any `WINDOW` in the sweep —
  always off by exactly `SINK=4`, since `WINDOW` and `RESIDUAL_SIZE` are both
  multiples of 32 and only `SINK` isn't. Resolved by quantizing sink and window-old
  *separately* (window-old alone is always a clean multiple of 32, no fix needed
  there), and padding sink's 4 real tokens up to one full group of 32 by *repeating
  its own first real token* — not zeros or garbage — so the padding doesn't stretch
  the group's min/max range and dilute precision on the 4 real values.
- **Metadata precision**: scale/zero-point stored fp16, not fp32. At `GROUP_SIZE=32`,
  fp32 metadata makes the *real* effective cost `0.5 bytes/element` (vs. the
  codes-only `0.25` Phase 2 used); fp16 metadata brings it to `0.375`. fp16 chosen
  since fp16's precision on the scale itself isn't the bottleneck — the 2-bit code
  resolution already dominates the error — and it halves a real, non-trivial overhead
  Phase 2's original formula never accounted for at all (flagged, not retroactively
  fixed in Phase 2's own numbers).

### Real bugs found only by running on hardware, in order

1. **Triton version mismatch.** The environment's pip-installed torch bundled Triton
   2.1.0, much older than what these kernels needed — `tl.range` doesn't exist in it
   (`AttributeError: module 'triton.language' has no attribute 'range'`), and its
   `@triton.jit` decorator's constexpr-detection internals broke on this Python/Triton
   combination in an unrelated way (`TypeError: argument of type 'dtype' is not
   iterable`) even on the *already-verified* dense kernel. Fixed with `pip install -U
   triton`, independent of torch's bundled version.
2. **`tl.math.round` doesn't exist on this Triton version** (`tl.math.exp` does, so
   it's specifically `round` that's missing). Replaced with `tl.floor(x + 0.5)`, a
   portable round-half-up substitute — exact `.5`-boundary tie-breaking differs
   slightly from Python's round-half-to-even, a sub-LSB-scale difference not worth
   chasing.
3. **Unsupported tensor row/column indexing.** `q[b*PACK_FACTOR+j, :]` — indexing a
   specific row out of an already-computed 2D tile with a non-Python-int index
   (`b`,`j` came from unrolled loops but still lowered to a runtime-typed index) isn't
   supported on this Triton version (`unsupported tensor index: int32[]`). This was
   flagged as an explicit unverified risk before ever running. Fixed by never
   materializing one big quantized tensor and slicing it — instead, compute each of
   the `PACK_FACTOR` packing positions via a *fresh*, independently strided load, and
   combine with plain elementwise ops. Same fix applied to K (row) and V (column,
   mirror image).
4. **Loop-carried dtype widening.** A `uint8` accumulator (`packed`) built inside a
   loop via `|=`/`<<` gets silently widened to `int32` by Triton's type promotion
   rules — but a loop-carried variable's declared type has to stay fixed across
   iterations (`Loop-carried variable packed has initial type uint8 but is
   re-assigned to int32`). Fixed by declaring the accumulator `int32` from the start
   and casting down to `uint8` once, at the store, not every iteration.
5. **A dtype mismatch in `_load_kv_tile_quantized`'s `tl.where`** (residual load
   stayed native fp16 while the dequantized sink/window branches were fp32) was
   hypothesized as the cause of `test_quantized_attention_correctness.py`'s first
   failure — **wrong hypothesis**, fixing it produced bit-identical results before and
   after, meaning Triton's `tl.where` was already promoting correctly. Real,
   legitimate improvement to make regardless (explicit is better than relying on
   implicit promotion), just not the bug in question — a useful negative result that
   narrowed the search.
6. **The real bug behind that failure: a tile-level clamp corrupted real, in-range
   data.** `win_tok_start = tl.maximum(slot_start - sink, 0)` was meant to keep
   addressing safe, but clamping the *shared, tile-level* starting offset broke the
   `tok_start + tok_in_tile` relationship for *every* position in the first tile, not
   just the genuinely invalid ones — shifting every real window-old position in that
   tile forward by exactly `sink` (4) tokens, which happens to equal `PACK_FACTOR`, so
   every one of them silently read the *adjacent* packed byte instead of its own.
   Symptom matched the mechanism exactly: real-looking-but-wrong-position data (not
   garbage), and `max_abs_diff` shrinking as `WINDOW` grew, since the bug was confined
   to one fixed-size tile whose share of the total output shrinks as the cache grows.
   Root-caused via `test_dequantize_tile_isolated.py` (a new test, written specifically
   to bypass `_load_kv_tile_quantized`'s composition and test `_dequantize_k_tile`/
   `_dequantize_v_tile` alone) coming back bit-perfect — that isolated the bug to the
   composition logic, not the dequant math itself. Fixed by *not* clamping the shared
   tile-level offset — letting it go negative — and instead clamping the per-position
   computed index, inside the dequant functions, which already had an upper-bound
   clamp and just needed a lower-bound one added.

### Optimization: two rounds, one failed, one worked

First benchmark (correct, but naive): `_load_kv_tile_quantized` computes all three
regime candidates (sink dequant, window-old dequant, residual load) unconditionally,
every tile, then selects via `tl.where` ("compute broadly, select narrowly" —
deliberate, for correctness/safety). Since `SINK=4` is tiny relative to `BLOCK_N=128`,
only the very first tile can ever contain a real sink position, yet every tile paid
for a full, wasted sink dequant; residual similarly wasted on tiles that are purely
sink+window-old. Measured: quantized kernel **slower than native sparse everywhere**
(`meas_mult` as low as 0.017–0.018 at large `WINDOW`, vs. a predicted ~5×) — the
opposite of the whole point of quantizing.

**Attempt 1 (failed): runtime `if`/`else`.** Guard each candidate's expensive
load/dequant behind `if slot_start < <regime boundary>:`, falling back to a
zero-filled placeholder otherwise (both branches must define the same variables with
matching types, same discipline as the loop-carried-dtype fix). Measured **~20x
slower**, not faster. Best-guess mechanism (unconfirmed without profiling access):
Triton's scalar `if`/`else` likely doesn't skip real work the way normal Python
control flow implies — plausibly predicated/masked execution under the hood — plus it
may break the loop pipelining tight GPU loops depend on. **Reverted.**

**Attempt 2 (worked): split the loop itself at compile time.** `sink`, `WIN_OLD_END`,
and `BLOCK_N` are all `tl.constexpr` — known when Triton compiles a given kernel
variant, not just at runtime. Restructured `_sparse_decode_attn_kernel_quantized`'s
single loop into three Python-level segments instead: segment 1 (tile 0, always
present, using the general 3-regime `_load_kv_tile_quantized`, since it's the only
tile that can ever mix regimes), segment 2 (the middle stretch, provably pure
window-old, using a new, genuinely branch-free `_load_kv_tile_window_old` — no
sink/residual code even exists in that loop body, not skipped, never generated), and
segment 3 (the window-old/residual boundary + pure residual tail, back to the general
helper). `WIN_OLD_END mod BLOCK_N` turns out to always equal exactly `SINK` for every
`WINDOW` in the sweep (since `WINDOW` and `RESIDUAL_SIZE` are both multiples of
`BLOCK_N`), so segments 1 and 3 stay fixed at ~1-2 tiles regardless of `WINDOW` — only
segment 2 scales, which is exactly the segment kept branch-free. This worked: measured
multiplier roughly doubled-to-tripled across the sweep (e.g. `W=16384`: ~0.33-0.39 →
~0.72; `W=128`: ~0.68 → ~0.96-0.97, essentially matching the 0.986 prediction).

### Final benchmark results (results/benchmark_results_quantized.csv)

- **Quantized-sparse vs. native sparse**: still slower everywhere (`meas_mult` never
  crosses 1.0), but no longer catastrophically so — from ~0.43× to ~0.97× depending on
  `WINDOW`, closest to break-even at `WINDOW=128`.
- **Quantized-sparse vs. dense**: faster for most of the sweep (e.g. `L=163840,
  W=16384`: dense≈90.5ms, quantized≈9.5ms, ~9.5× faster) — but *loses* to dense
  whenever `WINDOW` approaches `L` (e.g. `L=8192, W=8192`: dense=3.25ms,
  quantized=5.97ms). Makes sense mechanistically: at `W≈L` there's no real sparsity
  savings left to offset quantization/dequant overhead, so you're paying the tax with
  nothing to show for it. Not a bug — the honest boundary of where this combination
  actually helps.
- **Remaining gap has a plausible, understood, unchased mechanism**: `_dequantize_k_tile`/
  `_dequantize_v_tile` load scale/zero-point at full `(BLOCK_N, D_HEAD)` resolution even
  though they only vary once every `GROUP_SIZE=32` tokens — real redundant HBM traffic
  that scales with how much of the cache is window-old, matching the gap widening again
  at larger `WINDOW`. Same shape of decision as the dense kernel's own achieved-bandwidth
  question earlier in Phase 3: understood, real, explicitly not chased further —
  diminishing returns after two optimization rounds, with Phase 4 being clearly
  higher-value unexplored territory at this point.

### Correctness tests, in the order they were built

`test_quantize_correctness.py` (write-side: `quantize_k_cache`/`quantize_v_cache`
round-trip against a from-scratch plain-PyTorch reference, including the sink-padding
boundary case) → `test_dequantize_tile_isolated.py` (read-side alone, bypassing
`_load_kv_tile_quantized`'s composition, written specifically to isolate the tile-clamp
bug above) → `test_quantized_attention_correctness.py` (the full path, composing
already-independently-verified reference pieces rather than re-deriving from scratch,
tight ~1e-2 tolerance since both reference and kernel start from identical
already-quantized data). All pass.

## Open Threads Carried Forward

- **Resolved**: the int8/fp16 baseline mismatch between Phase 1/2's formulas and Phase
  3's actual fp16 kernels — see the "fp16 baseline correction" addendum at the end of
  Phase 2. Residual precision corrected from int8 to fp16 (2 bytes); AI ceiling, crossover
  window, and dominance-ratio numbers recalculated for fp16. Full reconciliation of every
  remaining int8-flavored number elsewhere in Phase 1/2 is *not* done — deliberately
  deferred to Phase 4, same scope discipline as the MLA/(c) deferrals below.
- Phase 4's MLA substitution needs an additive indexer-overhead term (DeepSeek's
  published numbers), not a pure `c` swap — flagged in Decision 3, not yet quantified.
  **Sharpened by Phase 2**: also can't shortcut via an AI-ratio multiplier (confirmed
  non-independent) — needs the full `Bytes(W,p)` solve with MLA's `c`, from scratch.
- If Phase 4 wants to compare against disagg's authoritative 830.59 req/s/chip / 5.82:1
  numbers, QKVO needs to be added back in the same way disagg did for dense decode.
- **Phase 3's native-kernel gap-hunt is closed out, not fully solved.** Tiling-boundary
  sawtooth: confirmed. Broader negative→positive `gap_%` trend: a plausible mechanism
  (GQA cache-reuse fading with L2 cache capacity) identified but not
  profiled/confirmed, and the user explicitly chose not to pursue it further — a
  stated scope call, not an abandoned thread. Don't re-open without a specific reason to.
- **Phase 3 is fully done, including the KIVI quant layer.** Real hardware numbers
  confirmed Phase 2's hand-derived residual-dilution effect is real but *not* the whole
  story — a naive kernel implementation (computing all three regimes unconditionally
  per tile) added its own large, separate real-hardware cost (~3x redundant memory
  traffic) that the byte-only hand-derivation structurally couldn't see. That cost was
  mostly (not fully) recovered via a loop-segmentation optimization; the residual gap
  (redundant scale/zero-point loads, ~62-86% below prediction depending on WINDOW) is
  understood but deliberately not chased further. Full narrative, all bugs found, and
  both optimization attempts (one that backfired ~20x, one that worked) are in the
  "Phase 3 continued — KIVI quant layer" section above.
- **Comparison target (c) — explicitly descoped**, see the scope-decision section
  above. Not an open thread; don't re-pick this up without a specific reason to revisit
  the decision.
- **Repo restructured** after Phase 3 closed out: `phase3_kernel_scaffold.py` (1092
  lines mixing native sparse kernel + full quant layer + benchmark harness) split into
  `constants.py`, `dense_kernel.py` (renamed from `dense_decode_reference.py`),
  `sparse_kernel.py`, `quant_kernel.py`, and `benchmark.py`; `triton_fused_attention_
  tutorial.py` deleted (confirmed unused — the reasoning for why it wasn't forked was
  already preserved in `dense_kernel.py`'s own header); CSVs moved into `results/`.
  All 5 test files' imports updated to match. If a file mentioned in an *older* part of
  this log (e.g. `phase3_kernel_scaffold.py`) doesn't exist anymore, this is why —
  check the current top-level file layout, not the historical name.
