# Project Spec: Numerics × Sparsity — the Levers Past GQA

**Continuity note:** this is project #5, following attention (prefill/decode),
MoE routing, and disaggregated serving. Reuses Llama-3-70B/GQA throughout —
every FLOPs/bytes/ridge-point formula from `prefill_notes.md`/`decode_notes.md`
is a direct input here, not re-derived. The new axis this project adds is
**context length itself**, swept from 8,192 (the value used everywhere so
far) up to long-context regimes — because sparsity's entire value
proposition is context-length-dependent in a way nothing prior actually
tested.

**Why this project, precisely:** two threads your own prior work opened and
explicitly declined to close now converge:
- `decode_notes.md` §6 flagged sparse/linear attention as "the natural next
  lever after GQA" — GQA was decode's *only* lever within SDPA, and that
  was stated as a boundary condition, not a dead end.
- `decode_notes.md` Phase 3 found a *hard, quantization-proof floor* for KV
  cache precision (crossover ≈0.033 bytes/element — unreachable). That
  finding was for **one lever in isolation**, at 8,192 context, GQA's bytes
  already baked in. Nobody's asked whether a *different* first lever
  (sparsity, changing which K/V get touched at all, not how many bytes each
  costs) changes where that floor sits.

The real question this project exists to answer: **are numerics and
sparsity independent, multiplicative levers on decode's regime, or does one
eat into the other's headroom** — the same category of question disagg's
Phase 2 answered for attention-vs-FFN (they didn't just add; FFN dominated
and attention became a rounding error). Don't assume the answer going in.

**A discipline to carry in explicitly, not rediscover:** disagg's own
cross-phase synthesis named this as the single thing that mattered most
across every phase — "a reused number or formula needs re-verification for
the *new* context, not just confirmation the formula is right in its
original source." It caught four real, distinct scope/context mismatches
(a borrowed `seq_len`, an incomplete SDPA-only scope, an ungrounded batch
size, a wrong assumed context length) — none were arithmetic errors, all
were reuse-without-recontextualization. This project reuses more than any
prior one (decode's roofline formulas, disagg's MLA cache formula, KIVI's
precision findings) — treat every reused number as a claim to re-check
against *this* project's own operating point, not a fact to import.

**Legend:** 🔧 = boilerplate/setup. 🧠 = your job.

---

## Phase 0 — Setup (🔧, with two 🧠 decisions)

### Reading (🔧)

Three literatures, read for real mechanisms, not just headline numbers:

- **Sparse/structural attention**: StreamingLLM (attention-sink — you've
  already read this once in disagg's Phase 3, re-read with a
  kernel-implementation lens this time, not a placement-policy one),
  Longformer/BigBird (sliding-window + global tokens), H2O (content-aware,
  dynamic eviction based on running attention scores — a genuinely
  different mechanism family from the first three, which are all
  position-based/static).
- **Quantized attention**: KIVI (already read once in disagg §3.1 — same
  re-read-with-new-lens note), SmoothQuant (activation quantization,
  different tensor than KIVI's KV-cache focus), and FlashAttention's own
  published quantized variants if they exist by the time you're doing this
  (search fresh — this space moves fast).
- **Sparsity × latent-compression composability** (new — confirms this
  isn't a speculative combination, it's shipping): DeepSeek V3.2's DSA
  (DeepSeek Sparse Attention) paired directly with MLA, GLM-5 adopting the
  same pairing, and "Alternating Sparse Attention" (arXiv 2511.00819) —
  which reports sparsity + latent-compression **compounding** (an
  additional 50% KV-cache reduction versus a sparse-only baseline, not a
  wash). Read this before Phase 1, not after — it directly informs how
  general to make the Phase 1 formula (see below).

### 🧠 Decision 1: which sparsity mechanism, and why *not* the others

Don't survey all four and hedge — pick one, stated with real reasoning,
mirroring how disagg's Phase 0 picked TPU 8i homogeneous over the
Groq-heterogeneous alternative and kept the rejected path on record. The
axis that actually matters for your decision: **static/position-based**
(sliding window + sink — cheap to reason about, cheap to implement, no
runtime scoring overhead) vs. **dynamic/content-aware** (H2O-style — better
theoretical quality/compression tradeoff, but the eviction-scoring logic
itself adds real compute and memory traffic that a hand-derivation has to
account for, not just the attention math). This choice determines whether
Phase 1's formulas are closed-form (static) or need a real approximation for
the scoring overhead (dynamic) — decide before Phase 1, not during it.

### 🧠 Decision 2: kernel environment

Triton needs a real GPU — Gemmini's RTL work solved its own equivalent
problem by moving to Farmshare (Linux x86_64 target); this project needs its
own answer (a cloud GPU instance, a university cluster allocation, whatever
you actually have access to). Resolve this before Phase 3, not when you get
there — same "check tool-representability/access before committing"
discipline as every prior project's Phase 0.

**Checkpoint:** one sparsity mechanism chosen and defended in a paragraph;
GPU environment identified and access confirmed.

### 🧠 Decision 3: how much MLA to carry, decided now rather than bolted on later

Real precedent (above) confirms sparsity and latent-compression compose —
this is no longer a speculative extension, so it's worth designing Phase 1's
formula to accommodate it from the start rather than retrofitting later.
**Scope call, stated explicitly**: don't run a second full derivation track
for MLA — that's real breadth creep on top of an already two-lever project.
Instead, **parameterize Phase 1's per-position cache-entry size as a
variable** (`c` bytes/position/layer) rather than hardcoding GQA's
`2×d_head×n_kv_heads` term inline. GQA's own value of `c` and MLA's
(disagg's own `(d_c+d_R_h)`, already derived in `disagg §2c.3`) both plug
into the same formula mechanically — primary derivation and kernel work
stay GQA-only (Phase 1–3), and Phase 4's synthesis substitutes MLA's `c` in
after the fact, reusing numbers instead of re-deriving them. Same
discipline as reusing prior projects' FLOPs formulas throughout this repo,
just applied one layer more generically this time because you know in
advance a second substitution is coming.

---

## Phase 1 — Hand-derive the sparsity lever, in isolation (🧠)

**Reuse directly, don't re-derive:** decode's FLOPs/bytes formulas
(`decode_notes.md` §1.1/§1.2), the 81,920-bytes/token KV-cache growth curve
(disagg §1.2), decode's ridge point (480.5, TPU v5e/int8 — or your project's
own chip choice, stated explicitly if you switch).

**The core derivation this phase owes you:** for your chosen mechanism
(e.g. sliding window of size W + S sink tokens), derive attention's FLOPs
and bytes as a function of **both** window size and true context length —
not just context length the way every prior project treated it. This is a
genuinely new formula, not a substitution into an old one. Per Decision 3
above, write the bytes term with cache-entry size as an explicit variable
(`c`) rather than GQA's specific formula inlined — costs nothing now, saves
a re-derivation in Phase 4.

**The question to resolve by derivation, not assumption:** GQA only ever
reduced *bytes* (FLOPs were identical for MHA/GQA, `prefill_notes.md` §1.1,
confirmed again in decode §1.1). Does sparsity reduce FLOPs too, since
you're skipping compute for masked/evicted positions entirely, not just
avoiding a memory fetch? If yes, that's a structurally different kind of
lever than every one used so far in this repo — work out whether it changes
AI's *numerator*, not just denominator, and what that does to the roofline
math.

**Sweep, don't pick one point:** context length from 8,192 (Llama-3's own
native cap, matches every prior project, direct comparability) out to
163,840 (DeepSeek-V2's real deployed cap, YaRN-scaled — now a grounded
anchor from disagg §2b.7, not a borrowed guess). Both endpoints are real,
sourced deployment values, not arbitrary sweep bounds — use them as the two
poles, not just a long-context afterthought. At each context length, derive:
at what window size does sparse attention's AI clear the ridge point, if it
doesn't already? Does the crossover context length (where sparsity starts
mattering at all) match the point where dense decode's KV cache starts
dominating HBM budget (disagg §1.4's own capacity math is your input here)?

**Check for a compute-bound asymptote, explicitly — don't assume sparsity's
gains are visible all the way up.** Disagg found the same structural pattern
independently four times (admission policy §3.3, KV-quantization headroom
§3.5, hot-expert residency §3.8–3.9, the simulator's own occupancy plateau
§4.7): once a system clears its compute-bound crossover, more headroom stops
buying throughput. Check whether that applies here — if sparsity pushes
attention's own AI far enough above the ridge, does something else (QKVO,
FFN, real kernel launch/occupancy overhead) become the actual bottleneck
first, making further sparsity gains real at the op level but invisible at
the system level? Don't assume yes or no; derive it the same way disagg did
each time — by checking where the crossover actually sits relative to your
real operating point, not just confirming the lever exists in principle.

**Checkpoint:** a real, derived formula for sparse-attention FLOPs/bytes as
a function of (window size, context length), plus a stated finding on
whether sparsity is a FLOPs lever, a bytes lever, or both — and why.

---

## Phase 2 — Layer numerics on top, quantify the interaction (🧠)

**Reuse directly:** decode Phase 3's coupled-vs-decoupled precision
framework (§4.1) and its crossover-point method (§4.2) — the *mechanism* for
finding a numerics crossover transfers exactly; what changes is the
starting AI you're solving from.

**The actual new question**: decode Phase 3 solved for the crossover
starting from GQA's AI≈16 (dense KV cache, full context). Now redo that
solve starting from Phase 1's *sparse* AI instead. Two possible outcomes,
don't assume which:

- **Multiplicative / independent**: sparsity moves AI up by some factor,
  numerics still needs to close whatever gap remains to the ridge — the
  crossover point (bytes/element) shifts by exactly sparsity's own AI
  multiplier, a clean, predictable interaction.
- **Non-independent**: something about combining the two mechanisms changes
  the relationship in a way a naive "just multiply the two AI gains
  together" model wouldn't predict — e.g., does quantizing a KV cache that's
  already been sparsified hit a different numerical-sensitivity regime (less
  data to average errors over) than quantizing the full dense cache KIVI was
  designed against? This is worth checking against KIVI's own stated
  precision-vs-error tradeoffs, not assumed away.

**A structural echo worth checking explicitly**: disagg's Phase 2 found FFN
dominance made attention "a rounding error" for chip-ratio purposes — a
combined-lever result that wasn't just the sum of parts. Check directly
whether something analogous happens here (does sparsity's savings dominate
so completely that numerics becomes marginal, the way FFN dominated
attention?), rather than assuming the two levers contribute comparably by
default.

**Checkpoint:** a stated, derived answer — independent/multiplicative or
not — with the mechanism explained, mirroring the rigor of disagg's own
FFN-dominance root-causing (§2.5), not just a number reported without
explanation.

---

## Phase 3 — Real Triton kernel (🔧 build, 🧠 interpret)

This is the project's real departure from prior methodology — validating
against actual measured hardware numbers instead of a simulator/RTL
generator with representability limits.

- Implement your chosen sparse mechanism (Phase 0 Decision 1) as a fused
  Triton kernel, quantized KV cache included (at whatever precision Phase 2
  concluded is the real, non-floor-hitting choice) — start from a reference
  dense FlashAttention Triton kernel (published, real, don't write from
  scratch) and modify, the same "read the real system before hypothesizing
  from nothing" discipline as disagg's Phase 0. Real precedent to design
  against, not guess at: disagg §3.7 confirmed (checking FlashAttention's
  real behavior, not a naive materialize-the-batch model) that fused kernels
  tile K/V and reuse one small, fixed SRAM buffer sequentially — footprint
  bounded by tile size, not by batch or context length. Your sparse kernel's
  SRAM footprint should follow the same shape; if it doesn't, that's worth
  understanding as a real finding, not a bug to paper over.
- Benchmark real wall-clock latency/throughput across the same context-length
  sweep as Phase 1, on real hardware.
- **Compare against three things, not one**: (a) your own Phase 1/2
  hand-derived predictions, (b) a dense FlashAttention baseline at the same
  context lengths, (c) whichever published reference system (vLLM's sliding
  window implementation, StreamingLLM's own reported numbers, or ASA's own
  reported ~50% additional KV-cache reduction if your mechanism resembles
  theirs closely enough to be a fair comparison) is closest to your
  mechanism — same three-way check disagg's Phase 4 ran against
  DistServe/Mooncake.
- **Gap-hunt every disagreement mechanistically** — the throughline across
  all four prior projects, restated because it's the actual point of this
  phase, not a formality.

---

## Phase 4 — Cross-project synthesis (🧠, capstone)

Three questions this project is positioned to answer that none of the first
four could:

1. **Does this change disagg's chip ratio?** Disagg's final, authoritative
   dense ratio is **~5.82:1** (§2.9, QKVO-corrected), against a decode
   throughput of **830.59 req/s/chip** and an FFN compute-bound crossover at
   **N≈296**. If sparse+quantized attention meaningfully changes decode's
   per-chip throughput at long context, check against these specific,
   real numbers, not an approximate range — does the ratio move the way
   MoE's own architecture barely moved it at matched 8,192 context
   (~5.97:1, disagg §2b.19), or does it diverge the way MoE did at its own
   real 163,840-token deployment cap (~1.31:1, §2b.16)? Disagg's own finding
   was that **context length, not architecture family, was the dominant
   lever on the ratio's magnitude** (§5.2) — check whether that finding
   holds for a *sparsity* change too, or whether sparsity moves the ratio by
   a mechanism context-length alone didn't cover.
2. **The MLA substitution (Decision 3), sharpened.** Disagg found MLA isn't
   a strict win over GQA — it's a **context-length-conditional trade**
   (§2b.20/§2b.23): MLA loses to GQA at 8,192 (compute term binds,
   0.0578µs/tok vs. GQA's 0.0128µs/tok) but wins 2.25× at 163,840 (bytes
   term dominates at long context). That means there's a real crossover
   *context length* between the two, not just two disconnected regimes.
   Swap disagg's MLA cache-entry size into Phase 1's parameterized formula
   and ask the sharper question directly: **does sparsity shift where that
   MLA-vs-GQA crossover sits** — pulling it toward shorter context (making
   MLA the better choice sooner) or pushing it further out? That's a more
   specific, more useful finding than "does the interaction hold," and it's
   now answerable because disagg already located the crossover's rough
   position. Separately, sanity-check the *combined* sparsity+MLA saving
   against ASA's reported ~50% additional-reduction figure, as before.
3. **Real vs. predicted gap, one more data point on a running theme.** Every
   project in this repo has found real, mechanistically-explained gaps
   between hand-analysis and ground truth (Timeloop's mapper local-optima,
   Gemmini's WS-only-is-a-control-constant finding, the FFN-omission
   correction in disagg). Does a real Triton kernel on real hardware surface
   a *new category* of gap those tools structurally couldn't have shown you
   (e.g. real memory-controller behavior, real occupancy/launch overhead) —
   or does it mostly confirm what the hand-derivation already predicted? Say
   which, and why, as the closing finding.

---

## Note on scope

The two-lever framing (sparsity + numerics) is already a real risk of
breadth. The MLA checkpoint added above (Decision 3, Phase 4 Q2) is
deliberately kept cheap *because* of that risk — a formula substitution and
a sanity check against a real published number, not a second full
derivation/kernel track. If it stops being cheap (if the substitution
doesn't fall out cleanly, or the regime interaction needs real new
derivation rather than reuse), that's a signal to write it up as a flagged
open thread for a future project, not to expand this one's scope
mid-stream — the same move this repo made with Groq/Cerebras, routing-aware
batching, and sparse/linear attention itself before this project existed.
Resist adding a genuinely new mechanism family on top of all this (e.g.
linear attention's totally different math, or MoE-style expert-sparsity-for-
attention) — that's a real fourth lever, not a cheap checkpoint, and belongs
in its own future project if it comes up.

## Fallback

Phases 1–2 (hand-derivation, no kernel) stand alone as a complete artifact —
a real, derived answer to "are sparsity and numerics independent levers,"
even without Phase 3's kernel or Phase 4's synthesis, if GPU access or time
runs short.