# Handoff — numerics-and-sparse-attn, about to start Phase 3

Read this + `notes.md` (full narrative log) + `spec.md` (the project brief) to pick up
exactly where the last session left off. Everything below is settled; nothing here needs
re-derivation unless you find a real error in it. **Phase 3 is a big scope shift — real
Triton kernel work — starting in a fresh chat, per the user.**

## How this project works (read this first)

This is a self-directed derivation project. Sections marked 🧠 in `spec.md` are the
user's own hand-derivation work — **the assistant's role is sounding board and
correctness-checker, not derivation engine.** Concretely, established over the last two
sessions:
- When the user proposes a formula/derivation step, check it, point out errors precisely
  (cite the source doc/section it should match), don't just hand them the answer.
- When the user explicitly delegates a mechanical step ("plug in numbers", "do the
  sweep", "just do it for me quickly", literature research), do it directly and report
  results plainly — don't over-interpret or conclude on their behalf.
- Don't do 🧠-marked derivation work unprompted, even when it would be faster.
- When the user questions whether a planned step is even worth doing, give an honest,
  concrete recommendation (not just "yes because the spec says so") — cite what's
  actually at stake (does it change a downstream decision, does it reveal a mechanism,
  is the effect size plausibly large) and let them decide. This came up twice in Phase 2
  (the crossover solve itself, and the residual-buffer check) — both times the honest
  answer was "yes, and here's why," but the reasoning had to be concrete, not reflexive.
- Push back on formula errors immediately and specifically — this repo's whole
  methodology is "gap-hunting is the highest-value activity," and that applies to
  catching the user's own mistakes mid-derivation, not just tool-vs-hypothesis gaps. Two
  real catches in Phase 2: a missing `batch` factor in a generalized bytes formula, and
  an inconsistent `min(W+S,L)` applied to bytes but not FLOPs.
- Real research (web search) is appropriate and was used for Phase 0's reading — this
  project explicitly wants real mechanisms, not assumed ones.

## Reference docs (read if you need the numbers behind anything below)

- `spec.md` — the project brief, full phase breakdown, all reused-number citations.
- `notes.md` — full narrative log of everything found/derived so far (Phase 0 reading in
  full, all three Decision write-ups, complete Phase 1 and Phase 2 derivations with
  sanity checks).
- `../workload-to-silicon/decode_notes.md` — dense GQA decode roofline (FLOPs/bytes/AI
  formulas this project's Phase 1 modified), Phase 3's coupled/decoupled numerics
  framework (§4.1) and crossover method (§4.2) — both reused directly in this project's
  own Phase 2.
- `../workload-to-silicon/disagg_and_placement_notes.md` — KV-cache capacity formula
  (§1), MLA cache-entry formula (§2c), dense/MoE chip ratios (§2.9/§2b), the
  compute-bound-asymptote pattern (§3.3/3.5/3.8-3.9/§4.7), FFN-dominance finding (§2.5,
  the pattern this project's Phase 2 checked for and did *not* find).

## Phase 0 — done

**Decision 1 (sparsity mechanism): sliding window (W) + attention sink (S=4),
StreamingLLM-style.** Chosen over H2O because it's provably closed-form. Full reasoning
in `notes.md`.

**Decision 2 (GPU environment): cloud GPU instance, provider/instance still TBD.**
**This is now blocking — Phase 3 needs it resolved first, before any kernel work
starts.** Nothing else deferred it further; this is the actual next action.

**Decision 3 (MLA carry-forward):** `c` bytes/position/layer parameterization confirmed
viable for the bytes term, refined to need an additive indexer-overhead term for Phase 4
(DeepSeek/GLM's separate lightning-indexer mechanism), and **further sharpened by Phase
2**: the substitution can't shortcut through an AI-ratio multiplier either (confirmed
non-independent interaction) — Phase 4 needs the full `Bytes(W,p)` solve redone with
MLA's own `c`, not a ratio trick.

**Scope confirmed: SDPA-only.** Flag carried forward: Phase 4 needs QKVO added back in
if comparing against disagg's authoritative 830.59 req/s/chip / 5.82:1 numbers.

## Phase 1 — done

**Derived formulas** (batch=32, n_heads=64, n_kv_heads=8, d_head=128, int8, S=4):
```
FLOPs(W) = 1,048,576 × (W+4)
Bytes(W) = 786,432 + 65,536×W
AI(W)    = 16(W+4)/(W+12)
```
Sanity check passed: recovers dense GQA's exact numbers at W+4=8192.

**Findings:**
1. Sparsity is a genuine FLOPs lever (unlike GQA), but bounded — fixed Q+O byte floor
   has no FLOPs analog, so AI is capped, not linearly improved.
2. **AI(W) ceiling = exactly 16** (dense GQA's own AI), strictly increasing, never
   exceeded. W=256→AI≈15.52, W=32→≈13.09, W=0→≈5.33.
3. Sparse AI can **never** clear the ridge (480.5) at any W, any context length.
4. Context-length sweep is an **AI non-event** — formula has zero `L` dependence once
   `L≥S+W`. This held for every later Phase 2 result too (precision-alone ratios were
   L-invariant, sparse bytes terms were L-invariant).
5. But **absolute savings ratio (dense/sparse) grows linearly with L**: ≈31.5× at
   L=8,192 vs. ≈630× at L=163,840. This became the mechanism behind Phase 2's dominance-
   ratio growth too.
6. Compute-bound-asymptote check: moot for sparsity alone — **resolved in Phase 2** (see
   below), still moot even combined with realistic precision.
7. disagg §1.4 HBM-capacity-crossover sub-question: judged moot, skipped (reasoning in
   `notes.md`).

## Phase 2 — done, checkpoint reached

Full derivation, all formulas, and all sanity checks are in `notes.md`'s Phase 2
section. Summary of what was resolved:

1. **Crossover solve**, generalizing decode's own method with cache precision `p` as a
   variable: `p(W) = 32/961 − 8/(W+4)`. Recovers decode's own ~0.033 dense floor exactly
   as `W→∞` (sanity check passed). Goes **negative below W≈236** — ridge is
   algebraically unreachable there, not just impractical. At W=256, `p(256)≈0.00253`
   bytes/element — ~100× smaller than KIVI's own smallest published group (2-bit≈0.25 B).
   **Precision cannot be a compute-bound lever, confirmed, at any window size.**

2. **Multiplicative-vs-non-independent check**: naive product-form prediction vs. the
   actual difference-form `p(W)` diverge both quantitatively (~12.8× at W=256) and
   qualitatively (naive never predicts the negative-p/unreachable regime). **Verdict:
   non-independent**, tracing to the same fixed Q+O bytes term that bounded Phase 1's AI
   — one structural cause, two manifestations.

3. **Dominance check** (disagg §2.5 FFN-dominance echo): ran a 2×2 factorial
   (dense/sparse × normal/low-precision) at W=256, p_low=0.25 (KIVI 2-bit). Idealized
   result: sparsity-alone 30.6×, precision-alone ~4×, naive-vs-actual gap only ~8.6% at
   L=8,192 (grows to ~8.9% at L=163,840, roughly stable). Dominance ratio (sparsity ÷
   precision) grows **linearly with L**, tracking the L-ratio almost exactly (20× for
   20×). **Verdict: not FFN-style dominance** — both levers contribute real, comparable-
   order-of-magnitude savings; sparsity's advantage over precision widens with context
   length but precision never becomes a rounding error.

4. **Residual-buffer correction** (the real headline result): KIVI's ~128-token
   uncompressed residual is only ~1.6% of a dense L=8,192 cache but **~49% of a W=256
   sparse cache**. Recomputed with the residual at int8 (this project's own baseline
   precision, not KIVI's paper-native FP16 — a deliberate, stated choice). Result:
   **combined realistic savings collapse from 112.3× (idealized) to 48.5×** — naive
   model now overstates by **>140%** (was ~8.6% before this correction). Precision's own
   *marginal* multiplier on top of sparsity drops from ~3.81× to **~1.59×**. Mechanism:
   not a 50/50 dilution — uncompressed tokens cost 4× more bytes/position than
   compressed ones, so the ~49%-by-count residual claims ~80% of the actual bytes.

**Phase 2's answer for what Phase 3 should implement:** KIVI's 2-bit (p=0.25
bytes/element) main cache **with an int8 residual for the most recent ~128 tokens** —
not the crossover-derived `p(W)` (unrealizable) and not an idealized uniform-precision
cache (Phase 2 showed this overstates savings by >2×).

## Next: Phase 3 — real Triton kernel (new chat)

Per `spec.md`:
- **Resolve Decision 2 first** (cloud GPU instance) — nothing else can start without it.
- Start from a **published, real dense FlashAttention Triton kernel** (don't write from
  scratch) and modify — same "read the real system before hypothesizing" discipline as
  disagg's Phase 0.
- Implement the chosen mechanism: sliding window (W) + sink (S=4), StreamingLLM-style,
  **with the KIVI-style quantized KV cache scoped above** (2-bit main + int8 residual,
  ~128 tokens) — this is the "real, non-floor-hitting choice" Phase 2 was asked to
  produce, and it now has one.
- Design the kernel's SRAM footprint against disagg §3.7's confirmed real behavior
  (tile-bounded, not batch/context-bounded) — if the sparse+quantized kernel's footprint
  doesn't follow that shape, that's a real finding, not a bug to paper over.
- Benchmark real wall-clock across the same context-length sweep as Phase 1 (8,192 →
  163,840).
- **Compare against three things**: (a) this project's own Phase 1/2 hand-derived
  predictions — including the residual-dilution effect, which is exactly the kind of
  thing real kernel overhead (dequant cost, non-contiguous residual/main-buffer access)
  could make *worse* than hand-derived, flagged as a live "real vs. predicted gap"
  candidate in `notes.md`'s open threads; (b) dense FlashAttention baseline; (c) a
  published reference system (vLLM sliding window, StreamingLLM's own numbers, or ASA's
  ~50% figure if the mechanism is close enough to compare fairly).
- Gap-hunt every disagreement mechanistically — the throughline across all four prior
  projects in this repo, not a formality here either.

Nothing else is blocking beyond Decision 2 (GPU access) — go straight into Phase 3 setup
once that's resolved.
