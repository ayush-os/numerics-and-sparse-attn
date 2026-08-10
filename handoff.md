# Handoff — numerics-and-sparse-attn, mid-Phase 3

Read this + `notes.md` (full narrative log, including the complete Phase 3 bug list and
gap-hunt) + `spec.md` (the project brief) to pick up exactly where the last session left
off. Everything below is settled unless explicitly marked open. **The sparse kernel is
built, correctness-verified, and benchmarked on real hardware (A100). Next real chunk of
work is finishing the gap-hunt and/or building the KIVI quant layer.**

## How this project works (read this first)

This is a self-directed derivation project. Sections marked 🧠 in `spec.md` are the
user's own hand-derivation/implementation work — **the assistant's role is sounding
board and correctness-checker, not derivation engine.** Established over Phases 0-2 and
reconfirmed through Phase 3's kernel work:

- When the user proposes a formula/derivation/code step, check it, point out errors
  precisely (cite the source/section it should match), don't just hand them the answer.
- When the user explicitly delegates a mechanical step ("plug in numbers," "do the
  sweep," "write the harness," literature research), do it directly and report results
  plainly — don't over-interpret or conclude on their behalf.
- Don't do 🧠-marked derivation/implementation work unprompted, even when it would be
  faster — but do the surrounding boilerplate/infra directly once it's genuinely
  separable from the content (grid/launch mechanics, online-softmax bookkeeping, a
  from-scratch dense-baseline kernel with no sparsity-selection logic to preserve, a
  correctness-test *harness* while leaving the reference's actual logic to the user or a
  real third-party source).
- **New in Phase 3, worth carrying forward: writing code has the same given/theirs split
  as writing math.** The sparse kernel's masking/selection logic (`_kv_indices`) stayed
  the user's own work through several correction rounds; the surrounding plumbing
  (kernel launch, benchmark harness, the dense reference kernel) didn't.
- **A correctness reference must be genuinely independent to mean anything.** Declined
  twice to write `test_sparse_correctness.py`'s reference mask myself, even when asked
  directly — since I'd already fed the user part of `_kv_indices`'s own logic earlier in
  conversation, writing the "independent" check myself would've meant checking my own
  answer against itself. Resolved by sourcing the mask from a real third party
  (`mit-han-lab/streaming-llm`'s actual `kv_cache.py`) instead of either of us
  re-deriving it.
- When the user questions whether a planned step is even worth doing, give an honest,
  concrete recommendation (not just "yes because the spec says so") — cite what's
  actually at stake. Came up repeatedly in Phase 3: whether to fork vLLM's kernel
  (declined — too large/unverifiable without a GPU) vs. write a compact purpose-built one
  (done); whether BLOCK_N tiling was worth doing before benchmarking (yes — it became a
  hard compiler-limit blocker, not just a style concern).
- Push back on errors immediately and specifically — this repo's whole methodology is
  "gap-hunting is the highest-value activity," now extended from formula errors to real
  code bugs and to interpreting benchmark data (e.g. correcting "small L OR small W
  causes overprediction" to the more precise "small W drives it, L shifts where the
  crossover sits" — backed by specific counter-example rows, not just asserted).
- Real research (web search, fetching real source code) is appropriate and was used
  repeatedly in Phase 3 — checking what vLLM's kernel actually contains before deciding
  whether to fork it, checking StreamingLLM's real reference implementation before
  writing a correctness mask, matches Phase 0's "real mechanisms, not assumed ones."
- **Git**: user wants granular commits ("more commits the merrier"), one per logical
  unit of work, not one per file-touch — established and repeated throughout Phase 3.
  Only commit/push when explicitly asked, but once asked, split sensibly without being
  asked how.

## Reference docs

- `spec.md` — the project brief, full phase breakdown, all reused-number citations.
- `notes.md` — full narrative log: Phase 0 reading, all three Decision write-ups,
  complete Phase 1/2 derivations, and Phase 3's full bug list + correctness methodology
  + benchmark gap-hunt (in progress).
- `../workload-to-silicon/decode_notes.md` — dense GQA decode roofline, Phase 3's
  coupled/decoupled numerics framework (§4.1) and crossover method (§4.2), both reused
  in this project's Phase 2.
- `../workload-to-silicon/disagg_and_placement_notes.md` — KV-cache capacity formula,
  MLA cache-entry formula, the compute-bound-asymptote pattern, FFN-dominance finding
  (checked for and *not* found in this project's Phase 2).

## Phases 0-2 — done

Full derivations in `notes.md`. Summary:

- **Decision 1**: sliding window (W) + attention sink (S=4), StreamingLLM-style —
  chosen over H2O for being provably closed-form.
- **Decision 2**: cloud GPU — **resolved in Phase 3**, see below.
- **Decision 3**: MLA `c`-substitution viable for bytes, needs an additive
  indexer-overhead term for Phase 4, and can't shortcut through an AI-ratio multiplier
  (non-independent interaction, confirmed in Phase 2).
- **Phase 1**: `FLOPs(W)=1,048,576×(W+4)`, `Bytes(W)=786,432+65,536×W` (int8),
  `AI(W)=16(W+4)/(W+12)`. AI ceiling = exactly 16, never clears ridge (480.5) at any W.
  Savings ratio (dense/sparse) grows linearly with L: ≈31.5× at L=8,192, ≈630× at
  L=163,840.
- **Phase 2**: sparsity and precision are **not independent multiplicative levers**
  (difference form, not product form — same root cause as Phase 1's bounded AI). KIVI's
  ~128-token residual is ~49% of a W=256 sparse cache (vs. ~1.6% of a dense L=8,192
  cache) — precision's marginal multiplier on top of sparsity collapses from ~3.81× to
  ~1.59× once this is accounted for. **Landed choice for Phase 3**: KIVI 2-bit main +
  int8 residual (~128 tokens), not the crossover-derived `p(W)` (unrealizable) or an
  idealized uniform-precision cache (overstates savings >2×).

## Phase 3 — in progress

**Decision 2 resolved**: cloud A100 (Paperspace), 80GB preferred. Reasoning: decode is
memory-bandwidth-bound throughout (Phase 1's own finding), so A100's bandwidth is
plenty and frontier compute (H100) isn't needed; sizing driven by the dense baseline's
K/V tensors at the largest sweep context (`L=163,840`) needing ≈20GiB alone.

**What's built and verified:**
- `dense_decode_reference.py` — compact, purpose-built dense-causal-GQA kernel
  (comparison target (b)). Not forked from a real system — both real candidates checked
  (Triton tutorial: no GQA; vLLM: unverifiable without a GPU at ~1,600 lines) were worse
  fits than writing something small enough to reason about directly. **Correctness
  verified** (`test_dense_correctness.py`, all pass, including past the int32 overflow
  crossover at L=131,072).
- `phase3_kernel_scaffold.py` — sparse sliding-window+sink decode kernel, native fp16
  (quant deliberately not yet added, per the staged plan). **Correctness verified**
  (`test_sparse_correctness.py`, all pass — pre-sink, mid-ramp-up, steady-state, and
  W=0). BLOCK_N=128 tiled with online-softmax accumulation.
- `triton_fused_attention_tutorial.py` — vendored Triton docs kernel, kept for
  reference, not the active dense baseline.
- Full 66-point benchmark sweep (`CONTEXT_LENGTHS × WINDOW_SIZES`) run clean.
  **Results exist only on the remote GPU box as `benchmark_results.csv` — not yet
  copied into this repo.**

**Real bugs found and fixed via iterative hardware testing (full detail + root causes
in notes.md's Phase 3 section)** — worth skimming even if not touching this code, since
several are genuinely reusable Triton lessons, not one-offs:
1. Missing kv_head offset / head-dim broadcast in early KV-tile loading.
2. `tl.arange` requires compile-time-constant bounds — can't use a runtime value
   directly, need a compile-time-sized range plus a runtime offset add.
3. Bare Python globals aren't visible inside `@triton.jit` functions in this Triton
   version — need explicit `tl.constexpr` parameters.
4. `tl.cat` needs `can_reorder=True`, and even then two separate `tl.cat` calls aren't
   guaranteed to reorder identically — real risk of index/mask arrays decoupling.
5. `tl.arange`'s range must be a power of 2.
6. **The real headline bug**: `_kv_indices` returned logical sequence positions instead
   of physical cache slots, present since the very first draft, invisible until it
   caused a genuine CUDA illegal-memory-access. Every earlier fix (masking, padding)
   preserved this same confusion instead of catching it.
7. Triton's max single-tensor size (1,048,576 elements) exceeded at large WINDOW with
   the original single-block design — fixed by restructuring to BLOCK_N-tiled
   online-softmax, which also eliminated the power-of-2-padding workaround from #5
   entirely.
8. int32 overflow in the dense kernel's offset arithmetic at large context lengths
   (crossover ≈67,653 for this workload's sizes) — fixed by typing `seq_len: tl.int64`.
   The sparse kernel is structurally immune for the current sweep, since its addressable
   range is bounded by `SINK+WINDOW`, never the full context length — the
   cache-compaction that's the whole point of the mechanism buys this safety margin,
   not luck.

**Benchmark gap-hunt — in progress, not resolved:**
- Dense-recovery sanity check (W≈L cases) passes clean on real hardware: gap_% ≈0.5-1.3%.
- Large systematic pattern in `gap_%`: strongly negative (formula overpredicts speedup)
  at small W, positive (underpredicts) at large W, crossover W shrinking as L grows.
- **Confirmed sub-mechanism**: `BLOCK_N=128` tiling creates a sawtooth in `gap_%`,
  empirically verified via a targeted `W=124/125/128/256` test (a 1-unit change from
  W=124→125 caused a +66% wall-clock jump — pure tile-count-boundary effect, not real
  work). Full mechanism and reasoning in notes.md.
- **Open**: whether this tiling artifact is the whole story behind the broader trend, or
  whether there's an additional smooth component (launch-overhead amortization,
  achieved-bandwidth differences) underneath it once the sawtooth is controlled for.

## Next: pick up here

No single blocking item — these are parallel, pick based on priority:

1. **Finish the gap-hunt** on the existing sweep data (see open question above). Doesn't
   need the GPU — `benchmark_results.csv` has everything, once it's copied off the
   remote box.
2. **Build the KIVI quant layer** (2-bit main + int8 residual, ~128 tokens) on top of
   the now-verified sparse kernel — Phase 2's landed choice, deliberately deferred until
   correctness was established. This is genuinely new kernel work (2-bit packing has no
   native Triton dtype — needs manual bit-packing/unpacking), needs its own staging
   discussion, and needs the GPU. Its own correctness check should follow the same
   independent-reference discipline as the sparse/dense kernels — don't let the
   assistant write the packing/dequant logic itself if it's the object of study, same as
   `_kv_indices` wasn't.
3. **Comparison target (c)**: a published reference system (vLLM sliding window,
   StreamingLLM's own numbers, or ASA's ~50% figure) — not started at all.
4. Copy `benchmark_results.csv` back from the remote GPU box into the repo if the raw
   sweep data should be preserved alongside the analysis.

GPU instance status: recommended shutting it down after the last verification
experiment (the W=124/125 boundary test) — no further GPU work was pending at handoff
time. A fresh instance (same A100 sizing guidance above) will need provisioning again
for items 2-4.
