# Handoff — numerics-and-sparse-attn, starting Phase 4

Read this + `notes.md` (full narrative log, including the complete Phase 3 bug list,
both quant-layer optimization attempts, and the final benchmark gap-hunt) + `spec.md`
(the project brief; `spec.md`'s Phase 3 section has an inline note on why comparison
target (c) was descoped) to pick up exactly where the last session left off. Everything
below is settled unless explicitly marked open.

**Status: Phase 3 is fully done** — sparse kernel and KIVI quant layer both built,
correctness-verified, and benchmarked on real hardware (A100). Gap-hunt closed out for
both (native sparse: a deliberate stopping point; quant layer: two optimization rounds,
one that backfired ~20x and one that worked, remaining gap understood and deliberately
not chased further). Comparison target (c) is explicitly descoped, not just skipped.
Repo was restructured after Phase 3 closed — see "Current file layout" below before
assuming any file name from an older part of this doc still exists.
**The next action is Phase 4 (cross-project synthesis) — jump to "Next: Phase 4" below.**

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
  real third-party source). Once the user has genuinely settled a design through their
  own back-and-forth (not just asked "is this syntax yet" reflexively), writing the
  actual Triton syntax is fair game if they ask — confirmed repeatedly in the quant
  layer build (`quantize_k_cache`, the dequant tile functions, `_load_kv_tile_quantized`
  were all written directly once design questions were actually resolved, not before).
- **Writing code has the same given/theirs split as writing math.** The sparse kernel's
  masking/selection logic (`_kv_indices`) stayed the user's own work through several
  correction rounds; surrounding plumbing (kernel launch, benchmark harness, the dense
  reference kernel) didn't.
- **A correctness reference must be genuinely independent to mean anything.** Declined
  to write `test_sparse_correctness.py`'s reference mask directly — sourced from a real
  third party (`mit-han-lab/streaming-llm`'s actual `kv_cache.py`) instead. Same
  discipline extended in the quant layer: `test_quantize_correctness.py`'s reference
  reused nowhere else, and `test_dequantize_tile_isolated.py` was written specifically
  to test the dequant kernels *without* going through the composition logic that had a
  real bug in it — isolating debugging surface, not just checking correctness once.
- When the user questions whether a planned step is even worth doing, give an honest,
  concrete recommendation — cite what's actually at stake, not just "yes because the
  spec says so." Came up repeatedly: whether to fork vLLM's kernel (declined), whether
  BLOCK_N tiling was worth doing before benchmarking (yes, became a hard compiler-limit
  blocker), and — new in the quant layer — whether a "obviously right" optimization
  (skip regime candidates a tile can't contain, via a runtime `if`) was actually worth
  doing before being honest that it was a real gamble, not a guaranteed win. It measured
  ~20x *slower*. A second, structurally different attempt (splitting the loop into
  compile-time-bounded segments instead of branching at runtime) worked. **Lesson
  worth carrying forward: an intuitive-sounding kernel optimization can be flatly wrong
  on real hardware — say so honestly before and after, don't retroactively rationalize
  either the guess or the failure.**
- Push back on errors immediately and specifically — this repo's whole methodology is
  "gap-hunting is the highest-value activity," extended from formula errors to real code
  bugs, benchmark-data interpretation, and now kernel-optimization intuitions too.
- Real research (web search, fetching real source code) is appropriate and was used
  repeatedly — checking what vLLM's kernel actually contains before deciding whether to
  fork it, checking StreamingLLM's real reference before writing a correctness mask.
- **Git**: user wants granular commits ("more commits the merrier"), one per logical
  unit of work. Only commit/push when explicitly asked, but once asked, split sensibly
  without being asked how.

## Current file layout

```
constants.py         shared workload constants (BATCH, N_HEADS, D_HEAD, SINK_SIZE,
                      RESIDUAL_SIZE, GROUP_SIZE, QUANT_BITS, sweep ranges, ...)
dense_kernel.py       dense causal-GQA decode kernel (comparison target (b))
sparse_kernel.py      sliding-window+sink decode kernel, native fp16 (comparison target (a))
quant_kernel.py       KIVI quant layer on top of sparse_kernel.py (2-bit main + fp16 residual)
benchmark.py          both sweeps (benchmark(), benchmark_quantized()), saves to results/
test_dense_correctness.py
test_sparse_correctness.py
test_quantize_correctness.py             quantize_k_cache/quantize_v_cache round-trip
test_dequantize_tile_isolated.py         dequant kernels alone, bypassing composition
test_quantized_attention_correctness.py  full quantized attention path
results/benchmark_results.csv            dense/sparse sweep (66 rows)
results/benchmark_results_quantized.csv  quantized sweep (48 rows)
spec.md, notes.md, handoff.md
```

Renamed/removed from earlier in this project's history: `phase3_kernel_scaffold.py`
(split into `sparse_kernel.py` + `quant_kernel.py` + `benchmark.py`),
`dense_decode_reference.py` (renamed `dense_kernel.py`),
`triton_fused_attention_tutorial.py` (deleted — confirmed unused; the reasoning for why
it wasn't forked is preserved in `dense_kernel.py`'s own header).

## Reference docs

- `spec.md` — the project brief, full phase breakdown, all reused-number citations.
- `notes.md` — full narrative log: Phase 0 reading, all three Decision write-ups,
  complete Phase 1/2 derivations, Phase 3's native-kernel bug list + gap-hunt, and
  Phase 3's KIVI quant-layer build (design, real bugs, both optimization attempts,
  final benchmark).
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
- **Decision 2**: cloud GPU — resolved (Paperspace A100), see Phase 3 below.
- **Decision 3**: MLA `c`-substitution viable for bytes, needs an additive
  indexer-overhead term for Phase 4, and can't shortcut through an AI-ratio multiplier
  (non-independent interaction, confirmed in Phase 2).
- **Phase 1**: `FLOPs(W)=1,048,576×(W+4)`, `Bytes(W)=786,432+65,536×W` (int8),
  `AI(W)=16(W+4)/(W+12)`. AI ceiling = exactly 16, never clears ridge (480.5) at any W.
  Savings ratio (dense/sparse) grows linearly with L: ≈31.5× at L=8,192, ≈630× at
  L=163,840.
- **Phase 2**: sparsity and precision are **not independent multiplicative levers**.
  **Landed choice for Phase 3**: KIVI 2-bit main + **fp16 residual** (~128 tokens).
  **Corrected post-kernel-build** (`notes.md`'s fp16-baseline addendum): Phase 1/2's
  formulas originally assumed an int8 baseline, but Phase 3's kernels are fp16
  throughout. Under the corrected baseline, precision's marginal multiplier collapses
  from ~7.17× (realistic, dense) to ~1.76× once layered on sparsity. Full reconciliation
  of the rest of Phase 1/2's int8-flavored numbers deliberately deferred to Phase 4.

## Phase 3 — done

**Decision 2 resolved**: cloud A100 (Paperspace), 80GB. Reasoning: decode is
memory-bandwidth-bound throughout (Phase 1's own finding), so A100's bandwidth is
plenty; sizing driven by the dense baseline's K/V tensors at the largest sweep context
(`L=163,840`) needing ≈20GiB alone.

**Native sparse kernel** (`dense_kernel.py`, `sparse_kernel.py`) — correctness-verified
(`test_dense_correctness.py`, `test_sparse_correctness.py`, all pass), benchmarked
(`results/benchmark_results.csv`, 66 rows). 8 real hardware bugs found and fixed (full
list in `notes.md`), headline one being `_kv_indices` originally returning logical
positions instead of physical slots — invisible until it caused a real CUDA
illegal-memory-access. Gap-hunt closed out (tiling-boundary sawtooth confirmed;
achieved-bandwidth question investigated then deliberately deprioritized — a stated
scope call, not an oversight).

**KIVI quant layer** (`quant_kernel.py`) — 2-bit main cache (K per-channel, V
per-token, group size 32) + fp16 residual (most recent 128 tokens), on top of the
sparse kernel. Correctness-verified three ways (`test_quantize_correctness.py`,
`test_dequantize_tile_isolated.py`, `test_quantized_attention_correctness.py`, all
pass), benchmarked (`results/benchmark_results_quantized.csv`, 48 rows). 6 more real
hardware bugs found (Triton version mismatch, `tl.math.round` missing, unsupported
tensor row/column indexing, loop-carried dtype widening, and — the real headline one —
a tile-level clamp that corrupted real in-range window-old data in the first tile,
root-caused via the isolated dequant test). Two optimization rounds after the first
benchmark came back with quantized *slower* than native sparse everywhere: a runtime
`if` to skip inapplicable regime computation measured ~20x slower (reverted); splitting
the outer loop into compile-time-bounded segments instead worked (measured multiplier
roughly doubled-to-tripled). Full narrative, mechanisms, and exact numbers in `notes.md`.

**Comparison target (c) — explicitly descoped**, not just skipped. Per spec.md Phase 3,
three comparisons were named: (a) hand-derived predictions — done, both kernels. (b)
dense baseline — done, both kernels. (c) a published reference system — decided
against; two versions considered (standing up vLLM for real, or a shallow literature
box-check against a paper's numbers), neither compelling relative to the KIVI quant
layer's own concrete payoffs. `spec.md` has an inline annotation recording this.

## Next: Phase 4

Cross-project synthesis — the capstone, and per the user's own stated interest, the
part of prior projects (workload-to-silicon's cross-project synthesis) that was
explicitly called out as "the coolest part," more engaging than single-project
derivation phases even when those produced expected results. Look for and lean into
cross-cutting resource-tradeoff connections, not siloed phase-by-phase analysis.

Three questions spec.md poses for Phase 4, all still fully open:

1. **Does this change disagg's chip ratio?** Compare against disagg's authoritative
   ~5.82:1 dense ratio, 830.59 req/s/chip decode throughput, and N≈296 FFN
   compute-bound crossover. Disagg's own finding was that context length, not
   architecture family, was the dominant lever on the ratio's magnitude (§5.2) — check
   whether that holds for a sparsity+quantization change too, or whether it moves the
   ratio by a mechanism context-length alone didn't cover. Needs QKVO added back in to
   compare against disagg's specific numbers (this project has been SDPA-only
   throughout, a scope call already flagged, not yet corrected).
2. **The MLA substitution, sharpened**: does sparsity shift where disagg's MLA-vs-GQA
   crossover context length sits? Substitute MLA's cache-entry size into Phase 1's
   parameterized `c` formula (Decision 3) and re-solve. Per Decision 3's own notes, this
   needs an additive indexer-overhead term too, not a pure `c` swap, and can't shortcut
   through an AI-ratio multiplier (Phase 2 already showed this project's own two levers
   aren't independent that way — the MLA substitution likely won't be either). Also
   sanity-check the combined sparsity+MLA saving against ASA's reported ~50%
   additional-reduction figure.
3. **Real vs. predicted gap, the closing finding across the whole repo**: does a real
   Triton kernel on real hardware surface a *new category* of gap the earlier
   projects' tools (Timeloop, Gemmini's RTL generator, the disagg simulator) structurally
   couldn't have shown — or does it mostly confirm what hand-derivation already
   predicted? This project now has real data either way: the native sparse kernel's
   achieved-bandwidth ceiling (only ~15-20% of A100 peak) is one candidate; the quant
   layer's own real-hardware-only bugs (six of them, none visible by inspection) and
   the optimization story (an intuitive fix that measured 20x slower, a less obvious one
   that worked) are a second, arguably sharper one — a class of finding neither
   Timeloop nor Gemmini's RTL generator could have produced, since it's specifically
   about how *this* Triton compiler/runtime version behaves on *this* hardware, not
   about the algorithm being wrong. Say which finding is the right closing one, and why,
   as this project's own contribution to that running cross-project theme.

Per spec.md's own scope note, Decision 3/Phase 4 Q2 (the MLA checkpoint) is deliberately
kept cheap — a formula substitution and a sanity check, not a second full
derivation/kernel track. If it stops being cheap, that's a signal to flag it as an open
thread for a future project, not expand this one's scope mid-stream.
