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

## Open Threads Carried Forward

- Phase 4's MLA substitution needs an additive indexer-overhead term (DeepSeek's
  published numbers), not a pure `c` swap — flagged in Decision 3, not yet quantified.
- If Phase 4 wants to compare against disagg's authoritative 830.59 req/s/chip / 5.82:1
  numbers, QKVO needs to be added back in the same way disagg did for dense decode.
- Phase 2 starts from a **worse** position than `decode_notes.md`'s original Phase 3:
  sparse AI is always ≤16 (never above dense GQA's own ceiling), so the numerics
  crossover point derived in Phase 2 may need to be *even more* extreme than decode's
  already-unrealizable ~0.033 bytes/element floor. Real tension to derive properly, not
  assume either direction.
