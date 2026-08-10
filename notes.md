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
