# Handoff — numerics-and-sparse-attn, about to start Phase 2

Read this + `notes.md` (full narrative log) + `spec.md` (the project brief) to pick up
exactly where the last session left off. Everything below is settled; nothing here needs
re-derivation unless you find a real error in it.

## How this project works (read this first)

This is a self-directed derivation project. Sections marked 🧠 in `spec.md` are the
user's own hand-derivation work — **the assistant's role is sounding board and
correctness-checker, not derivation engine.** Concretely, established over the last
session:
- When the user proposes a formula/derivation step, check it, point out errors precisely
  (cite the source doc/section it should match), don't just hand them the answer.
- When the user explicitly delegates a mechanical step ("plug in numbers", "do the
  sweep", literature research), do it directly and report results plainly.
- Don't do 🧠-marked derivation work unprompted, even when it would be faster.
- Push back on formula errors immediately and specifically — this repo's whole
  methodology is "gap-hunting is the highest-value activity," and that applies to
  catching the user's own mistakes mid-derivation, not just tool-vs-hypothesis gaps.
- Real research (web search) is appropriate and was used for Phase 0's reading — this
  project explicitly wants real mechanisms, not assumed ones.

## Reference docs (read if you need the numbers behind anything below)

- `spec.md` — the project brief, full phase breakdown, all reused-number citations.
- `notes.md` — full narrative log of everything found/derived so far (Phase 0 reading in
  full, all three Decision write-ups, complete Phase 1 derivation with sanity checks).
- `../workload-to-silicon/decode_notes.md` — dense GQA decode roofline (FLOPs/bytes/AI
  formulas this project's Phase 1 modifies), Phase 3's coupled/decoupled numerics
  framework (§4.1) and crossover method (§4.2) — **Phase 2's main input**.
- `../workload-to-silicon/disagg_and_placement_notes.md` — KV-cache capacity formula
  (§1), MLA cache-entry formula (§2c), dense/MoE chip ratios (§2.9/§2b), the
  compute-bound-asymptote pattern (§3.3/3.5/3.8-3.9/§4.7).
- `../workload-to-silicon/prefill_notes.md` — mostly background; less directly reused
  here since this project is decode-only.

## Phase 0 — done

**Decision 1 (sparsity mechanism): sliding window (W) + attention sink (S=4),
StreamingLLM-style.** Chosen over H2O because it's provably closed-form: fixed cache
size `S+W`, FIFO eviction, RoPE reassigned to cache-relative positions post-eviction, no
data-dependent branching. H2O's cumulative-attention-score eviction is real, confirmed
non-trivial added cost (O(k) data-dependent gather/scatter per step) — would break
closed-form. Longformer/BigBird ruled out (bidirectional encoders, no decode-eviction
analog). Full research writeup in `notes.md`.

**Decision 2 (GPU environment): cloud GPU instance**, provider/instance TBD closer to
Phase 3 — not blocking anything before then.

**Decision 3 (MLA carry-forward): `c` bytes/position/layer parameterization confirmed
viable for the bytes term**, but with a refinement over the spec's original framing —
real shipping systems (DeepSeek-V3.2 DSA, ASA) show the *selection* mechanism itself
needs adaptation when sparsity pairs with latent compression (DeepSeek: a separate
"lightning indexer" with its own cache/FLOPs, computed from raw hidden states not the
compressed latent; ASA: had to invent GLA because plain MLA broke block-selection).
**Action for Phase 4**: the MLA substitution should carry an explicit small additive
indexer-overhead term (from DeepSeek's published numbers), not be treated as a pure
`c` swap. If that stops being cheap, spec's own escape hatch applies (flag as open
thread, don't expand scope).

**Scope confirmed: SDPA-only** (QK^T → softmax → ·V), matching `decode_notes.md`'s
scope — correct here specifically because sparsity only changes which K/V positions get
touched, not QKVO/FFN. Flag: if Phase 4 wants to compare against disagg's authoritative
830.59 req/s/chip / 5.82:1 numbers, QKVO needs adding back in the same way disagg did.

## Phase 1 — done, checkpoint reached

**Workload constants** (reused unchanged from `decode_notes.md` §0): batch=32,
n_heads=64, n_kv_heads=8, d_head=128, int8 (1 byte). S=4. `seq_len_q=1` (fixed, decode's
defining feature, untouched by sparsity). `seq_len_kv = S+W` for the sweep range (ramp-up
transient where `L<S+W` is real but self-resolving and irrelevant at L≥8,192).

**Derived formulas:**
```
FLOPs(W) = 4 × batch × n_heads × d_head × (W+4) = 1,048,576 × (W+4)

c = 2 × d_head × n_kv_heads × precision = 2,048 bytes/position/layer   [the Decision-3 param]
Bytes(W) = 2×batch×n_heads×d_head×precision + batch×c×(W+4)
         = 524,288 + 65,536×(W+4) = 786,432 + 65,536×W

AI(W) = FLOPs(W)/Bytes(W) = 16(W+4)/(W+12)
```
Sanity check (passed): at `W+4=8192` (no sparsity), both formulas recover
`decode_notes.md`'s exact dense GQA numbers (FLOPs=2³³, bytes=537,395,200 B).

**Findings (full reasoning in `notes.md`):**
1. Sparsity is a genuine FLOPs lever, not just bytes (unlike GQA) — but FLOPs has no
   fixed term while bytes has a fixed Q+O floor (524,288), so AI is bounded, not
   improved.
2. **AI(W) is strictly increasing in W, ceiling = exactly 16** (dense GQA's own AI) —
   sparsity can only approach dense's AI, never exceed it. Real windows sit measurably
   below 16 (W=256→AI≈15.52; W=32→≈13.09; W=0→≈5.33) — this is a real, monotonic cost of
   sparsifying, not a wash, but too small relative to decode's ~30× ridge margin to
   matter for the regime question.
3. **Sparse AI can never clear the ridge (480.5) for any W at any context length** —
   ceiling (16) sits ~30× below ridge regardless. Same structural flavor as
   `decode_notes.md` Phase 3's quantization-proof floor.
4. **Context-length sweep (8,192→163,840) is a non-event for AI** — formula has no `L`
   dependence once `L≥S+W`. Confirmed numerically: AI≈15.52 at both endpoints (W=256).
5. **But absolute savings (dense/sparse ratio) grow linearly with L** — confirmed
   numerically: ≈31.5× at L=8,192 vs. ≈630× at L=163,840 (tracks the 20× L-ratio almost
   exactly). This is the real content of the sweep and confirms the spec's own premise
   that sparsity's value proposition is context-length-dependent.
6. Compute-bound-asymptote check: moot for sparsity alone (AI never clears ridge) —
   becomes live only in Phase 2.
7. disagg §1.4 HBM-capacity-crossover sub-question: judged moot, skipped — its premise
   (a discrete crossover) doesn't survive finding #4; also mechanistically circular
   (savings and capacity pressure are driven by the same growing quantity, not two
   independent numbers that happen to coincide).

## Next: Phase 2 — layer numerics on top, quantify the interaction

Per `spec.md`: reuse `decode_notes.md` §4.1's coupled-vs-decoupled precision framework
and §4.2's crossover-point method exactly — the mechanism transfers, only the starting
AI changes (sparse AI instead of dense GQA's AI≈16).

**The sharpened starting point, flagged and ready to use**: sparse AI is *always ≤16*,
never above dense GQA's own ceiling (Phase 1 finding #2). This means Phase 2 is solving
the crossover from a position that's **at best equal to, and realistically worse than**,
decode's original Phase 3 starting point (AI≈16 exactly) — so the quantization target
this project derives may need to be *even more extreme* than decode's own
already-unrealizable ~1/30 bytes/element (≈0.27 bits/element) floor. Don't assume this
either way — derive it.

**Two things to actually resolve in Phase 2** (per spec, unresolved either direction):
- Redo decode's crossover solve starting from `AI(W) = 16(W+4)/(W+12)` instead of a flat
  16 — is the interaction clean/multiplicative (crossover point shifts by exactly
  sparsity's own AI ratio) or does something non-obvious happen (e.g. quantizing an
  already-sparsified cache hits a different numerical-sensitivity regime — KIVI's own
  group-size-vs-accuracy ablation and its residual-buffer structure are the concrete
  mechanism to check this against, per the Phase 0 KIVI finding in `notes.md`).
- Check for the disagg-style "one lever dominates, the other becomes a rounding error"
  pattern (disagg §2.5's FFN-dominance finding) — does sparsity's savings dominate so
  completely that numerics becomes marginal, or vice versa?

Nothing else is blocking — go straight into Phase 2's derivation when ready.
