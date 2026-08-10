# Correctness check for the dense decode reference kernel (comparison
# target (b)). Never verified against a reference before — only ever
# smoke-tested for shape/NaN — despite it being what predicted_ratio /
# measured_ratio in phase3_kernel_scaffold.py's benchmark actually gets
# compared against. No sparsity-selection content here (dense attention has
# no masking/validity logic at all), so this whole file is boilerplate.

import torch

from dense_decode_reference import BATCH, N_HEADS, N_KV_HEADS, D_HEAD, dense_decode_attention

DEVICE = "cuda"


def _reference_dense_attention(q, k, v, sm_scale):
    """q: [BATCH, N_HEADS, D_HEAD]. k/v: [BATCH, N_KV_HEADS, seq_len, D_HEAD].

    GQA via reshape + einsum broadcast, not repeat_interleave — repeating k/v
    up to N_HEADS would materialize an 8x-larger copy, which at this file's
    large-seq_len case (the whole point of testing here, see below) would be
    tens of GB for no reason. Reshaping q into (N_KV_HEADS, GQA_GROUP) groups
    and letting einsum broadcast against the un-repeated k/v gets the same
    result without ever allocating that copy.
    """
    B, H, D = q.shape
    _, KV_H, N, _ = k.shape
    group = H // KV_H
    q = q.view(B, KV_H, group, D)
    qk = torch.einsum("bkgd,bknd->bkgn", q, k) * sm_scale
    p = torch.softmax(qk, dim=-1)
    o = torch.einsum("bkgn,bknd->bkgd", p, v)
    return o.reshape(B, H, D)


def test_dense_attention_correctness():
    torch.manual_seed(0)

    # 131_072 is deliberately included: past the int32 overflow crossover
    # (~67,653 for this workload's batch/head/dim sizes) that caused a real
    # illegal-memory-access on hardware — this checks the fix produces the
    # *correct* result at that scale, not just that it no longer crashes.
    seq_lens = [2, 8_192, 131_072]

    all_passed = True
    for seq_len in seq_lens:
        q = torch.randn(BATCH, N_HEADS, D_HEAD, dtype=torch.float32, device=DEVICE)
        k = torch.randn(BATCH, N_KV_HEADS, seq_len, D_HEAD, dtype=torch.float32, device=DEVICE)
        v = torch.randn(BATCH, N_KV_HEADS, seq_len, D_HEAD, dtype=torch.float32, device=DEVICE)
        sm_scale = D_HEAD ** -0.5

        o_kernel = dense_decode_attention(q.half(), k.half(), v.half(), seq_len=seq_len, sm_scale=sm_scale).float()
        o_ref = _reference_dense_attention(q, k, v, sm_scale)

        max_diff = (o_kernel - o_ref).abs().max().item()
        passed = max_diff < 1e-2  # loose tolerance — kernel runs fp16, reference fp32
        all_passed &= passed
        print(f"seq_len={seq_len:7d}  max_abs_diff={max_diff:.6f}  {'PASS' if passed else 'FAIL'}")

    if not all_passed:
        raise AssertionError("dense_decode_attention diverged from the reference — see above")


if __name__ == "__main__":
    test_dense_attention_correctness()
