# Shared workload constants for Phase 3's kernels (dense_kernel.py,
# sparse_kernel.py, quant_kernel.py) and benchmark.py. Single source of
# truth -- previously duplicated independently in dense_decode_reference.py
# and phase3_kernel_scaffold.py.

BATCH = 32
N_HEADS = 64
N_KV_HEADS = 8
GQA_GROUP = N_HEADS // N_KV_HEADS
D_HEAD = 128
BLOCK_N = 128

SINK_SIZE = 4          # StreamingLLM attention-sink size (notes.md Phase 0)
RESIDUAL_SIZE = 128    # KIVI: most recent tokens stay fp16, never quantized (notes.md Phase 2)
GROUP_SIZE = 32        # KIVI quantization group size, both K and V (notes.md Phase 2)
QUANT_BITS = 2         # KIVI main-cache precision (notes.md Phase 2)

CONTEXT_LENGTHS = [8_192, 16_384, 32_768, 65_536, 131_072, 163_840]
WINDOW_SIZES = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
