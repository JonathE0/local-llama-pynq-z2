# local-llama-pynq-z2 notes

Goal: run a small LLM on a PYNQ-Z2 (Zynq-7020: 2x Cortex-A9 @ 650 MHz, 512 MB DDR3 shared with Linux,
~1.2 GB/s usable through one 64-bit AXI HP port, 220 DSP48, ~630 KB BRAM). Decoding is bandwidth bound:
tok/s ≈ bandwidth / bytes of weights read per token.

## Plan
1. **CPU baseline**: llama2.c `runq.c` (int8, group 64) on the board, stories15M and stories110M.
   Build with `-O3 -mcpu=cortex-a9 -mfpu=neon -ffast-math -fopenmp`. Expect ~10–25 and ~1–3 tok/s.
2. **FPGA matmul**: offload `matmul` to an int8 Vitis HLS matrix-vector kernel streaming weights over AXI HP.
   Attention, softmax, RMSNorm and sampling stay on the ARM. Target for stories110M: 6–8 tok/s (ceiling ~10).
3. **Qwen2.5-0.5B-Instruct** at 4-bit (or a newer small model; pick after comparing perplexity on the PC).

Phases 1 and 2 are done; steps and results in [README.md](README.md). stories110M: 3.2 tok/s on the CPU,
7.9 tok/s with matmul on the FPGA (kernel at 142.86 MHz, ~80% of one HP port).
Phase 3: model chosen, quantized and exported to `.qwb` (PC side done). Next: `cma=` boot arg (~256+ MB),
4-bit kernel (int4 x int8, fp16 scales; int8 x int8 for the head), Qwen forward pass in C (QKV biases,
RoPE theta 1e6, GQA 14/2, head_ids sampling), byte-level BPE tokenizer in C, then test against llama.cpp on the PC.

## Decisions
- Phase 3 model: **Qwen2.5-Coder-0.5B-Instruct** (best code perplexity at 4-bit; numbers in README).
- Phase 3 format: llama.cpp `Q4_0` (symmetric 4-bit, group 32, fp16 scale) with an imatrix for all layer matrices,
  `Q8_0` for the tied embedding/head. The `.qwb` exporter repacks blocks into separate int4/int8 and fp16 scale
  planes for aligned AXI bursts. No K-quants. Activations: int8, group 32, like runq.
- Tied embeddings (152k x 896): the output head is a copy of the 49,152 most frequent rows (Q8, ~47 MB, in the FPGA
  buffer, logits mapped back through `head_ids`); the full table stays in the file for one-row-per-token input
  lookup (mmap from SD). ~250 MB of FPGA buffer instead of ~280 MB at full vocab.
- `runq.c` dequantizes the whole embedding table to float at load (~98 MB for 110M): change it to dequantize
  one row per token.
- Runtime is plain C: load the bitstream once, kernel registers via `/dev/mem`, weights in physically
  contiguous memory (bigger CMA, or reserve the top of RAM with a `mem=` boot arg). Needs one boot-arg change.
- Vivado/Vitis 2025.2 works with the stock PYNQ image; only rebuilding the image needs a matching version.

## Setup status (2026-10-26)
- Board: PYNQ image on Ubuntu 22.04 (`PynqLinux`), 32 GB card, 491 MB RAM visible to Linux, gcc 11.2.
- Board is cabled to the HP ENVY laptop; access details in [README.md](README.md#board-access).
- Vivado 2025.2 + Vitis are installed on the laptop (`C:\AMDDesignTools\2025.2`).
- Board clock is wrong (no internet, no NTP); harmless so far.
- PYNQ 3.1.1 allocates FPGA buffers through XRT/zocl from a 128 MiB CMA pool (phys 0x15900000–0x1D900000).
  stories110M's weights (~116 MB, packed into 8 MiB chunks) only just fit; Qwen (~220 MB) needs a bigger pool
  via a `cma=` boot arg first.
