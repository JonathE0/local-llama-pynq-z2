# local-ai-fpga notes

Goal: run a small LLM on a PYNQ-Z2 (Zynq-7020: 2x Cortex-A9 @ 650 MHz, 512 MB DDR3 shared with Linux,
~1.2 GB/s usable through one 64-bit AXI HP port, 220 DSP48, ~630 KB BRAM). Decoding is bandwidth bound:
tok/s ≈ bandwidth / bytes of weights read per token.

## Plan
1. **CPU baseline**: llama2.c `runq.c` (int8, group 64) on the board, stories15M and stories110M.
   Build with `-O3 -mcpu=cortex-a9 -mfpu=neon -ffast-math -fopenmp`. Expect ~10–25 and ~1–3 tok/s.
2. **FPGA matmul**: offload `matmul` to an int8 Vitis HLS matrix-vector kernel streaming weights over AXI HP.
   Attention, softmax, RMSNorm and sampling stay on the ARM. Target for stories110M: 6–8 tok/s (ceiling ~10).
3. **Qwen2.5-0.5B-Instruct** at 4-bit (or a newer small model; pick after comparing perplexity on the PC).

Milestone 1 deliverables (not started): one HLS kernel source, one C-sim testbench checking the kernel against
the CPU result, a small C driver called from `runq.c`, README with board and Vivado steps.

## Decisions
- Phase 3 format: llama.cpp `Q4_0` (symmetric 4-bit, group 32, fp16 scale), quantized with
  `llama-quantize --imatrix`, quality checked with `llama-perplexity`. Exporter repacks blocks into separate
  nibble and scale arrays for aligned AXI bursts. No K-quants.
- Tied embeddings (152k x 896): only the top ~32k rows (after vocab reorder) live in the FPGA buffer as the
  output head; the rest is mmap'd from SD for input lookup. ~220 MB buffer instead of ~280 MB.
- `runq.c` dequantizes the whole embedding table to float at load (~98 MB for 110M): change it to dequantize
  one row per token.
- Runtime is plain C: load the bitstream once, kernel registers via `/dev/mem`, weights in physically
  contiguous memory (bigger CMA, or reserve the top of RAM with a `mem=` boot arg). Needs one boot-arg change.
- Vivado/Vitis 2025.2 works with the stock PYNQ image; only rebuilding the image needs a matching version.

## Setup status (2026-10-26)
- Board boots from a 32 GB card with the PYNQ-Z2 v3.1.1 image (confirm: `lsb_release -ds` → Ubuntu 22.04).
- Board is cabled to the **desktop PC** (USB + Ethernet straight into the desktop).
- Desktop USB serial shows "USB Serial Port" with Code 28: install FTDI's VCP driver
  (ftdichip.com → Drivers → VCP → Windows setup executable), then replug. Console is 115200 baud.
- Network, pick one:
  - Board into the router: gets internet, reachable from any PC as `ssh xilinx@pynq`.
  - Internet Connection Sharing on the desktop: Wi-Fi → Properties → Sharing → share to Ethernet; leave the
    Ethernet IPv4 on automatic. Board gets a `192.168.137.x` address (`arp -a` to find it).
- Vivado 2025.2 + Vitis are installed on the HP ENVY laptop (`C:\AMDDesignTools\2025.2`); check the desktop.
