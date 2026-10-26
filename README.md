# local-ai-fpga

Small LLMs on a PYNQ-Z2. Plan and design decisions: [NOTES.md](NOTES.md).

## Phase 1: CPU baseline (llama2.c `runq`, int8)

On the PC (Python + numpy, no PyTorch needed):

```bash
git clone --depth 1 https://github.com/karpathy/llama2.c
mkdir -p models
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/stories110M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
python tools/bin2q80.py models/stories15M.bin models/stories15M_q80.bin
python tools/bin2q80.py models/stories110M.bin models/stories110M_q80.bin
```

Copy to the board, build and run (`pynq` is the SSH alias from [Board access](#board-access)):

```bash
ssh pynq mkdir -p llama2.c
scp llama2.c/runq.c llama2.c/tokenizer.bin models/*_q80.bin pynq:llama2.c/
ssh pynq 'cd llama2.c && gcc -O3 -ffast-math -mcpu=cortex-a9 -mfpu=neon -fopenmp runq.c -o runq -lm'
ssh pynq 'cd llama2.c && ./runq stories110M_q80.bin -n 256 -i "Once upon a time"'
```

Results (2026-10-26, `-t 0 -n 256`, prompt "Once upon a time", `OMP_NUM_THREADS` = threads):

| Model | Weights | 1 thread | 2 threads |
|---|---|---|---|
| stories15M int8 | 17 MB | 10.2 tok/s | 18.5 tok/s |
| stories110M int8 | 116 MB | 1.7 tok/s | 3.2 tok/s |

- Float `run.c` on stories15M with 2 threads: 10.1 tok/s.
- Check: at `-t 0` the int8 model writes the same text as the float model for the first ~50 tokens, then drifts (normal for quantization).
- stories110M at 3.2 tok/s reads ~370 MB/s of weights, about a third of what one AXI HP port can deliver.
  The A9 cores are compute bound, which is what the FPGA matmul in phase 2 addresses.

## Phase 2: matmul on the FPGA

`runq`'s `matmul()` (int8 matrix × vector, ~90% of the time) runs on an HLS kernel that streams the weights over
AXI HP0 at one 64-bit word (8 MACs) per clock; attention, softmax, RMSNorm and sampling stay on the ARM cores.

| Path | What |
|---|---|
| [hls/matvec.h](hls/matvec.h) | Kernel interface, the contract between kernel and driver |
| [hls/matvec.cpp](hls/matvec.cpp), [hls/matvec_tb.cpp](hls/matvec_tb.cpp) | Kernel and C-sim testbench (bit-exact vs runq.c's matmul) |
| [vivado/build_overlay.tcl](vivado/build_overlay.tcl) | Block design (PS7 + kernel on HP0/HP1, regs at 0x43C00000) → bitstream |
| [overlay/](overlay/) | Built `matvec.bit` + `matvec.hwh`, and `load_overlay.py` |
| [driver/](driver/) | XRT buffer allocation + register control (C API) |
| [runq/](runq/) | `runq.c` with the FPGA path (`-DFPGA`) and a Makefile |

Build on the PC (Vitis/Vivado 2025.2; HLS ~1 min, Vivado ~15 min):

```bash
C:/AMDDesignTools/2025.2/Vitis/bin/vitis-run.bat --mode hls --tcl hls/run_hls.tcl --work_dir hls/build
C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat -mode batch -source vivado/build_overlay.tcl -tclargs hls/build/matvec_prj/sol/impl/ip
```

On the board (as root, with PYNQ's environment; keep the `driver/ hls/ runq/ overlay/` layout under `~/fpga`):

```bash
ssh pynq mkdir -p fpga/driver fpga/hls fpga/runq fpga/overlay
scp driver/* pynq:fpga/driver/ && scp hls/xmatvec_hw.h pynq:fpga/hls/ && scp runq/* pynq:fpga/runq/ && scp overlay/* pynq:fpga/overlay/
ssh pynq 'cd ~/fpga/runq && make'
ssh pynq 'sudo bash -c "for f in /etc/profile.d/*.sh; do . \$f; done; python3 /home/xilinx/fpga/overlay/load_overlay.py"'
ssh pynq 'cd ~/llama2.c && sudo ~/fpga/runq/runq-fpga stories110M_q80.bin -n 256 -i "Once upon a time"'
```

- Load the overlay once per boot. It also sets the kernel clock to 142.86 MHz (closest to the 150 MHz Vivado timed)
  and drops the page cache so stories110M's ~116 MB of weights fit the 128 MiB CMA pool.
- `FPGA_CHECK=1` recomputes every matmul on the CPU and prints the largest difference (slow).

Results (2026-10-26, kernel at 142.86 MHz, `-t 0 -n 256`, same prompt as phase 1):

| Model | CPU, 2 threads | FPGA, 1 thread | FPGA, 2 threads | Speedup |
|---|---|---|---|---|
| stories15M int8 | 18.5 tok/s | 45.7 tok/s | **50.7 tok/s** | 2.7× |
| stories110M int8 | 3.2 tok/s | 7.3 tok/s | **7.9 tok/s** | 2.5× |

- At 125 MHz (the clock the overlay load alone leaves) stories110M ran at 7.1 tok/s.
- `FPGA_CHECK`: max abs difference 3.8e-6 on stories110M. The kernel keeps runq.c's float order exactly (bit-exact in
  simulation); the small difference comes from the CPU build's `-ffast-math`. Greedy text matches the CPU's for the
  first lines, then drifts at near-ties.
- stories110M now reads ~920 MB/s of weights, ~80% of what one 64-bit port can move at 142.86 MHz. The next lever is
  a second weight port (HP2), toward the DDR's ~1.5 GB/s practical limit.

## Board access

- The board's Ethernet goes to the laptop's USB Ethernet adapter ("Ethernet 2"), with Windows Internet Connection
  Sharing from Wi-Fi to Ethernet 2 (laptop = `192.168.137.1`).
- Sharing never answers the board's DHCP, so the board has a fixed `192.168.137.2` (alias `eth0:2` in
  `/etc/network/interfaces.d/eth0-laptop`). Serial console fallback: USB "PROG UART", COM12 here, 115200 baud.
- `xilinx` has passwordless sudo (`/etc/sudoers.d/xilinx-nopasswd`), needed for the FPGA runs.

- `~/.ssh/config` on the laptop (key installed in the board's `~/.ssh/authorized_keys`):

  ```
  Host pynq
      HostName 192.168.137.2
      User xilinx
      IdentityFile ~/.ssh/id_ed25519
  ```

- The board has no internet through sharing: download on the PC and `scp` files over.
