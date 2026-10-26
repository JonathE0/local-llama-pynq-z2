# local-llama-pynq-z2

Small LLMs running on a PYNQ-Z2 (Zynq-7020) with every matmul on an FPGA kernel: llama2.c TinyStories models
(stories110M at 7.9 tok/s) and Qwen2.5-Coder-0.5B-Instruct at 4-bit (3.7 tok/s), optionally shown on a 16×2 LCD. Plan and design decisions: [NOTES.md](NOTES.md).

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

## Phase 3: Qwen2.5-Coder-0.5B-Instruct

A 4-bit coding chat model runs on the board with every matmul on the FPGA:

| Run (2026-10-26, greedy, 128 new tokens, prompt "Write a Python function that checks if a number is prime.") | Prompt | Generation |
|---|---|---|
| ARM CPU only, 2 threads (`qwen`) | 1.3 tok/s | 2.0 tok/s |
| FPGA, 1 thread (`qwen-fpga`) | 3.9 tok/s | 3.5 tok/s |
| **FPGA, 2 threads** | **3.9 tok/s** | **3.7 tok/s** |

- `FPGA_CHECK=1`: max abs difference 2.4e-4 against the CPU over 9,216 matmuls (the kernel sums groups in a different order).
- [tools/qwen_ref.py](tools/qwen_ref.py), a numpy forward pass over the same `.qwb`, matches llama.cpp's greedy output on the GGUF once
  the head restriction and fp16-vs-fp32 activation scales are accounted for; the C program matches it within int8
  activation noise (greedy identical for 40–64 tokens).
- The kernel streams ~263 MB of weights per token (201 MB 4-bit layers + 62 MB 8-bit head), ~0.95 GB/s.

Deploy, build, load the overlay (once per boot) and run, from the PC:

```bash
ssh pynq mkdir -p qwen/driver qwen/hls4 qwen/qwen qwen/overlay qwen/models
scp driver/* pynq:qwen/driver/ && scp hls4/xmatvec4_hw.h pynq:qwen/hls4/ && scp qwen/*.c qwen/*.h qwen/Makefile pynq:qwen/qwen/
scp overlay/qwen.bit overlay/qwen.hwh overlay/load_overlay.py pynq:qwen/overlay/
scp models/qwen/qwen2.5-coder-0.5b-q4-head64k.qwb models/qwen/qwen_tokenizer.bin pynq:qwen/models/
ssh pynq 'cd ~/qwen/qwen && make'
ssh pynq 'sudo bash -c "for f in /etc/profile.d/*.sh; do . \$f; done; python3 /home/xilinx/qwen/overlay/load_overlay.py qwen"'
ssh pynq 'cd ~/qwen && sudo ./qwen/qwen-fpga models/qwen2.5-coder-0.5b-q4-head64k.qwb -i "Write a C function that reverses a string"'
```

Options: `-t` temperature (default 0.7, 0 = greedy), `-p` top-p (0.8), `-n` max new tokens (256), `-c` context
(1024), `-s` seed, `-m raw` (no chat template), `--lcd 1` (mirror the answer on the LCD), `FPGA_CHECK=1` (slow).
Build on the board with `make` in `qwen/` (`qwen` = CPU only, `qwen-fpga`).

| Piece | Files |
|---|---|
| 4-bit kernel: int4 or int8 weights, fp16 scales, 16 MACs/clock, II=1, 150 MHz timing | [hls4/](hls4/) (contract in `matvec4.h`) |
| Overlay: kernel on HP0/HP1 + LCD GPIO | `overlay/qwen.bit`, built with `vivado/build_overlay.tcl ... --vlnv xilinx.com:hls:matvec4 --name qwen --lcd` |
| Inference (Qwen2: QKV biases, NeoX RoPE θ=1e6, GQA 14/2, SwiGLU, head over `head_ids`) | [qwen/qwen.c](qwen/qwen.c), FPGA calls in [driver/](driver/) (`-DMATVEC4`) |
| Byte-level BPE tokenizer, id-for-id with HF `tokenizers` (1.18M test cases) | [qwen/tokenizer.c](qwen/tokenizer.c), data from `tools/qwen_tok_export.py`, tests `tools/qwen_tok_test.py` |
| 1602A LCD (line 1: scrolling answer, line 2: tok/s) | [qwen/lcd.c](qwen/lcd.c), pins in [vivado/lcd_pins.xdc](vivado/lcd_pins.xdc) |

The CMA pool is 320 MB (`/boot/uEnv.txt` with `cma=320M` in `bootargs`; delete that file from the SD card's boot
partition to undo). The weights take 263 MB of it in 8 MiB chunks.

### LCD wiring (1602A, 4-bit mode, PYNQ-Z2 Raspberry Pi header)

| LCD pin | Pi header pin |
|---|---|
| 1 VSS, 5 RW, 16 K | GND (6, 14) — RW must be on GND so the 5 V LCD never drives the 3.3 V FPGA pins |
| 2 VDD, 15 A | 5 V (2, 4) |
| 3 V0 | 10 kΩ pot wiper between 5 V and GND (or ~1 kΩ to GND) |
| 4 RS, 6 E | 11, 13 |
| 11–14 D4–D7 | 15, 16, 18, 22 |

### How the model was prepared (PC)

**Quality** (llama-perplexity, context 512, lower is better; code = held-out Python stdlib, English = wikitext-2):

| Model | Code F16 | Code 4-bit | English F16 | English 4-bit |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 4.273 | 4.554 (+6.6%) | 15.79 | 17.46 (+10.6%) |
| **Qwen2.5-Coder-0.5B-Instruct** | 4.202 | **4.428 (+5.4%)** | 34.76 | 37.89 (+9.0%) |

- 4-bit = `Q4_0` (group 32, fp16 scale) with an importance matrix, tied embedding/output head at `Q8_0`. Plain `Q4_0`
  without the imatrix was +6.3% (Coder) / +10.7% (Instruct) on code; the `Q8_0` head adds ~1 point of quality for
  ~15 MB more traffic per token.
- Both models handled FizzBuzz, palindromes and merging sorted lists; both wrote a buggy `key=value` parser.
- The official GGUFs store the tied head twice (`output.weight` = `token_embd.weight`); `tools/gguf_tie.py` drops the
  copy and `tools/imatrix_tie.py` moves its imatrix stats to `token_embd`.

**Output head:** logits only for 65,536 tokens (plus all special and byte tokens), ranked mostly by how often the model
itself emits them: 1,542 answers (401k tokens) generated with llama.cpp from the 514 prompts in
[tools/qwen_head_prompts.txt](tools/qwen_head_prompts.txt) (`tools/qwen_head_gen.py`), then English/Python/C text.
The set is [qwen/head_tokens_65536.npy](qwen/head_tokens_65536.npy) (`tools/qwen_head_tokens.py 65536`).

| Head set | Held-out model answers | English | Python | C |
|---|---|---|---|---|
| 49,152 from text frequencies only (first try) | 94.8% | 99.1% | 98.8% | 97.9% |
| 49,152 with model answers | 98.1% | 98.9% | 98.6% | 97.8% |
| **65,536 with model answers** | **99.0%** | 99.7% | 99.4% | 98.6% |

The first set lacked `` ``` ``, `_prime` and `Certainly`, so code fences lost their language and names came out as
`is prime`. What 65,536 still misses is rare identifier pieces and a few foreign-language word pieces.

**Board weight file** (`.qwb`, 390 MiB; layout in [tools/gguf2qwb.py](tools/gguf2qwb.py)): 4-bit matrices as packed
int4 + fp16 scales, the head and full embedding table at int8 + fp16 scales. `tools/qwb_check.py` walks the file the
way the board loader must and checks it against the GGUF.

Pipeline (PC; llama.cpp `b11889` Windows CPU build in `vendor/llama.cpp/`, venv with `numpy gguf tokenizers`):

```bash
curl -L -o models/qwen/qwen2.5-coder-0.5b-instruct-fp16.gguf https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-fp16.gguf
py -3 -I tools/qwen_calib_data.py
vendor/llama.cpp/llama-imatrix -m models/qwen/qwen2.5-coder-0.5b-instruct-fp16.gguf -f models/data/code/calib_mix.txt -o models/qwen/imatrix-coder.gguf -c 512 --process-output --no-ppl
.venv/Scripts/python tools/gguf_tie.py models/qwen/qwen2.5-coder-0.5b-instruct-fp16.gguf models/qwen/coder-fp16-tied.gguf
.venv/Scripts/python tools/imatrix_tie.py models/qwen/imatrix-coder.gguf models/qwen/imatrix-coder-tied.gguf
vendor/llama.cpp/llama-quantize --pure --imatrix models/qwen/imatrix-coder-tied.gguf --token-embedding-type q8_0 models/qwen/coder-fp16-tied.gguf models/qwen/coder-q4.gguf Q4_0
.venv/Scripts/python tools/gguf2qwb.py models/qwen/coder-q4.gguf qwen/head_tokens_65536.npy models/qwen/coder-q4.qwb
.venv/Scripts/python tools/qwb_check.py models/qwen/coder-q4.gguf models/qwen/coder-q4.qwb
```

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
