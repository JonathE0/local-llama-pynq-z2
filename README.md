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

## Board access

- The board's Ethernet goes to the laptop's USB Ethernet adapter ("Ethernet 2"), with Windows Internet Connection
  Sharing from Wi-Fi to Ethernet 2 (laptop = `192.168.137.1`).
- Sharing never answers the board's DHCP, so after each boot set a fixed address from the serial console
  (USB "PROG UART", COM12 here, 115200 baud):

  ```bash
  sudo ip addr add 192.168.137.2/24 dev eth0 && sudo ip route add default via 192.168.137.1
  ```

- `~/.ssh/config` on the laptop (key installed in the board's `~/.ssh/authorized_keys`):

  ```
  Host pynq
      HostName 192.168.137.2
      User xilinx
      IdentityFile ~/.ssh/id_ed25519
  ```

- The board has no internet through sharing: download on the PC and `scp` files over.
