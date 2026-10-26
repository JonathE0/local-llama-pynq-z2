"""Let the board's own model (the Q4_0 GGUF) answer tools/qwen_head_prompts.txt and record the token ids it generates.

usage (repo root; stop with Ctrl-C and rerun to resume, finished (prompt, seed) pairs are skipped):
  vendor/llama.cpp/llama-server.exe -m models/qwen/qwen2.5-coder-0.5b-instruct-q4_0-imat-emb-q8_0.gguf --port 8087 -t 6 -c 8192 --parallel 4 --no-webui
  .venv/Scripts/python tools/qwen_head_gen.py [samples_per_prompt=3] [port=8087]
writes models/qwen/head_gen.jsonl, one {"prompt", "seed", "stop", "tokens"} per line; tokens are the ids llama-server sampled
(including the final <|im_end|>), not a re-tokenization. Prompts are wrapped in the same chat template as qwen/qwen.c; a literal \\n
in a prompt line is a newline. Sampling: temperature 0.7, top-p 0.8, no top-k / repeat penalty (the board's sampler has neither).
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TMPL = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n")
OUT = ROOT / "models/qwen/head_gen.jsonl"


def complete(port, prompt, seed):
    body = {"prompt": TMPL % prompt.replace("\\n", "\n"), "n_predict": 400, "temperature": 0.7, "top_p": 0.8, "top_k": 0,
            "min_p": 0, "repeat_penalty": 1.0, "seed": seed, "return_tokens": True, "cache_prompt": False}
    r = json.load(urlopen(Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})))
    return {"prompt": prompt, "seed": seed, "stop": r["stop_type"], "tokens": r["tokens"]}


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8087
    prompts = (ROOT / "tools/qwen_head_prompts.txt").read_text(encoding="utf-8").splitlines()
    done = set()
    if OUT.exists():
        done = {(d["prompt"], d["seed"]) for d in map(json.loads, OUT.read_text(encoding="utf-8").splitlines())}
    todo = [(p, s) for s in range(samples) for p in prompts if (p, s) not in done]
    print(f"{len(prompts)} prompts x {samples} samples: {len(done)} done, {len(todo)} to go", flush=True)
    lock, n = threading.Lock(), [0]

    def job(ps):
        d = complete(port, *ps)
        with lock, open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n[0] += 1
            if n[0] % 50 == 0:
                print(f"{n[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(4) as ex:  # = llama-server --parallel
        list(ex.map(job, todo))
    print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
