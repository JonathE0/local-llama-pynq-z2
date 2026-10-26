"""Pick the K token ids the trimmed Qwen2.5 output head computes logits for (generation only; input embedding stays full).

usage (repo root):  .venv/Scripts/python tools/qwen_head_tokens.py [K=65536]
needs models/qwen/head_gen.jsonl first (tools/qwen_head_gen.py: the board's own model answering tools/qwen_head_prompts.txt).
writes qwen/head_tokens_K.npy (sorted int32 ids) and prints held-out coverage, the tokens that moved in and the ones still missing.
The baseline row "old 49152" is the existing qwen/head_tokens_49152.npy, made before the model corpus existed (rerun with K=49152
and it is overwritten).

Set = every added/special token (ids >= 151643) + the 256 byte tokens + the highest-scoring remaining tokens.
Score = 0.6*P_model + 0.2*P_english + 0.15*P_python + 0.05*P_c, each P a token distribution over its own corpus (so size
doesn't matter). The model's own chat answers dominate because that is what the head has to produce; the three text corpora
only fill the rest. Ties and never-seen tokens fall back to lower id (earlier BPE merge = more common). Rows >= 151665
(padding) are ignored.
Corpora split into counting and held-out parts:
  model: token ids the model sampled (temperature 0.7, top-p 0.8), crc32(prompt) % 4 == 0 held out (25%), the rest counted
  english: wikitext-2 train (count) / test (held out)     python: Python stdlib .py minus test dirs
  c: *.c/*.h in this repo + the Python include dir        (code files: crc32(relative path) parity, odd files held out)
Needs models/qwen/tokenizer-instruct/tokenizer.json (identical to the coder one, see sha256).
"""
import itertools
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
KS = (32768, 49152, 65536)
W = {"model": 0.6, "en": 0.2, "py": 0.15, "c": 0.05}
SKIP = {"test", "tests", "idle_test", "site-packages", "__pycache__", "build", ".venv", "vendor", ".git"}


def src_files(base, *pats):
    """[(even-half files, odd-half files)] of base/**/pat, minus SKIP dirs."""
    fs = sorted(p for pat in pats for p in base.rglob(pat) if p.is_file() and not SKIP & set(p.relative_to(base).parts))
    half = lambda p: zlib.crc32(p.relative_to(base).as_posix().encode()) & 1
    return [p for p in fs if half(p) == 0], [p for p in fs if half(p) == 1]


def read(paths):
    return [p.read_text(encoding="utf-8", errors="ignore") for p in paths]


def chunks(text, n=2000):  # whole-file strings are fine; wikitext is split on line ends to keep batches small
    lines = text.splitlines(True)
    return ["".join(lines[i:i + n]) for i in range(0, len(lines), n)]


def count(tok, texts):
    ids = itertools.chain.from_iterable(e.ids for e in tok.encode_batch(texts, add_special_tokens=False))
    return np.bincount(np.fromiter(ids, np.int64), minlength=tok.get_vocab_size())


def model_counts(n):
    """(counting, held-out) token counts of the generated answers; a prompt's samples stay on the same side"""
    path = ROOT / "models/qwen/head_gen.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing: run tools/qwen_head_gen.py first")
    c = [np.zeros(n, np.int64), np.zeros(n, np.int64)]
    for d in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
        c[zlib.crc32(d["prompt"].encode()) % 4 == 0] += np.bincount(d["tokens"], minlength=n)
    return c


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    tok = Tokenizer.from_file(str(ROOT / "models/qwen/tokenizer-instruct/tokenizer.json"))
    n = tok.get_vocab_size()

    wiki = ROOT / "models/data/wikitext-2-raw"
    base = Path(sys.base_prefix)  # real Python install, not the venv
    py_tr, py_te = src_files(base / "Lib", "*.py")
    (c_tr, c_te), (h_tr, h_te) = src_files(ROOT, "*.c", "*.h"), src_files(base / "include", "*.h")
    c_tr, c_te = c_tr + h_tr, c_te + h_te
    train = {"en": chunks((wiki / "wiki.train.raw").read_text(encoding="utf-8")), "py": read(py_tr), "c": read(c_tr)}
    test = {"en": chunks((wiki / "wiki.test.raw").read_text(encoding="utf-8")), "py": read(py_te), "c": read(c_te)}
    ctr = {k: count(tok, v) for k, v in train.items()}
    cte = {k: count(tok, v) for k, v in test.items()}
    ctr["model"], cte["model"] = model_counts(n)
    for k in W:
        print(f"{k}: counting {ctr[k].sum():,} tokens; held out {cte[k].sum():,}")

    added = list(tok.get_added_tokens_decoder())
    assert min(added) == 151643 and max(added) == n - 1, (min(added), max(added), n)
    bmap = byte_map()
    byte_ids = [tok.token_to_id(c) for c in bmap]
    assert None not in byte_ids, "a single-byte token is missing from the vocab"
    must = np.array(sorted(set(added + byte_ids)))

    def head(k, w):  # the k highest-scoring ids (+ forced ones), sorted
        score = sum(w[x] * ctr[x] / ctr[x].sum() for x in w)
        order = np.argsort(-score, kind="stable")  # ties -> lower id first
        return np.sort(np.concatenate([must, order[~np.isin(order, must)][:k - len(must)]])).astype(np.int32)

    def show(i):
        return ascii(bytes(bmap[c] for c in tok.id_to_token(int(i))).decode("utf-8", "backslashreplace"))

    old = np.load(ROOT / "qwen/head_tokens_49152.npy")  # the first set (English/Python/C only), the baseline to beat
    sets = [("old 49152", old)] + [(str(k), head(k, W)) for k in sorted({*KS, K})]
    print(f"{len(must)} forced tokens (added + byte); distinct tokens in the counting model answers: {(ctr['model'] > 0).sum():,}")
    print(f"{'set':>10} " + " ".join(f"{k:>8}" for k in W) + "   (% of held-out tokens inside the set)")
    for name, ids in sets:
        inside = np.zeros(n, bool)
        inside[ids] = True
        print(f"{name:>10} " + " ".join(f"{100 * cte[x][inside].sum() / cte[x].sum():8.3f}" for x in W))

    ids = head(K, W)
    out = ROOT / f"qwen/head_tokens_{K}.npy"
    np.save(out, ids)
    print(f"wrote {out} ({len(ids)} ids, int32)")
    new, gone = np.setdiff1d(ids, old), np.setdiff1d(old, ids)
    print(f"vs the old 49152 set: {len(new)} tokens in, {len(gone)} out. Most common new tokens in the model answers (id, token, count):")
    print("  " + "  ".join(f"{i} {show(i)} {ctr['model'][i]}" for i in new[np.argsort(-ctr['model'][new], kind="stable")][:30]))
    miss = np.setdiff1d(np.flatnonzero(cte["model"]), ids)
    print(f"held-out model tokens still outside the set: {len(miss)} distinct, {cte['model'][miss].sum()} of {cte['model'].sum()} total. Most common:")
    print("  " + "  ".join(f"{i} {show(i)} {cte['model'][i]}" for i in miss[np.argsort(-cte['model'][miss], kind="stable")][:30]))


def byte_map():  # GPT-2 byte-level BPE alphabet: printable char -> byte value (256 entries)
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    rest = [b for b in range(256) if b not in bs]
    return {chr(c): b for c, b in zip(bs + [256 + i for i in range(len(rest))], bs + rest)}


if __name__ == "__main__":
    main()
