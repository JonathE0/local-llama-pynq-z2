"""Pick the K token ids the trimmed Qwen2.5 output head computes logits for (generation only; input embedding stays full).

usage (repo root):  .venv/Scripts/python tools/qwen_head_tokens.py [K=32768]
writes models/qwen/head_tokens_K.npy (sorted int32 ids), prints held-out coverage and the tokens just past the cut.

Set = every added/special token (ids >= 151643) + the 256 byte tokens + the highest-scoring remaining tokens.
Score = 0.5*P_english + 0.4*P_python + 0.1*P_c, each P a token distribution over its own corpus (so size doesn't matter).
Ties and never-seen tokens fall back to lower id (earlier BPE merge = more common). Rows >= 151665 (padding) are ignored.
Corpora split by crc32(relative path) parity: even files count tokens, odd files are held out.
  english: wikitext-2 train (count) / test (held out)     python: Python stdlib .py minus test dirs
  c: *.c/*.h in this repo + the Python include dir
Needs models/qwen/tokenizer-instruct/tokenizer.json (identical to the coder one, see sha256).
"""
import itertools
import sys
import zlib
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
KS = (16384, 24576, 32768, 49152)
W = {"en": 0.5, "py": 0.4, "c": 0.1}
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


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
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
    for k in W:
        print(f"{k}: {len(train[k])} train chunks/files -> {ctr[k].sum():,} tokens; held out {len(test[k])} -> {cte[k].sum():,} tokens")

    score = sum(W[k] * ctr[k] / ctr[k].sum() for k in W)
    added = list(tok.get_added_tokens_decoder())
    assert min(added) == 151643 and max(added) == n - 1, (min(added), max(added), n)
    bmap = byte_map()
    byte_ids = [tok.token_to_id(c) for c in bmap]
    assert None not in byte_ids, "a single-byte token is missing from the vocab"
    must = np.array(sorted(set(added + byte_ids)))
    order = np.argsort(-score, kind="stable")  # ties -> lower id first
    order = order[~np.isin(order, must)]

    def head(k):
        return np.sort(np.concatenate([must, order[:k - len(must)]])).astype(np.int32)

    print(f"{len(must)} forced tokens (added + byte); tokens seen in counting corpus: {(score > 0).sum():,} of {n:,}")
    print(f"{'K':>6} " + " ".join(f"{k:>8}" for k in W) + "   (% of held-out tokens inside the set)")
    for k in sorted({*KS, K}):
        inside = np.zeros(n, bool)
        inside[head(k)] = True
        print(f"{k:>6} " + " ".join(f"{100 * cte[x][inside].sum() / cte[x].sum():8.3f}" for x in W))

    ids = head(K)
    out = ROOT / f"models/qwen/head_tokens_{K}.npy"
    np.save(out, ids)
    print(f"wrote {out} ({len(ids)} ids, int32)")
    print(f"first 20 tokens past the K={K} cut (rank, id, token, train counts en/py/c):")
    for r, i in enumerate(order[K - len(must):K - len(must) + 20], K + 1):
        print(f"{r:>6} {i:>6} {ascii(bytes(bmap[c] for c in tok.id_to_token(int(i))).decode('utf-8', 'backslashreplace')):<24} {ctr['en'][i]}/{ctr['py'][i]}/{ctr['c'][i]}")


def byte_map():  # GPT-2 byte-level BPE alphabet: printable char -> byte value (256 entries)
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    rest = [b for b in range(256) if b not in bs]
    return {chr(c): b for c, b in zip(bs + [256 + i for i in range(len(rest))], bs + rest)}


if __name__ == "__main__":
    main()
