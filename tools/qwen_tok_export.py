#!/usr/bin/env python3
"""Export Qwen2/Qwen2.5 tokenizer.json to the compact binary read by qwen/tokenizer.c.

    .venv/Scripts/python tools/qwen_tok_export.py [tokenizer.json] [out.bin]
    (defaults: models/qwen/tokenizer-instruct/tokenizer.json -> models/qwen/qwen_tokenizer.bin)

The C code hard-codes the Qwen2 pre-tokenizer (GPT-4 style Split regex + ByteLevel without prefix space), the NFC
normalizer and plain BPE, so this script refuses any tokenizer.json that is configured differently. What it exports:

  * the vocabulary as raw bytes (the byte-to-unicode mapping of GPT-2 undone), for decode;
  * the merges as id triples (left, right, merged) in rank order, so the C side never needs token strings;
  * the added tokens (<|im_start|> ... <|file_sep|>);
  * one Unicode table: per code point the pre-tokenizer class (letter / number / white space / other), the
    canonical combining class and an "NFC may change this" flag; plus NFD decompositions and the NFC composition
    pairs. Hangul is left to the algorithm in C.

Unicode data comes from Python's unicodedata (3.12: Unicode 15.0) and is then reconciled with the `tokenizers`
library itself (HF bundles Unicode 16 for \\p{L}/\\p{N}, and its normalization tables differ from unicodedata for a
handful of code points, e.g. U+1DFA, U+16FF0, U+11938). `tokenizers` is the reference, so it wins; the number of
differences is printed. Without the `tokenizers` module the script stops (exactness would be lost).

Binary format (all integers little-endian, every section starts 4-byte aligned, zero padded):

    char     magic[8]  = "QWTOK001"
    uint32   hdr[12]   = n_vocab, n_added, n_merges, blob_len, n_uni, n_dec, dec_pool_len, n_comp, 0, 0, 0, 0
    uint8    tok_len[n_vocab]               byte length of token i (ids 0 .. n_vocab-1; must be <= 255)
    uint8    tok_blob[blob_len]             the token bytes, concatenated in id order
    uint32   merges[3 * n_merges]           (left id, right id, merged id) per merge; merge rank = index
    added tokens, n_added times:            uint32 id, uint32 len, uint8 text[len], zero padding to 4 bytes
    uint32   uni[n_uni]                     run-length table: entry = start | payload << 21 covers code points
                                            [start, next start); payload bits: 0-1 class (0 other, 1 letter,
                                            2 number, 3 white space), 2 NFC-dirty, 3-10 combining class
    uint32   dec_cp[n_dec], dec_off[n_dec+1], dec_pool[dec_pool_len]
                                            NFD of dec_cp[i] = dec_pool[dec_off[i] .. dec_off[i+1]) (sorted by cp)
    uint32   comp[3 * n_comp]               (first, second, composite), sorted by (first, second)

Id space: n_vocab base tokens (151643), the added tokens follow (151643 .. 151664), 151665 ids in total.
"""
import collections
import json
import os
import struct
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the exact Split regex the C pre-tokenizer implements; anything else is refused
QWEN2_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
               r"|\s*[\r\n]+|\s+(?!\S)|\s+")
MAXCP = 0x110000
SURR = range(0xD800, 0xE000)
HANGUL = range(0xAC00, 0xAC00 + 11172)
CLS_OTHER, CLS_LETTER, CLS_NUMBER, CLS_SPACE = 0, 1, 2, 3


def bytes_to_unicode():  # GPT-2's byte <-> printable char map
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def check_config(d):
    pre = d["pre_tokenizer"]
    assert d["normalizer"] == {"type": "NFC"}, d["normalizer"]
    assert pre["type"] == "Sequence" and len(pre["pretokenizers"]) == 2, pre
    sp, bl = pre["pretokenizers"]
    assert sp["type"] == "Split" and sp["pattern"] == {"Regex": QWEN2_REGEX}, sp
    assert sp["behavior"] == "Isolated" and not sp["invert"], sp
    assert bl["type"] == "ByteLevel" and not bl["add_prefix_space"] and not bl["use_regex"], bl
    m = d["model"]
    assert m["type"] == "BPE" and not m.get("dropout") and not m.get("unk_token"), "unsupported BPE settings"
    assert not m.get("byte_fallback") and not m.get("ignore_merges"), "unsupported BPE settings"
    assert not m.get("continuing_subword_prefix") and not m.get("end_of_word_suffix"), "unsupported BPE settings"
    for a in d["added_tokens"]:
        assert not (a["lstrip"] or a["rstrip"] or a["single_word"] or a["normalized"]), a


def vocab_and_merges(d):
    u2b = {c: b for b, c in bytes_to_unicode().items()}
    vocab = d["model"]["vocab"]
    n = len(vocab)
    toks = [None] * n
    for s, i in vocab.items():
        toks[i] = bytes(u2b[c] for c in s)
    assert all(t is not None for t in toks) and len(set(toks)) == n, "vocab ids must be dense and unique"
    assert max(map(len, toks)) < 256
    assert len({t for t in toks if len(t) == 1}) == 256, "all 256 single-byte tokens must exist"
    merges = []
    for m in d["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ")
        merges.append((vocab[a], vocab[b], vocab[a + b]))
    assert len({(a, b) for a, b, _ in merges}) == len(merges), "duplicate merges"
    added = sorted((a["id"], a["content"].encode("utf8")) for a in d["added_tokens"])
    assert [i for i, _ in added] == list(range(n, n + len(added))), "added token ids must follow the vocab"
    return toks, merges, added


# ---------------------------------------------------------------------------------------------------------------
# Unicode tables
# ---------------------------------------------------------------------------------------------------------------
def batch(norm, strs, chunk=1 << 16):
    """norm.normalize_str on many strings at once ('\\n' is a starter, so it separates them safely)."""
    out = []
    for i in range(0, len(strs), chunk):
        part = strs[i:i + chunk]
        r = norm.normalize_str("\n".join(part)).split("\n")
        assert len(r) == len(part)
        out += r
    return out


def unicode_tables(d):
    try:
        from tokenizers import Regex, normalizers, pre_tokenizers
    except ImportError:
        raise SystemExit("needs the `tokenizers` package (use the repo venv: .venv/Scripts/python)")
    NFC, NFD = normalizers.NFC(), normalizers.NFD()
    split = pre_tokenizers.Split(Regex(d["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]), "isolated")
    cps = [c for c in range(MAXCP) if c not in SURR and c != 10]  # '\n' is the batch separator; it is trivial
    diffs = collections.Counter()

    # -- pre-tokenizer class: probe the real regex with "a<c><c>a"
    cls = {}
    for cp in cps:
        c = chr(cp)
        r = [p for p, _ in split.pre_tokenize_str("a" + c + c + "a")]
        if cp == 13: k = CLS_SPACE  # "two CRs" looks like punctuation in this probe
        elif r == ["a" + c + c + "a"]: k = CLS_LETTER
        elif r == ["a", c, c, "a"]: k = CLS_NUMBER
        elif r == ["a", c, c + "a"]: k = CLS_SPACE
        elif r == ["a", c + c, "a"]: k = CLS_OTHER
        else: raise SystemExit(f"cannot classify U+{cp:04X}: {r}")
        cls[cp] = k
        ucat = unicodedata.category(c)[0]
        py = CLS_LETTER if ucat == "L" else CLS_NUMBER if ucat == "N" else CLS_OTHER
        if cls[cp] != CLS_SPACE and py != k:
            diffs["class vs unicodedata"] += 1
    cls[10] = CLS_SPACE
    spaces = [hex(c) for c in sorted(cls) if cls[c] == CLS_SPACE]
    assert len(spaces) == 25, spaces  # the Unicode White_Space set, as the C code's comments assume

    # -- NFD / NFC of every single code point
    sc = [chr(c) for c in cps]
    nfd = dict(zip(cps, (s for s in batch(NFD, sc))))
    nfc1 = dict(zip(cps, batch(NFC, sc)))
    dec = {c: [ord(x) for x in nfd[c]] for c in cps if nfd[c] != chr(c) and c not in HANGUL}
    for c in cps:
        if nfd[c] != unicodedata.normalize("NFD", chr(c)) and c not in HANGUL:
            diffs["NFD vs unicodedata"] += 1
    for c in HANGUL:  # Hangul is algorithmic in C: check the formula against the library
        s = c - 0xAC00
        f = [0x1100 + s // 588, 0x1161 + s % 588 // 28] + ([0x11A7 + s % 28] if s % 28 else [])
        assert [ord(x) for x in nfd[c]] == f and nfc1[c] == chr(c)

    # -- canonical combining class (only matters for code points that survive NFD)
    stable = [c for c in cps if nfd[c] == chr(c)]
    cand = set(c for c in stable if unicodedata.combining(chr(c)))
    fa = batch(NFD, ["a" + chr(c) + "\u0334" for c in stable])
    fb = batch(NFD, ["a\u0345" + chr(c) for c in stable])
    for c, x, y in zip(stable, fa, fb):
        if x != "a" + chr(c) + "\u0334" or y != "a\u0345" + chr(c):
            cand.add(c)
    def jumps(x, ys):  # does mark y move in front of x in "a"+x+y ?
        r = batch(NFD, ["a" + chr(x) + chr(y) for y in ys])
        return [o != "a" + chr(x) + chr(y) for o, y in zip(r, ys)]
    refs = {}  # combining class value -> a code point that really has that class in the library
    by_val = collections.defaultdict(list)
    for c in stable:
        v = unicodedata.combining(chr(c))
        if v:
            by_val[v].append(c)
    for v in sorted(by_val):
        for c in by_val[v][:6]:
            if all(jumps(r, [c])[0] == (v < w) for w, r in refs.items()) and \
               all(jumps(c, [r])[0] == (w < v) for w, r in refs.items()):
                refs[v] = c
                break
    rv = sorted(refs)
    ccc = {}
    for c in sorted(cand):
        j = {v: jumps(refs[v], [c])[0] for v in rv}  # c after ref v moves before it iff 0 < ccc(c) < v
        ok = [v for v in rv if not j[v]]
        if len(ok) == len(rv):  # never moved: 0, or at least the largest class
            moved = batch(NFD, ["a" + chr(c) + "\u0334"])[0] != "a" + chr(c) + "\u0334"
            w = rv[-1] if moved else 0
        else:
            w = max(ok) if ok else 0
            if not ok or not all(j[v] for v in rv if v > w):
                raise SystemExit(f"inconsistent combining class for U+{c:04X}")
        if w:
            ccc[c] = w
        if w != unicodedata.combining(chr(c)):
            diffs["ccc vs unicodedata"] += 1  # tokenizers treats e.g. U+1DFA and U+16FF0 as starters

    # -- composition pairs (first, second) -> composite, derived from the library's own NFD / NFC
    comp = {}
    cc = [c for c in dec if len(dec[c]) >= 2]
    firsts = batch(NFC, ["".join(map(chr, dec[c][:-1])) for c in cc])
    pair = [(f, chr(dec[c][-1])) for c, f in zip(cc, firsts)]
    back = batch(NFC, [f + s for f, s in pair])
    for c, (f, s), r in zip(cc, pair, back):
        if len(f) == 1 and r == chr(c):
            comp[(ord(f), ord(s))] = c
    py = {}
    for c in cps:
        dd = unicodedata.decomposition(chr(c))
        if dd and not dd.startswith("<"):
            p = [int(x, 16) for x in dd.split()]
            if len(p) == 2 and unicodedata.normalize("NFC", chr(p[0]) + chr(p[1])) == chr(c):
                py[tuple(p)] = c
    diffs["composition pairs vs unicodedata"] = len(set(py.items()) ^ set(comp.items()))

    # -- per code point payload; "dirty" = NFC may change the text around it
    second = {s for _, s in comp} | set(range(0x1161, 0x1176)) | set(range(0x11A8, 0x11C3))
    nfc_changed = [c for c in cps if nfc1[c] != chr(c)]
    pay = [0] * MAXCP
    for c in cps:
        dirty = nfc1[c] != chr(c) or ccc.get(c, 0) or c in second
        pay[c] = cls[c] | (4 if dirty else 0) | (ccc.get(c, 0) << 3)
    pay[10] = CLS_SPACE
    runs = [i for i in range(MAXCP) if i == 0 or pay[i] != pay[i - 1]]
    uni = [i | pay[i] << 21 for i in runs]
    print("  differences between `tokenizers` and unicodedata", unicodedata.unidata_version + ":", dict(diffs))
    print(f"  {len(uni)} uni runs, {len(dec)} decompositions, {len(comp)} compositions, {len(ccc)} non-starters, "
          f"{len(nfc_changed)} NFC-changing code points")
    return uni, dec, comp


# ---------------------------------------------------------------------------------------------------------------
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models/qwen/tokenizer-instruct/tokenizer.json")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "models/qwen/qwen_tokenizer.bin")
    d = json.load(open(src, encoding="utf8"))
    check_config(d)
    toks, merges, added = vocab_and_merges(d)
    t0 = time.time()
    uni, dec, comp = unicode_tables(d)
    print(f"  unicode tables in {time.time() - t0:.0f}s")

    blob = b"".join(toks)
    decs = sorted(dec)
    pool = [x for c in decs for x in dec[c]]
    offs = [0]
    for c in decs:
        offs.append(offs[-1] + len(dec[c]))
    comps = sorted((a, b, c) for (a, b), c in comp.items())

    def pad(b):
        return b + b"\0" * (-len(b) % 4)

    def u32(a):
        return struct.pack(f"<{len(a)}I", *a)
    out = [b"QWTOK001",
           u32([len(toks), len(added), len(merges), len(blob), len(uni), len(decs), len(pool), len(comps), 0, 0, 0, 0]),
           pad(bytes(map(len, toks))), pad(blob), u32([x for m in merges for x in m])]
    for i, s in added:
        out.append(u32([i, len(s)]) + pad(s))
    out += [u32(uni), u32(decs), u32(offs), u32(pool), u32([x for t in comps for x in t])]
    data = b"".join(out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(data)
    print(f"wrote {dst}: {len(data)} bytes ({len(toks)} tokens, {len(added)} added, {len(merges)} merges)")


if __name__ == "__main__":
    main()
