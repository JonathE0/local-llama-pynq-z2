#!/usr/bin/env python3
"""Compare the C tokenizer (qwen/tok_test) with Hugging Face `tokenizers`, ids for ids. Builds nothing.

    .venv/Scripts/python tools/qwen_tok_test.py [--tool "<cmd prefix>"] [--json tokenizer.json] [--quick]

--tool is the command that runs the C CLI up to and including the .bin argument; "enc"/"dec" and flags are appended.
Default: qwen/tok_test(.exe) models/qwen/qwen_tokenizer.bin. Use forward slashes. On the board (from the laptop):
    --tool "ssh pynq ~/qwen/tok_test ~/qwen/qwen_tokenizer.bin"
Build first: gcc -O2 -std=c99 -o tok_test qwen/tok_test.c qwen/tokenizer.c   (export the .bin with qwen_tok_export.py)

Test groups (every case is encoded separately by both sides and compared id for id; allow_special on = `-s`, off = plain
text; "off" is HF with the 22 added tokens removed). Each group is also decoded: C decode of the HF ids must equal the
text after NFC (HF normalizer, per segment between added tokens):
  files        wiki.test.raw, eval_code.py.txt, repo docs/tools: whole file as one case, special off and on
  stress       hand-written cases: CJK, emoji/ZWJ, accents/combining, RTL, digits, whitespace, contractions, long runs,
               chat template / FIM / tool-call text (special off and on)
  nfc          every composition pair, Hangul jamo triples, random combining-mark soup
  fuzz         random strings over a pool that hits every branch of the regex (special off)
  allcp        every Unicode code point (not surrogates) in a few contexts (class / NFC tables), special off
--write-stress FILE writes the stress cases NUL-separated (feed it to `tok_test <bin> enc -s`).
Exit status 0 only if there are no mismatches.
"""
import argparse
import difflib
import os
import random
import re
import shlex
import subprocess
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_tok_export import bytes_to_unicode  # noqa: E402  (same byte<->unicode map; only used to print tokens)

from tokenizers import Tokenizer, normalizers  # noqa: E402
import json  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# test material
# ---------------------------------------------------------------------------------------------------------------
PY = "def f(x):\n    if x > 0:\n        return [i ** 2 for i in range(x)]  # squares\n    return None\n\n\nclass A:\n\tpass\n"
STRESS = {
    "ascii": ["", "a", " ", "\n", "\t", "Hello, world!", "The quick brown fox jumps over the lazy dog.", "  leading", "trailing  ",
              "   ", "x" * 7, "a\x01b\x7fc\x1fd", "\x1c\x1d\x1e\x1f", "tab\there", "a_b_c", "x=1+2*3;", "#include <stdio.h>\nint main(){return 0;}\n"],
    "contractions": ["I'm you're he's she'd we'll they've can't won't", "DON'T WE'LL IT'S I'M YOU'RE THEY'VE HE'D", "we'Ll don'T She'S I'ld THEY'rE",
                     "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D", "'sometimes 'tis 'twas 'dd 'ree 'lll rock'n'roll o'clock", "''s '''t ' 's' 'x '1 '\u00e9",
                     "it\u2019s the \u2018quote\u2019 can\u2019t", "\u017f \u2019\u017f '\u017f '\u017ftill '\u017f'\u017f", "'re're'll'd", "a's's's", "'s\n's\r\n's"],
    "digits": ["0123456789" * 20, "1234567890123", "3.14159", "1,234,567.89", "x2y33z444", "\uff12\uff10\uff12\uff14", "\u0663\u0664\u0665", "\u2460\u2461\u2462",
               "\u00bd \u00bc \u00be", "\u2167 \u2160", "1e10 0x1F 0b101 1_000", "2024-01-31T12:34:56Z", "+1 (555) 123-4567", "\u0e51\u0e52\u0e53 \u0966\u0967", "12 345", "9" * 300],
    "whitespace": ["a  b", "a   b", "a    ", "a\n\nb", "a\n\n\nb", "a \n b", "a\r\nb\r\n", "a\r\n\r\nb", "\r\r\n", " \n", "\n ", "  \n  \n  x",
                   "x\n  ", " " * 100, "\n" * 100, " \n" * 50, "\t\t\tx", "a\tb\t\tc", "\ta", " \ta", "a \t b", "\u00a0nbsp", "a\u00a0b", "a\u00a0\u00a0b",
                   "\u2003em\u2002space", "\u3000ideographic\u3000", "a\u2028b\u2029c", "\u0085nel", "a\x0b\x0cb", "line1\r\nline2\rline3\nline4", " " * 7 + "x",
                   "word \n", "word  \n\n", "word\n \n", "  \n", "\n\n  ", "!  ", "!  \n", "! \n!", ". . .", "a\u200bb\u200c\u200d", "\ufeffBOM at start", "x \u3000 y", "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"],
    "punct": ["!!!???...", "a---b", "((((", "<html><body></body></html>", "foo.bar(baz);", "// comment\n/* c */", "\"quoted\" 'single'", "$$$ %%% ^^^", "...\n\n...",
              "\u3002\u3002\u3002", "a, b.\n\nc", "?!\n", ".\n\n", "!\n\n\n", " !", " !\n", "  !x", "\u2014\u2026\u201c\u201d", "\u00a1Hola! \u00bfQu\u00e9?", "~`!@#$%^&*()_+-=[]{}|\\;:'\",.<>/?",
              "http://example.com/a?b=c&d=e#f", "user@example.com", "C:\\Users\\x\\file.txt", "/usr/local/bin/python3"],
    "cjk": ["\u4f60\u597d\uff0c\u4e16\u754c\uff01\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002\u3072\u3089\u304c\u306a\u30ab\u30bf\u30ab\u30ca\u6f22\u5b57",
            "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8\uc785\ub2c8\ub2e4 \ud6c8\ubbfc\uc815\uc74c", "Hello \u4e16\u754c 123 \u4e16\u754cabc \u4e16\u754c\n\u4e16\u754c", "\u4e2d\u6587" * 500, "\u3042\u3044\u3046\u3048\u304a " * 40,
            "\u5929\u5730\u7384\u9ec3\uff0c\u5b87\u5b99\u6d2a\u8352\u3002", "\U00020000\U00020001\U0002a6d6 ext B/C", "\uff21\uff22\uff23 \uff41\uff42\uff43 \uff11\uff12\uff13", "\u3000\u3000\u3000\u6bb5\u843d", "\u30fb\u3001\u300c\u300d\u300e\u300f"],
    "emoji": ["\U0001f600\U0001f603\U0001f604", "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466", "\U0001f3f3\ufe0f\u200d\U0001f308", "\U0001f44d\U0001f3fd", "\U0001f1fa\U0001f1f8\U0001f1ef\U0001f1f5",
              "1\ufe0f\u20e32\ufe0f\u20e3", "\u2764\ufe0f \u00a9\ufe0f \u2122\ufe0f", "\U0001f9d1\u200d\U0001f4bb", "\U0001f469\U0001f3fb\u200d\U0001f91d\u200d\U0001f468\U0001f3ff", "\u2764\ufe0f\u200d\U0001f525",
              "text \U0001f600 text \U0001f600\U0001f600 text", "\U0001f600" * 800, "\U0001f4a9\n\U0001f4a9\n", "a\U0001f600b", " \U0001f600", "\U0001f600 ", "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"],
    "accents": ["caf\u00e9 na\u00efve fa\u00e7ade Z\u00fcrich", "cafe\u0301 nai\u0308ve", "\u00c5ngstr\u00f6m \u212bngstr\u00f6m", "\u1e9b\u0323", "\u01c5 \ufb01 \u0130stanbul \u00df \u1e9e", "n\u0303 e\u0301 e\u0301\u0323 e\u0323\u0301",
                "a\u0308\u0304 A\u030a", "\u1112\u1161\u11ab\u1100\u1173\u11af \ud55c\uae00", "q\u0307\u0323", "\u0301abc", "x\u0301\u0301\u0301", "Z\u0351\u0361a\u036cl\u0352g\u0315o\u0335",
                "\u1ec7 e\u0302\u0323 e\u0323\u0302", "\u0958 \u0915\u093c", "\u0344 \u0308\u0301", "\u2126 \u03a9", "\u1f80 \u1f00\u0345", "\u09c7\u09be \u09cb", "\u0b47\u0b3e", "\u0cc6\u0cc2\u0cd5",
                "\ufb1d \ufb2c \u05e9\u05bc\u05c1", "\U0001d15e \U0001d157\U0001d165", "\u0f73 \u0f71\u0f72", "\u2adc \u2add\u0338", "A\u0338\u0338", "\u00e9\u0301", "e\u0301\u0301", "\u0627\u0653 \u0622",
                "\u1dfa\u0316 \u0316\u1dfa", "\U00016ff0\u0301 a\U00016ff0", "\u0898\u0316 \U00010d24\u0301", "\U00010d24\u0316", "\U00011938 \U00011935\U00011930", "o\u0338\u0301", "a\u0316\u0301\u0323"],
    "rtl": ["\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd", "Hello \u05e9\u05dc\u05d5\u05dd 123 \u0645\u0631\u062d\u0628\u0627", "\u200eabc\u200f \u202bxyz\u202c", "\u0661\u0662\u0663\u0664",
            "\u0628\u0650\u0633\u0652\u0645\u0650 \u0671\u0644\u0644\u0651\u064e\u0670\u0647\u0650", "\u0645\u0631\u062d\u0628\u0627\n\u05e9\u05dc\u05d5\u05dd\n", "\u062a\u0640\u0640\u0640\u0640\u0640\u0640\u0640\u0640\u0633", "\u06f1\u06f2\u06f3 \u067e\u0686\u0698\u06af"],
    "scripts": ["\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01", "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e", "\u0e21\u0e19\u0e38\u0e29\u0e22\u0e4c", "\u03a7\u03b1\u03af\u03c1\u03b5\u03c4\u03b5 \u03ba\u03cc\u03c3\u03bc\u03b5 \u03ac\u03ad\u03ae\u03af\u03cc\u03cd\u03ce",
                "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u0419\u0451\u0436", "\u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0", "\u0540\u0561\u0575\u0565\u0580\u0565\u0576", "\u1200\u1208\u121e \u12d3\u1208\u121d", "\u1780\u17d2\u1798\u17c2\u179a", "\u0f56\u0f7c\u0f51\u0f0b\u0f61\u0f72\u0f42",
                "\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02", "\u0b95\u0ba4\u0bc8", "\U00010400\U00010428 \U0001d400\U0001d41a \U0001f130", "\u2126\u212a\u212b \u2160 \u24b6 \u00b2\u00b3\u00b9 \u2070", "\u0130i\u0307 \u0131", "\u01c6 \u01c5 \u01c4", "\ua7dc\u1c89 \U000105c0 \U00010d40 \U0002ee5d"],
    "long": ["a" * 5000, "ab" * 3000, " " * 3000 + "x", "\n" * 1000, "1" * 3000, "!" * 2000, "\u00e9" * 1500, ".\n" * 500, "word " * 1000, "x\n" * 700, "A" * 300 + "b" * 300,
             "".join(chr(65 + i % 26) for i in range(2000)), "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
             "hello" * 1000, "=" * 500 + "\n" + "-" * 500, "ab cd " * 400, "\u4e2d\u6587 " * 700],
    "code": [PY, '{"name": "Qwen", "list": [1, 2.5, null, true], "nested": {"k": "v\\n\\u00e9"}}\n', "SELECT * FROM t WHERE a >= 10 AND b <> 'x';\n",
             "<div class=\"a\" id='b'>text &amp; more</div>", "def \u00fcber(\u03b1, \u03b2):\n    return \u03b1 + \u03b2  # \u2713\n", "printf(\"%d\\n\", x++);\r\n\tx--;\r\n",
             "aGVsbG8gd29ybGQ=" * 30, "0x7fffffffe000 deadbeef cafebabe", "a.b.c.d.e.f.g.h", "git commit -m \"fix: handle 'quotes'\" --amend\n", "$ ls -la /usr/bin | grep python\n", "x = [1,2,3][0]**2 // 3 % 4 << 1 >= 5 != 6"],
}
CHAT = [
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a haiku about caf\u00e9s.<|im_end|>\n<|im_start|>assistant\nSteam on the window\u2014\n<|im_end|>\n",
    "<|im_start|><|im_end|>", "<|endoftext|>", "x<|im_end|>y", "<|im_start|", "<|im_start|>|>", "<<|im_start|>>", "<|IM_START|>", "<|im_start| >", "<|im_start|><|im_start|><|im_start|>",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call>", "<|fim_prefix|>def f():\n    <|fim_suffix|>\n    return 1<|fim_middle|>", "<|repo_name|>x<|file_sep|>a.py\n",
    "<|vision_start|><|image_pad|><|vision_end|>", "<|im_end|>\n<|im_end|>", "e<|im_end|>\u0301", "\u00e9<|im_end|>\u0301a", "e\u0301<|endoftext|>\u0323", "<|endoftext|><|endoftext|>", "a <|im_start|> b", "\n<|im_start|>\n",
    "<|im_start|>user\nCafe\u0301 \u4f60\u597d \U0001f600<|im_end|>", "<|object_ref_start|>a<|object_ref_end|><|box_start|>b<|box_end|><|quad_start|>c<|quad_end|>", "<|video_pad|><|vision_pad|>",
    "<tool_call><tool_call></tool_call></tool_call>", "<|file_sep|><|fim_pad|><|repo_name|>", "<|", "|>", "<>", "<|im_end|", "<|im_end|>>", "<<|im_end|>", " <|im_end|> ", "\n\n<|im_end|>\n\n",
    "x" * 300 + "<|im_end|>" + "y" * 300, "<|im_start|>" * 40,
]

POOL = ["a", "B", "z", "Q", "e", "s", "t", "r", "l", "d", "m", "v", "S", "T", "R", "E", "L", "D", "M", "V", "1", "7", "0", ".", ",", "!", "?", "'", "'", "'", "-", "(", ")", "_", "/", "<", "|", ">",
        " ", " ", " ", " ", "\t", "\n", "\n", "\r", "\x0b", "\x0c", "\u00a0", "\u0085", "\u2028", "\u3000", "\u200b", "\u200d", "\ufe0f", "\u00e9", "\u00f1", "\u00df", "\u017f", "\u0130", "\u4e2d",
        "\u65e5", "\ud55c", "\u0627", "\u05e9", "\u0663", "\u00b2", "\u00bd", "\u2167", "\U0001f600", "\U0001f468", "\u3002", "\uff0c", "\u0301", "\u0323", "\u0338", "\u0345", "\u3099", "\u094d", "\u093c",
        "\u1dfa", "\U00016ff0", "\u0898", "\U00010d24", "\u1100", "\u1161", "\u11a8", "\uac00", "\uac01", "\u09c7", "\u09be", "\u0b47", "\u0b3e", "\u212b", "\u0958", "\u0344", "\ufb1d", "\U0001d15e",
        "\u0f73", "\u2126", "\u00c5", "\u1e9b", "\U0001e4ec", "\x1c", "\x7f", "\U000e0067"]


# ---------------------------------------------------------------------------------------------------------------
class Ref:
    def __init__(self, path):
        d = json.load(open(path, encoding="utf8"))
        self.tok = Tokenizer.from_str(json.dumps(d))
        d2 = dict(d)
        d2["added_tokens"] = []
        self.plain = Tokenizer.from_str(json.dumps(d2))  # same model without added tokens = allow_special off
        self.nfc = normalizers.NFC()
        u2b = {c: b for b, c in bytes_to_unicode().items()}
        self.tb = {i: bytes(u2b[c] for c in s) for s, i in d["model"]["vocab"].items()}
        for a in d["added_tokens"]:
            self.tb[a["id"]] = a["content"].encode("utf8")
        self.special_re = re.compile("|".join(re.escape(a["content"]) for a in sorted(d["added_tokens"], key=lambda a: -len(a["content"]))))

    def encode(self, texts, special):
        t = self.tok if special else self.plain
        out = []
        for i in range(0, len(texts), 20000):  # batches: an Encoding object is big
            out += [e.ids for e in t.encode_batch(texts[i:i + 20000], add_special_tokens=False)]
        return out

    def expected_bytes(self, text, special):  # decode(encode(text)) as bytes: NFC per segment between added tokens
        if not special:
            return self.nfc.normalize_str(text).encode("utf8")
        out, pos = [], 0
        for m in self.special_re.finditer(text):
            out += [self.nfc.normalize_str(text[pos:m.start()]), m.group()]
            pos = m.end()
        out.append(self.nfc.normalize_str(text[pos:]))
        return "".join(out).encode("utf8")

    def show(self, ids):
        return [self.tb.get(i, b"?").decode("utf8", "backslashreplace") for i in ids]


class CTool:
    def __init__(self, cmd):
        self.cmd = shlex.split(cmd)

    def run(self, args, data):
        r = subprocess.run(self.cmd + args, input=data, capture_output=True)
        if r.returncode:
            sys.exit(f"C tool failed ({r.returncode}): {r.stderr.decode(errors='replace')[:500]}")
        return r.stdout, r.stderr.decode(errors="replace")

    def encode(self, texts, special, timing=False):
        data = b"".join(t.encode("utf8") + b"\0" for t in texts)
        out, err = self.run(["enc"] + (["-s"] if special else []) + (["-t"] if timing else []), data)
        lines = out.split(b"\n")[:-1]
        assert len(lines) == len(texts), f"C tool printed {len(lines)} lines for {len(texts)} cases"
        return [[int(x) for x in l.split()] for l in lines], err

    def decode(self, idlists):
        data = b"".join(b" ".join(b"%d" % i for i in ids) + b"\n" for ids in idlists)
        out, _ = self.run(["dec"], data)
        res = out.split(b"\0")[:-1]
        assert len(res) == len(idlists)
        return res


def report(ref, name, text, exp, got, special, limit_blocks=4):
    print(f"  MISMATCH {name!r}: HF {len(exp)} ids, C {len(got)} ids, special={'on' if special else 'off'}")
    if len(exp) + len(got) < 400000:
        ops = [o for o in difflib.SequenceMatcher(None, exp, got, autojunk=False).get_opcodes() if o[0] != "equal"]
    else:
        i = next((k for k, (x, y) in enumerate(zip(exp, got)) if x != y), min(len(exp), len(got)))
        ops = [("replace", i, len(exp), i, len(got))]
    blob = ref.expected_bytes(text, special)
    print(f"    {len(ops)} differing block(s); showing {min(len(ops), limit_blocks)}")
    for tag, i1, i2, j1, j2 in ops[:limit_blocks]:
        off = sum(len(ref.tb.get(x, b"")) for x in exp[:i1])
        snip = blob[max(0, off - 40):off + 40].decode("utf8", "backslashreplace")
        print(f"    at token {i1} (byte {off}) near {snip!r}")
        print(f"      HF: {exp[max(0, i1 - 3):i2 + 3]} {ref.show(exp[max(0, i1 - 3):i2 + 3])}")
        print(f"      C : {got[max(0, j1 - 3):j2 + 3]} {ref.show(got[max(0, j1 - 3):j2 + 3])}")


def check(ref, tool, label, cases, special, timing=False, max_report=4):
    """cases: list of (name, text). Returns the number of mismatching cases (ids + decode)."""
    names, texts = [c[0] for c in cases], [c[1] for c in cases]
    t0 = time.time()
    exp = ref.encode(texts, special)
    t1 = time.time()
    got, err = tool.encode(texts, special, timing)
    t2 = time.time()
    bad = [k for k in range(len(texts)) if exp[k] != got[k]]
    for k in bad[:max_report]:
        report(ref, names[k], texts[k], exp[k], got[k], special)
    dec = tool.decode(exp)
    dbad = [k for k in range(len(texts)) if dec[k] != ref.expected_bytes(texts[k], special)]
    for k in dbad[:max_report]:
        want = ref.expected_bytes(texts[k], special)
        i = next((j for j, (a, b) in enumerate(zip(dec[k], want)) if a != b), min(len(dec[k]), len(want)))
        print(f"  DECODE MISMATCH {names[k]!r}: byte {i}: C {dec[k][max(0, i - 20):i + 20]!r} want {want[max(0, i - 20):i + 20]!r}")
    ntok = sum(map(len, exp))
    print(f"{'FAIL' if bad or dbad else 'ok  '} {label:<26} special={'on ' if special else 'off'} cases={len(cases):<8} ids={ntok:<9} "
          f"encode mismatches={len(bad)} decode mismatches={len(dbad)}   (HF {t1 - t0:.1f}s, C+io {t2 - t1:.1f}s)")
    if timing and err.strip():
        print("     C: " + err.strip().replace("\n", "\n     C: "))
    return len(bad) + len(dbad)


# ---------------------------------------------------------------------------------------------------------------
def build_cases(quick, seed):
    rng = random.Random(seed)
    stress = [(f"{g}[{i}]", t) for g, ts in STRESS.items() for i, t in enumerate(ts)]
    chat = [(f"chat[{i}]", t) for i, t in enumerate(CHAT)]
    stress.append(("stress-joined", "\n".join(t for _, t in stress)))
    stress.append(("chat-joined", "".join(CHAT)))
    # NFC: composition pairs, Hangul, mark soup
    pairs = {}
    for c in range(0x110000):
        dd = unicodedata.decomposition(chr(c)) if not 0xD800 <= c < 0xE000 else ""
        if dd and not dd.startswith("<"):
            p = [int(x, 16) for x in dd.split()]
            if len(p) == 2:
                pairs[tuple(p)] = c
    nfc = []
    for (a, b), c in sorted(pairs.items()):
        nfc.append((f"pair U+{a:04X} U+{b:04X}", f"x{chr(a)}{chr(b)}y"))
        nfc.append((f"pairrev U+{a:04X} U+{b:04X}", f"{chr(b)}{chr(a)}"))
        nfc.append((f"pair+mark U+{a:04X} U+{b:04X}", f"{chr(a)}{chr(b)}\u0323\u0301"))
        nfc.append((f"composite U+{c:04X}", f"{chr(c)}\u0301"))
    for l in range(0x1100, 0x1113):
        for v in range(0x1161, 0x1176):
            nfc.append((f"jamo {l:X} {v:X}", "".join(chr(l) + chr(v) + (chr(t) if t > 0x11A7 else "") for t in range(0x11A7, 0x11C3))))
    marks = [chr(c) for c in range(0x110000) if not 0xD800 <= c < 0xE000 and unicodedata.combining(chr(c))]
    seconds = sorted({chr(b) for (_, b) in pairs})
    firsts = sorted({chr(a) for (a, _) in pairs})
    for i in range(3000 if quick else 20000):
        n = rng.randint(1, 10)
        s = "".join(rng.choice(firsts if rng.random() < 0.4 else ["a", "e", "o", "x", "\n", " "] if rng.random() < 0.5 else marks if rng.random() < 0.6 else seconds)
                    for _ in range(n))
        nfc.append((f"soup[{i}]", s))
    fuzz = []
    for i in range(5000 if quick else 40000):
        fuzz.append((f"fuzz[{i}]", "".join(rng.choice(POOL) for _ in range(rng.randint(1, 30)))))
    allcp = []
    cps = [c for c in range(1, 0x110000) if not 0xD800 <= c < 0xE000]
    if quick:
        cps = [c for c in cps if c < 0x3000 or 0x1F000 <= c < 0x20000 or rng.random() < 0.02]
    for c in cps:
        ch = chr(c)
        allcp.append((f"U+{c:04X}", f"\n{ch}\na{ch}{ch}a '{ch}a {ch}\n"))
    return stress, chat, nfc, fuzz, allcp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    exe = os.path.join(ROOT, "qwen", "tok_test" + (".exe" if os.name == "nt" else ""))
    ap.add_argument("--tool", default=f"{exe.replace(os.sep, '/')} {ROOT.replace(os.sep, '/')}/models/qwen/qwen_tokenizer.bin")
    ap.add_argument("--json", default=os.path.join(ROOT, "models/qwen/tokenizer-instruct/tokenizer.json"))
    ap.add_argument("--quick", action="store_true", help="smaller fuzz / sampled code points")
    ap.add_argument("--no-allcp", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write-stress", metavar="FILE")
    ap.add_argument("--files", nargs="*", help="extra text files (default: wikitext test, eval_code, repo docs)")
    a = ap.parse_args()

    ref, tool = Ref(a.json), CTool(a.tool)
    stress, chat, nfc, fuzz, allcp = build_cases(a.quick, a.seed)
    if a.write_stress:
        open(a.write_stress, "wb").write(b"".join(t.encode("utf8") + b"\0" for _, t in stress + chat))
        print("wrote", a.write_stress)
    files = a.files if a.files is not None else [
        "models/data/wikitext-2-raw/wiki.test.raw", "models/data/code/eval_code.py.txt", "README.md", "NOTES.md", "CONTEXT.md",
        "tools/qwen_tok_test.py", "qwen/tokenizer.c"]
    fcases = []
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(ROOT, f)
        if os.path.exists(p):
            fcases.append((os.path.basename(p), open(p, "rb").read().decode("utf8").replace("\0", "")))
        else:
            print("skip missing", p)
    bad = 0
    t0 = time.time()
    # big files first, one process per file so the C-side timing / memory lines belong to the file
    for name, text in fcases:
        for sp in (False, True):
            bad += check(ref, tool, f"file {name} ({len(text.encode()) // 1024} KB)", [(name, text)], sp, timing=True)
    for sp in (False, True):
        bad += check(ref, tool, "stress", stress, sp)
        bad += check(ref, tool, "chat template", [(n, t) for n, t in chat], sp)
    bad += check(ref, tool, "nfc", nfc, False)
    bad += check(ref, tool, "nfc (special on)", nfc[:2000], True)
    bad += check(ref, tool, "fuzz", fuzz, False)
    if not a.no_allcp:
        bad += check(ref, tool, "allcp", allcp, False)
    # robustness (HF cannot take invalid UTF-8, so only sanity): random bytes must not crash or hang; ASCII mixed with
    # stray bytes that can never form UTF-8 (0x80-0xC1, 0xF5-0xFF) must round-trip exactly
    rng = random.Random(5)
    for label, junk in (("random bytes", rng.randbytes(200000).replace(b"\0", b"\1")),
                        ("ascii + stray bytes", bytes(rng.choice(b"abc XY\n\t.,'1" + bytes(range(0x80, 0xC2)) + bytes(range(0xF5, 0x100)))
                                                      for _ in range(100000)))):
        t1 = time.time()
        out, _ = tool.run(["enc", "-s"], junk + b"\0")
        ids = [int(x) for x in out.split()]
        back = tool.run(["dec"], out)[0][:-1]
        ok = bool(ids) and (back == junk or label == "random bytes")
        print(f"{'ok  ' if ok else 'FAIL'} {label:<26} {len(junk)} bytes -> {len(ids)} ids, decode "
              f"{'round-trips' if back == junk else 'differs (NFC of valid sequences)'} ({time.time() - t1:.1f}s)")
        bad += not ok
    print(f"\n{'ALL OK: zero mismatches' if not bad else f'{bad} MISMATCHES'} in {time.time() - t0:.0f}s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
