"""numpy reference forward pass of Qwen2.5 from the board's .qwb file, with exactly qwen/qwen.c's arithmetic:
activations quantized to int8 per group of 32 (scale = max|x| * (1/127), q = roundf(x * (1/scale))), int sums inside a
group, out[i] = sum_g (isum * wscale) * xscale; logits only for the head_ids rows. Greedy decoding.

usage:
  python tools/qwen_ref.py model.qwb --prompt "text" [--raw] [-n 40]   chat template (HF tokenizers) -> ids, greedy
  python tools/qwen_ref.py model.qwb --ids 1,2,3 [-n 40]               same from token ids
  python tools/qwen_ref.py model.qwb --ids ... --logits c.bin          compare with qwen.c's --logits dump (run -t 0):
      feeds the prompt, then qwen.c's own greedy choices, and compares the logits at every position
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TEMPLATE = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
STOP = (151645, 151643)  # <|im_end|>, <|endoftext|>
F = np.float32


class Model:
    def __init__(self, path):
        m = np.memmap(path, dtype=np.uint8, mode="r")
        assert bytes(m[:4]) == b"QWB1" and m[4:8].view("<u4")[0] == 1, "not a .qwb v1 file"
        (self.dim, self.hidden, self.L, self.heads, self.kv_heads, self.vocab, self.head_k, _, self.gs,
         emb_bits) = (int(v) for v in m[8:48].view("<i4"))
        self.theta, self.eps = (float(v) for v in m[48:56].view("<f4"))
        assert self.gs == 32 and emb_bits == 8
        self.hs = self.dim // self.heads
        self.kv = self.hs * self.kv_heads
        self.m, self.off = m, 256
        L, dim, kv, hid = self.L, self.dim, self.kv, self.hidden
        vec = lambda n: [self.take(n * 4).view("<f4") for _ in range(L)]
        self.attn_norm, self.ffn_norm, self.bq, self.bk, self.bv = vec(dim), vec(dim), vec(dim), vec(kv), vec(kv)
        self.out_norm = self.take(dim * 4).view("<f4")
        self.head_ids = self.take(self.head_k * 4).view("<i4").astype(np.int64)
        mats = lambda d, n: [self.matrix(d, n, 4) for _ in range(L)]
        self.wq, self.wk, self.wv, self.wo = mats(dim, dim), mats(kv, dim), mats(kv, dim), mats(dim, dim)
        self.w1, self.w2, self.w3 = mats(hid, dim), mats(dim, hid), mats(hid, dim)
        self.head = self.matrix(self.head_k, dim, 8)
        self.emb_q = self.take(self.vocab * dim).view(np.int8).reshape(self.vocab, dim)
        self.emb_s = self.take(self.vocab * dim // 32 * 2).view("<f2").reshape(self.vocab, dim // 32)
        assert self.off == len(m), "file size does not match the header"

    def take(self, nbytes):
        a = self.m[self.off:self.off + nbytes]
        self.off += nbytes + (-(self.off + nbytes) % 64)
        return a

    def matrix(self, d, n, bits):
        """(weights as float32 [n/32, d, 32] for exact group sums through BLAS, scales float32 [d, n/32])"""
        p = self.take(d * n * bits // 8)
        s = self.take(d * n // 32 * 2).view("<f2").reshape(d, n // 32).astype(F)
        if bits == 4:
            p = p.reshape(d, n // 2).view(np.int8)
            q = np.empty((d, n), np.int8)
            q[:, 0::2] = (p << 4) >> 4  # low nibble, sign-extended
            q[:, 1::2] = p >> 4  # high nibble (arithmetic shift)
        else:
            q = p.view(np.int8).reshape(d, n)
        return np.ascontiguousarray(q.reshape(d, n // 32, 32).transpose(1, 0, 2), dtype=F), s


def quantize(x):
    """int8 values (as float32, [n/32, 32]) and float32 scales, like qwen.c's quantize()"""
    g = x.reshape(-1, 32)
    s = (np.abs(g).max(1) * F(1 / 127)).astype(F)
    inv = (F(1) / np.where(s > 0, s, F(1))) * (s > 0)
    v = g * inv[:, None].astype(F)
    t = np.trunc(v)
    return (t + np.sign(v) * (np.abs(v - t) >= 0.5)).astype(F), s  # roundf: halves away from zero


def matvec(w, xq):
    wg, ws = w
    q, xs = xq
    isum = np.matmul(wg, q[:, :, None])[:, :, 0]  # [groups, d], exact integers (|sum| < 2^24)
    return ((isum.T * ws) * xs).sum(1, dtype=F)


def rmsnorm(x, w, eps):
    ss = F(np.sum(x * x, dtype=F) / F(len(x)) + F(eps))
    return w * (F(1) / np.sqrt(ss) * x)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum(dtype=F)


class Runner:
    def __init__(self, M, ctx=1024):
        self.M = M
        self.kc = np.zeros((M.L, ctx, M.kv), F)
        self.vc = np.zeros((M.L, ctx, M.kv), F)
        half = M.hs // 2
        self.freq = M.theta ** (-2.0 * np.arange(half) / M.hs)  # float64, like qwen.c

    def rope(self, v, pos):
        ang = pos * self.freq
        c, s = np.cos(ang).astype(F), np.sin(ang).astype(F)
        v = v.reshape(-1, self.M.hs)
        a, b = v[:, :self.M.hs // 2].copy(), v[:, self.M.hs // 2:].copy()
        v[:, :self.M.hs // 2] = a * c - b * s  # NeoX "rotate half": pairs (i, i + 32)
        v[:, self.M.hs // 2:] = b * c + a * s

    def forward(self, tok, pos):
        M = self.M
        x = (M.emb_q[tok].reshape(-1, 32) * M.emb_s[tok].astype(F)[:, None]).reshape(-1)
        rep = M.heads // M.kv_heads
        for l in range(M.L):
            xq = quantize(rmsnorm(x, M.attn_norm[l], M.eps))
            q = matvec(M.wq[l], xq) + M.bq[l]
            k = matvec(M.wk[l], xq) + M.bk[l]
            v = matvec(M.wv[l], xq) + M.bv[l]
            self.rope(q, pos)
            self.rope(k, pos)
            self.kc[l, pos], self.vc[l, pos] = k, v
            K = self.kc[l, :pos + 1].reshape(pos + 1, M.kv_heads, M.hs)
            V = self.vc[l, :pos + 1].reshape(pos + 1, M.kv_heads, M.hs)
            att = np.empty(M.dim, F)
            for h in range(M.heads):
                qh = q[h * M.hs:(h + 1) * M.hs]
                a = softmax((K[:, h // rep] @ qh) / F(np.sqrt(M.hs)))
                att[h * M.hs:(h + 1) * M.hs] = a @ V[:, h // rep]
            x = x + matvec(M.wo[l], quantize(att))
            xq = quantize(rmsnorm(x, M.ffn_norm[l], M.eps))
            g, u = matvec(M.w1[l], xq), matvec(M.w3[l], xq)
            with np.errstate(over="ignore"):
                hb = g * (F(1) / (F(1) + np.exp(-g))) * u
            x = x + matvec(M.w2[l], quantize(hb))
        return matvec(M.head, quantize(rmsnorm(x, M.out_norm, M.eps)))  # logits of the head_ids rows


def tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qwb")
    ap.add_argument("--ids", help="comma-separated prompt token ids")
    ap.add_argument("--prompt", help="prompt text (chat template unless --raw)")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("-n", type=int, default=40, help="max new tokens (greedy)")
    ap.add_argument("--logits", help="qwen.c --logits dump to compare against")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models/qwen/tokenizer-coder/tokenizer.json"))
    a = ap.parse_args()
    tok = tokenizer(a.tokenizer) if (a.prompt is not None or os.path.exists(a.tokenizer)) else None
    if a.ids:
        ids = [int(t) for t in a.ids.split(",")]
    else:
        ids = tok.encode(a.prompt if a.raw else TEMPLATE.format(a.prompt), add_special_tokens=False).ids
    print("prompt ids:", ",".join(map(str, ids)), flush=True)
    M = Model(a.qwb)
    R = Runner(M)

    if a.logits:
        c = np.fromfile(a.logits, dtype="<f4").reshape(-1, M.head_k)
        seq, worst, top5_same, first_diff = list(ids), 0.0, 0, None
        for pos in range(len(c)):
            r = R.forward(seq[pos], pos)
            diff = float(np.abs(r - c[pos]).max())
            worst = max(worst, diff)
            t5r, t5c = np.argsort(-r)[:5], np.argsort(-c[pos])[:5]
            top5_same += bool(np.array_equal(t5r, t5c))
            if pos >= len(ids) - 1 and t5r[0] != t5c[0] and first_diff is None:
                first_diff = pos
            print(f"pos {pos:4d} in {seq[pos]:6d}: max|ref-c| {diff:.3e}  ref top5 {M.head_ids[t5r].tolist()} "
                  f"{'==' if np.array_equal(t5r, t5c) else '!='} c top5 {M.head_ids[t5c].tolist()}")
            if pos + 1 >= len(seq):
                seq.append(int(M.head_ids[int(np.argmax(c[pos]))]))  # follow qwen.c's greedy choice
        gen = len(c) - len(ids) + 1
        print(f"{len(c)} positions: max abs diff {worst:.3e}, top-5 identical (ids and order) at {top5_same}/{len(c)}, "
              f"greedy tokens identical for {gen if first_diff is None else first_diff - len(ids) + 1}/{gen}")
        return

    out = []
    for pos in range(len(ids) + a.n - 1):
        logits = R.forward(ids[pos] if pos < len(ids) else out[-1], pos)
        if pos >= len(ids) - 1:
            out.append(int(M.head_ids[int(np.argmax(logits))]))
            if out[-1] in STOP:
                break
    print("generated ids:", ",".join(map(str, out)))
    if tok:
        print("text:", repr(tok.decode(out, skip_special_tokens=False)))


if __name__ == "__main__":
    main()
