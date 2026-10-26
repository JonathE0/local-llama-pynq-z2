"""Convert a legacy llama2.c float checkpoint (.bin, version 0) to the Q8_0 int8 format runq.c reads (version 2).
Same output as `export.py --version 2`, without needing PyTorch.

usage: python tools/bin2q80.py models/stories15M.bin models/stories15M_q80.bin
"""
import struct
import sys

import numpy as np


def q80(w, gs):
    w = w.reshape(-1, gs)
    s = np.abs(w).max(axis=1) / 127.0
    q = np.round(w / np.where(s == 0, 1, s)[:, None]).astype(np.int8)
    return q, s.astype(np.float32), np.abs(q * s[:, None] - w).max()


src, dst = sys.argv[1], sys.argv[2]
dim, hidden, n_layers, n_heads, n_kv, vocab, seq_len = np.fromfile(src, np.int32, 7)
data = np.fromfile(src, np.float32, offset=28)
shared, vocab = vocab > 0, abs(vocab)  # legacy format: negative vocab = separate classifier
head = dim // n_heads
pos = 0


def take(*shape):
    global pos
    n = int(np.prod(shape))
    pos += n
    return data[pos - n:pos].reshape(shape)


emb = take(vocab, dim)
rms_att = take(n_layers, dim)
wq, wk, wv, wo = take(n_layers, dim, dim), take(n_layers, n_kv * head, dim), take(n_layers, n_kv * head, dim), take(n_layers, dim, dim)
rms_ffn = take(n_layers, dim)
w1, w2, w3 = take(n_layers, hidden, dim), take(n_layers, dim, hidden), take(n_layers, hidden, dim)
rms_final = take(dim)
take(2, seq_len, head // 2)  # freq_cis real/imag, runq computes RoPE itself
wcls = [] if shared else [take(vocab, dim)]
assert pos == data.size, "unexpected checkpoint layout"

gs = 64
while dim % gs:
    gs //= 2

with open(dst, "wb") as f:
    f.write(struct.pack("<Ii7iBi", 0x616B3432, 2, dim, hidden, n_layers, n_heads, n_kv, vocab, seq_len, int(shared), gs))
    f.write(b"\0" * (256 - f.tell()))
    for t in (rms_att, rms_ffn, rms_final):
        f.write(t.tobytes())
    maxerr = 0.0
    for t in [emb, *wq, *wk, *wv, *wo, *w1, *w2, *w3, *wcls]:
        q, s, err = q80(t, gs)
        f.write(q.tobytes())
        f.write(s.tobytes())
        maxerr = max(maxerr, float(err))
print(f"wrote {dst}: dim={dim} layers={n_layers} vocab={vocab} group={gs} max quant error={maxerr:.5f}")
