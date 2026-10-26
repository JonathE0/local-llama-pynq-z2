"""Walk a .qwb file exactly as the board loader must (layout in tools/gguf2qwb.py) and compare tensors to the GGUF.

usage: python tools/qwb_check.py model.gguf model.qwb
"""
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

gguf_path, path = sys.argv[1:3]
m = np.memmap(path, dtype=np.uint8, mode="r")
assert bytes(m[:4]) == b"QWB1" and m[4:8].view("<u4")[0] == 1
dim, hidden, L, heads, kv_heads, vocab, head_k, ctx, gs, emb_bits = m[8:48].view("<i4")
kv = dim // heads * kv_heads
off = 256


def take(nbytes):
    global off
    a = m[off:off + nbytes]
    off += nbytes + (-(off + nbytes) % 64)
    return a


def matrix(rows, n, bits):
    p = take(rows * n * bits // 8)
    s = take(rows * n // gs * 2).view("<f2").reshape(rows, n // gs).astype(np.float32)
    if bits == 4:
        p = p.reshape(rows, n // 2)
        q = np.empty((rows, n), np.int8)
        q[:, 0::2], q[:, 1::2] = p & 0x0F, p >> 4
        q = np.where(q > 7, q - 16, q)
    else:
        q = p.view(np.int8).reshape(rows, n)
    return (q.reshape(rows, n // gs, gs) * s[:, :, None]).reshape(rows, n)


g = {t.name: t for t in GGUFReader(gguf_path).tensors}


def ref(name):
    t = g[name]
    return dequantize(np.asarray(t.data), t.tensor_type).reshape(int(np.prod(t.shape[1:])), int(t.shape[0]))


small = {k: [take(n * 4).view("<f4") for _ in range(L)] for k, n in
         (("attn_norm", dim), ("ffn_norm", dim), ("attn_q.bias", dim), ("attn_k.bias", kv), ("attn_v.bias", kv))}
assert np.array_equal(small["attn_k.bias"][L - 1], g[f"blk.{L - 1}.attn_k.bias"].data)
take(dim * 4)  # output_norm
head_ids = take(head_k * 4).view("<i4")
shapes = {"attn_q": (dim, dim), "attn_k": (kv, dim), "attn_v": (kv, dim), "attn_output": (dim, dim),
          "ffn_gate": (hidden, dim), "ffn_down": (dim, hidden), "ffn_up": (hidden, dim)}
for kind, (rows, n) in shapes.items():
    for l in range(L):
        w = matrix(rows, n, 4)
        if l in (0, L - 1):
            assert np.array_equal(w, ref(f"blk.{l}.{kind}.weight")), (kind, l)
emb = ref("token_embd.weight")
assert np.array_equal(matrix(head_k, dim, emb_bits), emb[head_ids])
assert np.array_equal(matrix(vocab, dim, emb_bits), emb)
assert off == len(m), (off, len(m))
print(f"OK: {path} matches {gguf_path} (dim={dim} hidden={hidden} layers={L} heads={heads}/{kv_heads} "
      f"vocab={vocab} head={head_k} ctx={ctx} emb={emb_bits}-bit)")
