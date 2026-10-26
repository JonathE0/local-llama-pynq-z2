"""Export a Qwen2 GGUF (Q4_0 matrices, Q4_0 or Q8_0 tied embedding) to the board's .qwb weight file.

usage: python tools/gguf2qwb.py model.gguf qwen/head_tokens_65536.npy out.qwb

Layout (little-endian, every array padded to 64 bytes):
  header, 256 bytes: "QWB1", version 1, then int32 dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
      head_k, max_seq_len, group_size (32), emb_bits (4 or 8), then float32 rope_theta, rms_eps
  float32: attn_norm[L][dim], ffn_norm[L][dim], bq[L][dim], bk[L][kv_dim], bv[L][kv_dim], output_norm[dim]
  int32 head_ids[head_k]: token id of each output-head row (ascending)
  matrices, each a weight plane then a scale plane (float16 per group of 32 along a row), in this order:
      wq[L], wk[L], wv[L], wo[L], w1[L] (gate), w2[L] (down), w3[L] (up)   4-bit
      head (rows head_ids of the embedding), embedding (all vocab_size rows)  emb_bits
  A matrix W (d rows, n columns) is row-major. 4-bit planes hold two's complement int4, element 2i of a row in the
  low nibble of byte i and 2i+1 in the high nibble; 8-bit planes hold int8. Value = int * scale of its group.
Every matrix is checked to dequantize to exactly what gguf's own dequantizer gives.
"""
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import GGUFReader
from gguf.quants import dequantize

QK = 32
src, head_path, dst = sys.argv[1:4]
r = GGUFReader(src)
tensors = {t.name: t for t in r.tensors}


def meta(key):
    f = r.fields[key]
    return f.parts[f.data[0]][0].item()


def ints_scales(t):
    """int8 values [rows, n] and float16 scales [rows, n/32] of a Q4_0 or Q8_0 tensor"""
    n, rows = int(t.shape[0]), int(np.prod(t.shape[1:]))
    size = {T.Q4_0: 18, T.Q8_0: 34}.get(t.tensor_type)
    if size is None:
        raise SystemExit(f"{t.name}: {t.tensor_type.name} not supported, quantize with --pure")
    b = np.asarray(t.data, dtype=np.uint8).reshape(rows, n // QK, size)
    d = b[:, :, :2].copy().view(np.float16).reshape(rows, n // QK)
    if t.tensor_type == T.Q4_0:  # 16 bytes: element j in the low nibble of byte j, element j+16 in the high one
        qs = b[:, :, 2:]
        q = np.concatenate([(qs & 0x0F).astype(np.int8) - 8, (qs >> 4).astype(np.int8) - 8], axis=2)
    else:
        q = b[:, :, 2:].copy().view(np.int8)
    q = q.reshape(rows, n)
    deq = (q.reshape(rows, n // QK, QK) * d[:, :, None].astype(np.float32)).reshape(rows, n)
    assert np.array_equal(deq, dequantize(b.reshape(rows, -1), t.tensor_type).reshape(rows, n)), t.name
    return q, d


def pack4(q):
    u = (q & 0x0F).astype(np.uint8)
    p = u[:, 0::2] | (u[:, 1::2] << 4)
    back = np.empty_like(q)
    back[:, 0::2], back[:, 1::2] = p & 0x0F, p >> 4
    assert np.array_equal(np.where(back > 7, back - 16, back), q)  # round-trips
    return p


def put(f, a):
    f.write(np.ascontiguousarray(a).tobytes())
    f.write(b"\0" * (-f.tell() % 64))


def put_matrix(f, q, d, bits):
    put(f, pack4(q) if bits == 4 else q)
    put(f, d)


L = meta("qwen2.block_count")
emb = tensors["token_embd.weight"]
emb_bits = 4 if emb.tensor_type == T.Q4_0 else 8
head_ids = np.load(head_path).astype(np.int32)
assert np.all(np.diff(head_ids) > 0), "head ids must be ascending and unique"
cfg = [meta("qwen2.embedding_length"), meta("qwen2.feed_forward_length"), L, meta("qwen2.attention.head_count"),
       meta("qwen2.attention.head_count_kv"), int(emb.shape[1]), len(head_ids), meta("qwen2.context_length"),
       QK, emb_bits]


def f32(name):
    t = tensors[name]
    assert t.tensor_type == T.F32, f"{name}: expected F32, got {t.tensor_type.name}"
    return np.asarray(t.data, dtype=np.float32)


with open(dst, "wb") as f:
    f.write(struct.pack("<4sI10i2f", b"QWB1", 1, *cfg, meta("qwen2.rope.freq_base"),
                        meta("qwen2.attention.layer_norm_rms_epsilon")))
    f.write(b"\0" * (256 - f.tell()))
    for kind in ("attn_norm.weight", "ffn_norm.weight", "attn_q.bias", "attn_k.bias", "attn_v.bias"):
        for l in range(L):
            put(f, f32(f"blk.{l}.{kind}"))
    put(f, f32("output_norm.weight"))
    put(f, head_ids)
    for kind in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"):
        for l in range(L):
            t = tensors[f"blk.{l}.{kind}.weight"]
            if t.tensor_type != T.Q4_0:
                raise SystemExit(f"{t.name}: {t.tensor_type.name}, expected Q4_0 (quantize with --pure)")
            put_matrix(f, *ints_scales(t), 4)
    q, d = ints_scales(emb)
    put_matrix(f, q[head_ids], d[head_ids], emb_bits)
    put_matrix(f, q, d, emb_bits)
    size = f.tell()
print(f"wrote {dst}: {size / 2**20:.1f} MiB, dim={cfg[0]} layers={L} vocab={cfg[5]} head={cfg[6]} "
      f"embedding={emb_bits}-bit, all matrices verified against gguf's dequantizer")
