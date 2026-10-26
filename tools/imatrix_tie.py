# imatrix collected on the official (untied) GGUF has the output-head stats under output.weight.*;
# the tied GGUF has no output.weight, so re-key them to token_embd.weight.* (same tensor, same role as head).
import sys, gguf
src, dst = sys.argv[1:3]
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(dst, arch="imatrix", endianess=r.endianess)
for f in r.fields.values():
    if f.name in ("general.architecture",) or f.name.startswith("GGUF."): continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
out = []
for t in r.tensors:
    name = "token_embd.weight" + t.name[len("output.weight"):] if t.name.startswith("output.weight.") else t.name
    out.append((name, t))
    w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for _, t in out: w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print("wrote", dst, sum(n.startswith("token_embd") for n, _ in out), "token_embd entries")
