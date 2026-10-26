# Copy a GGUF, dropping output.weight (identical to token_embd.weight) so llama.cpp loads it as a tied model.
import sys, numpy as np, gguf
src, dst = sys.argv[1:3]
r = gguf.GGUFReader(src)
t = {x.name: x for x in r.tensors}
assert np.array_equal(np.asarray(t["output.weight"].data), np.asarray(t["token_embd.weight"].data)), "not a duplicate"
w = gguf.GGUFWriter(dst, arch=r.fields["general.architecture"].contents(), endianess=r.endianess)
for f in r.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."): continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
keep = [x for x in r.tensors if x.name != "output.weight"]
for x in keep: w.add_tensor_info(x.name, x.data.shape, x.data.dtype, x.data.nbytes, x.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for x in keep: w.write_tensor_data(x.data, tensor_endianess=r.endianess)
w.close()
print("wrote", dst, len(keep), "tensors")
