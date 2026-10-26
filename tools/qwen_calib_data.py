"""Build imatrix calibration text (wikitext-2 train slices + Python stdlib code) and a held-out code eval file.

usage: py -3 -I tools/qwen_calib_data.py   (needs models/data/wikitext-2-raw/; writes models/data/code/)
"""
import pathlib, random, sysconfig
STD = pathlib.Path(sysconfig.get_paths()["stdlib"])
OUT = pathlib.Path(__file__).resolve().parents[1] / "models" / "data"
SKIP = {"test", "tests", "idlelib", "lib2to3", "site-packages", "__pycache__", "ensurepip", "turtledemo", "tkinter"}
HEAD = 20_000  # per-file head cap -> diversity across modules
files = []
for p in STD.rglob("*.py"):
    rel = p.relative_to(STD)
    if any(part in SKIP for part in rel.parts) or p.name.startswith("test_") or p.stat().st_size < 3000:
        continue
    files.append(rel)
files.sort()
# module = top-level name; split by MODULE so eval/calib share no module/package
mods = sorted({f.parts[0].removesuffix(".py") for f in files})
random.Random(0).shuffle(mods)
def build(mod_list, target):
    chosen = set(mod_list); names = []; n = 0
    for f in files:
        if f.parts[0].removesuffix(".py") in chosen:
            names.append(f)
    random.Random(1).shuffle(names)
    out = []
    for f in names:
        t = (STD / f).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")[:HEAD]
        t = t[: t.rfind("\n") + 1]  # cut at line boundary
        out.append(t); n += len(t.encode()); used.append(str(f))
        if n >= target: break
    return "".join(out), n
used = []
# eval: ~300 KB from first modules in shuffled order; calib: from the *rest*
ev_mods, acc, k = [], 0, 0
sizes = {m: sum(min(HEAD, (STD / f).stat().st_size) for f in files if f.parts[0].removesuffix(".py") == m) for m in mods}
for m in mods:
    ev_mods.append(m); acc += sizes[m]
    if acc >= 330_000: break
cal_mods = [m for m in mods if m not in ev_mods]
ev_txt, ev_n = build(ev_mods, 300_000); ev_files = list(used); used.clear()
ca_txt, ca_n = build(cal_mods, 700_000); ca_files = list(used)
assert not ({f.split("\\")[0].removesuffix(".py") for f in ev_files} & {f.split("\\")[0].removesuffix(".py") for f in ca_files})
d = OUT / "code"; d.mkdir(exist_ok=True)
(d / "eval_code.py.txt").write_bytes(ev_txt.encode()); (d / "calib_code.py.txt").write_bytes(ca_txt.encode())
# wiki calibration: 35 evenly spaced 14 KB slices of wiki.train (cut at line boundaries)
w = (OUT / "wikitext-2-raw" / "wiki.train.raw").read_text(encoding="utf-8")
S, L, parts = 35, 14_000, []
for i in range(S):
    a = w.find("\n", i * (len(w) - L) // (S - 1)) + 1
    parts.append(w[a : w.rfind("\n", a, a + L) + 1])
(d / "calib_wiki.txt").write_bytes("".join(parts).encode())
code = ca_txt.encode()
(d / "calib_mix.txt").write_bytes("".join(parts).encode() + code[: code.rfind(b"\n", 0, 415_000) + 1])  # ~415 KB of code
(d / "manifest.txt").write_text("EVAL modules: %s\nEVAL files (%d):\n%s\n\nCALIB modules: %s\nCALIB files (%d):\n%s\n" % (
    " ".join(sorted(ev_mods)), len(ev_files), "\n".join(ev_files), " ".join(sorted(cal_mods)), len(ca_files), "\n".join(ca_files)))
print("eval", ev_n, len(ev_files), "files;", "calib code", ca_n, len(ca_files), "files; wiki calib", len("".join(parts).encode()))
print("eval modules:", sorted(ev_mods)[:40])
