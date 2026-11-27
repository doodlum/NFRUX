"""Makes NFRUX's embedded-model headers (core/model/<name>_int8*.h) for a retrained NFRU-MFG VGF of the same shape,
using the committed NFRU headers as the template (same graph interface; only the SPIR-V graph and the constant bytes change).
   python vgf_to_nfrux_header.py VGF NAME OUT_DIR      e.g. ... output_tag0.tosa.vgf nfru_mfg_v15_int8 core/model
Check: running it on the committed model's own VGF reproduces its headers byte for byte (verified for v1, 2026-11-27)."""
import re, subprocess, sys, tempfile, os
from pathlib import Path

vgf, name, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
TPL = Path(__file__).resolve().parents[1] / 'core' / 'model'
OLD = next(p.stem for p in TPL.glob('nfru_mfg_v*_int8.h'))
DUMP = r'C:\dev\modelgym\venv\Scripts\vgf_dump.exe'
tmp = Path(tempfile.mkdtemp())


def dump(flag, idx):
    p = tmp / f'{flag}_{idx}.bin'
    subprocess.run([DUMP, '-i', vgf, '-o', str(p), f'--dump-{flag}', str(idx)], check=True, capture_output=True)
    return p.read_bytes()


def arr(b):
    return ',\n'.join(','.join(f'0x{x:02x}' for x in b[i:i + 16]) for i in range(0, len(b), 16))


def put(text, var, data):
    pat = re.compile(r'(static const unsigned char ' + re.escape(var) + r'\[\] = \{\n)(.*?)(\n\};)', re.S)
    assert pat.search(text), var
    return pat.sub(lambda m: m.group(1) + arr(data) + m.group(3), text, count=1)


out.mkdir(parents=True, exist_ok=True)
g = (TPL / f'{OLD}_graph_0.h').read_text()
spv = dump('spirv', 0)
g = put(g, f'g_{OLD}_graph_0_data', spv)
g = re.sub(rf'(g_{OLD}_graph_0_data_size = )\d+', lambda m: m.group(1) + str(len(spv)), g)
c = (TPL / f'{OLD}_constants.h').read_text()
n = int(re.search(rf'g_{OLD}_constants_data_size\[\] = \{{(.*?)\}}', c).group(1).count(','))
sizes = []
for i in range(n):
    d = dump('constant', i); sizes.append(len(d)); c = put(c, f'g_{OLD}_constants_data_{i}', d)
old_sizes = [int(x) for x in re.search(rf'g_{OLD}_constants_data_size\[\] = \{{(.*?)\}}', c).group(1).split(',') if x.strip()]
assert sizes == old_sizes, ('constant sizes differ: not the same network shape', sizes, old_sizes)
h = (TPL / f'{OLD}.h').read_text()
for fn, t in ((f'{name}.h', h), (f'{name}_graph_0.h', g), (f'{name}_constants.h', c)):
    (out / fn).write_text(t.replace(OLD, name), newline='\n')
meta = Path(vgf).parent / 'nfru_mfg_int8_metadata.json'
if meta.exists():
    (out / f'{name}_metadata.json').write_bytes(meta.read_bytes())
print('wrote', out, 'graph', len(spv), 'bytes,', n, 'constants')
