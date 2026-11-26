"""Decodes the tensor descriptors every ML pass hard-codes and writes graph.json.

    python decode_graph.py CAPTURE_DIR

Reads CAPTURE_DIR/glsl/mlfi_*_passN.glsl (to_glsl.ps1) and CAPTURE_DIR/log.txt (ffx_capture). In each pass the
generator emits the tensor arguments as a run of constants at the top of main(), field by field:
  Tensor4*  (activations, NHWC / HWCN)   logical, storage, byteStrides, padBegin, padEnd (uvec4) + byteOffset (uint)
  Blocked5* (activations, N Cb H W C)    the five uvec4 above + blockSizes (uvec4) + blockStride (uint) + byteOffset
  Blocked6* (weights, Nb Cb H W N C ...)  the five uvec4 above + blockSizes (uvec4) + blockStrides (uvec2) + byteOffset
  Tensor1f  (bias)                        logical, storage, byteStride, padBegin, padEnd, byteOffset (uint each)
The uvec3 constants that follow (stride / offset / tile) and the float literals and functions are kept too.
"""
import json, os, re, sys, glob

DECL = re.compile(r'^\s+(uvec4|uvec2|uint) _\d+ = (?:uvec4|uvec2)?\(?([\du, ]+)\)?;$')
VEC3 = re.compile(r'^\s+[ui]vec3 _\d+ = [ui]vec3\(([-\du, ]+)\);$')

def tokens(lines):
    out, started = [], False
    for l in lines:
        m = DECL.match(l)
        if m:
            vals = [int(v) for v in re.findall(r'\d+', m.group(2))]
            kind = m.group(1)
            if kind == 'uvec4' and len(vals) == 1: vals *= 4
            if kind == 'uvec2' and len(vals) == 1: vals *= 2
            out.append((kind, vals)); started = True
        elif started:
            break
    return out

def parse(tok):
    ts, i = [], 0
    k = lambda j: tok[j][0] if j < len(tok) else None
    while i < len(tok):
        if [k(i + j) for j in range(6)] == ['uint'] * 6:
            ts.append({'kind': 'Tensor1f', 'logical': tok[i][1][0], 'storage': tok[i + 1][1][0], 'stride': tok[i + 2][1][0],
                       'padBegin': tok[i + 3][1][0], 'padEnd': tok[i + 4][1][0], 'offset': tok[i + 5][1][0]}); i += 6; continue
        if [k(i + j) for j in range(5)] == ['uvec4'] * 5:
            base = {'logical': tok[i][1], 'storage': tok[i + 1][1], 'strides': tok[i + 2][1], 'padBegin': tok[i + 3][1], 'padEnd': tok[i + 4][1]}
            if k(i + 5) == 'uint':
                ts.append(dict(base, kind='Tensor4', offset=tok[i + 5][1][0])); i += 6; continue
            if k(i + 5) == 'uvec4' and k(i + 6) == 'uvec2':
                ts.append(dict(base, kind='Blocked6', blocks=tok[i + 5][1], blockStrides=tok[i + 6][1], offset=tok[i + 7][1][0])); i += 8; continue
            if k(i + 5) == 'uvec4' and k(i + 6) == 'uint':
                ts.append(dict(base, kind='Blocked5', blocks=tok[i + 5][1], blockStride=tok[i + 6][1][0], offset=tok[i + 7][1][0])); i += 8; continue
        ts.append({'kind': 'unparsed', 'token': tok[i]}); i += 1
    return ts

def main(cap):
    dispatch = {}
    for l in open(os.path.join(cap, 'log.txt')):
        m = re.match(r'\d+ s\d+ D (\d+),(\d+),(\d+) pipe\d+ (mlfi_\w+)', l)
        if m: dispatch[m.group(4)] = [int(m.group(i)) for i in (1, 2, 3)]
    graph = []
    for p in sorted(glob.glob(os.path.join(cap, 'glsl', 'mlfi_*_pass*.glsl')), key=lambda p: int(re.search(r'pass(\d+)', p).group(1))):
        name = os.path.basename(p)[:-5]
        lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
        try: start = next(i for i, l in enumerate(lines) if l.startswith('void main'))
        except StopIteration: continue
        body = lines[start:]
        tok = tokens(body)
        vec3 = [[int(v) for v in re.findall(r'-?\d+', m.group(1))] for m in (VEC3.match(l) for l in body[:600]) if m][:6]
        text = '\n'.join(body)
        structs = re.findall(r'^struct (\w*Tensor\w*)', '\n'.join(lines[:start]), re.M)
        graph.append({
            'pass': int(re.search(r'pass(\d+)', name).group(1)), 'name': name, 'dispatch': dispatch.get(name),
            'structs': structs, 'tensors': parse(tok), 'vec3': vec3,
            'floats': sorted(set(re.findall(r'\b\d+\.\d+(?:e[-+]?\d+)?\b', text)) - {'0.0', '1.0'}),
            'leaky': '0.20000000298023223876953125' in text, 'exp': bool(re.search(r'\bexp\(', text)),
            'residual': bool(re.search(r'm_matrix\.m_matrix \+ _\d+\[', text) or re.search(r'float _\d+ = _\d+\[_\d+\] \+ _\d+\[_\d+\];', text)),
            'weights': 'InitializerBuffer' in text, 'fp8': 'floate4m3_t' in text, 'lines': len(lines),
            'tensorSizeIdx': sorted({int(x) for x in re.findall(r'tensorSizes\[(\d+)\]', text)}),
        })
    json.dump(graph, open(os.path.join(cap, 'graph.json'), 'w'), indent=1)
    for g in graph:
        ts = ' | '.join(f"{t['kind']}{t.get('logical')}@{t.get('offset')}" for t in g['tensors'])
        print(f"{g['pass']:3d} {g['dispatch']} leaky={int(g['leaky'])} exp={int(g['exp'])} fp8={int(g['fp8'])} sz={g['tensorSizeIdx'][:4]} vec3={g['vec3'][:2]} :: {ts}")

if __name__ == '__main__':
    main(sys.argv[1])
