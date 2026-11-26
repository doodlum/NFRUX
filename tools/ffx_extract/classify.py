"""Groups the ML passes by template: the tensor struct types a pass declares, its matrix types, the float literals and
math functions it uses, and whether it reads weights. One line per pass plus a count per template.

    python classify.py GLSL_DIR [OUT_TXT]
"""
import os, re, sys, glob, collections

def sig(t):
    structs = tuple(sorted(set(re.findall(r'^struct (\w*Tensor\w*)', t, re.M))))
    cm = tuple(sorted(set(re.findall(r'coopmat<(\w+), gl_ScopeSubgroup, (\d+)u, (\d+)u, gl_MatrixUse(\w+)>', t))))
    floats = tuple(sorted(set(re.findall(r'\b(\d+\.\d+(?:e[-+]?\d+)?)\b', t)) - {'0.0', '1.0'}))[:8]
    funcs = tuple(sorted(set(re.findall(r'\b(exp|exp2|log|log2|tanh|sigmoid|max|min|clamp|mix|sqrt|inversesqrt|pow|floor|fract|round|abs|textureLod|texelFetch|imageStore|imageLoad|packHalf2x16|unpackHalf2x16)\s*\(', t))))
    weights = 'InitializerBuffer' in t and 'coopMatLoad' in t or 'InitializerBuffer._m0[' in t
    return structs, cm, floats, funcs, weights

def main(d, out=None):
    rows = []
    for p in glob.glob(os.path.join(d, 'mlfi_*_pass*.glsl')):
        name = os.path.basename(p)[:-5]
        rows.append((int(re.search(r'pass(\d+)$', name).group(1)), name, sig(open(p, encoding='utf-8', errors='replace').read())))
    rows.sort()
    groups = collections.defaultdict(list)
    for n, name, s in rows: groups[s].append(n)
    f = open(out, 'w') if out else sys.stdout
    for i, (s, ns) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        print(f'T{i:02d} x{len(ns):3d} passes {ns[:12]}{"..." if len(ns) > 12 else ""}', file=f)
        print(f'     structs {s[0]}', file=f)
        print(f'     coopmat {s[1]} weights {s[4]}', file=f)
        print(f'     floats {s[2]} funcs {s[3]}', file=f)

if __name__ == '__main__':
    main(*sys.argv[1:3])
