"""Summarises what one disassembled SPIR-V pass does: cooperative-matrix types (use, rows x cols, component type),
GLSL.std.450 / other extended instructions, conversions, barriers, integer constants, and the order of the main
arithmetic opcodes. Helps classify passes (convolution, element-wise, resize, concat) before checking them numerically.

    python pass_summary.py PASS.spvasm [PASS.spvasm ...]
"""
import re, sys, collections

USE = {'0': 'A', '1': 'B', '2': 'Acc'}

def summary(path):
    t = open(path, encoding='utf-8', errors='replace').read()
    ids = {}
    for m in re.finditer(r'(%\w+) = OpConstant %(\w+) (-?[\d.e+-]+)', t):
        ids[m.group(1)] = (m.group(2), m.group(3))
    types = {}
    for m in re.finditer(r'(%\w+) = OpType(Float|Int) (\d+)(?: (\d+))?(?: (\w+))?', t):
        types[m.group(1)] = f'{m.group(2)[0].lower()}{m.group(3)}' + ('' if m.group(2) == 'Float' else ('s' if m.group(4) == '1' else 'u')) + (f'/{m.group(5)}' if m.group(5) else '')
    cm = []
    for m in re.finditer(r'(%\w+) = OpTypeCooperativeMatrixKHR (%\w+) (%\w+) (%\w+) (%\w+) (%\w+)', t):
        comp, scope, rows, cols, use = m.groups()[1:]
        v = lambda x: ids.get(x, ('', x))[1]
        cm.append(f'{USE.get(v(use), v(use))}:{v(rows)}x{v(cols)}:{types.get(comp, comp)}')
    ext = collections.Counter(re.findall(r'OpExtInst %\w+ %\w+ (\w+)', t))
    ops = collections.Counter(re.findall(r'= Op(F\w+|I\w+|S\w+|U\w+|Convert\w+|Cooperative\w+|Bitcast|Select|Dot|Atomic\w+)\b', t))
    consts = sorted({int(v) for k, (ty, v) in ids.items() if re.fullmatch(r'-?\d+', v) and abs(int(v)) > 64})
    ep = re.search(r'OpEntryPoint \w+ %\w+ "(\w+)"', t).group(1)
    ls = re.search(r'LocalSize (\d+) (\d+) (\d+)', t)
    return ep, ls.groups() if ls else None, cm, ext, ops, consts, t.count('OpControlBarrier')

for p in sys.argv[1:]:
    ep, ls, cm, ext, ops, consts, bar = summary(p)
    print(f'== {ep} local {ls} barriers {bar}')
    print('  coopmat', sorted(set(cm)))
    print('  ext', dict(ext))
    print('  ops', dict(ops.most_common(25)))
    print('  big int constants', consts[:60], '...' if len(consts) > 60 else '')
