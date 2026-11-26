"""For every pass with one weight tensor, finds the variant that reproduces AMD's output exactly and writes spec.json.

    python solve_convs.py CAPTURE_DIR DUMP_DIR WEIGHTS.bin [--frame 2]

Variants tried: weight layout (IO / OI), activation (none, relu, leaky 0.2, exp, clamp to +-15 for fp8 stores),
residual add of the input before the activation, and the NVIDIA-only zeroing of channels when a pass's channel
count is a multiple of 8 but not 16 (the shaders zero accumulator lanes 16..31 assuming AMD's register layout; on
NVIDIA those lanes hold rows 4-7 and 12-15 of the block). The spec records the AMD semantics: the last
(16 - C % 16) channels are zero.
"""
import json, os, sys, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlfg_ref import *

ACTS = {'none': lambda v: v, 'relu': torch.relu, 'leaky': leaky, 'exp': torch.exp}

def nv_zero(y, C):
    """NVIDIA's result of AMD's lane>=16 zeroing on the last block of a C-channel tensor (C % 16 == 8)."""
    if C % 16 == 0 or C % 8 != 0: return y
    base = C // 16 * 16
    y = y.clone()
    for r in list(range(4, 8)) + list(range(12, 16)):
        if base + r < y.shape[0]: y[base + r] = 0
    return y

def solve(g, sizes, wbuf, before, after):
    ts = g['tensors']
    acts = [t for t in ts if t['kind'] in ('Tensor4', 'Blocked5')]
    wts = [t for t in ts if t['kind'] == 'Blocked6']; bias = [t for t in ts if t['kind'] == 'Tensor1f']
    out_t = acts[-1]
    ow, oh = sizes[g['tensorSizeIdx'][0]]
    stride2 = bool(g['vec3']) and g['vec3'][0] == [2, 2, 1]
    k = wts[0]['logical'][0]
    transposed = k == 2 and not stride2
    iw, ih = (ow * 2, oh * 2) if stride2 else ((ow // 2, oh // 2) if transposed else (ow, oh))
    xin = dict(acts[0]); xin['logical'] = list(acts[0]['logical']); xin['logical'][2] = (acts[0]['logical'][2] + 15) // 16 * 16
    x = read_act(before, xin, iw, ih)
    C = out_t['logical'][2]
    ro = dict(out_t); ro['logical'] = list(out_t['logical']); ro['logical'][2] = (C + 15) // 16 * 16 if out_t['kind'] == 'Blocked5' or out_t['storage'][2] >= (C + 15) // 16 * 16 else C
    ref = read_act(after, ro, ow, oh)
    w = read_weights(wbuf, wts[0]); b = read_bias(wbuf, bias[0])
    e_out = elem(out_t)
    best = None
    for oi, act, res in itertools.product((True, False), ACTS, (False, True)):
        ww = w if oi else w.permute(0, 1, 3, 2)
        I, O = ww.shape[2], ww.shape[3]
        n, m = min(I, x.shape[0]), min(O, ref.shape[0])
        bb = torch.zeros(m); bb[:min(m, b.shape[0])] = b[:m]
        y = conv(x[:n], ww[:, :, :n, :m], bb, 2 if stride2 else 1, transposed, True)
        if res:
            if x.shape[1:] != y.shape[1:] or x.shape[0] < m: continue
            y = y + x[:m]
        y = ACTS[act](y)
        y = nv_zero(y, C)
        y = q(y, e_out)
        err = (y - ref[:m]).abs().max().item()
        if best is None or err < best['err']:
            best = dict(err=err, layout='IO' if oi else 'OI', act=act, residual=res)
    tol = 1e-2 * max(1.0, ref.abs().max().item()) if e_out == 2 else 0.13 * max(1.0, ref.abs().max().item())
    best.update(op='conv_transpose' if transposed else 'conv', k=k, stride=2 if stride2 else 1, cin=x.shape[0], cout=C,
                exact=best['err'] <= tol, ref_absmax=ref.abs().max().item(), in_size=[int(iw), int(ih)], out_size=[int(ow), int(oh)])
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('capture'); ap.add_argument('dumps'); ap.add_argument('weights')
    ap.add_argument('--frame', type=int, default=2); a = ap.parse_args()
    graph = {g['pass']: g for g in json.load(open(os.path.join(a.capture, 'graph.json')))}
    sizes = sizes_table(a.capture); wbuf = open(a.weights, 'rb').read()
    have = sorted(int(f.split('_')[1][:3]) for f in os.listdir(a.dumps) if f.startswith(f'{a.frame}_'))
    dump = lambda n: open(os.path.join(a.dumps, f'{a.frame}_{n:03d}.bin'), 'rb').read()
    spec = {}
    for p, g in sorted(graph.items()):
        if sum(t['kind'] == 'Blocked6' for t in g['tensors']) != 1: continue
        s = solve(g, sizes, wbuf, dump(max(h for h in have if h < p)), dump(p))
        spec[p] = s
        print(f"pass {p:3d} {s['op']:14s} k{s['k']} s{s['stride']} {s['cin']:3d}->{s['cout']:3d} {s['in_size']}->{s['out_size']} "
              f"{s['layout']} act={s['act']:5s} res={int(s['residual'])} err={s['err']:.4f} {'OK' if s['exact'] else 'MISMATCH'} (absmax {s['ref_absmax']:.2f})")
    json.dump(spec, open(os.path.join(a.capture, 'spec_convs.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
