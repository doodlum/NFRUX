"""Exports AMD FSR FG 4's network weights into one self-describing file for the reference model and the port.

    python export_weights.py CAPTURE_DIR WEIGHTS.bin OUT.npz

Every layer becomes  L{pass}_{name}_w  [O, I, KH, KW] float32 holding the exact fp16 / fp8 values AMD stores, and
L{pass}_{name}_b  [O] float32. A JSON manifest (OUT.json) lists, per pass: op, kernel, stride, activation, residual,
the weight dtype, the channel counts and the runtime size index. Sources: graph.json (decode_graph.py) and
spec_convs.json (solve_convs.py; layouts verified against AMD's output on the GPU). FP8 multi-weight passes
(FasterNet blocks) are decoded from the shader code: partial 3x3 conv on the first P channels, 1x1 expand + ReLU,
1x1 project + residual; pass 176 adds a 2x2 transposed up-conv and a skip add.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlfg_ref import read_weights, read_bias

def main(cap, wpath, out):
    graph = {g['pass']: g for g in json.load(open(os.path.join(cap, 'graph.json')))}
    spec = {int(k): v for k, v in json.load(open(os.path.join(cap, 'spec_convs.json'))).items()}
    wbuf = open(wpath, 'rb').read()
    arrays, manifest = {}, {}

    def put(name, wt, b, layout_io):
        if wt['kind'] == 'Tensor4':   # dense HWCN fp8 (pass 176's up-conv): element (kw, kh, c, n) at kw*s0 + kh*s1 + c*s2 + n*s3
            kw, kh, C, N = wt['logical']; s = wt['strides']
            raw = np.frombuffer(wbuf, np.uint8, count=wt['storage'][0] * s[0] * 2 + 1, offset=wt['offset'])
            w = torch.zeros(kh, kw, C, N)
            for y in range(kh):
                for x in range(kw):
                    blk = np.stack([raw[x * s[0] + y * s[1] + c * s[2]: x * s[0] + y * s[1] + c * s[2] + N * s[3]: s[3]] for c in range(C)])
                    w[y, x] = torch.from_numpy(blk.copy()).view(torch.float8_e4m3fn).float()
            # transposed conv: c = output channel, n = input channel -> stored [O=C, I=N, KH, KW]
            arrays[name + '_w'] = w.permute(2, 3, 0, 1).numpy().astype(np.float32)
            arrays[name + '_b'] = read_bias(wbuf, b).numpy().astype(np.float32)
            return {'w': name + '_w', 'b': name + '_b', 'shape': list(arrays[name + '_w'].shape), 'dtype': 'fp8', 'transposed': True}
        w = read_weights(wbuf, wt)                       # [KH, KW, A, B]
        w = w.permute(3, 2, 0, 1) if layout_io else w.permute(2, 3, 0, 1)   # -> [O, I, KH, KW]
        arrays[name + '_w'] = w.numpy().astype(np.float32)
        arrays[name + '_b'] = read_bias(wbuf, b).numpy().astype(np.float32)
        return {'w': name + '_w', 'b': name + '_b', 'shape': list(w.shape), 'dtype': 'fp8' if min(wt['blockStrides']) == 1 else 'fp16'}

    for p, g in sorted(graph.items()):
        ts = g['tensors']
        ws = [t for t in ts if t['kind'] == 'Blocked6' or (t['kind'] == 'Tensor4' and t['logical'][0] <= 3 and t['logical'][1] <= 3)]
        bs = [t for t in ts if t['kind'] == 'Tensor1f']
        if not ws: continue
        entry = {'pass': p, 'sizeIdx': g['tensorSizeIdx'], 'fp8': g['fp8']}
        if p in spec:
            s = spec[p]
            entry.update(op=s['op'], k=s['k'], stride=s['stride'], act=s['act'], residual=s['residual'], cin=s['cin'], cout=s['cout'],
                         exact=s['exact'])
            if p == 188: entry.update(act='softmax_max0')
            if p in (166, 186): entry.update(act='none', residual=False)   # FP8 multiply: undefined on NVIDIA, semantics from the shader
            entry['layer'] = put(f'L{p}', ws[0], bs[0], s['layout'] == 'IO')
        elif len(ws) in (3, 4):
            # FasterNet block: [pconv 3x3, expand 1x1, project 1x1] (+ [up 2x2] for pass 176). Weight tensor kinds: Blocked6 or,
            # for 176's up-conv, a Tensor4-described HWCN block (read as Blocked6 with the same field meaning).
            names = ['pconv', 'expand', 'project', 'up']
            entry.update(op='fasternet' + ('_up_skip' if len(ws) == 4 else ''), act='relu_after_expand', residual=True,
                         pconv_channels=ws[0]['logical'][2], channels=ws[1]['logical'][2])
            for i, (wt, bt) in enumerate(zip(ws, bs)):
                entry[names[i]] = put(f'L{p}_{names[i]}', wt, bt, True)
        manifest[p] = entry
    np.savez(out, **arrays)
    json.dump(manifest, open(os.path.splitext(out)[0] + '.json', 'w'), indent=1)
    n = sum(a.size for k, a in arrays.items())
    print(f'{len(manifest)} layers, {n} parameters, {sum(1 for e in manifest.values() if e.get("exact"))} verified on the GPU')

if __name__ == '__main__':
    main(*sys.argv[1:4])
