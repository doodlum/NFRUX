"""Reference implementation of AMD FSR Frame Generation 4's network, rebuilt from graph.json (decode_graph.py),
and a checker that recomputes every convolution pass from the scratch dump before it and compares with AMD's own
output in the dump after it (ffx_capture --dump-passes).

    python mlfg_ref.py CAPTURE_DIR DUMP_DIR WEIGHTS.bin [--frame 2] [--passes 2,4,8]

Tensor storage, as the shaders address it (byte offsets into the scratch / initializer buffers):
  Tensor4   element (x, y, c, n)  = offset + (x+padB.x)*s.x + (y+padB.y)*s.y + c*s.z + n*s.w
  Blocked5  element (x, y, c, n)  = offset + (x+padB.x)*s.x + (y+padB.y)*s.y + (c//B)*s.z + n*s.w + (c%B)*elem
  Blocked6  weights (kw, kh, i, o) with 16x16 blocks: offset + kw*s.x + kh*s.y + (i//16)*s.z + (o//16)*s.w
            + (i%16)*bs[0] + (o%16)*bs[1]   (NbCb: bs = (elem, 16*elem); CbNb: the roles of the two block axes swap)
  elem = 2 bytes (fp16) or 1 byte (fp8 e4m3); a Blocked5 / Blocked6 is fp8 when its element stride is 1.
"""
import json, os, re, sys, argparse
import numpy as np
import torch

F8 = torch.float8_e4m3fn

def sizes_table(cap):
    for l in open(os.path.join(cap, 'log.txt')):
        if ' mlfi_' in l and 's0b257' in l:
            m = re.search(r's0b257\[0\]=t6:buf0\+(\d+),', l)
            h = dict(re.findall(r'\| buf\d+\+(\d+)=([0-9a-f]+)', l)).get(m.group(1))
            if h: return np.frombuffer(bytes.fromhex(h), np.uint32).reshape(-1, 4)[:, :2].copy()

def elem(t):
    if t['kind'] == 'Blocked5': return t['blockStride']
    if t['kind'] == 'Blocked6': return min(t['blockStrides'])
    if t['kind'] == 'Tensor4': return t['strides'][2]
    return 4

def as_float(raw, e):
    if e == 2: return torch.from_numpy(np.frombuffer(raw, np.float16).copy()).float()
    return torch.from_numpy(np.frombuffer(raw, np.uint8).copy()).view(F8).float()

def read_act(buf, t, W, H):
    """Activation tensor -> float32 [C, H, W] for the live size W x H."""
    e = elem(t); s = t['strides']; pb = t['padBegin']; C = t['logical'][2]
    out = torch.zeros(C, H, W)
    if t['kind'] == 'Tensor4':
        for y in range(H):
            base = t['offset'] + (y + pb[1]) * s[1] + pb[0] * s[0]
            row = np.frombuffer(buf, np.uint8, count=(W - 1) * s[0] + C * e, offset=base)
            px = np.lib.stride_tricks.as_strided(row, (W, C * e), (s[0], 1))
            out[:, y, :] = as_float(px.copy().tobytes(), e).view(W, C).t()
        return out
    B = t['blocks'][2]
    for cb in range((C + B - 1) // B):
        n = min(B, C - cb * B)
        for y in range(H):
            base = t['offset'] + (y + pb[1]) * s[1] + pb[0] * s[0] + cb * s[2]
            row = np.frombuffer(buf, np.uint8, count=(W - 1) * s[0] + B * e, offset=base)
            px = np.lib.stride_tricks.as_strided(row, (W, B * e), (s[0], 1))
            out[cb * B:cb * B + n, y, :] = as_float(px.copy().tobytes(), e).view(W, B)[:, :n].t()
    return out

def read_weights(wbuf, t):
    """Blocked6 -> float32 [KH, KW, A, B] with A, B the two logical channel axes (logical[2], logical[3])."""
    kw, kh, A, Bn = t['logical']; s = t['strides']; bs = t['blockStrides']; e = min(bs)
    w = torch.zeros(kh, kw, A, Bn)
    for y in range(kh):
        for x in range(kw):
            for ab in range((A + 15) // 16):
                for bb in range((Bn + 15) // 16):
                    off = t['offset'] + x * s[0] + y * s[1] + ab * s[2] + bb * s[3]
                    raw = np.frombuffer(wbuf, np.uint8, count=16 * 16 * e, offset=off)
                    blk = as_float(raw.tobytes(), e)
                    # element (a, b) at a*bs[0] + b*bs[1]
                    ia = torch.arange(16).view(16, 1) * (bs[0] // e) + torch.arange(16).view(1, 16) * (bs[1] // e)
                    blk = blk[ia]
                    na, nb = min(16, A - ab * 16), min(16, Bn - bb * 16)
                    w[y, x, ab * 16:ab * 16 + na, bb * 16:bb * 16 + nb] = blk[:na, :nb]
    return w

def read_bias(wbuf, t):
    return torch.from_numpy(np.frombuffer(wbuf, np.float32, count=t['logical'], offset=t['offset']).copy())

def q(x, e):
    """Round to the storage type the pass writes."""
    return x.to(torch.float16).float() if e == 2 else x.to(F8).float()

def conv(x, w, b, stride, transposed=False, oi=True):
    """x [C,H,W]; w [KH,KW,I,O] if oi else [KH,KW,O,I]."""
    wt = w.permute(3, 2, 0, 1) if oi else w.permute(2, 3, 0, 1)  # -> [O, I, KH, KW]
    if transposed:
        return torch.nn.functional.conv_transpose2d(x[None], wt.permute(1, 0, 2, 3), b, stride=2)[0]
    pad = (w.shape[0] - 1) // 2 if stride == 1 else 0
    return torch.nn.functional.conv2d(x[None], wt, b, stride=stride, padding=pad)[0]

def leaky(x): return torch.where(x > 0, x, 0.2 * x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture'); ap.add_argument('dumps'); ap.add_argument('weights')
    ap.add_argument('--frame', type=int, default=2); ap.add_argument('--passes', default='')
    a = ap.parse_args()
    graph = {g['pass']: g for g in json.load(open(os.path.join(a.capture, 'graph.json')))}
    sizes = sizes_table(a.capture)
    wbuf = open(a.weights, 'rb').read()
    dump = lambda n: open(os.path.join(a.dumps, f'{a.frame}_{n:03d}.bin'), 'rb').read()
    have = sorted(int(f.split('_')[1][:3]) for f in os.listdir(a.dumps) if f.startswith(f'{a.frame}_'))
    sel = [int(x) for x in a.passes.split(',')] if a.passes else [p for p, g in graph.items() if g['weights'] and g['tensors']]
    for p in sel:
        g = graph[p]; ts = g['tensors']
        prev = max(h for h in have if h < p)
        before, after = dump(prev), dump(p)
        acts = [t for t in ts if t['kind'] in ('Tensor4', 'Blocked5')]
        out_t = acts[-1]
        ow, oh = sizes[g['tensorSizeIdx'][0]]
        wts = [t for t in ts if t['kind'] == 'Blocked6']
        bias = [t for t in ts if t['kind'] == 'Tensor1f']
        stride2 = g['vec3'] and g['vec3'][0] == [2, 2, 1]
        k = wts[0]['logical'][0]
        transposed = k == 2 and not stride2
        iw, ih = (ow * 2, oh * 2) if stride2 else ((ow // 2, oh // 2) if transposed else (ow, oh))
        xin = dict(acts[0]); xin['logical'] = list(acts[0]['logical']); xin['logical'][2] = (acts[0]['logical'][2] + 15) // 16 * 16
        x = read_act(before, xin, iw, ih)  # the shaders load whole 16-channel windows, so padded channels count
        ref = read_act(after, out_t, ow, oh)
        best = None
        if len(wts) == 1:
            w = read_weights(wbuf, wts[0]); b = read_bias(wbuf, bias[0])
            cin, cout = x.shape[0], out_t['logical'][2]
            for oi in (True, False):
                ww = w if oi else w
                I = ww.shape[2] if oi else ww.shape[3]
                O = ww.shape[3] if oi else ww.shape[2]
                if O < cout: continue
                n = min(I, cin)
                wt = ww[:, :, :n, :cout] if oi else ww[:, :, :cout, :n]
                y = conv(x[:n], wt, b[:cout], 2 if stride2 else 1, transposed, oi)
                if g.get('residual'): y = y + x[:cout]
                if g['leaky']: y = leaky(y)
                if g['exp']: y = torch.exp(y)
                y = q(y, elem(out_t))
                err = (y - ref).abs().max().item()
                if best is None or err < best[0]: best = (err, oi)
        print(f"pass {p:3d} k{k} {'s2' if stride2 else ('up' if transposed else 's1')} {x.shape[0]}->{out_t['logical'][2]} "
              f"{iw}x{ih}->{ow}x{oh} fp8={g['fp8']} leaky={g['leaky']} exp={g['exp']} nw={len(wts)}: "
              f"max err {best[0] if best else 'n/a'} (layout {'IO' if best and best[1] else 'OI'}) ref absmax {ref.abs().max().item():.3f}")

if __name__ == '__main__':
    main()
