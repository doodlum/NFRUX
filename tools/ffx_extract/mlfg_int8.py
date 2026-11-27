"""Int8 reference of AMD FSR Frame Generation 4's ML interpolation network (FidelityFX SDK v2.4.0, MIT; docs/FFX_MLFG.md):
mlfg_net.AmdMlfgNet with every convolution in the integer semantics of an int8 export (ng_model_gym amd_mlfg export_amd.py
--qat: OUT_int8.npz + OUT_int8.json). This is the specification NFRUX's int8 tiers (core/mlfg, AmdPrecision=int8) follow.

    from mlfg_int8 import Int8MlfgNet
    net = Int8MlfgNet('amd_mlfg_qat_int8.npz', glue='amd')    # glue 'amd': the runtime's fp16 glue; 'train': QAT's fp32
    out, aux = net(prev_rgb, cur_rgb, cur_depth, cur_mv, hist, A, B, t)     # as AmdMlfgNet

Integer semantics (symmetric int8, value = int * scale, integers in [-127, 127]):
  quantise    Q(v, s) = clamp(rne(fp32(v * r)), -127, 127), r = fp32(1 / s) (the correctly rounded fp32 reciprocal; the
              export's x / s is computed as a multiplication by it: identical except where x / s lies within an fp32
              rounding of a half integer - an occasional one-step difference, inside QAT's own run-to-run noise)
  layer       acc[o] = sum wq[o, c, tap] * x_int (exact, int32); y = fp32(fp32(fp32(acc) * m[o]) + b[o]) with
              m[o] = fp32(ws[o] * s_in) (ws[o] with a fold); residual / skip: y = fp32(y + fp32(R * s_res))
  block entry the layers fed by glue passes (L2/L10 encoders, L20/L57/L86/L115 flow blocks, L154 stage 2) quantise their
              fp16 / fp32 input per channel: x_int[c] = Q(x[c], fold[c])
  activation  with a 'pre' quantiser: p = fp32(Q(y, s_pre) * s_pre), a = leaky 0.2 (fp32(p * 0.2) below 0) / relu, out
              Q(a, s_out); without: out Q(y, s_out)
  fp16 out    flow heads L52 / 81 / 110 / 147 and the logits L188: y rounded to fp16 (glue 'amd') / kept fp32 ('train')
  encoders    their int8 outputs dequantised fp32(O * s_o) into net_in (fp16 there with glue 'amd')
  FasterNet   expand input H = cat(Q(pconv y, s_h), Q(fp32(X[P:] * s_x), s_h)) (one per-tensor scale q_h)
Everything outside the convolutions is mlfg_net's (prepass, glue, splats, stage-2 input, softmax, composite) at
precision 'amd' (glue='amd', what NFRUX runs) or 'train' (glue='train', what QAT trains: then the result is QAT's
fake-quant forward up to fp32 summation order inside QAT's float convolutions).
SPDX-License-Identifier: MIT
"""
import io, json, os, types
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

import mlfg_net as N

QMAX = 127.0


# ------------------------------------------------------------------------------------------------ export -> layers
def runtime_keys():
    """npz layer keys in the order core/mlfg/mlfg_network.cpp runs the 105 layers (tools/mlfg_weights.layers)."""
    k = []
    for base in (2, 10):
        k += [f'L{base}', f'L{base + 2}', f'L{base + 4}', f'L{base + 6}']
    specs = [(20, 22, 24, [26, 29, 32, 35, 38, 41, 44, 47], 50, 52), (57, 59, 61, list(range(63, 78, 2)), 79, 81),
             (86, 88, 90, list(range(92, 107, 2)), 108, 110), (115, 117, 119, list(range(121, 143, 3)), 145, 147)]
    for d1, c1, d2, res, u1, u2 in specs:
        k += [f'L{d1}', f'L{c1}', f'L{d2}'] + [f'L{p}' for p in res] + [f'L{u1}', f'L{u2}']
    k += ['L154', 'L156']
    fn = lambda p: [f'L{p}_pconv', f'L{p}_expand', f'L{p}_project']
    for p in (158, 160, 162, 164): k += fn(p)
    k += ['L166']
    for p in (168, 170, 172, 174, 176): k += fn(p)
    k += ['L176_up']
    for p in (178, 180, 182, 184): k += fn(p)
    return k + ['L186', 'L188']


def runtime_modules(net):
    """The modules in runtime order (tools/mlfg_weights.layers without importing it): (module, transposed)."""
    out = []
    for enc in (net.encA, net.encB):
        out += [(enc.down, False), (enc.c1, False), (enc.c2, False), (enc.up, True)]
    for b in (net.block1, net.block2, net.block3, net.block4):
        out += [(b.d1, False), (b.c1, False), (b.d2, False)] + [(r, False) for r in b.res] + [(b.u1, True), (b.u2, True)]
    s = net.stage2
    out += [(s.inp, False), (s.d1, False)]
    fn = lambda blocks: sum([[(m.pconv, False), (m.expand, False), (m.project, False)] + ([(m.up, True)] if m.up is not None else [])
                             for m in blocks], [])
    out += fn(s.b1) + [(s.d2, False)] + fn(s.b2) + fn(s.b3) + [(s.up, True), (s.head, False)]
    return out


@dataclass
class QLayer:
    key: str
    transposed: bool
    k: int
    stride: int
    cin: int
    cout: int
    wq: np.ndarray            # int8, torch layout of the module (conv [O,I,k,k], transposed [I,O,2,2])
    ws: np.ndarray            # fp32 [cout]
    b: np.ndarray             # fp32 [cout]
    fold: np.ndarray = None   # fp32 [cin] (block entries)
    q_in: str = None          # quantiser names (export json)
    q_pre: str = None
    q_out: str = None
    q_res: str = None
    s_in: float = 0.0         # per-tensor scales (0: none)
    s_pre: float = 0.0
    s_out: float = 0.0
    s_res: float = 0.0
    act: int = 0              # 0 none, 1 leaky 0.2, 2 relu
    m: np.ndarray = field(default=None, repr=False)   # fp32 [cout] epilogue multiplier


def _module_view(arrays, keys, value):
    """Loads arrays (K_w := value(K), K_b) through mlfg_net.load_amd: returns {key: module weight tensor}, so the
    module layout / slicing is exactly mlfg_net's (passes 186 / 188 included)."""
    a = {}
    for k in keys:
        a[k + '_w'] = value(k).astype(np.float32)
        a[k + '_b'] = arrays[k + '_b'].astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, **a)
    buf.seek(0)
    net = N.AmdMlfgNet(weights=None)
    net.load_amd(buf)
    return {k: m.weight.detach().clone() for k, (m, _) in zip(keys, runtime_modules(net))}, net


def load_int8(path):
    """An export_amd int8 npz (+ its json) -> [QLayer] in runtime order, act scales {name: np.ndarray}."""
    z = np.load(path)
    arrays = {k: z[k] for k in z.files}
    meta = json.load(open(os.path.splitext(path)[0] + '.json'))
    layers = meta['layers']
    keys = runtime_keys()
    assert set(keys) == set(layers), 'export layer list differs from the runtime\'s'
    act = {k[4:]: v.astype(np.float32) for k, v in arrays.items() if k.startswith('act:')}

    def bcast(k, axis_key, vec_key):
        w = arrays[k + '_wq']
        shp = [1] * w.ndim; shp[layers[k][axis_key]] = -1
        v = arrays[k + vec_key] if (k + vec_key) in arrays else np.ones(w.shape[layers[k][axis_key]], np.float32)
        return np.broadcast_to(v.reshape(shp), w.shape)
    wq, net = _module_view(arrays, keys, lambda k: arrays[k + '_wq'])
    ws, _ = _module_view(arrays, keys, lambda k: bcast(k, 'w_out_axis', '_ws'))
    fo, _ = _module_view(arrays, keys, lambda k: bcast(k, 'w_in_axis', '_fold'))
    out = []
    for k, (mod, tr) in zip(keys, runtime_modules(net)):
        L = layers[k]
        assert not L.get('w_float'), f'{k}: float weights (--qat-fp-heads) are not supported by the int8 runtime'
        W = wq[k]
        cin, cout = (W.shape[0], W.shape[1]) if tr else (W.shape[1], W.shape[0])
        wsv = (ws[k][0, :, 0, 0] if tr else ws[k][:, 0, 0, 0]).numpy().astype(np.float32)
        q = QLayer(key=k, transposed=tr, k=W.shape[2], stride=mod.stride[0], cin=cin, cout=cout,
                   wq=W.numpy().round().astype(np.int8), ws=wsv, b=mod.bias.detach().numpy().astype(np.float32),
                   q_in=L['input'], q_pre=L.get('pre'), q_out=L.get('output'), q_res=L.get('residual'))
        assert np.array_equal(W.numpy(), q.wq.astype(np.float32)), k
        if L.get('fold'):
            q.fold = (fo[k][:, 0, 0, 0] if tr else fo[k][0, :, 0, 0]).numpy().astype(np.float32)
        else:
            q.s_in = float(act[q.q_in].reshape(-1)[0]) if act[q.q_in].size == 1 else None
            assert q.s_in, f'{k}: input quantiser {q.q_in} is not per-tensor'
        for n in ('pre', 'out', 'res'):
            name = getattr(q, 'q_' + n)
            if name is not None:
                assert act[name].size == 1, f'{k}: {n} quantiser {name} is not per-tensor'
                setattr(q, 's_' + n, float(act[name][0]))
        q.m = q.ws.copy() if q.fold is not None else (q.ws * np.float32(q.s_in)).astype(np.float32)
        out.append(q)
    # activations: stage 2 ReLU (inp, expand), stage 1 leaky; exactly the layers with a pre quantiser
    for q in out:
        if q.q_pre is not None:
            q.act = 2 if q.key in ('L154',) or q.key.endswith('_expand') else 1
    return out, act


# ------------------------------------------------------------------------------------------------ integer arithmetic
f32 = np.float32


def rcp(s):
    return f32(1.0) / f32(s)


def quant(v, r):
    """Q: v fp32 (numpy), r fp32 reciprocal (scalar or per-channel broadcastable) -> int8 values as float64."""
    return np.clip(np.rint((v.astype(np.float32) * r).astype(np.float32)), -QMAX, QMAX).astype(np.float64)


def deq(x, s):
    return (x.astype(np.float32) * f32(s)).astype(np.float32)


def conv_int(q, x):
    """x: int values (float64 numpy [B,C,H,W]) -> exact integer sums (float64 numpy [B,O,H',W'])."""
    xt = torch.from_numpy(np.ascontiguousarray(x))
    w = torch.from_numpy(q.wq.astype(np.float64))
    if q.transposed:
        y = F.conv_transpose2d(xt, w, None, q.stride)
    else:
        y = F.conv2d(xt, w, None, q.stride, padding=1 if q.k == 3 else 0)
    return y.numpy()


def epilogue(q, acc, res=None):
    """acc exact sums -> fp32 y = (acc * m + b) [+ R * s_res]."""
    sh = (1, -1, 1, 1)
    y = (acc.astype(np.float32) * q.m.reshape(sh)).astype(np.float32)
    y = (y + q.b.reshape(sh)).astype(np.float32)
    if res is not None:
        y = (y + deq(res, q.s_res)).astype(np.float32)
    return y


def layer(q, x, res=None):
    """One int8 layer: int input -> int output (or fp32 y for the fp16-output layers: q_out None)."""
    y = epilogue(q, conv_int(q, x), res)
    if q.q_out is None:
        return y
    if q.act:
        p = deq(quant(y, rcp(q.s_pre)), q.s_pre)
        p = np.where(p > 0, p, (p * f32(0.2)).astype(np.float32)) if q.act == 1 else np.maximum(p, f32(0.0))
        return quant(p, rcp(q.s_out))
    return quant(y, rcp(q.s_out))


def quant_in(q, x):
    """Block entry: per-channel quantisation of a glue tensor (torch, any float) with the layer's fold."""
    r = (f32(1.0) / q.fold.astype(np.float32)).reshape(1, -1, 1, 1)
    return quant(x.detach().float().cpu().numpy(), r)


# ------------------------------------------------------------------------------------------------ the network
class Int8MlfgNet(N.AmdMlfgNet):
    def __init__(self, int8_npz, glue='amd', sampler='point', flow_t='scale'):
        super().__init__(weights=None, precision='amd' if glue == 'amd' else 'train', sampler=sampler)
        self.flow_t = flow_t
        self.glue = glue
        self.q, self.act_scales = load_int8(int8_npz)
        self.Q = {q.key: q for q in self.q}
        self.trace = {}   # per-call int tensors of interest (tests)
        net = self
        self.encA.forward = types.MethodType(lambda m, x, amd=False: net._enc(x, 0), self.encA)
        self.encB.forward = types.MethodType(lambda m, x, amd=False: net._enc(x, 4), self.encB)
        for i, b in enumerate((self.block1, self.block2, self.block3, self.block4)):
            b.forward = types.MethodType(lambda m, x, amd=False, i=i: net._block(x, i), b)
        self.stage2.forward = types.MethodType(lambda m, x, amd=False: net._stage2(x), self.stage2)

    def _out(self, y):
        t = torch.from_numpy(np.ascontiguousarray(y))
        return N.q16(t, self.glue == 'amd')

    def _enc(self, x, first):
        L = self.q[first:first + 4]
        h = quant_in(L[0], x)
        for q in L:
            h = layer(q, h)
        return self._out(deq(h, L[3].s_out))

    def _block(self, x, i):
        L = self.q[8 + 13 * i: 8 + 13 * (i + 1)]
        h = quant_in(L[0], x)
        h = layer(L[0], h); h = layer(L[1], h); h = layer(L[2], h)
        for r in range(8):
            h = layer(L[3 + r], h, res=h)
        h = layer(L[11], h)
        return self._out(layer(L[12], h))

    def _fasternet(self, i, x, skip=None):
        pc, ex, pr = self.q[i:i + 3]
        P = pc.cin
        hp = layer(pc, x[:, :P])
        hr = quant(deq(x[:, P:], pc.s_in), rcp(pc.s_out))
        h = layer(ex, np.concatenate([hp, hr], 1))
        y = layer(pr, h, res=x)
        if skip is not None:
            y = layer(self.q[i + 3], y, res=skip)
        return y

    def _stage2(self, x):
        Q = self.Q
        h = quant_in(Q['L154'], x)
        h = layer(Q['L154'], h)
        h = layer(Q['L156'], h)
        i = runtime_keys().index('L158_pconv')
        for j in range(4):
            h = self._fasternet(i + 3 * j, h)
        skip = h
        h = layer(Q['L166'], h)
        i = runtime_keys().index('L168_pconv')
        for j in range(4):
            h = self._fasternet(i + 3 * j, h)
        h = self._fasternet(i + 12, h, skip=skip)
        i = runtime_keys().index('L178_pconv')
        for j in range(4):
            h = self._fasternet(i + 3 * j, h)
        h = layer(Q['L186'], h)
        y = self._out(layer(Q['L188'], h))
        self.trace['logits'] = y
        if self.glue == 'amd':   # AMD's softmax (fp16 logits, running max from 0, fp16 store), as Stage2.forward
            e = torch.exp(y - torch.clamp(y.amax(1, keepdim=True), min=0.0))
            return N.q16(e / e.sum(1, keepdim=True))
        return torch.softmax(y.float(), 1)


# ------------------------------------------------------------------------------------------------ self-check vs QAT
def check_vs_qat(ckpt, int8_npz, weights=N.DEFAULT_WEIGHTS, W=512, H=288, ts=(0.5, 0.25)):
    """The QAT checkpoint's fake-quant forward (ng_model_gym export_amd.build_from_ckpt, frozen scales) against
    Int8MlfgNet(glue='train') on export_amd's synthetic inputs: PSNR of the frames and int agreement of the logits."""
    import sys
    sys.path.insert(0, r'C:\dev\modelgym\ng-model-gym\src')
    from ng_model_gym.usecases.amd_mlfg.export_amd import build_from_ckpt, synthetic_inputs
    from ng_model_gym.usecases.amd_mlfg.model import run
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    qnet, _ = build_from_ckpt(ckpt, weights)
    qnet.flow_t = 'scale'
    inet = Int8MlfgNet(int8_npz, glue='train', sampler=qnet.sampler)
    x = synthetic_inputs(W, H)
    res = {}
    with torch.no_grad():
        for t in ts:
            a, aa = run(qnet, x, t)
            b, ab = run(inet, x, t)
            e = float(((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean())
            res[t] = dict(psnr=99.0 if e == 0 else 10 * np.log10(1.0 / e), max=float((a - b).abs().max()),
                          flow_max=float((aa['flow'] - ab['flow']).abs().max()),
                          weights_max=float((aa['weights'] - ab['weights']).abs().max()))
    return res


def check_layers(ckpt, int8_npz, weights=N.DEFAULT_WEIGHTS, W=512, H=288):
    """Layer by layer: every QAT quantiser's integers (fake-quant forward, frozen scales) fed to the int8 layer here;
    returns {key: (fraction of output integers that differ, max difference)} (fp16-output layers: max |y| difference).
    A cascade of rare one-step flips (fp32 summation order inside QAT's float convolutions) is all that separates the
    whole networks."""
    import sys
    sys.path.insert(0, r'C:\dev\modelgym\ng-model-gym\src')
    from ng_model_gym.usecases.amd_mlfg.export_amd import build_from_ckpt, synthetic_inputs
    from ng_model_gym.usecases.amd_mlfg.model import run
    from ng_model_gym.usecases.amd_mlfg.qat import act_quantizers
    qn, _ = build_from_ckpt(ckpt, weights)
    qn.flow_t = 'scale'
    L, act = load_int8(int8_npz)
    cap, heads, hs = {}, {}, []
    for n, m in act_quantizers(qn).items():
        hs.append(m.register_forward_hook(lambda m, i, o, n=n: cap.__setitem__(n, (i[0].detach(), o.detach().double()))))
    for k, (mod, _) in zip(runtime_keys(), runtime_modules(qn)):
        hs.append(mod.register_forward_hook(lambda m, i, o, k=k: heads.__setitem__(k, o.detach().double())))
    with torch.no_grad():
        run(qn, synthetic_inputs(W, H), 0.5)
    for h in hs:
        h.remove()
    ints = lambda n: torch.round(cap[n][1] / torch.from_numpy(act[n].astype(np.float64)).view(1, -1, 1, 1)).numpy()
    res = {}
    for q in L:
        xi = quant_in(q, cap[q.q_in][0]) if q.fold is not None else ints(q.q_in)
        if q.key.endswith('_pconv'):
            xi = xi[:, :q.cin]
        y = layer(q, xi, ints(q.q_res) if q.q_res else None)
        if q.q_out is None:
            res[q.key] = (None, float(np.abs(y - heads[q.key].numpy()).max()))
            continue
        r = ints(q.q_out)[:, :y.shape[1]]
        d = np.abs(y - r)
        res[q.key] = (float(np.mean(d > 0)), float(d.max()))
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('int8_npz')
    a = ap.parse_args()
    lr = check_layers(a.ckpt, a.int8_npz)
    fr = [v[0] for v in lr.values() if v[0] is not None]
    print(f'layer by layer: {len(fr)} int8-output layers, integers differing: mean {np.mean(fr) * 100:.4f}%, worst '
          f'{max(fr) * 100:.4f}%, max step {max(v[1] for v in lr.values() if v[0] is not None):.0f}; fp16-output layers max '
          f'{max(v[1] for v in lr.values() if v[0] is None):.3g}')
    for t, r in check_vs_qat(a.ckpt, a.int8_npz).items():
        print(f't={t}: PSNR {r["psnr"]:.2f} dB, max {r["max"]:.4g}, flow max {r["flow_max"]:.4g}, weights max {r["weights_max"]:.4g}')
