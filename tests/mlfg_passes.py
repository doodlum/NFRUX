"""Checks NFRUX's port of AMD FSR Frame Generation 4's hand-written passes (core/shaders/mlfg, kMlfg*) on the GPU, on D3D12
and Vulkan (build/Release/mlfg_passes.exe runs one pass at a time from raw files), against
  * the PyTorch transcriptions tools/ffx_extract/mlfg_io.py / mlfg_glue.py (MODE 'glsl': AMD's shader roundings),
  * AMD's own GPU output in the captures (io_b / io_c image + scratch dumps, dump_a scratch dumps), and
  * each other (D3D12 and Vulkan must give the same bytes).
All at t = 0.5, plus checks that t enters where it should.

    python tests/mlfg_passes.py [--exe build-ship/Release/mlfg_passes.exe] [--work D:/nfrux_ffx/work/mlfg_test] [--api d3d12,vulkan]

Colour / flow reads: AMD's binding-320 sampler is nearest / clamp on every vendor (its DLL creates no other), so the
shaders point-sample with mlfg_io.sample_point's texel rule (G_SAMPLER_IS_POINT True), and pass 0 is
mlfg_glue.pass0(filt='capture_nearest').
SPDX-License-Identifier: MIT
"""
import argparse, os, struct, subprocess, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'ffx_extract'))
import mlfg_io as io  # noqa: E402
import mlfg_glue as glue  # noqa: E402
import mlfg_ref  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--exe', default=os.path.join(HERE, '..', 'build-ship', 'Release', 'mlfg_passes.exe'))
ap.add_argument('--work', default=r'D:\nfrux_ffx\work\mlfg_test')
ap.add_argument('--api', default='d3d12,vulkan')
ap.add_argument('--io-b', default=r'D:\nfrux_ffx\capture\io_b')
ap.add_argument('--io-c', default=r'D:\nfrux_ffx\capture\io_c')
ap.add_argument('--dump-a', default=r'D:\nfrux_ffx\capture\dump_a')
A = ap.parse_args()
APIS = A.api.split(',')
os.makedirs(A.work, exist_ok=True)
FRAME = 2
RESULTS = []


def path(n):
    return os.path.join(A.work, n)


def save(name, arr):
    p = path(name)
    np.ascontiguousarray(arr).tofile(p)
    return p


def cbytes(*rows):
    """Constant rows: each a tuple of 4 ('i' ints or 'f' floats) -> bytes; a list of 16 floats = 4 rows."""
    out = b''
    for kind, vals in rows:
        vals = list(vals) + [0] * (4 * ((len(vals) + 3) // 4) - len(vals))
        out += struct.pack('<%d%s' % (len(vals), 'i' if kind == 'i' else 'f'), *[int(v) if kind == 'i' else float(v) for v in vals])
    return out


def groups(w, h):
    return (w + 7) // 8, (h + 7) // 8


class Job:
    """One dispatch: lines of the harness job file, outputs {name: (file name, dtype, shape)}."""
    def __init__(self, name, shader, cb, gx, gy):
        self.name, self.lines, self.outs = name, [f'pass {shader}'], {}
        self.lines.append(f'cb {save(name + "_cb.bin", np.frombuffer(cb, np.uint8))}')
        self.g = (gx, gy)

    def tex(self, b, fmt, arr):
        h, w = arr.shape[:2]
        self.lines.append(f'tex {b} {fmt} {w} {h} {save(f"{self.name}_t{b}.bin", arr)}')
        return self

    def buf(self, b, arr):
        self.lines.append(f'buf {b} {arr.nbytes} {save(f"{self.name}_b{b}.bin", arr)}')
        return self

    def utex(self, b, fmt, w, h, out, dtype, ch):
        self.lines.append(f'utex {b} {fmt} {w} {h} - @{out}')
        self.outs[out] = (dtype, (h, w, ch))
        return self

    def ubuf(self, b, nbytes, out, dtype, shape):
        self.lines.append(f'ubuf {b} {nbytes} - @{out}')
        self.outs[out] = (dtype, shape)
        return self

    def run(self):
        """Runs on every API; returns {api: {out: array}}."""
        res = {}
        for api in APIS:
            lines = [l.replace('@', path(api + '_' + self.name + '_')) for l in self.lines] + ['run %d %d' % self.g]
            jf = path(f'{api}_{self.name}.job')
            open(jf, 'w').write('\n'.join(lines) + '\n')
            r = subprocess.run([A.exe, api, jf], capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f'{api} {self.name}: {r.stdout}{r.stderr}')
            res[api] = {o: np.fromfile(path(api + '_' + self.name + '_' + o), dt).reshape(shp) for o, (dt, shp) in self.outs.items()}
        if len(APIS) > 1:
            for o in self.outs:
                same = all(np.array_equal(res[APIS[0]][o].view(np.uint8), res[a][o].view(np.uint8)) for a in APIS[1:])
                report(f'{self.name}.{o}: {" vs ".join(APIS)}', 'bytes identical' if same else 'BYTES DIFFER', same)
        return res[APIS[0]]


def report(name, msg, ok, info=False):
    if info:  # context only: where the transcription itself is known to differ from AMD's capture GPU
        print(f'  [..] {name:58s} {msg}')
        return
    RESULTS.append((name, ok))
    print(f'  [{"ok" if ok else "!!"}] {name:58s} {msg}')


def capture_diff(name, got, ref, amd, fp16=True):
    """got == ref is checked elsewhere; here: how far both are from AMD's capture, and that the shader is no further."""
    g, r, a = (np.asarray(x, np.float64) for x in (got, ref, amd))
    dg, dr = np.abs(g - a), np.abs(r - a)
    msg = f'shader: max {dg.max():.4g}, {int((dg > 0).sum())} differ; transcription: max {dr.max():.4g}, {int((dr > 0).sum())} differ'
    if fp16:
        msg += f'; max fp16 ulp {int(ulp16(np.asarray(got), np.asarray(amd)).max())}'
    report(name, msg, True, info=True)
    if not np.array_equal(dg, dr):
        report(name + ' (shader no further than transcription)', 'DIFFERENT from the transcription\'s distance', False)


def ulp16(a, b):
    ia = a.astype(np.float16).view(np.int16).astype(np.int32)
    ib = b.astype(np.float16).view(np.int16).astype(np.int32)
    ia = np.where(ia < 0, -32768 - ia, ia); ib = np.where(ib < 0, -32768 - ib, ib)
    return np.abs(ia - ib)


def compare(name, got, ref, exact=True, tol_ulp=0, fp16=True):
    got = np.asarray(got, np.float64) if not np.issubdtype(np.asarray(got).dtype, np.integer) else np.asarray(got)
    ref = np.asarray(ref, np.float64) if not np.issubdtype(np.asarray(ref).dtype, np.integer) else np.asarray(ref)
    if got.shape != ref.shape:
        report(name, f'SHAPE {got.shape} vs {ref.shape}', False)
        return
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    bad = int((d > 0).sum())
    msg = f'max abs err {d.max():.4g}, mismatched {bad}/{d.size}'
    ok = bad == 0
    if fp16 and not np.issubdtype(got.dtype, np.integer):
        u = ulp16(got, ref)
        msg += f', max fp16 ulp {int(u.max())}'
        if not exact:
            ok = int(u.max()) <= tol_ulp
    report(name, msg, ok)


# ---------------------------------------------------------------------------------------------- capture access
def raw_images(cap):
    """{(seq, binding): (np array of the raw texels, fmt)}"""
    d = os.path.join(cap, 'images')
    out = {}
    FMT = {83: (np.float16, 2), 100: (np.float32, 1), 101: (np.uint32, 2), 37: (np.uint8, 4)}
    for l in open(os.path.join(d, 'index.txt')):
        f = l.split()
        W, H = (int(x) for x in f[1].split('x'))
        fmt = int(f[3])
        dt, c = FMT[fmt]
        seq, b = int(f[0].split('_')[1]), int(f[0].rsplit('_b', 1)[1].split('.')[0])
        out[(seq, b)] = (np.fromfile(os.path.join(d, f[0]), dt).reshape(H, W, c), fmt)
    return out


def dispatch_cbs(cap):
    """[(entry, cb)] of the frame's image-pass dispatches in image-dump sequence order."""
    import re
    disp = io.frame_dispatches(cap, FRAME)
    return [(d[1], d[3], d[0]) for d in disp if d[1] == 'main' or re.search(r'_pass(151|152|190)$', d[1])]


def tex_of(arr):
    """Capture raw texels -> (fmt name, array for the harness)."""
    if arr.dtype == np.float16: return 'rg16f', arr
    if arr.dtype == np.float32: return 'r32f', arr
    if arr.dtype == np.uint8: return 'rgba8', arr
    raise ValueError(arr.dtype)


def nhwc(t, pitch):
    """torch [C, H, W] (fp16 values) -> numpy fp16 [H, W, pitch]."""
    C, H, W = t.shape
    a = np.zeros((H, W, pitch), np.float16)
    a[..., :C] = t.permute(1, 2, 0).numpy().astype(np.float16)
    return a


# ---------------------------------------------------------------------------------------------- prepass (io_b)
def test_prepass():
    print('prepass (io_b frame 2): pipe191 dilate, pipe192 splats, pipe193 forward mv')
    img = raw_images(A.io_b)
    tim, _ = io.load_images(A.io_b)
    seqs = dispatch_cbs(A.io_b)
    # pipe191
    s = [i for i, (e, cb, pid) in enumerate(seqs) if pid == 191][0]
    cb = seqs[s][1]
    inv = 'Inv' if cb['depthInverted'] else ''
    W, H = cb['dispatchSize']
    k = cbytes(('i', cb['renderSize'] + cb['dispatchSize']), ('f', cb['motionVectorScale']),
               ('f', cb['generationRectMinUV'] + cb['generationRectMaxUV']), ('f', cb['colorRectMinUV'] + cb['colorRectMaxUV']))
    j = Job('dilate', 'kMlfgDilate' + inv, k, *groups(W, H))
    j.tex(1, 'rg16f', img[(s, 0)][0]).tex(2, 'r32f', img[(s, 1)][0])
    j.utex(3, 'rg16f', W, H, 'mv', np.float16, 2).utex(4, 'r32f', W, H, 'depth', np.float32, 1)
    out = j.run()
    rmv, rdp = io.pipe191_dilate(tim[(s, 0)], tim[(s, 1)], cb)
    compare('pipe191 mv vs transcription', out['mv'], rmv.numpy())
    compare('pipe191 mv vs AMD capture', out['mv'], img[(s, 128)][0])
    compare('pipe191 depth vs AMD capture', out['depth'], img[(s, 129)][0], fp16=False)
    # pipe192 x3
    for s in [i for i, (e, cb, pid) in enumerate(seqs) if pid == 192]:
        cb = seqs[s][1]
        inv = 'Inv' if cb['depthInverted'] else ''
        W, H = cb['dstSize']
        fl, dp = img[(s, 0)][0], img[(s, 1)][0]
        k = cbytes(('i', [W, H, fl.shape[1], fl.shape[0]]), ('f', [cb['projAmount'], -1.0 if cb['flowIsUv'] else 1.0]),
                   ('f', cb['generationRectMinUV'] + cb['generationRectMaxUV']))
        tag = f'splat{s}'
        j = Job(tag + 'k', 'kMlfgSplatKey' + inv, k, *groups(W, H))
        j.tex(1, 'rg16f', fl).tex(2, 'r32f', dp).utex(4, 'r32ui', W, H, 'key', np.uint32, 1)
        key = j.run()['key']
        j = Job(tag + 'f', 'kMlfgSplatFlow' + inv, k, *groups(W, H))
        j.tex(1, 'rg16f', fl).tex(2, 'r32f', dp).tex(3, 'r32ui', key).utex(5, 'r32ui', W, H, 'flow', np.uint32, 1)
        word = j.run()['flow']
        amd = img[(s, 128)][0].astype(np.int64)
        ref = io.pipe192_splat(tim[(s, 0)], tim[(s, 1)], cb).numpy()
        name = f'pipe192 seq{s} proj {cb["projAmount"]} uv {cb["flowIsUv"]}'
        compare(name + ' (flow, key) vs AMD capture', np.concatenate([word, key], -1).astype(np.int64), amd, fp16=False)
        compare(name + ' vs transcription', np.concatenate([word, key], -1).astype(np.int64), ref, fp16=False)
    # pipe193
    s = [i for i, (e, cb, pid) in enumerate(seqs) if pid == 193][0]
    cb = seqs[s][1]
    mv0, d0, sp = img[(s, 0)][0], img[(s, 1)][0], img[(s, 2)][0]
    H, W = mv0.shape[:2]
    k = cbytes(('f', cb['prevPstoPs']), ('f', cb['prevPsToPrev2Ps']), ('i', cb['displaySize'] + [W, H]), ('i', [d0.shape[1], d0.shape[0]]))
    j = Job('fwdmv', 'kMlfgForwardMv', k, *groups(W, H))
    j.tex(1, 'rg16f', mv0).tex(2, 'r32f', d0).tex(3, 'r32ui', np.ascontiguousarray(sp[..., 1:2])).tex(4, 'r32ui', np.ascontiguousarray(sp[..., 0:1]))
    j.utex(5, 'rg16f', W, H, 'mv', np.float16, 2)
    out = j.run()
    ref = io.pipe193_forward_mv(tim[(s, 0)], tim[(s, 1)], tim[(s, 2)], cb).numpy()
    compare('pipe193 forward mv vs transcription', out['mv'], ref)
    compare('pipe193 forward mv vs AMD capture', out['mv'], img[(s, 128)][0])


# ---------------------------------------------------------------------------------------------- 151 / 152 / 190 (io_b, io_c)
def run_flowout(name, f149, flow112, S69, S70, t_note=''):
    """f149: [>=4, H, W]; flow112 [4, H, W] -> (flowout, img0, img1) from the GPU."""
    W, H = S69
    head = nhwc(f149[:, :H, :W], 16)
    prev = nhwc(flow112[:, :H, :W], 4)
    k = cbytes(('i', [W, W]), ('i', [W, H, W, H]), ('i', list(S70)), ('f', [1.0, 1.0]))
    gw, gh = max(W, S70[0]), max(H, S70[1])
    j = Job(name, 'kMlfgFlowOut', k, *groups(gw, gh))
    j.buf(1, head).buf(2, prev)
    j.ubuf(3, W * H * 8, 'flow', np.float16, (H, W, 4))
    j.utex(4, 'rg16f', S70[0], S70[1], 'img0', np.float16, 2).utex(5, 'rg16f', S70[0], S70[1], 'img1', np.float16, 2)
    return j.run()


def frame_cb_rows(cb):
    return cb['generationRectMinUV'] + cb['generationRectMaxUV'], cb['minMaxLuminance']


def test_151_152_190():
    print('passes 151 / 152 / 190 (io_b, io_c frame 2)')
    sizes = mlfg_ref.sizes_table(A.io_b)
    S = lambda i: (int(sizes[i][0]), int(sizes[i][1]))
    img = raw_images(A.io_b)
    tim, _ = io.load_images(A.io_b)
    seqs = dispatch_cbs(A.io_b)
    pas = os.path.join(A.io_b, 'passes')
    # 151: AMD's flowout (scratch after pass 150) through FlowOut with a zero residual.
    s = [i for i, (e, cb, pid) in enumerate(seqs) if e.endswith('_pass151')][0]
    fo = glue.read_t(open(os.path.join(pas, f'{FRAME}_150.bin'), 'rb').read(), 'flowout', *S(69))
    out = run_flowout('flow151', fo, torch.zeros_like(fo), S(69), S(70))
    r0, r1 = glue.pass151(fo, S(70))
    compare('pass151 flow0toT vs transcription', out['img0'], r0.permute(1, 2, 0).numpy())
    compare('pass151 flow1toT vs transcription', out['img1'], r1.permute(1, 2, 0).numpy())
    compare('pass151 flow0toT vs AMD capture', out['img0'], img[(s, 129)][0])
    compare('pass151 flow1toT vs AMD capture', out['img1'], img[(s, 130)][0])
    # 152
    s = [i for i, (e, cb, pid) in enumerate(seqs) if e.endswith('_pass152')][0]
    cb = seqs[s][1]
    print('    frame cb:', {k: cb[k] for k in ('reset', 'backBufferTransferFunction', 'hudlessAttachedFactor', 'dispatchFlags')})
    rgb0, rgb1, fl0, fl1, sp0, sp1 = (img[(s, b)][0] for b in range(1, 7))
    W72, H72 = S(72)
    gr, lum = frame_cb_rows(cb)
    amd = io.read_pass152(open(os.path.join(pas, f'{FRAME}_152.bin'), 'rb').read(), sizes).numpy()
    k = cbytes(('i', [W72, H72, W72, H72]), ('i', [rgb0.shape[1], rgb0.shape[0], fl0.shape[1], fl0.shape[0]]),
               ('i', [sp0.shape[1], sp0.shape[0], cb['backBufferTransferFunction'], 0]), ('f', gr), ('f', lum))
    j = Job('s2in', 'kMlfgStage2Input', k, *groups(W72, H72))
    j.tex(1, 'rgba8', rgb0).tex(2, 'rgba8', rgb1).tex(3, 'rg16f', fl0).tex(4, 'rg16f', fl1)
    j.tex(5, 'r32ui', np.ascontiguousarray(sp0[..., 1:2])).tex(6, 'r32ui', np.ascontiguousarray(sp0[..., 0:1]))
    j.tex(7, 'r32ui', np.ascontiguousarray(sp1[..., 1:2])).tex(8, 'r32ui', np.ascontiguousarray(sp1[..., 0:1]))
    j.ubuf(9, W72 * H72 * 64, 'out', np.float16, (H72, W72, 32))
    got = j.run()['out']
    ref = io.pass152_stage2_input(*(tim[(s, b)] for b in range(1, 7)), cb, sizes).numpy()
    compare('pass152 ch 0-23 vs transcription', got[..., :24], ref)
    capture_diff('pass152 vs AMD scratch dump (luma ch 20-23: NV fma)', got[..., :24], ref, amd)
    report('pass152 ch 24-31 zero', '', bool((got[..., 24:] == 0).all()))
    # 190 (io_c: weights from its pass-189 dump)
    imc = raw_images(A.io_c)
    timc, _ = io.load_images(A.io_c)
    seqc = dispatch_cbs(A.io_c)
    s = [i for i, (e, cb, pid) in enumerate(seqc) if e.endswith('_pass190')][0]
    cb = seqc[s][1]
    print('    frame cb:', {k: cb[k] for k in ('reset', 'backBufferTransferFunction', 'hudlessAttachedFactor', 'dispatchFlags')})
    wts = io.read_pass188_weights(open(os.path.join(A.io_c, 'passes', f'{FRAME}_189.bin'), 'rb').read(), sizes)
    TH, TW = wts.shape[:2]
    wbuf = np.zeros((TH, TW, 8), np.float16)
    wbuf[..., :5] = wts[..., :5].numpy().astype(np.float16)
    rgb0, rgb1, fl0, fl1, sp0, sp1, hud = (imc[(s, b)][0] for b in range(1, 8))
    H, W = rgb0.shape[:2]
    gr, lum = frame_cb_rows(cb)
    hl = hud if cb['hudlessAttachedFactor'] else rgb1
    for t in (0.5, 0.25):
        k = cbytes(('i', [W, H, fl0.shape[1], fl0.shape[0]]), ('i', [sp0.shape[1], sp0.shape[0], TW, 0]),
                   ('i', [TW, TH, cb['reset'], cb['hudlessAttachedFactor']]), ('i', [cb['backBufferTransferFunction'], cb['dispatchFlags']]),
                   ('f', gr), ('f', lum + [t]), ('f', cb['debugBarColor']))
        j = Job(f'comp_{t}', 'kMlfgCompositeRgba8', k, *groups(W, H))
        j.buf(1, wbuf).tex(2, 'rgba8', rgb0).tex(3, 'rgba8', rgb1).tex(4, 'rg16f', fl0).tex(5, 'rg16f', fl1)
        j.tex(6, 'r32ui', np.ascontiguousarray(sp0[..., 1:2])).tex(7, 'r32ui', np.ascontiguousarray(sp0[..., 0:1]))
        j.tex(8, 'r32ui', np.ascontiguousarray(sp1[..., 1:2])).tex(9, 'r32ui', np.ascontiguousarray(sp1[..., 0:1]))
        j.tex(10, 'rgba8', hl).utex(11, 'rgba8', W, H, 'out', np.uint8, 4)
        got = j.run()['out']
        args = (torch.from_numpy(wbuf.astype(np.float32)),) + tuple(timc[(s, b)] for b in range(1, 7))
        if t == 0.5:
            ref = io.to_unorm8(io.pass190_composite(*args, cb, sizes, timc.get((s, 7)))).numpy().astype(np.uint8)
        else:
            ref = composite_at_t(*args, cb, sizes, t).numpy().astype(np.uint8)
        compare(f'pass190 (t={t}) rgb vs transcription', got[..., :3], ref, fp16=False)
        report(f'pass190 (t={t}) alpha 0', '', bool((got[..., 3] == 0).all()))
        if t == 0.5:
            capture_diff('pass190 rgb vs AMD capture (unorm8 levels)', got[..., :3], ref, imc[(s, 129)][0][..., :3], fp16=False)


def composite_at_t(weights6, rgb0, rgb1, flow0, flow1, splat0, splat1, cb, sizes, t):
    """mlfg_io.pass190_composite with the plain-blend term at t: rgb0 * h((1-t) w4) + rgb1 * h(t w4) (no HUD-less, no
    debug bars). At t = 0.5 this is AMD's term."""
    h = io.h
    W, H = rgb0.shape[1], rgb0.shape[0]
    gx, gy = io.pix_grid(W, H)
    uv = io.texel_uv(torch.stack([gx, gy], -1), [float(W), float(H)])
    u, v = uv[..., 0], uv[..., 1]
    lo = torch.tensor(cb['generationRectMinUV']); hi = torch.tensor(cb['generationRectMaxUV'])
    out_rect = (uv > hi).any(-1) | (uv < lo).any(-1)
    p0 = rgb0[..., :3]; p1 = rgb1[..., :3]
    TW, TH = (int(x) for x in sizes[90])
    xs = torch.clamp((gx.float() + 0.5) * 0.25 - 0.5, min=0.0); ys = torch.clamp((gy.float() + 0.5) * 0.25 - 0.5, min=0.0)
    x0 = torch.floor(xs).long().clamp(0, TW - 1); y0 = torch.floor(ys).long().clamp(0, TH - 1)
    fx = (xs - x0.float()).clamp(0, 1)[..., None]; fy = (ys - y0.float()).clamp(0, 1)[..., None]
    x1 = (x0 + 1).clamp(0, TW - 1); y1 = (y0 + 1).clamp(0, TH - 1)
    tt = weights6[..., :6]
    w = h(io.mix(h(io.mix(tt[y0, x0], tt[y0, x1], fx)), h(io.mix(tt[y1, x0], tt[y1, x1], fx)), fy))
    m1 = io.splat_valid_weight(splat1, u, v)[..., None]; m0 = io.splat_valid_weight(splat0, u, v)[..., None]
    s0 = io.splat_flow_at(splat0, u, v, W); s1 = io.splat_flow_at(splat1, u, v, W)
    cl = lambda q: torch.minimum(torch.maximum(q, lo), hi)
    rd = lambda img, q: io.sample_g(img, q[..., 0], q[..., 1])[..., :3]
    c1 = rd(rgb1, cl(uv - s1)); c0 = rd(rgb0, cl(uv - s0))
    f0 = h(io.sample_g(flow0, u, v)); f1 = h(io.sample_g(flow1, u, v))
    one_t = torch.tensor(1.0 - t, dtype=torch.float32)  # fp32 (1 - t), as the shader computes it
    w40 = h(one_t * w[..., 4:5]); w41 = h(torch.tensor(t, dtype=torch.float32) * w[..., 4:5])
    out = ((((c0 * (1 - m1) + c1 * m1) * w[..., 0:1] + (c1 * (1 - m0) + c0 * m0) * w[..., 1:2])
            + rd(rgb0, cl(uv + f0)) * w[..., 2:3]) + rd(rgb1, cl(uv + f1)) * w[..., 3:4]) + p0 * w40 + p1 * w41
    gen = (~out_rect) & (cb['reset'] == 0)
    return io.to_unorm8(torch.where(gen[..., None], out, p1))


# ---------------------------------------------------------------------------------------------- glue passes (dump_a)
def test_glue():
    print('glue passes 0 / 18 / 54 / 83 / 112 / 149 (dump_a frame 2; reference mlfg_glue MODE glsl, AMD dump = NVIDIA arithmetic)')
    dumps = os.path.join(A.dump_a, 'passes')
    sizes = mlfg_ref.sizes_table(A.dump_a)
    sz = lambda i: (int(sizes[i][0]), int(sizes[i][1]))
    have = sorted(int(f.split('_')[1][:3]) for f in os.listdir(dumps) if f.startswith(f'{FRAME}_'))
    dump = lambda n: open(os.path.join(dumps, f'{FRAME}_{n:03d}.bin'), 'rb').read()
    before = lambda p: dump(max(x for x in have if x < p))

    def both(name, got, mode_ref, amd):
        compare(name + ' vs transcription (glsl)', got, mode_ref)
        capture_diff(name + ' vs AMD dump (NVIDIA arithmetic)', got, mode_ref, amd)

    # pass 0 (the synthetic frames the dump_a capture rendered; AMD's nearest sampler = 'capture_nearest').
    S0 = sz(0)
    prev = glue.synth_frame(FRAME - 1); cur = glue.synth_frame(FRAME)
    rgba = lambda f: np.ascontiguousarray(np.concatenate([f.transpose(1, 2, 0), np.full(f.shape[1:] + (1,), 255, np.uint8)], -1))
    k = cbytes(('i', [S0[0], S0[1], S0[0], S0[1]]), ('i', [1920, 1080, 0, 0]), ('f', [0, 0, 1, 1]), ('f', [0, 0]))
    j = Job('pass0', 'kMlfgNetInput', k, *groups(*S0))
    j.tex(1, 'rgba8', rgba(prev)).tex(2, 'rgba8', rgba(cur)).ubuf(3, S0[0] * S0[1] * 64, 'net', np.float16, (S0[1], S0[0], 32))
    got = j.run()['net']
    lw, lh = min(S0[0], 240), min(S0[1], 135)
    ref = glue.pass0(torch.from_numpy(prev.astype(np.float32)) / 255.0, torch.from_numpy(cur.astype(np.float32)) / 255.0, lw, lh, S0[0], S0[1], 'capture_nearest')
    compare('pass0 net_in ch 0-17 vs transcription', got[..., :18], ref.permute(1, 2, 0).numpy())
    compare('pass0 net_in ch 0-5 vs AMD dump', got[..., :6], glue.read_t(dump(0), 'net_in', *S0).permute(1, 2, 0).numpy()[..., :6])
    report('note', 'glue below: AMD dump = NVIDIA arithmetic (mlfg_glue nv mode); the shaders follow AMD\'s shader (glsl mode)', True, info=True)
    report('pass0 net_in ch 18-31 zero', '', bool((got[..., 18:] == 0).all()))

    glue.MODE = 'glsl'
    # pass 18
    S1, S10 = sz(1), sz(10)
    net = glue.read_t(before(18), 'net_in', *S1)
    k = cbytes(('i', [S1[0], S1[0], S1[1]]), ('i', [S10[0], S10[1], S10[0], S10[1]]), ('f', [4.0]))
    j = Job('pass18', 'kMlfgDownsample', k, *groups(*S10))
    j.buf(1, nhwc(net, 32)).ubuf(2, S10[0] * S10[1] * 64, 'inq', np.float16, (S10[1], S10[0], 32))
    got = j.run()['inq']
    both('pass18 in_q', got[..., :18], glue.pass18(net, S1, S10).permute(1, 2, 0).numpy(),
         glue.read_t(dump(18), 'in_q', *S10).permute(1, 2, 0).numpy())
    report('pass18 in_q ch 18-31 zero', '', bool((got[..., 18:] == 0).all()))

    def refine(name, mode, net, Snet, head, Shead, prev, So, flowsz, scale, fscale, t=0.5):
        k = cbytes(('i', [mode]), ('i', [Snet[0], Snet[0], Snet[1]]), ('i', [Shead[0], Shead[0], Shead[1]]),
                   ('i', [prev.shape[2] if prev is not None else 1]), ('i', [So[0], So[1], So[0], So[1]]), ('i', [flowsz[0]]),
                   ('f', [scale, fscale, t]))
        j = Job(name, 'kMlfgRefine', k, *groups(*So))
        j.buf(1, nhwc(net, 32)).buf(2, nhwc(head[:13], 16))
        j.buf(3, nhwc(prev, 4) if prev is not None else np.zeros(4, np.float16))
        j.ubuf(4, So[0] * So[1] * 64, 'x', np.float16, (So[1], So[0], 32)).ubuf(5, flowsz[0] * flowsz[1] * 8, 'flow', np.float16, (flowsz[1], flowsz[0], 4))
        return j.run()

    P = lambda t: t.permute(1, 2, 0).numpy()
    # pass 54
    b, a = before(54), dump(54)
    S23, S24 = sz(23), sz(24)
    net, f54 = glue.read_t(b, 'net_in', *S1), glue.read_t(b, 'f54', *S23)
    out = refine('pass54', 0, net, S1, f54, S23, None, S24, (2 * S24[0], 2 * S24[1]), 0.25, 4.0)
    rf, rx = glue.pass54(net, f54, S1, S23, S24)
    both('pass54 flow54', out['flow'], P(rf), P(glue.read_t(a, 'flow54', 2 * S24[0], 2 * S24[1])))
    both('pass54 x54', out['x'], P(rx), P(glue.read_t(a, 'x54', *S24)))
    # pass 83
    b, a = before(83), dump(83)
    S38, S39 = sz(38), sz(39)
    net, f83, fl54 = glue.read_t(b, 'net_in', *S1), glue.read_t(b, 'f83', *S38), glue.read_t(b, 'flow54', *S39)
    out = refine('pass83', 1, net, S1, f83, S38, fl54, S39, S39, 0.5, 2.0)
    rf, rx = glue.pass83(net, f83, fl54, S1, S38, S39)
    both('pass83 flow83', out['flow'], P(rf), P(glue.read_t(a, 'flow83', *S39)))
    both('pass83 x83', out['x'], P(rx), P(glue.read_t(a, 'x83', *S39)))
    # the timestep: channel 18 = fp16(t), the rest unchanged
    out3 = refine('pass83_t03', 1, net, S1, f83, S38, fl54, S39, S39, 0.5, 2.0, t=0.3)
    report('pass83 t=0.3: ch 18 == fp16(0.3)', '', bool((out3['x'][..., 18] == np.float16(0.3)).all()))
    report('pass83 t=0.3: other channels as t=0.5', '', bool(np.array_equal(np.delete(out3['x'], 18, -1), np.delete(out['x'], 18, -1))))
    # pass 112
    b, a = before(112), dump(112)
    S53, S54 = sz(53), sz(54)
    net, f112, fl83 = glue.read_t(b, 'net_in', *S53), glue.read_t(b, 'f112', *S54), glue.read_t(b, 'flow83', *S54)
    out = refine('pass112', 2, net, S53, f112, S54, fl83, S54, S54, 0.0, 1.0)
    rf, rx = glue.pass112(net, f112, fl83, S53, S54)
    both('pass112 flow112', out['flow'], P(rf), P(glue.read_t(a, 'flow112', *S54)))
    both('pass112 x112', out['x'], P(rx), P(glue.read_t(a, 'x112', *S54)))
    # pass 149 (+151 images of the result)
    b, a = before(149), dump(149)
    S69 = sz(69)
    f149, fl112 = glue.read_t(b, 'f149', *S69), glue.read_t(b, 'flow112', *S69)
    out = run_flowout('pass149', f149, fl112, S69, tuple(int(x) for x in sizes[70]))
    both('pass149 flowout', out['flow'], P(glue.pass149(f149, fl112, S69)), P(glue.read_t(a, 'flowout', *S69)))
    r0, r1 = glue.pass151(glue.pass149(f149, fl112, S69), sizes[70])
    compare('pass149+151 flow0toT vs transcription', out['img0'], P(r0))
    compare('pass149+151 flow1toT vs transcription', out['img1'], P(r1))


# ---------------------------------------------------------------------------------------------- softmax5
def test_softmax():
    print('softmax5 (pass 188 activation; synthetic fp16 logits)')
    W, H = 480, 272
    g = torch.Generator().manual_seed(1)
    y = (torch.randn(5, H, W, generator=g) * 4.0).to(torch.float16).float()
    y[:, :8] = -(y[:, :8].abs()) - 1.0  # rows where every logit is negative: the running max starts at 0
    y[:, 8:12] = y[:, 8:12] * 6.0
    y = y.to(torch.float16).float()  # the logits are fp16
    logits = nhwc(y, 16)
    j = Job('softmax5', 'kMlfgSoftmax5', cbytes(('i', [W, H, W, W])), *groups(W, H))
    j.buf(1, logits).ubuf(2, W * H * 16, 'w', np.float16, (H, W, 8))
    got = j.run()['w']
    mx = torch.clamp(y.amax(0, keepdim=True), min=0.0)  # mlfg_model.Emulator.conv_pass, act 'softmax_max0'
    e = torch.exp(y - mx)
    ref = (e / e.sum(0, keepdim=True)).to(torch.float16).float()
    compare('softmax5 ch 0-4 vs torch (exp: polynomial vs libm)', got[..., :5], ref.permute(1, 2, 0).numpy(), exact=False, tol_ulp=1)
    report('softmax5 ch 5-7 zero', '', bool((got[..., 5:] == 0).all()))


if __name__ == '__main__':
    torch.set_num_threads(4)
    test_prepass()
    test_151_152_190()
    test_glue()
    test_softmax()
    bad = [n for n, ok in RESULTS if not ok]
    print(f'{len(RESULTS) - len(bad)}/{len(RESULTS)} checks ok' + ('' if not bad else '; failing: ' + '; '.join(bad)))
    sys.exit(1 if bad else 0)
