"""Checks mlfg_net.AmdMlfgNet (precision 'amd', t = 0.5) against the byte-exact emulator / glue / io transcriptions and
AMD's GPU dumps.

    python verify_mlfg_net.py [--part all|glue|blocks|chain|io|stage2]
"""
import argparse, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlfg_model, mlfg_glue, mlfg_io, mlfg_ref
from mlfg_model import Emulator, read
import mlfg_net as N

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
CAP = r'D:\nfrux_ffx\capture\nv_1080p_spoof'
DUMP_A = r'D:\nfrux_ffx\capture\dump_a'
IO_B = r'D:\nfrux_ffx\capture\io_b'
IO_C = r'D:\nfrux_ffx\capture\io_c'
NPZ = N.DEFAULT_WEIGHTS
dev = 'cuda'


def rep(name, a, b, extra=''):
    a = a.float().cpu(); b = b.float().cpu()
    d = (a - b).abs()
    ref = b.abs().max().item()
    print(f'{name:44s} max {d.max().item():.4g}  mean {d.mean().item():.3g}  ref|max| {ref:.3g}  '
          f'mismatch {(d > 0).float().mean().item() * 100:.2f}% {extra}', flush=True)
    return d.max().item()


def load(d, n):
    return torch.from_numpy(np.fromfile(os.path.join(d, 'passes', f'2_{n:03d}.bin'), np.uint8))


def rt(buf, name, W, H, C=None):
    """mlfg_glue.read_t on a torch uint8 buffer -> [1,C,H,W] on dev."""
    return mlfg_glue.read_t(buf.cpu().numpy().tobytes(), name, W, H, C)[None].to(dev)


def wt(scr, name, v, W, H):
    """Write fp16 [C,H,W] into the emulator scratch (glue layout, 1-px halo)."""
    off, sx, sy, Cn = mlfg_glue.L[name]
    C = v.shape[0]
    h = scr.view(torch.float16)
    idx = (off // 2 + (torch.arange(H, device=scr.device).view(1, H, 1) + 1) * (sy // 2)
           + (torch.arange(W, device=scr.device).view(1, 1, W) + 1) * (sx // 2) + torch.arange(C, device=scr.device).view(C, 1, 1))
    h[idx] = v[:, :H, :W].to(scr.device).half()


def sizes(cap):
    S = mlfg_ref.sizes_table(cap)
    return lambda i: (int(S[i][0]), int(S[i][1]))


# ------------------------------------------------------------------------------------------------ glue
def part_glue(net):
    print('== glue passes (mlfg_net amd vs mlfg_glue MODE=glsl, and vs the dump_a GPU dumps)')
    sz = sizes(DUMP_A)
    g = N.Glue(True)
    mlfg_glue.MODE = 'glsl'
    S1, S10, S23, S24, S38, S39, S53, S54, S69 = (sz(i) for i in (1, 10, 23, 24, 38, 39, 53, 54, 69))
    b, a = load(DUMP_A, 17), load(DUMP_A, 18)
    net_in = rt(b, 'net_in', *S1)
    y = g.pass18(net_in, S1, S10)
    rep('pass18 vs glue', y[0], mlfg_glue.pass18(net_in[0].cpu(), S1, S10)); rep('pass18 vs dump', y, rt(a, 'in_q', *S10))
    b, a = load(DUMP_A, 53), load(DUMP_A, 54)
    net_in = rt(b, 'net_in', *S1); f54 = rt(b, 'f54', *S23)
    fl, x = g.pass54(net_in, f54, S1, S23, S24, 0.5)
    rfl, rx = mlfg_glue.pass54(net_in[0].cpu(), f54[0].cpu(), S1, S23, S24)
    rep('pass54 flow vs glue', fl[0], rfl); rep('pass54 x vs glue', x[0], rx)
    rep('pass54 flow vs dump', fl, rt(a, 'flow54', 2 * S24[0], 2 * S24[1])); rep('pass54 x vs dump', x, rt(a, 'x54', *S24))
    b, a = load(DUMP_A, 82), load(DUMP_A, 83)
    net_in = rt(b, 'net_in', *S1); f83 = rt(b, 'f83', *S38); fl54 = rt(b, 'flow54', *S39)
    fl, x = g.pass83(net_in, f83, fl54, S1, S38, S39, 0.5)
    rfl, rx = mlfg_glue.pass83(net_in[0].cpu(), f83[0].cpu(), fl54[0].cpu(), S1, S38, S39)
    rep('pass83 flow vs glue', fl[0], rfl); rep('pass83 x vs glue', x[0], rx)
    rep('pass83 flow vs dump', fl, rt(a, 'flow83', *S39)); rep('pass83 x vs dump', x, rt(a, 'x83', *S39))
    b, a = load(DUMP_A, 111), load(DUMP_A, 112)
    net_in = rt(b, 'net_in', *S53); f112 = rt(b, 'f112', *S54); fl83 = rt(b, 'flow83', *S54)
    fl, x = g.pass112(net_in, f112, fl83, S53, S54, 0.5)
    rfl, rx = mlfg_glue.pass112(net_in[0].cpu(), f112[0].cpu(), fl83[0].cpu(), S53, S54)
    rep('pass112 flow vs glue', fl[0], rfl); rep('pass112 x vs glue', x[0], rx)
    rep('pass112 flow vs dump', fl, rt(a, 'flow112', *S54)); rep('pass112 x vs dump', x, rt(a, 'x112', *S54))
    b, a = load(DUMP_A, 148), load(DUMP_A, 149)
    fo = g.pass149(rt(b, 'f149', *S69), rt(b, 'flow112', *S69))
    rep('pass149 vs dump', fo, rt(a, 'flowout', *S69))
    mlfg_glue.MODE = 'nv'


# ------------------------------------------------------------------------------------------------ blocks
BLOCKS = [  # name, dump before, first, last, input (glue name, size idx, C), output (glue name, size idx, C)
    ('encA', 1, 2, 8, ('net_in', 1, 18), ('net_in', 5, 18)),
    ('encB', 9, 10, 16, ('net_in', 1, 18), ('net_in', 9, 18)),
    ('block1', 19, 20, 52, ('in_q', 10, 18), ('f54', 23, 13)),
    ('block2', 56, 57, 81, ('x54', 24, 32), ('f83', 38, 13)),
    ('block3', 85, 86, 110, ('x83', 39, 32), ('f112', 53, 13)),
    ('block4', 114, 115, 147, ('x112', 54, 32), ('f149', 68, 13)),
]


def run_block(net, name, x):
    if name == 'encA': return torch.cat([x[:, :6], net.encA(x[:, 0:3], True), x[:, 12:18]], 1)
    if name == 'encB': return torch.cat([x[:, :12], net.encB(x[:, 3:6], True)], 1)
    return getattr(net, name)(x, True)


def part_blocks(net):
    print('== conv blocks: mlfg_net amd vs Emulator (NV_LANES False = AMD semantics), and vs dump_a (NVIDIA GPU)')
    em = Emulator(CAP, NPZ, dev)
    sz = sizes(DUMP_A)
    for name, d0, a, b, (iname, isz, ic), (oname, osz, oc) in BLOCKS:
        x = rt(load(DUMP_A, d0), iname, *sz(isz), ic)
        y = run_block(net, name, x)
        res = {}
        for lanes in (False, True):
            mlfg_model.NV_LANES = lanes
            s = load(DUMP_A, d0).to(dev)
            em.run_passes(s, a, b)
            res[lanes] = rt(s, oname, *sz(osz), oc)
        mlfg_model.NV_LANES = False
        ref = rt(load(DUMP_A, b), oname, *sz(osz), oc)
        rep(f'{name} ({a}-{b}) vs emulator', y, res[False])
        rep(f'{name} vs dump (NVIDIA)', y, ref)
        if name == 'block4':
            net.block4.nv_lanes = True
            rep('block4 nv_lanes=True vs dump', run_block(net, name, x), ref)
            rep('emulator NV_LANES=True vs dump', res[True], ref)
            net.block4.nv_lanes = False


# ------------------------------------------------------------------------------------------------ whole stage 1
def glue_hooks(em, sz):
    S = sz
    S1, S10, S23, S24, S38, S39, S53, S54, S69 = (S(i) for i in (1, 10, 23, 24, 38, 39, 53, 54, 69))
    G = mlfg_glue

    def r(s, n, w, h, c=None): return rt(s, n, w, h, c)[0].cpu()

    def p18(s, ctx): wt(s, 'in_q', G.pass18(r(s, 'net_in', *S1), S1, S10), *S10)

    def p54(s, ctx):
        fl, x = G.pass54(r(s, 'net_in', *S1), r(s, 'f54', *S23), S1, S23, S24)
        wt(s, 'flow54', fl, 2 * S24[0], 2 * S24[1]); wt(s, 'x54', x, *S24)

    def p83(s, ctx):
        fl, x = G.pass83(r(s, 'net_in', *S1), r(s, 'f83', *S38), r(s, 'flow54', *S39), S1, S38, S39)
        wt(s, 'flow83', fl, *S39); wt(s, 'x83', x, *S39)

    def p112(s, ctx):
        fl, x = G.pass112(r(s, 'net_in', *S53), r(s, 'f112', *S54), r(s, 'flow83', *S54), S53, S54)
        wt(s, 'flow112', fl, *S54); wt(s, 'x112', x, *S54)

    def p149(s, ctx): wt(s, 'flowout', G.pass149(r(s, 'f149', *S69), r(s, 'flow112', *S69), S69), *S69)
    em.glue.update({18: p18, 54: p54, 83: p83, 112: p112, 149: p149})


def part_chain(net):
    print('== stage 1 end to end from pass-0 output (dump_a frame 2): mlfg_net vs Emulator+glue chain, vs dump')
    sz = sizes(DUMP_A)
    em = Emulator(CAP, NPZ, dev)
    glue_hooks(em, sz)
    msz = N.mlfg_sizes(1920, 1080)
    x0 = rt(load(DUMP_A, 0), 'net_in', *sz(0))
    S69 = sz(69)
    for lanes, glue_mode in ((False, 'glsl'), (True, 'nv')):
        mlfg_model.NV_LANES = lanes; mlfg_glue.MODE = glue_mode
        s = load(DUMP_A, 0).to(dev)
        em.run_passes(s, 1, 149)
        ref_em = rt(s, 'flowout', *S69)
        net.block4.nv_lanes = lanes
        fl, (f0, f1), aux = net.stage1(None, None, 0.5, msz, x6=x0[:, :6])
        tag = 'NV lanes + nv glue' if lanes else 'AMD semantics'
        rep(f'stage1 flow vs emulator chain ({tag})', fl, ref_em)
        if lanes:
            rep('stage1 flow (nv_lanes) vs dump_a 149', fl, rt(load(DUMP_A, 149), 'flowout', *S69))
            rep('emulator chain (NV) vs dump_a 149', ref_em, rt(load(DUMP_A, 149), 'flowout', *S69))
    net.block4.nv_lanes = False
    mlfg_model.NV_LANES = False; mlfg_glue.MODE = 'nv'


# ------------------------------------------------------------------------------------------------ io captures
def hw(x):
    return x.permute(2, 0, 1)[None].to(dev).float()


def packed(img):
    return (img[..., 1].to(torch.int64) << 32) | img[..., 0].to(torch.int64)


def part_io(net):
    print('== prepass / stage-1 flows / pass152 / pass190 vs the io_b / io_c GPU image dumps (point sampler)')
    net.sampler = 'point'
    img, _ = mlfg_io.load_images(IO_B)
    disp = mlfg_io.frame_dispatches(IO_B, 2)
    cbs = [d[3] for d in disp if d[1] == 'main']
    msz = N.mlfg_sizes(1920, 1080)
    # pipe191
    mv, d = N.pipe191_dilate(hw(img[(0, 0)])[:, :2], hw(img[(0, 1)]), msz['pre'], (1.0, 1.0), True, True)
    rep('pipe191 dilated mv', mv, hw(img[(0, 128)])[:, :2]); rep('pipe191 depth', d, hw(img[(0, 129)]))
    # pipe192 x3, pipe193
    for s, (proj, uv) in ((1, (1.0, 0)), (3, (0.5, 1)), (4, (0.5, 0))):
        assert cbs[s]['projAmount'] == proj and cbs[s]['flowIsUv'] == uv
        sp = N.pipe192_splat(hw(img[(s, 0)])[:, :2], hw(img[(s, 1)]), proj, uv)
        ref = packed(img[(s, 128)]).to(dev)[None]
        print(f'pipe192 splat seq {s} (proj {proj}, uv {uv}): exact {bool((sp == ref).all())}, '
              f'mismatching texels {(sp != ref).float().mean().item() * 100:.4f}%')
    cb = cbs[2]
    A = torch.tensor(cb['prevPstoPs']).view(1, 4, 4).to(dev); B = torch.tensor(cb['prevPsToPrev2Ps']).view(1, 4, 4).to(dev)
    fw = N.pipe193_forward_mv(hw(img[(2, 0)])[:, :2], hw(img[(2, 1)]), packed(img[(2, 2)]).to(dev)[None], A, B, (1920, 1080))
    rep('pipe193 forward mv', fw, hw(img[(2, 128)])[:, :2])
    # whole prepass chain from the raw inputs + history images
    pre = net.prepass(hw(img[(0, 0)])[:, :2], hw(img[(0, 1)]), N.MlfgHistory(hw(img[(2, 0)])[:, :2], hw(img[(2, 1)])), A, B, 0.5, (1920, 1080))
    for k, s in (('splat1to0', 1), ('splat0t', 3), ('splat1t', 4)):
        ref = packed(img[(s, 128)]).to(dev)[None]
        print(f'prepass chain {k}: exact {bool((pre[k] == ref).all())}, mismatching {(pre[k] != ref).float().mean().item() * 100:.4f}%')
    rep('prepass chain fwd0', pre['fwd0'], hw(img[(2, 128)])[:, :2])
    # stage 1 from the colour images
    rgb0, rgb1 = hw(img[(6, 1)]), hw(img[(6, 2)])
    for lanes in (False, True):
        net.block4.nv_lanes = lanes
        fl, (f0, f1), _ = net.stage1(rgb0, rgb1, 0.5, msz)
        t = 'nv_lanes' if lanes else 'AMD lanes'
        rep(f'stage1 flow_0to05 ({t}) vs pass151 image', f0, hw(img[(5, 129)]))
        rep(f'stage1 flow_1to05 ({t}) vs pass151 image', f1, hw(img[(5, 130)]))
        rep(f'stage1 flowout ({t}) vs scratch 150', fl, rt(load(IO_B, 150), 'flowout', 256, 160))
    net.block4.nv_lanes = False
    # pass152 with the GPU's own inputs
    sp0 = N.splat_decode(packed(img[(6, 5)]).to(dev)[None]); sp1 = N.splat_decode(packed(img[(6, 6)]).to(dev)[None])
    x2 = net.pass152(rgb0, rgb1, hw(img[(6, 3)]), hw(img[(6, 4)]), sp0, sp1, msz)
    ref = mlfg_io.read_pass152(load(IO_B, 152).numpy().tobytes(), mlfg_ref.sizes_table(IO_B))
    rep('pass152 (24 ch) vs scratch 152', x2[0, :, :270], ref.permute(2, 0, 1)[:, :270])
    e = (x2[0, :, :270].cpu() - ref.permute(2, 0, 1)[:, :270]).abs().amax((1, 2))
    print('    per-channel max err', [round(v, 5) for v in e.tolist()])
    # pass190 with io_c's GPU weights and inputs
    img_c, _ = mlfg_io.load_images(IO_C)
    s = max(k[0] for k in img_c)
    w6 = mlfg_io.read_pass188_weights(load(IO_C, 189).numpy().tobytes(), mlfg_ref.sizes_table(IO_C))
    w5 = w6.permute(2, 0, 1)[None, :5].to(dev)
    sp0 = N.splat_decode(packed(img_c[(s, 5)]).to(dev)[None]); sp1 = N.splat_decode(packed(img_c[(s, 6)]).to(dev)[None])
    out = net.pass190(F_pad(w5, msz['s2']), hw(img_c[(s, 1)]), hw(img_c[(s, 2)]), hw(img_c[(s, 3)]), hw(img_c[(s, 4)]), sp0, sp1, 0.5, msz)
    o8 = torch.floor(out.clamp(0, 1) * 255 + 0.5)
    rep('pass190 composite vs image (unorm8 levels)', o8, hw(img_c[(s, 129)])[:, :3] * 255)
    net.sampler = 'linear'


def F_pad(w, s2):
    return torch.nn.functional.pad(w, (0, s2[0] - w.shape[3], 0, s2[1] - w.shape[2]))


# ------------------------------------------------------------------------------------------------ stage 2
def part_stage2(net):
    print('== stage 2 (passes 153-189) vs Emulator from the io_b pass-152 scratch (emulator weights for 186/188 '
          'transposed to the bias-consistent orientation; FP8 passes are undefined on NVIDIA, so no GPU reference)')
    em = Emulator(CAP, NPZ, dev)
    em.w['L186_w'] = em.w['L186_w'].permute(1, 0, 2, 3).contiguous()   # -> [O=32, I=48]
    em.w['L188_w'] = em.w['L188_w'].permute(1, 0, 2, 3).contiguous()   # -> [O=16, I=32]
    s = load(IO_B, 152).to(dev)
    x2 = mlfg_io.read_pass152(s.cpu().numpy().tobytes(), mlfg_ref.sizes_table(IO_B)).permute(2, 0, 1)[None].to(dev)
    em.run_passes(s, 153, 189)
    g = em.graph
    st = net.stage2
    # intermediates
    x = N.q16(torch.relu(st.inp(x2)))
    W2, H2 = em.size(73)
    rd = lambda p, W_, H_: read(s, [t for t in g[p]['tensors'] if t['kind'] in ('Tensor4', 'Blocked5')][-1], W_, H_,
                                [t for t in g[p]['tensors'] if t['kind'] in ('Tensor4', 'Blocked5')][-1]['logical'][2])[None]
    y = st(x2, True)
    w_em = rd(188, *em.size(90))
    rep('stage2 softmax weights vs emulator', y, w_em[:, :5])
    # one more consistency check: the 184 output (fp8, 48 ch)
    with torch.no_grad():
        x = N.q16(torch.relu(st.inp(x2))); x = N.q8(st.d1(x))
        for b in st.b1: x = b(x, True)
        skip = x; x = N.q8(st.d2(x))
        for b in st.b2[:-1]: x = b(x, True)
        x = st.b2[-1](x, True, skip=skip)
        for b in st.b3: x = b(x, True)
    rep('stage2 pass-184 output (fp8) vs emulator', x, rd(184, *em.size(88)))
    print('    weights mean', [round(v, 4) for v in y.mean((0, 2, 3)).tolist()], ' sum range',
          (y.sum(1).min().item(), y.sum(1).max().item()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--part', default='all')
    a = ap.parse_args()
    net = N.AmdMlfgNet(precision='amd').to(dev).eval()
    with torch.no_grad():
        for p in (['glue', 'blocks', 'chain', 'io', 'stage2'] if a.part == 'all' else a.part.split(',')):
            globals()['part_' + p](net)


if __name__ == '__main__':
    main()
