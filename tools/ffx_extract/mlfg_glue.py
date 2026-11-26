"""Exact PyTorch transcriptions of the non-convolution "glue" compute passes of AMD FSR Frame Generation 4's ML network
(FidelityFX SDK v2.4.0, mlfi_4k_pass{0,18,54,83,112,149,151}), checked against the GPU's own scratch-buffer dumps.

    python mlfg_glue.py [--capture D:/nfrux_ffx/capture/nv_1080p_spoof] [--dumps D:/nfrux_ffx/capture/dump_a/passes]
                        [--frame 2] [--mode nv|glsl]

Result on dump_a frame 2 with --mode nv: every pass bit-exact (0 mismatched elements).

Conventions
  * Tensors are torch float32 [C, H, W] holding fp16-representable values (the scratch buffer stores fp16).
  * Scratch addressing: byte = offset + (x+1)*sx + (y+1)*sy + c*2  (1-pixel zero halo).
  * "Level 0" = 1/8 of the display resolution (240x135 live for 1080p; the runtime size table pads it to 256x160 and
    every tensor is zero beyond 240x135). Flows are in pixels of the level they are stored at, (dx, dy).
  * fp16 rounding points: MODE='glsl' reproduces the decompiled shader literally (each lerp of a bilinear tap computed
    in fp32 and rounded to fp16, horizontal pair first, then vertical; fp16 sums rounded after every add).
    MODE='nv' (default) reproduces what the capture GPU actually computed -- see r16 / mix / div_s below.
    The two differ by <= ~2 fp16 ulp per output.
  * The two "frames" of the network input: image 0 = r_input_rgb0 (ch 0-2), image 1 = r_input_rgb1 (ch 3-5).
    Flow A (channels 0,1 of every flow tensor) warps image 0 + its features, flow B (channels 2,3) warps image 1.
    The final flows are written to rw_output_flow_0to05 / rw_output_flow_1to05, i.e. A = flow from the
    interpolated (t=0.5) frame into frame 0, B = into frame 1 (backward-warp convention: sample frame at p + flow).
"""
import os, sys, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlfg_ref  # noqa: E402  (sizes_table)

# ----------------------------------------------------------------------------------------------------------------
# scratch-buffer layout of the tensors these passes touch: name -> (byte offset, sx, sy, channels)
L = {
    'net_in':  (0,        36, 17352, 18),  # level 0: rgb0(3) rgb1(3) zeros(12)        -- written by pass 0
    'in_q':    (5032080,  36, 4680,  18),  # level 2 (1/4 of level 0) copy of net_in    -- pass 18
    'f54':     (5774736,  32, 4160,  16),  # level 2 decoder head (flow residual etc.)  -- read by pass 54
    'x54':     (6082576,  64, 16512, 32),  # level 1 warped stack                       -- pass 54
    'flow54':  (8493328,  8,  3856,  4),   # level 0 flow after pass 54                 -- pass 54
    'f83':     (5032080,  32, 8256,  16),  # level 1 decoder head                       -- read by pass 83
    'flow83':  (6237456,  8,  3856,  4),   # level 0 flow after pass 83                 -- pass 83
    'x83':     (9611568,  64, 30848, 32),  # level 0 warped stack                       -- pass 83
    'f112':    (14665904, 32, 15424, 16),  # level 0 decoder head                       -- read by pass 112
    'x112':    (19138864, 64, 30848, 32),  # level 0 warped stack                       -- pass 112
    'flow112': (5032080,  8,  3856,  4),   # level 0 flow after pass 112                -- pass 112
    'f149':    (9484592,  32, 15424, 16),  # level 0 final head (ch0-3 = last residual) -- read by pass 149
    'flowout': (6150320,  8,  3856,  4),   # level 0 final flow                         -- pass 149
}


def q16(x):
    return x.to(torch.float16).to(torch.float32)


# Rounding of *intermediate* fp16 values (the GLSL's float16_t(mix(..)), fp16 sums, f16 flow used for addressing).
# MODE 'glsl': round at every point the decompiled shader does (what an AMD GPU / a strict compiler does).
# MODE 'nv'  : what the capture GPU (NVIDIA RTX 4070, driver 610.88) actually computes, found empirically: it drops
#              the fp16 rounding of the two *horizontal* lerps of every bilinear tap (keeps them fp32) and keeps all
#              other fp16 roundings (tags in NV_ROUND); plus fma-mix and approximate-rcp division (below).
#              With MODE='nv' every pass here matches the dump bit-exactly.
MODE = 'nv'
NV_ROUND = {'flow', 'lerp_v', 'sum'}


def r16(x, tag=''):
    return q16(x) if (MODE == 'glsl' or tag in NV_ROUND) else x


def mix(a, b, t):
    """GLSL mix in fp32. 'nv': fma(t, b-a, a) (b-a rounded to fp32, then one fused multiply-add -- emulated in fp64,
    which is exact for the product; the final fp64->fp32 step can in theory double-round, never observed).
    'glsl': plain a + t*(b-a). The caller rounds to fp16 where the shader does."""
    if MODE == 'nv':
        return (t.double() * (b - a).double() + a.double()).float()
    return a + t * (b - a)


# NVIDIA computes x / S as x * rcp(S) with its approximate MUFU.RCP. rcp(S) differs from the correctly rounded
# 1/S by NV_RCP_ULP[S] fp32 ulps; only the sizes of this capture are known (255 = 256-1 -> -1 ulp, 159 -> 0).
NV_RCP_ULP = {255.0: -1, 159.0: 0}


def div_s(x, S):
    """x / S as the capture GPU does it ('nv') or exactly ('glsl'). S: tensor [2,1,1] float32."""
    if MODE != 'nv':
        return x / S
    r = 1.0 / S
    up = torch.nextafter(r, torch.full_like(r, float('inf'))) - r
    k = torch.tensor([float(NV_RCP_ULP.get(float(s), 0)) for s in S.flatten()]).view_as(S)
    return x * (r + k * up)


def read_t(buf, name, W, H, C=None):
    """Scratch tensor `name` -> float32 [C, H, W] (live region W x H, halo stripped)."""
    off, sx, sy, Cn = L[name]
    C = Cn if C is None else C
    h = np.frombuffer(buf, np.float16)
    idx = (off // 2 + (np.arange(H)[None, :, None] + 1) * (sy // 2) + (np.arange(W)[None, None, :] + 1) * (sx // 2)
           + np.arange(C)[:, None, None])
    return torch.from_numpy(h[idx].astype(np.float32))


# ----------------------------------------------------------------------------------------------------------------
# sampling helpers

def _taps(src, ix0, iy0, ix1, iy1):
    """src [C,H,W]; integer index maps [h,w] -> four taps [C,h,w] (a=x0y0, b=x1y0, c=x0y1, d=x1y1)."""
    return src[:, iy0, ix0], src[:, iy0, ix1], src[:, iy1, ix0], src[:, iy1, ix1]


def bilerp16(a, b, c, d, fx, fy):
    """The shader's bilinear: fp16(mix(fp16(mix(a,b,fx)), fp16(mix(c,d,fx)), fy))."""
    return r16(mix(r16(mix(a, b, fx), 'lerp_h'), r16(mix(c, d, fx), 'lerp_h'), fy), 'lerp_v')


def resample(src, Sw, Sh, OW, OH, scale):
    """Separable-free bilinear resize used by passes 18/54/83: sample point p = max((o+0.5)*scale-0.5, 0),
    i0 = clamp(floor p, 0, S-1), i1 = clamp(i0+1, 0, S-1), f = clamp(p-i0, 0, 1) (clamp-to-edge, align_corners=False).
    src [C,>=Sh,>=Sw]; returns [C, OH, OW] fp16-rounded."""
    def axis(n, S):
        p = torch.clamp((torch.arange(n, dtype=torch.float32) + 0.5) * scale - 0.5, min=0.0)
        i0 = torch.clamp(torch.floor(p).long(), 0, S - 1)
        i1 = torch.clamp(i0 + 1, 0, S - 1)
        f = torch.clamp(p - i0.float(), 0.0, 1.0)
        return i0, i1, f
    x0, x1, fx = axis(OW, Sw)
    y0, y1, fy = axis(OH, Sh)
    X0, Y0 = x0[None, :].expand(OH, OW), y0[:, None].expand(OH, OW)
    X1, Y1 = x1[None, :].expand(OH, OW), y1[:, None].expand(OH, OW)
    FX, FY = fx[None, :].expand(OH, OW), fy[:, None].expand(OH, OW)
    return bilerp16(*_taps(src, X0, Y0, X1, Y1), FX, FY)


def warp(src, flow, Sw, Sh):
    """Backward warp: out(p) = bilinear(src, p + flow(p)), p = integer pixel coords (no +0.5), clamp-to-edge on
    [0,S-1]. Position math reproduces the shader literally in fp32: ((p+f)*2/(S-1))*0.5*(S-1).
    flow [2,H,W] fp16 values; src [C,>=Sh,>=Sw]."""
    H, W = flow.shape[1:]
    S = torch.tensor([Sw - 1.0, Sh - 1.0], dtype=torch.float32).view(2, 1, 1)
    gy, gx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
    pos = ((div_s((torch.stack([gx, gy]) + flow) * 2.0, S)) * 0.5) * S
    fl = torch.floor(pos)
    fr = pos - fl
    i = fl.long()
    x0 = torch.clamp(i[0], 0, Sw - 1); x1 = torch.clamp(i[0] + 1, 0, Sw - 1)
    y0 = torch.clamp(i[1], 0, Sh - 1); y1 = torch.clamp(i[1] + 1, 0, Sh - 1)
    return bilerp16(*_taps(src, x0, y0, x1, y1), fr[0], fr[1])


def warp_pair(net_in, flow, Sw, Sh):
    """Warp both frames of the level-0 input: A = flow[0:2] warps rgb0 (ch0-2) + feat0 (ch6-11),
    B = flow[2:4] warps rgb1 (ch3-5) + feat1 (ch12-17). Returns (wa [9,...] = rgb0,feat0 ; wb [9,...] = rgb1,feat1)."""
    a = warp(net_in[[0, 1, 2, 6, 7, 8, 9, 10, 11]], flow[0:2], Sw, Sh)
    b = warp(net_in[[3, 4, 5, 12, 13, 14, 15, 16, 17]], flow[2:4], Sw, Sh)
    return a, b


def stack32(wa, wb, head9, flow):
    """The 32-channel stack written by passes 54/83/112 (before any averaging):
       0-2 warped rgb0 | 3-5 warped rgb1 | 6-11 warped feat0 | 12-17 warped feat1 | 18 const 0.5 |
       19-27 head channels 4..12 (9 ch) | 28-31 flow (A.x, A.y, B.x, B.y)."""
    H, W = flow.shape[1:]
    return torch.cat([wa[0:3], wb[0:3], wa[3:9], wb[3:9], torch.full((1, H, W), 0.5), head9, flow], 0)


# ----------------------------------------------------------------------------------------------------------------
# pass 0

def synth_frame(frame, W=1920, H=1080, rounding='trunc'):
    """The synthetic RGBA8 colour texture used for the dump_a capture (u = x - 8*frame). -> uint8 [3,H,W]."""
    x = np.arange(W)[None, :]; y = np.arange(H)[:, None]
    u = x - 8 * frame
    f = np.float32  # the capture generated the pattern in fp32 (matters where R lands next to an integer)
    r = f(128) + f(100) * np.sin(u.astype(f) * f(0.05)) * np.cos(y.astype(f) * f(0.031))
    r = np.floor(r) if rounding == 'trunc' else np.round(r)
    g = np.where((((u >> 4) ^ (y >> 4)) & 1) != 0, 200, 40) + 0 * y
    b = (u * 3 + y) & 255
    return np.stack([np.broadcast_to(r, (H, W)), np.broadcast_to(g, (H, W)), np.broadcast_to(b, (H, W))]).astype(np.uint8)


def _nearest_index(n, size):
    """Texel index the capture GPU fetched for uv = (i+0.5)/(size/8) with the (unbound -> point) sampler:
    uv = (i+0.5)*rcp(size/8) in fp32, quantised by truncation to 21 fractional bits, times size, floored.
    Gives 8i+3 except where fp32 rounding of uv lands on/above the texel corner (then 8i+4). Fitted to this
    capture (exact for every row and column)."""
    uv = (np.arange(n, dtype=np.float32) + np.float32(0.5)) * (np.float32(1) / (np.float32(size) * np.float32(0.125)))
    return np.floor(np.floor(uv.astype(np.float64) * 2.0 ** 21) / 2.0 ** 21 * size).astype(np.int64)


def pass0(rgb0, rgb1, live_w, live_h, out_w=None, out_h=None, filt='linear'):
    """Network input. Each colour texture (UNORM in [0,1], [3,H,W] float; rgb0 = r_input_rgb0, rgb1 = r_input_rgb1 --
    in this capture rgb0 held the OLDER frame (index 1) and rgb1 the NEWER one (index 2)) is sampled ONCE per level-0
    pixel with g_sampler_linear at uv = (x+0.5)/(tex/8). That point is the corner shared by texels 8x+3..8x+4 /
    8y+3..8y+4, so with the intended linear clamp sampler it is a 2x2 box average -- an 8x downsample that only looks
    at the central 2x2 of each 8x8 block (filt='linear').
    In the dump_a capture the sampler at binding 320 was not bound by the capture harness and the GPU point-sampled
    (filt='capture_nearest' reproduces that: texel (8x+3|4, 8y+3|4), see _nearest_index).
    backBufferTransferFunction 0/1 = passthrough (2 = PQ encode of scRGB, not reproduced: unused here).
    Output [18, out_h, out_w], fp16: ch0-2 rgb0, ch3-5 rgb1, ch6-17 zero (they are filled later by the feature
    encoder passes 2..16, which write into the same tensor); zero outside live_w x live_h (= min(tensorSizes[0],
    tex/8)) and outside generationRect (here [0,1]^2). Scratch 'net_in' (offset 0, 18 ch, tensorSizes[0])."""
    out_w = out_w or live_w; out_h = out_h or live_h
    out = torch.zeros(18, out_h, out_w)
    for k, img in enumerate((rgb0, rgb1)):
        img = img.float()
        if filt == 'linear':
            tl = img[:, 3::8, 3::8][:, :live_h, :live_w]; tr = img[:, 3::8, 4::8][:, :live_h, :live_w]
            bl = img[:, 4::8, 3::8][:, :live_h, :live_w]; br = img[:, 4::8, 4::8][:, :live_h, :live_w]
            v = ((tl + tr) * 0.5 + (bl + br) * 0.5) * 0.5
        else:
            iy = torch.from_numpy(_nearest_index(live_h, img.shape[1])); ix = torch.from_numpy(_nearest_index(live_w, img.shape[2]))
            v = img[:, iy][:, :, ix]
        out[3 * k:3 * k + 3, :live_h, :live_w] = q16(v)
    return out

# ----------------------------------------------------------------------------------------------------------------
# pass 18

def pass18(net_in, S1, S10):
    """4x bilinear downsample of the 18-channel network input to level 2 (the coarsest flow level):
    sample at (x+0.5)*4-0.5 = 4x+1.5 -> equal-weight average of the 2x2 texels 4x+1..4x+2 (clamped to
    tensorSizes[1]-1). In: 'net_in' (tensorSizes[1]). Out: 'in_q' [18, S10.h, S10.w] (tensorSizes[10])."""
    return q16(resample(net_in, S1[0], S1[1], S10[0], S10[1], 4.0))


# ----------------------------------------------------------------------------------------------------------------
# pass 54 / 83 / 112 : flow refinement + warping at three levels

def pass54(net_in, f54, S1, S23, S24):
    """Coarse-to-fine step level 2 -> level 1. Computed at level-0 resolution and box-averaged 2x2 to level 1.
      flow0 = 4 * bilinear_up4(f54[0:4])           (level-2 flow in level-2 px, x4 -> level-0 px; f54 at tensorSizes[23])
      warp rgb0/feat0 by flow0.A and rgb1/feat1 by flow0.B (level-0 'net_in', clamp to tensorSizes[1])
      head = bilinear_up4(f54[4:13]) (9 channels passed through upsampled)
    Outputs: 'flow54' = flow0 [4, 2*S24] (level-0 px); 'x54' [32, S24] = fp16 2x2 sum-then-/4 of
    stack32(warpA, warpB, head, flow0) with the flow channels additionally /2 (-> level-1 px) and ch18 = 0.5.
    Sum order: ((s00 + s10) + s01) + s11."""
    W0, H0 = 2 * S24[0], 2 * S24[1]
    up = resample(f54[:13], S23[0], S23[1], W0, H0, 0.25)
    flow = r16(4.0 * up[0:4], 'flow')
    wa, wb = warp_pair(net_in, flow, S1[0], S1[1])
    st = stack32(wa, wb, up[4:13], flow)
    s = st[:, 0::2, 0::2]
    for dy, dx in ((0, 1), (1, 0), (1, 1)):
        s = r16(s + st[:, dy::2, dx::2], 'sum')
    x = s / 4.0
    x[28:32] = x[28:32] / 2.0
    x = q16(x)
    x[18] = 0.5
    return q16(flow), x


def pass83(net_in, f83, flow54, S1, S38, S39):
    """Coarse-to-fine step level 1 -> level 0.
      flow = 2 * bilinear_up2(f83[0:4]) + flow54   (residual in level-1 px, x2; fp16 mul then fp16 add)
      warp both frames by it (clamp to tensorSizes[1]); head = bilinear_up2(f83[4:13]).
    In: 'f83' (tensorSizes[38]), 'flow54', 'net_in'. Out: 'flow83' [4,S39], 'x83' = stack32(...) [32,S39]."""
    up = resample(f83[:13], S38[0], S38[1], S39[0], S39[1], 0.5)
    flow = r16(r16(2.0 * up[0:4], 'flow') + flow54[:, :S39[1], :S39[0]], 'flow')
    wa, wb = warp_pair(net_in, flow, S1[0], S1[1])
    return q16(flow), q16(stack32(wa, wb, up[4:13], flow))


def pass112(net_in, f112, flow83, S53, S54):
    """Level-0 refinement: flow = f112[0:4] + flow83 (fp16 add, same resolution); warp both frames
    (clamp to tensorSizes[53]); head = f112[4:13] copied. Out: 'x112' = stack32(...) [32,S54] and
    'flow112' = flow [4,S54] (written over the old 'f83' region at offset 5032080)."""
    flow = r16(f112[0:4, :S54[1], :S54[0]] + flow83[:, :S54[1], :S54[0]], 'flow')
    wa, wb = warp_pair(net_in, flow, S53[0], S53[1])
    return q16(flow), q16(stack32(wa, wb, f112[4:13, :S54[1], :S54[0]], flow))


# ----------------------------------------------------------------------------------------------------------------
# pass 149 / 151 : final flow

def pass149(f149, flow112, S69):
    """Final residual: flowout = f149[0:4] + flow112 (fp16 add). Out: 'flowout' [4, S69]."""
    return q16(f149[0:4, :S69[1], :S69[0]] + flow112[:, :S69[1], :S69[0]])


def pass151(flowout, S70):
    """Export: rw_output_flow_0to05 = flowout[0:2] / (W,H), rw_output_flow_1to05 = flowout[2:4] / (W,H), where
    (W,H) = tensorSizes[70] = the true level-0 size (240x135 for 1080p), division in fp32 then fp16.
    Images are VK_FORMAT_R16G16_SFLOAT, W x H (display/8), flow in UV units (fraction of the image), (dx,dy).
    Returns (img0 [2,H,W], img1 [2,H,W])."""
    W, H = int(S70[0]), int(S70[1])
    d = torch.tensor([float(W), float(H)]).view(2, 1, 1)
    f = flowout[:, :H, :W]
    return q16(f[0:2] / d), q16(f[2:4] / d)


# ----------------------------------------------------------------------------------------------------------------
# verification

def ulps(a, b):
    """max distance in fp16 ulps (both exactly fp16)."""
    ia = torch.from_numpy(a.numpy().astype(np.float16).view(np.int16).astype(np.int32))
    ib = torch.from_numpy(b.numpy().astype(np.float16).view(np.int16).astype(np.int32))
    ia = torch.where(ia < 0, -32768 - ia, ia); ib = torch.where(ib < 0, -32768 - ib, ib)
    return (ia - ib).abs().max().item()


def report(name, y, ref):
    d = (y - ref).abs()
    n_bad = (d > 0).sum().item()
    print(f"pass {name:>8}: max abs err {d.max().item():.3g}  max ulp {ulps(y, ref)}  "
          f"mismatched {n_bad}/{d.numel()}  ref absmax {ref.abs().max().item():.3g}")
    return d.max().item()


def verify(capture, dumps, frame=2):
    log_dir = os.path.dirname(dumps.rstrip('/\\'))
    S = mlfg_ref.sizes_table(log_dir) if os.path.exists(os.path.join(log_dir, 'log.txt')) else None
    if S is None:
        S = mlfg_ref.sizes_table(capture)
    sz = lambda i: (int(S[i][0]), int(S[i][1]))
    have = sorted(int(f.split('_')[1][:3]) for f in os.listdir(dumps) if f.startswith(f'{frame}_'))
    dump = lambda n: open(os.path.join(dumps, f'{frame}_{n:03d}.bin'), 'rb').read()
    before = lambda p: dump(max(h for h in have if h < p))
    res = {}

    # pass 0: no texture in the dump -> rebuild the synthetic frames.
    a0 = dump(0)
    S0 = sz(0)
    ref = read_t(a0, 'net_in', S0[0], S0[1])
    lw, lh = min(S0[0], 1920 // 8), min(S0[1], 1080 // 8)
    best = None
    for rnd in ('trunc', 'round'):
        cur = torch.from_numpy(synth_frame(frame, rounding=rnd).astype(np.float32)) / 255.0
        prev = torch.from_numpy(synth_frame(frame - 1, rounding=rnd).astype(np.float32)) / 255.0
        for tag, (r0, r1) in (('rgb0=cur', (cur, prev)), ('rgb0=prev', (prev, cur))):
            for filt in ('linear', 'capture_nearest'):
                y = pass0(r0, r1, lw, lh, S0[0], S0[1], filt)
                e = (y - ref).abs().max().item()
                print(f"  pass 0 candidate {tag}, R {rnd}, {filt}: max err {e:.4g}")
                if best is None or e < best[0]: best = (e, f'{tag}, R {rnd}, {filt}', y)
    print(f"pass 0 synthetic match: best = {best[1]}; ch6-17 absmax {ref[6:].abs().max().item()}; "
          f"padding absmax {ref[:, lh:, :].abs().max().item()}, {ref[:, :, lw:].abs().max().item()}")
    res[0] = report('0', best[2], ref)

    # pass 18
    b, a = before(18), dump(18)
    net = read_t(b, 'net_in', *sz(1))
    res[18] = report('18', pass18(net, sz(1), sz(10)), read_t(a, 'in_q', *sz(10)))

    # pass 54
    b, a = before(54), dump(54)
    S24 = sz(24)
    flow, x = pass54(read_t(b, 'net_in', *sz(1)), read_t(b, 'f54', *sz(23)), sz(1), sz(23), S24)
    e1 = report('54 flow', flow, read_t(a, 'flow54', 2 * S24[0], 2 * S24[1]))
    e2 = report('54 x', x, read_t(a, 'x54', *S24))
    res[54] = max(e1, e2)

    # pass 83
    b, a = before(83), dump(83)
    S39 = sz(39)
    flow, x = pass83(read_t(b, 'net_in', *sz(1)), read_t(b, 'f83', *sz(38)), read_t(b, 'flow54', *S39),
                     sz(1), sz(38), S39)
    res[83] = max(report('83 flow', flow, read_t(a, 'flow83', *S39)), report('83 x', x, read_t(a, 'x83', *S39)))

    # pass 112
    b, a = before(112), dump(112)
    S54 = sz(54)
    flow, x = pass112(read_t(b, 'net_in', *sz(53)), read_t(b, 'f112', *S54), read_t(b, 'flow83', *S54), sz(53), S54)
    res[112] = max(report('112 flow', flow, read_t(a, 'flow112', *S54)), report('112 x', x, read_t(a, 'x112', *S54)))

    # pass 149
    b, a = before(149), dump(149)
    S69 = sz(69)
    fo = pass149(read_t(b, 'f149', *S69), read_t(b, 'flow112', *S69), S69)
    res[149] = report('149', fo, read_t(a, 'flowout', *S69))

    # pass 151: images are not in the dump; verify its input is the pass-149 output (unchanged since) and report.
    b = before(151)
    fo151 = read_t(b, 'flowout', *S69)
    i0, i1 = pass151(fo151, S[70])
    print(f"pass 151: input 'flowout' unchanged since pass 149: {torch.equal(fo151, read_t(dump(149), 'flowout', *S69))};"
          f" images {tuple(i0.shape)} R16G16_SFLOAT, |uv| max {i0.abs().max().item():.4g} / {i1.abs().max().item():.4g}")
    print('summary (max abs err):', {k: float(f'{v:.3g}') for k, v in res.items()})
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', default=r'D:\nfrux_ffx\capture\nv_1080p_spoof')
    ap.add_argument('--dumps', default=r'D:\nfrux_ffx\capture\dump_a\passes')
    ap.add_argument('--frame', type=int, default=2)
    ap.add_argument('--mode', default='nv', choices=['nv', 'glsl'], help="'nv' = capture GPU's arithmetic (bit-exact), 'glsl' = literal shader roundings")
    a = ap.parse_args()
    MODE = a.mode
    verify(a.capture, a.dumps, a.frame)
