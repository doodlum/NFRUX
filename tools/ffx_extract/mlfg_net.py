"""Clean, trainable PyTorch implementation of AMD FSR Frame Generation 4's ML frame interpolation (FidelityFX SDK v2.4.0,
MIT; see docs/FFX_MLFG.md), generalised to an arbitrary timestep t in (0, 1).

    from mlfg_net import AmdMlfgNet, mlfg_sizes
    net = AmdMlfgNet(precision='train').cuda()            # or precision='amd' (AMD's fp16 / fp8 roundings)
    hist = net.history(prev_mv, prev_depth, display)       # frame 0's dilated mv / depth (pipe191 of the previous frame)
    out, aux = net(prev_rgb, cur_rgb, cur_depth, cur_mv, hist, prevPstoPs, prevPsToPrev2Ps, t)
    next_hist = aux['history']                             # keep for the next frame pair

Everything works on plain [B, C, H, W] tensors (no scratch bytes). Sources of truth: mlfg_model.py (layer semantics),
mlfg_glue.py (stage-1 glue), mlfg_io.py (prepass / stage-2 input / composite), all bit-exact against AMD's GPU dumps;
verify_mlfg_net.py checks this module against them.

Inputs (frame "0" = previous presented frame, "1" = current frame, t = generated time between them)
  prev_rgb, cur_rgb  [B,3,Hd,Wd] display-referred swapchain colour in [0,1] (what the network sees; tf 0/1 passthrough)
  cur_depth          [B,1,Hr,Wr] device depth of frame 1 at render res (depth_inverted=True: 1 = near)
  cur_mv             [B,2,Hr,Wr] (x, y) motion of frame 1 in render pixels * mv_scale; FFX convention mv = p_prev - p_cur
  hist               MlfgHistory(mv [B,2,Hp,Wp] uv, depth [B,1,Hp,Wp]) = pipe191 output of frame 0 (net.history(...))
  prevPstoPs         [B,4,4] frame-0 clip -> frame-1 clip, ROW-vector convention (clip' = [x y z 1] @ M), i.e. the
                     transpose of the usual column-vector VP_1 @ inv(VP_0)
  prevPsToPrev2Ps    [B,4,4] frame-0 clip -> frame-(-1) clip, row-vector (transpose of VP_-1 @ inv(VP_0))
  t                  float timestep; 0.5 reproduces AMD's output
  reset              bool: output = current colour (AMD's reset path)

Network (78 layers, ~5 M parameters, all weights from amd_mlfg_4k.npz):
  pass0      level-0 input: each colour sampled once per 8x8 block (linear sampler -> 2x2 box of the block centre)
  EncA/EncB  passes 2-8 / 10-16, same template: conv2x2/2 3->32 lrelu, conv3x3 32 lrelu x2, convT2x2 32->6.
             EncA sees rgb0 only, EncB rgb1 only (the 16-channel windows AMD reads have zero weights elsewhere).
             net_in (18 ch, level 0) = rgb0 | rgb1 | feat0 | feat1
  pass18     4x bilinear downsample of net_in -> level 2
  FlowBlock  4 coarse-to-fine U-shaped blocks (no skips): conv2x2/2 lrelu, conv3x3 lrelu, conv2x2/2 lrelu,
             8 x residual conv3x3 (lrelu(conv(x)+x)), convT2x2 lrelu, convT2x2 -> 13 ch
             (ch 0-3 flow residual A.xy B.xy, ch 4-12 'head' features carried to the next block's input)
               block1 level2  18->96->192   (passes 20-52)      block2 level1 32->64->128 (57-81)
               block3 level0  32->48->96    (86-110)            block4 level0 32->24->48  (115-147)
  glue 54/83/112/149: flow upsample + residual add, backward warps of rgb/feat of both frames, 32-ch stacks
             (warped rgb0|rgb1|feat0|feat1, T, head 9, flow 4)  <- THE TIMESTEP ENTERS HERE (ch 18, AMD: const 0.5)
  pass151    flow export: A = flow t->0, B = flow t->1 in uv of the level-0 live grid (display/8)
  prepass    pipe191 mv dilation + depth downscale (display/3), pipe192 64-bit forward splats, pipe193 frame-0
             forward motion from camera matrices (fallback: splat of frame-1 mvs onto frame 0)
             splat1to0 (proj 1), splat0->t (proj t), splat1->t (proj 1-t)          <- t enters (AMD: 0.5, 0.5)
  pass152    24-ch stage-2 input at display/4 (splat-warped, flow-warped and plain colours, coverages, luma diffs)
  stage2     FasterNet U-Net (FP8 on AMD): conv3x3 24->32 relu, conv2x2/2 32->48, 4 x FN(48, p16), conv2x2/2
             48->64, 5 x FN(64, p32), convT2x2 64->48 + skip, 4 x FN(48, p16), convT2x2 48->32, conv3x3 32->5 softmax
             FN(C, P): x + proj(relu(expand(cat(pconv3x3(x[:P]), x[P:]))))   (expand = 1x1 C->2C, proj = 1x1 2C->C)
  pass190    composite at display res: 5-way blend of splat-warped (holes filled from the other frame), flow-warped and
             plain colours with the softmax weights.  plain blend generalised from 0.5/0.5 to (1-t)/t   <- t enters

Timestep in stage 1: net.flow_t = 'channel' (default; the stack constant becomes t) or 'scale' (stage 1 at 0.5, final
flows scaled by 2t / 2(1-t)). The network was trained at 0.5 only and follows the t channel just partially (flow
magnitude ratio ~0.7 instead of 0.5 at t=.25), so 'scale' is slightly better at 4x. Identical at t = 0.5.

Precision: 'train' = fp32 everywhere (differentiable; integer/scatter parts of the prepass carry no gradient), 'amd' =
AMD's roundings at the emulator's points (fp16 stores of every fp16 tensor and glue intermediate, fp8 e4m3 stores of the
stage-2 tensors), applied with straight-through gradients when autograd is on.

NB weights of passes 186 and 188 are used in the orientation their bias lengths require ([I=48,O=32] convT,
[I=32,O=16->5] conv); amd_mlfg_4k.json / mlfg_model.Emulator use the other orientation for these two layers (it was
unverifiable on NVIDIA, and the emulator's conv_pass fails on them with a bias-size mismatch).
"""
import math, os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = r'D:\nfrux_ffx\work\amd_mlfg_4k.npz'
F8 = torch.float8_e4m3fn
F16MAX = 65504.0
INVALID2 = 0x7BFF7BFF
LUMA = (0.298999994993210, 0.587000012397766, 0.114000000059605)


# ================================================================================================ sizes
def mlfg_sizes(W, H):
    """All sizes AMD derives from the display size (checked against the runtime size tables of 1920x1080 and 1280x720
    captures). Returns dict of (w, h)."""
    cdiv = lambda a, b: -(-a // b)
    return dict(
        display=(W, H),
        l0_live=(W // 8, H // 8),                               # tensorSizes[70]: flow export grid
        l0=(cdiv(W, 256) * 32, cdiv(H, 256) * 32),              # tensorSizes[0]: level-0 allocation (multiple of 32)
        s2_live=(W // 4, H // 4),
        s2=(cdiv(W, 16) * 4, cdiv(H, 16) * 4),                  # tensorSizes[72]: stage-2 allocation (multiple of 4)
        pre=(W // 3, H // 3),                                   # prepass images (pipe191 dispatchSize)
    )


# ================================================================================================ rounding helpers
def _ste(x, q):
    return x + (q - x).detach() if (torch.is_grad_enabled() and x.requires_grad) else q


def q16(x, on=True):
    return _ste(x, x.to(torch.float16).float()) if on else x


def q8(x, on=True):
    return _ste(x, x.to(F8).float()) if on else x


def leaky(x):
    return F.leaky_relu(x, 0.2)


# ================================================================================================ sampling helpers
def _gather(src, ix, iy):
    """src [B,C,H,W]; ix, iy long [B,h,w] (already clamped) -> [B,C,h,w]."""
    B, C, H, W = src.shape
    idx = (iy * W + ix).reshape(B, 1, -1).expand(B, C, -1)
    return src.reshape(B, C, H * W).gather(2, idx).reshape(B, C, *ix.shape[1:])


def _fetch0(src, ix, iy):
    """texelFetch with out-of-range texels reading 0."""
    H, W = src.shape[-2:]
    ok = ((ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)).unsqueeze(1)
    return torch.where(ok, _gather(src, ix.clamp(0, W - 1), iy.clamp(0, H - 1)), torch.zeros((), device=src.device))


def _bc(x, B):
    return x.expand(B, *x.shape[1:]) if x.shape[0] == 1 else x


def sample_linear(img, u, v, frac_bits=None):
    """Bilinear clamp-to-edge at uv (texel centres at (i+0.5)/size). img [B,C,H,W]; u, v [B|1,h,w]."""
    B, C, H, W = img.shape
    u, v = _bc(u, B), _bc(v, B)
    x = u * W - 0.5; y = v * H - 0.5
    x0 = torch.floor(x); y0 = torch.floor(y)
    fx = x - x0; fy = y - y0
    if frac_bits is not None:
        s = float(1 << frac_bits)
        fx = _ste(fx, torch.floor(fx * s + 0.5) / s); fy = _ste(fy, torch.floor(fy * s + 0.5) / s)
    x0 = x0.long(); y0 = y0.long()
    cx = lambda i: i.clamp(0, W - 1); cy = lambda i: i.clamp(0, H - 1)
    a = _gather(img, cx(x0), cy(y0)); b = _gather(img, cx(x0 + 1), cy(y0))
    c = _gather(img, cx(x0), cy(y0 + 1)); d = _gather(img, cx(x0 + 1), cy(y0 + 1))
    fx = fx.unsqueeze(1); fy = fy.unsqueeze(1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


UV_FRAC_BITS = 21


def _point_idx(a, n):
    s = float(1 << UV_FRAC_BITS)
    return torch.floor(torch.floor(a.double() * s) / s * n).long().clamp(0, n - 1)


def sample_point(img, u, v):
    """Point sampling as the capture GPU does it (uv truncated to 21 fractional bits, then floor(uv*size))."""
    B, C, H, W = img.shape
    return _gather(img, _point_idx(_bc(u, B), W), _point_idx(_bc(v, B), H))


def sample_nearest(img, u, v):
    B, C, H, W = img.shape
    return _gather(img, torch.floor(_bc(u, B) * W).long().clamp(0, W - 1), torch.floor(_bc(v, B) * H).long().clamp(0, H - 1))


def texel_uv(W, H, device):
    """(p + 0.5) * fp32 rcp(size) for a W x H grid -> u, v [1,H,W] (the GPU's reciprocal multiply, not a division)."""
    gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    ru = float(np.float32(1.0 / np.float64(W))); rv = float(np.float32(1.0 / np.float64(H)))
    return ((gx.float() + 0.5) * ru)[None], ((gy.float() + 0.5) * rv)[None]


# ================================================================================================ network modules
class Encoder(nn.Module):
    """Passes 2-8 (EncA, rgb0) / 10-16 (EncB, rgb1): 3 -> 6 feature channels at level 0."""

    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(3, 32, 2, 2)
        self.c1 = nn.Conv2d(32, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.up = nn.ConvTranspose2d(32, 6, 2, 2)

    def forward(self, x, amd=False):
        x = q16(leaky(self.down(x)), amd)
        x = q16(leaky(self.c1(x)), amd)
        x = q16(leaky(self.c2(x)), amd)
        return q16(self.up(x), amd)


class FlowBlock(nn.Module):
    """One coarse-to-fine flow block: 2 stride-2 levels, 8 residual convs, 2 transposed convs -> 13 channels."""

    def __init__(self, cin, c1, c2, nres=8, cout=13):
        super().__init__()
        self.d1 = nn.Conv2d(cin, c1, 2, 2)
        self.c1 = nn.Conv2d(c1, c1, 3, padding=1)
        self.d2 = nn.Conv2d(c1, c2, 2, 2)
        self.res = nn.ModuleList([nn.Conv2d(c2, c2, 3, padding=1) for _ in range(nres)])
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.u2 = nn.ConvTranspose2d(c1, cout, 2, 2)
        self.nv_lanes = False  # block 4 only: reproduce NVIDIA's lane bug (zero ch 20-23 of 24-ch tensors)

    def _nv(self, x):
        if self.nv_lanes and x.shape[1] == 24:
            x = x.clone(); x[:, 20:24] = 0
        return x

    def forward(self, x, amd=False):
        x = self._nv(q16(leaky(self.d1(x)), amd))
        x = self._nv(q16(leaky(self.c1(x)), amd))
        x = q16(leaky(self.d2(x)), amd)
        for r in self.res:
            x = q16(leaky(r(x) + x), amd)
        x = self._nv(q16(leaky(self.u1(x)), amd))
        return q16(self.u2(x), amd)


class FasterNet(nn.Module):
    """AMD's FP8 FasterNet block: partial 3x3 conv on the first P channels, 1x1 expand + ReLU, 1x1 project + residual.
    With up=True (pass 176) the result is also upsampled by a 2x2 transposed conv and added to a skip tensor."""

    def __init__(self, C, P, up_out=None):
        super().__init__()
        self.P = P
        self.pconv = nn.Conv2d(P, P, 3, padding=1)
        self.expand = nn.Conv2d(C, 2 * C, 1)
        self.project = nn.Conv2d(2 * C, C, 1)
        self.up = nn.ConvTranspose2d(C, up_out, 2, 2) if up_out else None

    def forward(self, x, amd=False, skip=None):
        pc = q8(self.pconv(x[:, :self.P]), amd)
        h = torch.cat([pc, x[:, self.P:]], 1)
        h = q8(torch.relu(self.expand(h)), amd)
        y = q8(self.project(h) + x, amd)
        if self.up is not None:
            y = q8(self.up(y) + skip, amd)
        return y


class Stage2(nn.Module):
    """Passes 154-188: 24-ch input at display/4 -> 5 softmax blend weights."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Conv2d(24, 32, 3, padding=1)               # 154 fp16, relu
        self.d1 = nn.Conv2d(32, 48, 2, 2)                        # 156 -> fp8
        self.b1 = nn.ModuleList([FasterNet(48, 16) for _ in range(4)])   # 158-164
        self.d2 = nn.Conv2d(48, 64, 2, 2)                        # 166
        self.b2 = nn.ModuleList([FasterNet(64, 32) for _ in range(4)] + [FasterNet(64, 32, up_out=48)])  # 168-176
        self.b3 = nn.ModuleList([FasterNet(48, 16) for _ in range(4)])   # 178-184
        self.up = nn.ConvTranspose2d(48, 32, 2, 2)               # 186 fp8 in, fp16 out
        self.head = nn.Conv2d(32, 5, 3, padding=1)               # 188 softmax

    def forward(self, x, amd=False):
        x = q16(torch.relu(self.inp(x)), amd)
        x = q8(self.d1(x), amd)
        for b in self.b1: x = b(x, amd)
        skip = x
        x = q8(self.d2(x), amd)
        for b in self.b2[:-1]: x = b(x, amd)
        x = self.b2[-1](x, amd, skip=skip)
        for b in self.b3: x = b(x, amd)
        x = q16(self.up(x), amd)
        y = self.head(x)
        if amd:  # AMD: fp16 logits, running max initialised at 0, fp16 store
            y = q16(y)
            e = torch.exp(y - torch.clamp(y.amax(1, keepdim=True), min=0.0))
            return q16(e / e.sum(1, keepdim=True))
        return torch.softmax(y, 1)


# ================================================================================================ stage-1 glue
class Glue:
    """Batched transcriptions of mlfg_glue's passes (MODE 'glsl' = the shader's own fp16 roundings when amd=True)."""

    def __init__(self, amd):
        self.amd = amd

    def r16(self, x):
        return q16(x, self.amd)

    def mix(self, a, b, t):
        return a + t * (b - a)

    def bilerp(self, a, b, c, d, fx, fy):
        fx = fx.unsqueeze(1); fy = fy.unsqueeze(1)
        return self.r16(self.mix(self.r16(self.mix(a, b, fx)), self.r16(self.mix(c, d, fx)), fy))

    def resample(self, src, Sw, Sh, OW, OH, scale):
        """Bilinear resize (align_corners=False, clamp to [0, S-1]) sampled at max((o+.5)*scale-.5, 0)."""
        dev = src.device
        B = src.shape[0]

        def axis(n, S):
            p = torch.clamp((torch.arange(n, dtype=torch.float32, device=dev) + 0.5) * scale - 0.5, min=0.0)
            i0 = torch.clamp(torch.floor(p).long(), 0, S - 1)
            return i0, torch.clamp(i0 + 1, 0, S - 1), torch.clamp(p - i0.float(), 0.0, 1.0)
        x0, x1, fx = axis(OW, Sw); y0, y1, fy = axis(OH, Sh)
        e = lambda a, b: torch.broadcast_to(a[None, None, :] if b == 'x' else a[None, :, None], (B, OH, OW))
        X0, X1, FX = e(x0, 'x'), e(x1, 'x'), e(fx, 'x')
        Y0, Y1, FY = e(y0, 'y'), e(y1, 'y'), e(fy, 'y')
        return self.bilerp(_gather(src, X0, Y0), _gather(src, X1, Y0), _gather(src, X0, Y1), _gather(src, X1, Y1), FX, FY)

    def warp(self, src, flow, Sw, Sh):
        """Backward warp at integer pixel p: bilinear(src, p + flow), clamp to [0, S-1]."""
        B, _, H, W = flow.shape
        dev = flow.device
        gy, gx = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=dev),
                                torch.arange(W, dtype=torch.float32, device=dev), indexing='ij')
        px = gx[None] + flow[:, 0]; py = gy[None] + flow[:, 1]
        if self.amd:  # the shader's literal fp32 position math ((p+f)*2/(S-1))*0.5*(S-1)
            sx, sy = float(Sw - 1), float(Sh - 1)   # (full_like: CUDA turns division by a scalar into a rcp multiply)
            px = ((px * 2.0) / torch.full_like(px, sx) * 0.5) * sx; py = ((py * 2.0) / torch.full_like(py, sy) * 0.5) * sy
        fl_x = torch.floor(px); fl_y = torch.floor(py)
        fx = px - fl_x; fy = py - fl_y
        ix = fl_x.long(); iy = fl_y.long()
        x0 = ix.clamp(0, Sw - 1); x1 = (ix + 1).clamp(0, Sw - 1)
        y0 = iy.clamp(0, Sh - 1); y1 = (iy + 1).clamp(0, Sh - 1)
        return self.bilerp(_gather(src, x0, y0), _gather(src, x1, y0), _gather(src, x0, y1), _gather(src, x1, y1), fx, fy)

    def warp_pair(self, net_in, flow, Sw, Sh):
        a = self.warp(torch.cat([net_in[:, 0:3], net_in[:, 6:12]], 1), flow[:, 0:2], Sw, Sh)
        b = self.warp(torch.cat([net_in[:, 3:6], net_in[:, 12:18]], 1), flow[:, 2:4], Sw, Sh)
        return a, b

    @staticmethod
    def stack32(wa, wb, head9, flow, t):
        B, _, H, W = flow.shape
        return torch.cat([wa[:, 0:3], wb[:, 0:3], wa[:, 3:9], wb[:, 3:9],
                          torch.full((B, 1, H, W), float(t), device=flow.device, dtype=flow.dtype), head9, flow], 1)

    def pass18(self, net_in, S1, S10):
        return self.r16(self.resample(net_in, S1[0], S1[1], S10[0], S10[1], 4.0))

    def pass54(self, net_in, f54, S1, S23, S24, t):
        W0, H0 = 2 * S24[0], 2 * S24[1]
        up = self.resample(f54[:, :13], S23[0], S23[1], W0, H0, 0.25)
        flow = self.r16(4.0 * up[:, 0:4])
        wa, wb = self.warp_pair(net_in, flow, S1[0], S1[1])
        st = self.stack32(wa, wb, up[:, 4:13], flow, t)
        s = st[:, :, 0::2, 0::2]
        for dy, dx in ((0, 1), (1, 0), (1, 1)):
            s = self.r16(s + st[:, :, dy::2, dx::2])
        x = s / 4.0
        x = torch.cat([x[:, :18], torch.full_like(x[:, 18:19], float(t)), x[:, 19:28], x[:, 28:32] / 2.0], 1)
        x = self.r16(x)
        return self.r16(flow), x

    def pass83(self, net_in, f83, flow54, S1, S38, S39, t):
        up = self.resample(f83[:, :13], S38[0], S38[1], S39[0], S39[1], 0.5)
        flow = self.r16(self.r16(2.0 * up[:, 0:4]) + flow54[:, :, :S39[1], :S39[0]])
        wa, wb = self.warp_pair(net_in, flow, S1[0], S1[1])
        return self.r16(flow), self.r16(self.stack32(wa, wb, up[:, 4:13], flow, t))

    def pass112(self, net_in, f112, flow83, S53, S54, t):
        flow = self.r16(f112[:, 0:4, :S54[1], :S54[0]] + flow83[:, :, :S54[1], :S54[0]])
        wa, wb = self.warp_pair(net_in, flow, S53[0], S53[1])
        return self.r16(flow), self.r16(self.stack32(wa, wb, f112[:, 4:13, :S54[1], :S54[0]], flow, t))

    def pass149(self, f149, flow112):
        return self.r16(f149[:, 0:4] + flow112)

    def pass151(self, flowout, live):
        W, H = live
        d = torch.tensor([float(W), float(H)], device=flowout.device).view(1, 2, 1, 1)
        f = flowout[:, :, :H, :W]
        return self.r16(f[:, 0:2] / d), self.r16(f[:, 2:4] / d)


def _nearest_index(n, size, device):
    """pass0's point-sampled texel (capture GPU): uv=(i+0.5)*rcp(size/8), uv truncated to 21 frac bits, floor(uv*size)."""
    uv = (np.arange(n, dtype=np.float32) + np.float32(0.5)) * (np.float32(1) / (np.float32(size) * np.float32(0.125)))
    return torch.from_numpy(np.floor(np.floor(uv.astype(np.float64) * 2.0 ** 21) / 2.0 ** 21 * size).astype(np.int64)).to(device)


def pass0(rgb0, rgb1, live, l0, amd, point=False):
    """Level-0 network input [B,6,l0h,l0w]: rgb0 | rgb1, one sample per 8x8 block (linear sampler: 2x2 box of texels
    8x+3..8x+4), zero outside the live region (min(l0, tex/8))."""
    B, _, Hd, Wd = rgb0.shape
    lw, lh = min(live[0], Wd // 8, l0[0]), min(live[1], Hd // 8, l0[1])
    outs = []
    for img in (rgb0, rgb1):
        if point:
            iy = _nearest_index(lh, Hd, img.device); ix = _nearest_index(lw, Wd, img.device)
            v = img[:, :3][:, :, iy][:, :, :, ix]
        else:
            tl = img[:, :3, 3::8, 3::8][:, :, :lh, :lw]; tr = img[:, :3, 3::8, 4::8][:, :, :lh, :lw]
            bl = img[:, :3, 4::8, 3::8][:, :, :lh, :lw]; br = img[:, :3, 4::8, 4::8][:, :, :lh, :lw]
            v = ((tl + tr) * 0.5 + (bl + br) * 0.5) * 0.5
        outs.append(F.pad(q16(v, amd), (0, l0[0] - lw, 0, l0[1] - lh)))
    return torch.cat(outs, 1)


# ================================================================================================ prepass
@dataclass
class MlfgHistory:
    mv: torch.Tensor     # [B,2,Hp,Wp] dilated mv of frame 0 in uv (pipe191 output of the previous frame)
    depth: torch.Tensor  # [B,1,Hp,Wp] its nearest-downscaled depth


def pipe191_dilate(mv, depth, pre, mv_scale=(1.0, 1.0), depth_inverted=True, amd=True):
    """MV dilation + depth downscale to the prepass size (generation / colour rects = full frame, no distortion).
    mv [B,2,Hr,Wr] (x,y) render px; depth [B,1,Hr,Wr]. Returns (mv uv [B,2,Hp,Wp] fp16, depth [B,1,Hp,Wp])."""
    B, _, Hr, Wr = depth.shape
    Wp, Hp = pre
    dev = depth.device
    gy, gx = torch.meshgrid(torch.arange(Hp, device=dev), torch.arange(Wp, device=dev), indexing='ij')

    def src(ox, oy):
        sx = torch.div((gx + ox) * Wr + Wr // 2, Wp, rounding_mode='trunc')
        sy = torch.div((gy + oy) * Hr + Hr // 2, Hp, rounding_mode='trunc')
        return sx, sy
    cx, cy = src(0, 0)
    bx = cx[None].expand(B, -1, -1); by = cy[None].expand(B, -1, -1)
    best_d = _fetch0(depth, bx, by)[:, 0]
    d_out = best_d
    for ox, oy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        sx, sy = src(ox, oy)
        sx = sx.clamp(0, Wr - 1)[None].expand(B, -1, -1); sy = sy.clamp(0, Hr - 1)[None].expand(B, -1, -1)
        d = _fetch0(depth, sx, sy)[:, 0]
        take = (d > best_d) if depth_inverted else (d < best_d)
        best_d = torch.where(take, d, best_d)
        bx = torch.where(take, sx, bx); by = torch.where(take, sy, by)
    m = _fetch0(mv, bx, by)[:, :2]
    m = (m * torch.tensor(mv_scale, device=dev).view(1, 2, 1, 1)) / torch.tensor([float(Wr), float(Hr)], device=dev).view(1, 2, 1, 1)
    return q16(m, amd), d_out[:, None]


def _depth_key(d, inverted):
    d = torch.abs(torch.clamp(d, min=1e-37))
    k = (d if inverted else 1.0 - d).float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.clamp(k, min=1)


def _f16bits(x):
    return x.to(torch.float16).contiguous().view(torch.int16).to(torch.int64) & 0xFFFF


def pipe192_splat(flow, depth, proj, flow_is_uv, depth_inverted=True):
    """Forward splat with 64-bit atomic max (nearest surface wins). flow [B,2,H,W] uv (x == 65504: invalid), depth
    [B,1,H,W], both at the destination size. Returns packed int64 [B,H,W] (key << 32 | f16 v.y << 16 | f16 v.x)."""
    B, _, H, W = flow.shape
    dev = flow.device
    gy, gx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    valid = flow[:, 0] != F16MAX
    f = torch.where(valid[:, None], flow, torch.zeros((), device=dev)) * float(proj) * (-1.0 if flow_is_uv else 1.0)
    key = _depth_key(depth[:, 0], depth_inverted)
    tx = torch.trunc((0.5 + gx.float())[None] + f[:, 0] * W).long()
    ty = torch.trunc((0.5 + gy.float())[None] + f[:, 1] * H).long()
    v = torch.where(valid[:, None], f, torch.full((), F16MAX, device=dev))
    packed = (key << 32) | (_f16bits(v[:, 1]) << 16) | _f16bits(v[:, 0])
    out = torch.zeros(B, H * W, dtype=torch.int64, device=dev)
    for dy in (0, 1):
        for dx in (0, 1):
            x = tx + dx; y = ty + dy
            ok = (x >= 0) & (y >= 0) & (x < W) & (y < H)
            idx = torch.where(ok, y * W + x, torch.zeros_like(x))
            src = torch.where(ok, packed, torch.zeros_like(packed))  # 0 never beats a written texel (key >= 1)
            out.scatter_reduce_(1, idx.reshape(B, -1), src.reshape(B, -1), reduce='amax')
    return out.view(B, H, W)


def splat_decode(packed):
    """packed [B,H,W] -> (flow [B,2,H,W] uv with 0 where none, hit [B,1,H,W] = written with a valid flow)."""
    lo = packed & 0xFFFFFFFF
    fx = _bits_to_f16(lo & 0xFFFF)
    fy = _bits_to_f16((lo >> 16) & 0xFFFF)
    ok = fx != F16MAX
    hit = ((packed >> 32) >= 1) & (lo != INVALID2)
    flow = torch.where(ok[:, None], torch.stack([fx, fy], 1), torch.zeros((), device=packed.device))
    return flow, hit[:, None].float()


def _bits_to_f16(b):
    b = torch.where(b >= 32768, b - 65536, b)
    return b.to(torch.int16).view(torch.float16).float()


def pipe193_forward_mv(mv0, depth0, splat1to0, A, B_, display, amd=True):
    """Frame-0 forward motion (value uv0 - uv1, FFX mv sign): camera motion from the matrices where frame 0's own mv
    agrees with the camera reprojection into frame -1 (<= 2 display px), else the splat of frame 1's mvs onto frame 0
    (65504 where nothing landed). mv0/depth0 = MlfgHistory; A = prevPstoPs, B_ = prevPsToPrev2Ps (row-vector)."""
    Bn, _, H, W = mv0.shape
    dev = mv0.device
    gy, gx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    disp = torch.tensor([float(display[0]), float(display[1])], device=dev).view(1, 2, 1, 1)
    msz = torch.tensor([float(W), float(H)], device=dev).view(1, 2, 1, 1)
    g = torch.stack([gx, gy], 0)[None].float()
    p86 = g * (disp / msz)
    uvc = p86 * (1.0 / disp) + 0.5 / msz
    dsz = torch.tensor([float(depth0.shape[3]), float(depth0.shape[2])], device=dev).view(1, 2, 1, 1)
    di = torch.trunc(uvc * dsz).long().expand(Bn, -1, -1, -1)
    z = torch.clamp(_fetch0(depth0, di[:, 0], di[:, 1])[:, 0], min=1e-37)
    uv = p86 / disp
    ndc = uv * 2.0 - 1.0
    clip = torch.stack([ndc[:, 0].expand(Bn, -1, -1), -ndc[:, 1].expand(Bn, -1, -1), z, torch.ones_like(z)], -1)
    q = torch.einsum('bhwi,bij->bhwj', clip, B_.float())
    q = q / q[..., 3:4]
    prev2 = torch.stack([q[..., 0], -q[..., 1]], 1) * 0.5 + 0.5
    mi = torch.trunc(uvc * msz).long().expand(Bn, -1, -1, -1)
    m = _fetch0(mv0, mi[:, 0], mi[:, 1])[:, :2]
    err = ((uv - m * -1.0) - prev2) * disp
    r = torch.einsum('bhwi,bij->bhwj', clip, A.float())
    r = r / r[..., 3:4]
    rc = torch.stack([r[..., 0] - clip[..., 0], r[..., 1] - clip[..., 1]], 1)
    cam = -((((rc * torch.tensor([0.25, -0.25], device=dev).view(1, 2, 1, 1)) * disp) / disp * 2.0))
    fl, hit = splat_decode(splat1to0)
    fb = torch.where(hit > 0, fl, torch.full((), F16MAX, device=dev))
    use_cam = ~(torch.maximum(err[:, 0].abs(), err[:, 1].abs()) > 2.0)
    return q16(torch.where(use_cam[:, None], cam, fb), amd)


def splat_valid_weight(hit, u, v):
    """Bilinear coverage of a splat (clamp addressing, weights fract(uv*size+0.5)). hit [B,1,H,W] -> [B,1,h,w]."""
    return sample_linear(hit, u, v)


def splat_flow_at(flow, u, v, ref_width):
    """pass152/190's single-texel splat-flow read: texel trunc(uv*size - 0.5*size.x/ref_width), OOB -> 0."""
    B, _, H, W = flow.shape
    k = 0.5 * (W / ref_width)
    sx = torch.trunc(_bc(u, B) * W - k).long(); sy = torch.trunc(_bc(v, B) * H - k).long()
    return _fetch0(flow, sx, sy)


# ================================================================================================ the model
class AmdMlfgNet(nn.Module):
    def __init__(self, weights=DEFAULT_WEIGHTS, precision='train', sampler='linear', nv_lanes=False):
        """precision 'train' | 'amd'. sampler: 'linear' (AMD's intended g_sampler_linear) or 'point' (what the
        NVIDIA captures did: the sampler at binding 320 was unbound). nv_lanes: reproduce NVIDIA's lane bug in flow
        block 4 (for comparing with NVIDIA scratch dumps only)."""
        super().__init__()
        self.precision = precision
        self.sampler = sampler
        self.composite_sampler = None   # None: same as sampler (override for pass190's colour / flow reads only)
        # how stage 1 gets t: 'channel' = the constant stack channel AMD sets to 0.5 becomes t (literal generalisation);
        # 'scale' = stage 1 at t=0.5, final flows scaled linearly (A * 2t, B * 2(1-t)). Identical at t = 0.5.
        self.flow_t = 'channel'
        self.encA, self.encB = Encoder(), Encoder()
        self.block1 = FlowBlock(18, 96, 192)
        self.block2 = FlowBlock(32, 64, 128)
        self.block3 = FlowBlock(32, 48, 96)
        self.block4 = FlowBlock(32, 24, 48)
        self.block4.nv_lanes = nv_lanes
        self.stage2 = Stage2()
        if weights:
            self.load_amd(weights)

    @property
    def amd(self):
        return self.precision == 'amd'

    # ---------------------------------------------------------------------------- weights
    @torch.no_grad()
    def load_amd(self, path):
        z = np.load(path)
        w = lambda k: torch.from_numpy(z[k + '_w'].copy())
        b = lambda k: torch.from_numpy(z[k + '_b'].copy())

        def conv(m, k, o=None, i=None):
            W_, B_ = w(k), b(k)
            o = slice(None) if o is None else o
            i = slice(None) if i is None else i
            m.weight.copy_(W_[o][:, i]); m.bias.copy_(B_[o])

        def convT(m, k, o=None, i=None, io=False):
            W_, B_ = w(k), b(k)                     # stored [O, I] (or [I, O] when io)
            if io: W_ = W_.permute(1, 0, 2, 3)
            o = slice(None) if o is None else o
            i = slice(None) if i is None else i
            m.weight.copy_(W_[o][:, i].permute(1, 0, 2, 3)); m.bias.copy_(B_[o])

        for enc, base, win in ((self.encA, 2, slice(0, 3)), (self.encB, 10, slice(1, 4))):
            conv(enc.down, f'L{base}', i=win)       # 16-ch windows: ch 0..15 (rgb0 at 0..2) / ch 2..17 (rgb1 at 1..3)
            conv(enc.c1, f'L{base + 2}'); conv(enc.c2, f'L{base + 4}')
            convT(enc.up, f'L{base + 6}', o=slice(0, 6))
        specs = [(self.block1, 20, 22, 24, [26, 29, 32, 35, 38, 41, 44, 47], 50, 52),
                 (self.block2, 57, 59, 61, list(range(63, 78, 2)), 79, 81),
                 (self.block3, 86, 88, 90, list(range(92, 107, 2)), 108, 110),
                 (self.block4, 115, 117, 119, list(range(121, 143, 3)), 145, 147)]
        for blk, d1, c1, d2, res, u1, u2 in specs:
            ci, c1o = blk.d1.in_channels, blk.d1.out_channels
            conv(blk.d1, f'L{d1}', o=slice(0, c1o), i=slice(0, ci))
            conv(blk.c1, f'L{c1}', o=slice(0, c1o), i=slice(0, c1o))
            conv(blk.d2, f'L{d2}', i=slice(0, c1o))
            for m, p in zip(blk.res, res): conv(m, f'L{p}')
            convT(blk.u1, f'L{u1}', o=slice(0, c1o))
            convT(blk.u2, f'L{u2}', o=slice(0, 13), i=slice(0, c1o))
        s = self.stage2
        conv(s.inp, 'L154', i=slice(0, 24)); conv(s.d1, 'L156'); conv(s.d2, 'L166')
        for blocks, passes in ((s.b1, [158, 160, 162, 164]), (s.b2, [168, 170, 172, 174, 176]), (s.b3, [178, 180, 182, 184])):
            for m, p in zip(blocks, passes):
                conv(m.pconv, f'L{p}_pconv'); conv(m.expand, f'L{p}_expand'); conv(m.project, f'L{p}_project')
                if m.up is not None: convT(m.up, f'L{p}_up')
        convT(s.up, 'L186', io=True)                # stored [I=48, O=32] (bias has 32 entries)
        W188 = w('L188').permute(1, 0, 2, 3)        # stored [I=32, O=16]; outputs 0..4 are real
        s.head.weight.copy_(W188[:5]); s.head.bias.copy_(b('L188')[:5])

    # ---------------------------------------------------------------------------- stage 1
    def stage1(self, rgb0, rgb1, t, sz, x6=None):
        """-> (flow [B,4,l0h,l0w] level-0 px: A.xy = t->0, B.xy = t->1; flow images (f0, f1) [B,2,live] uv; aux).
        x6: optional precomputed pass-0 output [B,6,l0h,l0w] (verification)."""
        amd = self.amd
        g = Glue(amd)
        t_out = t
        if self.flow_t == 'scale':   # run the network at its training timestep, scale the final flows linearly
            t = 0.5
        l0 = sz['l0']; S10 = (l0[0] // 4, l0[1] // 4); S24 = (l0[0] // 2, l0[1] // 2)
        if x6 is None:
            x6 = pass0(rgb0, rgb1, sz['l0_live'], l0, amd, point=self.sampler == 'point')
        net_in = torch.cat([x6, self.encA(x6[:, 0:3], amd), self.encB(x6[:, 3:6], amd)], 1)
        f54 = self.block1(g.pass18(net_in, l0, S10), amd)
        flow, x = g.pass54(net_in, f54, l0, S10, S24, t)
        f83 = self.block2(x, amd)
        flow, x = g.pass83(net_in, f83, flow, l0, S24, l0, t)
        f112 = self.block3(x, amd)
        flow, x = g.pass112(net_in, f112, flow, l0, l0, t)
        f149 = self.block4(x, amd)
        flow = g.pass149(f149, flow)
        if self.flow_t == 'scale' and t_out != 0.5:
            flow = q16(torch.cat([flow[:, 0:2] * (2.0 * t_out), flow[:, 2:4] * (2.0 * (1.0 - t_out))], 1), amd)
        f0, f1 = g.pass151(flow, sz['l0_live'])
        return flow, (f0, f1), dict(net_in=net_in, f54=f54, f83=f83, f112=f112, f149=f149)

    # ---------------------------------------------------------------------------- prepass
    def history(self, mv, depth, display, mv_scale=(1.0, 1.0), depth_inverted=True):
        """pipe191 of a frame (its mv/depth) -> the MlfgHistory the next frame pair needs."""
        m, d = pipe191_dilate(mv, depth, mlfg_sizes(*display)['pre'], mv_scale, depth_inverted, self.amd)
        return MlfgHistory(m, d)

    @torch.no_grad()
    def prepass(self, cur_mv, cur_depth, hist, A, B_, t, display, mv_scale=(1.0, 1.0), depth_inverted=True):
        sz = mlfg_sizes(*display)
        mv1, d1 = pipe191_dilate(cur_mv, cur_depth, sz['pre'], mv_scale, depth_inverted, self.amd)
        s10 = pipe192_splat(mv1, d1, 1.0, False, depth_inverted)
        fwd0 = pipe193_forward_mv(hist.mv, hist.depth, s10, A, B_, display, self.amd)
        s0t = pipe192_splat(fwd0, hist.depth, t, True, depth_inverted)        # AMD: proj 0.5
        s1t = pipe192_splat(mv1, d1, 1.0 - t, False, depth_inverted)          # AMD: proj 0.5
        return dict(mv1=mv1, d1=d1, splat1to0=s10, fwd0=fwd0, splat0t=s0t, splat1t=s1t,
                    history=MlfgHistory(mv1, d1))

    # ---------------------------------------------------------------------------- stage-2 input / composite
    def _sg(self, img, u, v, composite=False):
        s = self.composite_sampler if (composite and self.composite_sampler) else self.sampler
        return sample_point(img, u, v) if s == 'point' else sample_linear(img, u, v)

    def pass152(self, rgb0, rgb1, f0img, f1img, sp0, sp1, sz):
        """24-ch stage-2 input at the stage-2 allocation size (zero beyond display/4)."""
        amd = self.amd
        W, H = sz['s2']
        dev = rgb0.device
        u, v = texel_uv(W, H, dev)
        (fl0, hit0), (fl1, hit1) = sp0, sp1
        s0 = splat_flow_at(fl0, u, v, W); s1 = splat_flow_at(fl1, u, v, W)
        cl = lambda a: a.clamp(0.0, 1.0)
        rd = lambda img, uu, vv: self._sg(img[:, :3], cl(uu), cl(vv))
        c14 = rd(rgb0, u, v); c17 = rd(rgb1, u, v)
        c3 = rd(rgb1, u - s1[:, 0], v - s1[:, 1]); c0 = rd(rgb0, u - s0[:, 0], v - s0[:, 1])
        f0 = q16(self._sg(f0img, u, v), amd); f1 = q16(self._sg(f1img, u, v), amd)
        c6 = rd(rgb0, u + f0[:, 0], v + f0[:, 1]); c9 = rd(rgb1, u + f1[:, 0], v + f1[:, 1])
        m0 = splat_valid_weight(hit0, u, v); m1 = splat_valid_weight(hit1, u, v)
        L = torch.tensor(LUMA, device=dev).view(1, 3, 1, 1)
        Y = lambda c: (c * L).sum(1, keepdim=True)
        y0, y6 = Y(c0), Y(c6)
        ch = torch.cat([c0, c3, c6, c9, m0, m1, c14, c17, (Y(c14) - Y(c17)).abs(), (y0 - Y(c3)).abs(),
                        (y0 - y6).abs(), (y6 - Y(c9)).abs()], 1)
        gy = torch.arange(H, device=dev).view(1, 1, H, 1); gx = torch.arange(W, device=dev).view(1, 1, 1, W)
        live = (gx < rgb0.shape[3] // 4) & (gy < rgb0.shape[2] // 4)
        return q16(torch.where(live, ch, torch.zeros((), device=dev)), amd)

    def pass190(self, w6, rgb0, rgb1, f0img, f1img, sp0, sp1, t, sz):
        """Display-res composite. w6 [B,5,s2h,s2w] softmax weights."""
        amd = self.amd
        B, _, H, W = rgb0.shape
        dev = rgb0.device
        u, v = texel_uv(W, H, dev)
        TW, TH = sz['s2']
        gy, gx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
        xs = torch.clamp((gx.float() + 0.5) * 0.25 - 0.5, min=0.0); ys = torch.clamp((gy.float() + 0.5) * 0.25 - 0.5, min=0.0)
        x0 = torch.floor(xs).long().clamp(0, TW - 1); y0 = torch.floor(ys).long().clamp(0, TH - 1)
        fx = (xs - x0.float()).clamp(0, 1)[None, None]; fy = (ys - y0.float()).clamp(0, 1)[None, None]
        x1 = (x0 + 1).clamp(0, TW - 1); y1 = (y0 + 1).clamp(0, TH - 1)
        e = lambda a: a[None].expand(B, -1, -1)
        tap = lambda xx, yy: _gather(w6, e(xx), e(yy))
        mix = lambda a, b, f: a + (b - a) * f
        w = q16(mix(q16(mix(tap(x0, y0), tap(x1, y0), fx), amd), q16(mix(tap(x0, y1), tap(x1, y1), fx), amd), fy), amd)
        (fl0, hit0), (fl1, hit1) = sp0, sp1
        m1 = splat_valid_weight(hit1, u, v); m0 = splat_valid_weight(hit0, u, v)
        s0 = splat_flow_at(fl0, u, v, W); s1 = splat_flow_at(fl1, u, v, W)
        cl = lambda a: a.clamp(0.0, 1.0)
        rd = lambda img, uu, vv: self._sg(img[:, :3], cl(uu), cl(vv), True)
        c1 = rd(rgb1, u - s1[:, 0], v - s1[:, 1]); c0 = rd(rgb0, u - s0[:, 0], v - s0[:, 1])
        f0 = q16(self._sg(f0img, u, v, True), amd); f1 = q16(self._sg(f1img, u, v, True), amd)
        wb0 = q16((1.0 - t) * w[:, 4:5], amd); wb1 = q16(t * w[:, 4:5], amd)   # AMD: both f16(0.5 * w4)
        p0 = rgb0[:, :3]; p1 = rgb1[:, :3]
        out = ((((c0 * (1 - m1) + c1 * m1) * w[:, 0:1] + (c1 * (1 - m0) + c0 * m0) * w[:, 1:2])
                + rd(rgb0, u + f0[:, 0], v + f0[:, 1]) * w[:, 2:3]) + rd(rgb1, u + f1[:, 0], v + f1[:, 1]) * w[:, 3:4]) \
            + p0 * wb0 + p1 * wb1
        return out

    # ---------------------------------------------------------------------------- full forward
    def forward(self, prev_rgb, cur_rgb, cur_depth, cur_mv, hist, prevPstoPs, prevPsToPrev2Ps, t=0.5, reset=False,
                mv_scale=(1.0, 1.0), depth_inverted=True, pre=None):
        B, _, H, W = cur_rgb.shape
        sz = mlfg_sizes(W, H)
        if pre is None:
            pre = self.prepass(cur_mv, cur_depth, hist, prevPstoPs, prevPsToPrev2Ps, t, (W, H), mv_scale, depth_inverted)
        flow, (f0, f1), aux1 = self.stage1(prev_rgb, cur_rgb, t, sz)
        sp0, sp1 = splat_decode(pre['splat0t']), splat_decode(pre['splat1t'])
        x2 = self.pass152(prev_rgb, cur_rgb, f0, f1, sp0, sp1, sz)
        w5 = self.stage2(x2, self.amd)
        out = self.pass190(w5, prev_rgb, cur_rgb, f0, f1, sp0, sp1, t, sz)
        if reset:
            out = cur_rgb[:, :3]
        return out, dict(flow=flow, flow0=f0, flow1=f1, stage2_in=x2, weights=w5, history=pre['history'], pre=pre, **aux1)
