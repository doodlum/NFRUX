"""Image-space passes around AMD FSR Frame Generation 4's network, transcribed from the decompiled shaders, and a
checker against AMD's own GPU output (ffx_capture --dump-images / --dump-passes, see main()).

    python mlfg_io.py [CAP_A] [CAP_B]      CAP_A: --patch-cb --dump-images 2 --dump-passes 2 150 152
                                           CAP_B: --patch-cb --dump-images 2 --dump-passes 2 189 189

Frame "1" is the current (just rendered) frame, frame "0" the previous one, 0.5 the generated midpoint.
Motion vectors follow FFX: mv(p1) = p0 - p1 (current -> previous), and after pipe191 they are in UV units.

Per-frame pass order (one 1920x1080 frame, 4K-compiled network, all prepass images 640x360 = 1/3 res):
  pipe191 dilate      mv/depth (render res) -> dilated mv (uv) + nearest-downscaled depth, both 640x360
  pipe192 splat       (dilated mv, depth, proj=1.0)        -> splat1to0   (current pixels pushed onto frame 0)
  pipe193 fwd mv      (prev dilated mv, prev depth, splat1to0, camera matrices) -> fwd mv of frame 0 ("mv_0_to_1")
  pipe192 splat       (fwd mv, prev depth, proj=0.5, uv flag) -> splat0to05 (frame-0 pixels pushed to t=0.5)
  pipe192 splat       (dilated mv, depth, proj=0.5)          -> splat1to05 (frame-1 pixels pushed to t=0.5)
  network passes 0..150 (stage 1, 1/8 res flows), pass151 (flows -> 240x135 uv images),
  pass152 (stage-2 input, 24 ch at 1/4 res), passes 153..189 (stage 2; pass188 writes the 6-ch blend weights),
  pass190 (full-res composite -> output image), then the host copies the current colour into the history.
"""
import os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

F16MAX = 65504.0
INVALID2 = 0x7BFF7BFF  # packHalf2x16(65504, 65504): "no flow" marker of a splat texel

def h(x):
    """Round to fp16 (RNE) and back."""
    return x.to(torch.float16).float()

def mix(a, b, t):
    return a + (b - a) * t

# ---------------------------------------------------------------------------------------------- texture access
# Images are torch tensors [H, W, C] (float32, or int64 for packed splat images). Coordinates follow Vulkan:
# texel (x, y) has its centre at uv = ((x, y) + 0.5) / size. The linear sampler is clamp-to-edge; NVIDIA filters
# with 8 fractional bits of sub-texel position (FRAC_BITS; None = exact fp32 weights).
FRAC_BITS = 8

def fetch(img, ix, iy):
    """texelFetch with integer tensors; out-of-range texels read as 0 (what this GPU returned)."""
    H, W = img.shape[:2]
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    v = img[iy.clamp(0, H - 1), ix.clamp(0, W - 1)]
    return torch.where(ok[..., None], v, torch.zeros_like(v))

def _frac(f):
    if FRAC_BITS is None: return f
    s = float(1 << FRAC_BITS)
    return torch.floor(f * s + 0.5) / s

def sample_linear(img, u, v):
    H, W = img.shape[:2]
    x = u * W - 0.5; y = v * H - 0.5
    x0 = torch.floor(x); y0 = torch.floor(y)
    fx = _frac(x - x0)[..., None]; fy = _frac(y - y0)[..., None]
    x0 = x0.long(); y0 = y0.long()
    cx = lambda i: i.clamp(0, W - 1)
    cy = lambda i: i.clamp(0, H - 1)
    a = img[cy(y0), cx(x0)]; b = img[cy(y0), cx(x0 + 1)]
    c = img[cy(y0 + 1), cx(x0)]; d = img[cy(y0 + 1), cx(x0 + 1)]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

def sample_point(img, u, v):
    """Point sampling as this GPU does it: the normalized coordinate is truncated to UV_FRAC_BITS (21) fractional
    bits, then texel = floor(uv_fx * size) exactly. This only matters where uv*size lands on a texel edge, which
    pass152 does on every column (its 1/4-res centres map to uv*1920 = 4x+2); fitted on those edges."""
    H, W = img.shape[:2]
    s = float(1 << UV_FRAC_BITS)
    fx = lambda a, n: torch.floor(torch.floor(a.double() * s) / s * n).long().clamp(0, n - 1)
    return img[fx(v, H), fx(u, W)]

UV_FRAC_BITS = 21

# The shaders sample the flow images at vec2(f16vec2(uv)); this GPU's results match sampling at the fp32 uv (the
# conversion pair is dropped by the compiler), verified on pass152 rows whose f16(uv) crosses a flow texel edge.
FLOW_UV_F16 = False

def texel_uv(p, size):
    """(p + 0.5) / size as the GPU evaluates it: multiplication by the fp32 reciprocal (not a correctly rounded
    division; bit-exactness of the edge cases above depends on it)."""
    return (p.float() + 0.5) * (1.0 / torch.tensor(size, dtype=torch.float64)).float()

# The FFX shaders call their binding-320 sampler g_sampler_linear, but in this capture it behaves as a point
# sampler (pass152's 1/4-res colour reads return single texels, never blends).
G_SAMPLER_IS_POINT = True

def sample_g(img, u, v):
    return sample_point(img, u, v) if G_SAMPLER_IS_POINT else sample_linear(img, u, v)

def sample_nearest(img, u, v):
    H, W = img.shape[:2]
    return img[torch.floor(v * H).long().clamp(0, H - 1), torch.floor(u * W).long().clamp(0, W - 1)]

def gather(img, u, v, comp):
    """textureGather(img, uv, comp) -> [..., 4] in Vulkan order (x0,y1) (x1,y1) (x1,y0) (x0,y0)."""
    H, W = img.shape[:2]
    x0 = torch.floor(u * W - 0.5).long(); y0 = torch.floor(v * H - 0.5).long()
    t = lambda xx, yy: img[yy.clamp(0, H - 1), xx.clamp(0, W - 1), comp]
    return torch.stack([t(x0, y0 + 1), t(x0 + 1, y0 + 1), t(x0 + 1, y0), t(x0, y0)], -1)

def splat_valid_weight(splat, u, v):
    """Bilinear coverage of a packed splat image at uv: texels count when they were written (key >= 1) and carry a
    valid flow (low word != INVALID2). splat: int64 [H, W, 2] = (low32, high32)."""
    H, W = splat.shape[:2]
    g = (gather(splat, u, v, 1) >= 1).float() * (gather(splat, u, v, 0) != INVALID2).float()
    f = torch.frac(torch.stack([u * W, v * H], -1) + 0.5)  # fract(uv*size + 0.5), as the shader computes it
    fx, fy = f[..., 0], f[..., 1]
    return mix(mix(g[..., 3], g[..., 2], fx), mix(g[..., 0], g[..., 1], fx), fy)

def unpack_flow(low32):
    """Low word of a splat texel -> (flow, ok): two fp16 (x in bits 0-15, y in 16-31); x == 65504 means none."""
    lo = torch.from_numpy((low32.numpy() & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32))
    hi = torch.from_numpy(((low32.numpy() >> 16) & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32))
    ok = lo != F16MAX
    return torch.where(ok[..., None], torch.stack([lo, hi], -1), torch.zeros(lo.shape + (2,))), ok

def splat_flow_at(splat, u, v, ref_width):
    """The single-texel flow read of pass152/190: texel ivec2(uv*size - 0.5*size.x/ref_width) (truncating)."""
    H, W = splat.shape[:2]
    sx = u * W - 0.5 * (W / ref_width); sy = v * H - 0.5 * (W / ref_width)
    t = fetch(splat, torch.trunc(sx).long(), torch.trunc(sy).long())
    return unpack_flow(t[..., 0])[0]

def pix_grid(W, H):
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    return x, y

# ---------------------------------------------------------------------------------------------- pipe191
def pipe191_dilate(mv, depth, cb, distortion=None):
    """MV dilation + depth downscale to cb['dispatchSize'] (here 640x360 from 1920x1080).
    mv [Hr,Wr,2] render-res motion vectors (pixels*scale), depth [Hr,Wr,1]. For each output texel g:
      src = (g*renderSize + renderSize/2) / colorRectSize        (integer division; nearest render pixel)
      depth_out = depth[src]
      pick the nearest-to-camera depth among src and its 8 neighbours (neighbours clamped; inverted depth: max,
      first one wins ties, order centre,(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1) in (x,y))
      mv_out = mv[picked] * motionVectorScale * colorRectUVsize / renderSize      (UV units)
    Outside the generation rect: depth = 0 (inverted) / 1, mv = 0. The distortion field (lens distortion offsets
    in uv) is not bound by this host; it reads as 0."""
    Wd, Hd = cb['dispatchSize']; rs = torch.tensor(cb['renderSize'])
    gx, gy = pix_grid(Wd, Hd)
    dsz = torch.tensor([float(Wd), float(Hd)])
    g = torch.stack([gx, gy], -1).float()
    uv = (g + 0.5) / dsz
    dist = torch.zeros_like(uv) if distortion is None else sample_linear(distortion, uv[..., 0], uv[..., 1])[..., :2]
    uvd = uv + dist
    outside = (uvd > torch.tensor(cb['generationRectMaxUV'])).any(-1) | (uvd < torch.tensor(cb['generationRectMinUV'])).any(-1)
    base = torch.trunc(g + dist * dsz - torch.tensor(cb['colorRectMinUV']) * dsz).long()
    rect = torch.tensor(cb['colorRectMaxUV']) - torch.tensor(cb['colorRectMinUV'])
    rsz = torch.trunc(rect * dsz).long()
    half = rs // 2
    src = lambda o: torch.div(((base + torch.tensor(o)) * rs + half), rsz, rounding_mode='trunc')
    c0 = src((0, 0))
    best_d = fetch(depth, c0[..., 0], c0[..., 1])[..., 0]
    d_out = best_d.clone()
    best = c0.clone()
    inv = cb['depthInverted'] != 0
    for o in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        c = torch.minimum(torch.maximum(src(o), torch.zeros(2, dtype=torch.long)), rs - 1)
        d = fetch(depth, c[..., 0], c[..., 1])[..., 0]
        take = (d > best_d) if inv else (d < best_d)
        best_d = torch.where(take, d, best_d)
        best = torch.where(take[..., None], c, best)
    m = fetch(mv, best[..., 0], best[..., 1])[..., :2]
    m = ((m * torch.tensor(cb['motionVectorScale'])) * rect) / rs.float()
    d_out = torch.where(outside, torch.tensor(0.0 if inv else 1.0), d_out)
    m = torch.where(outside[..., None], torch.zeros(2), m)
    return h(m), d_out[..., None]

# ---------------------------------------------------------------------------------------------- pipe192
def depth_key(d, inverted):
    """32-bit sort key of a depth: bigger = nearer. The shader computes max(asuint(1 - z), 1) with z = 1-|d| for
    inverted depth; NVIDIA's compiler folds 1-(1-|d|) to |d| (verified bit-exact), so that is what is used here."""
    d = torch.abs(torch.clamp(d, min=1e-37))
    k = torch.from_numpy((d if inverted else 1.0 - d).numpy().astype(np.float32).view(np.uint32).astype(np.int64))
    return torch.clamp(k, min=1)

def pipe192_splat(flow, depth, cb, clear=0):
    """Forward splat with 64-bit atomic max. For each source texel p of dstSize (flow, depth at the same res,
    nearest-sampled at the generation-rect-clamped centre):
      f = (flow if flow.x != 65504 else 0) * projAmount * (-1 if flowIsUv else 1)          (uv units)
      t = ivec2(p + 0.5 + f*dstSize)                                                       (truncating)
      for the 2x2 texels t + {0,1}^2 inside dstSize (only the upper bound is tested; negative coords are dropped
      by the image unit): atomicMax(key << 32 | f16(v.y) << 16 | f16(v.x)), v = f (or 65504,65504 if invalid).
    The nearest surface wins; among equal depths the larger packed flow bits win. Returns int64 [H,W,2]
    (low32, high32) like the R32G32_UINT view."""
    W, H = cb['dstSize']
    gx, gy = pix_grid(W, H)
    dsz = torch.tensor([float(W), float(H)])
    lo = torch.trunc(torch.tensor(cb['generationRectMinUV']) * dsz).long()
    hi = torch.trunc(torch.tensor(cb['generationRectMaxUV']) * dsz).long() - 1
    p = torch.stack([gx, gy], -1)
    uv = (torch.minimum(torch.maximum(p, lo), hi).float() + 0.5) / dsz
    fl = sample_nearest(flow, uv[..., 0], uv[..., 1])[..., :2]
    valid = fl[..., 0] != F16MAX
    f = torch.where(valid[..., None], fl, torch.zeros(2)) * cb['projAmount'] * (-1.0 if cb['flowIsUv'] else 1.0)
    d = sample_nearest(depth, uv[..., 0], uv[..., 1])[..., 0]
    key = depth_key(d, cb['depthInverted'] != 0)
    t = torch.trunc((0.5 + p.float()) + f * dsz).long()
    v = torch.where(valid[..., None], f, torch.full((2,), F16MAX))
    vb = torch.from_numpy(v.numpy().astype(np.float16).view(np.uint16).astype(np.int64))
    packed = (key << 32) | (vb[..., 1] << 16) | vb[..., 0]
    out = torch.full((H * W,), clear, dtype=torch.int64)
    for dy in (0, 1):
        for dx in (0, 1):
            tx = t[..., 0] + dx; ty = t[..., 1] + dy
            ok = (tx >= 0) & (ty >= 0) & (tx < W) & (ty < H)
            out.scatter_reduce_(0, (ty * W + tx)[ok], packed[ok], reduce='amax')
    out = out.view(H, W)
    return torch.stack([out & 0xFFFFFFFF, out >> 32], -1)

# ---------------------------------------------------------------------------------------------- pipe193
def pipe193_forward_mv(mv0, depth0, splat1to0, cb):
    """Forward motion of the previous frame ("mv_0_to_1", value = uv0 - uv1 at the frame-0 pixel, i.e. the same
    sign convention as FFX mvs, so splatting it with flowIsUv=1 negates it back to a 0->1 displacement).
    mv0/depth0: frame 0's dilated mv / depth (640x360, the copies of last frame's pipe191 outputs).
    Camera check: frame-0 clip position (ndc from the texel's top-left uv, y up, z = depth) * prevPsToPrev2Ps
    reprojects into frame -1; if |uv0 + mv0 - reproj| (in display pixels) <= 2 the pixel moves with the camera and
    the output is the camera-only motion -(proj(prevPstoPs) - ndc) * (0.5, -0.5) (i.e. uv0 - uv1);
    otherwise it falls back to the splat of the current frame's mvs onto frame 0 (pipe192, proj 1): its flow, or
    65504 where nothing landed. Matrices are row-vector style: clip' = [x y z 1] @ M (M as stored row-major).
    prevPstoPs = frame-0 clip -> frame-1 clip, prevPsToPrev2Ps = frame-0 clip -> frame-(-1) clip."""
    H, W = mv0.shape[:2]
    gx, gy = pix_grid(W, H)
    disp = torch.tensor([float(cb['displaySize'][0]), float(cb['displaySize'][1])])
    msz = torch.tensor([float(W), float(H)])
    g = torch.stack([gx, gy], -1).float()
    p86 = g * (disp / msz)
    uvc = p86 * (1.0 / disp) + 0.5 / msz
    dsz = torch.tensor([float(depth0.shape[1]), float(depth0.shape[0])])
    di = torch.trunc(uvc * dsz).long()
    z = torch.clamp(fetch(depth0, di[..., 0], di[..., 1])[..., 0], min=1e-37)
    uv = p86 / disp
    ndc = uv * 2.0 - 1.0
    clip = torch.stack([ndc[..., 0], -ndc[..., 1], z, torch.ones_like(z)], -1)
    A = torch.tensor(cb['prevPstoPs'], dtype=torch.float32).view(4, 4)
    B = torch.tensor(cb['prevPsToPrev2Ps'], dtype=torch.float32).view(4, 4)
    q = clip @ B
    q = q / q[..., 3:4]
    prev2 = torch.stack([q[..., 0], -q[..., 1]], -1) * 0.5 + 0.5
    mi = torch.trunc(uvc * msz).long()
    m = fetch(mv0, mi[..., 0], mi[..., 1])[..., :2]
    err = ((uv - m * -1.0) - prev2) * disp
    r = clip @ A
    r = r / r[..., 3:4]
    cam = -((((r[..., :2] - clip[..., :2]) * torch.tensor([0.25, -0.25])) * disp) / disp * 2.0)
    s = splat1to0
    fl, ok = unpack_flow(s[..., 0])
    hit = (s[..., 1] > 0) & ok
    fb = torch.where(hit[..., None], fl, torch.full((2,), F16MAX))
    use_cam = ~(torch.maximum(err[..., 0].abs(), err[..., 1].abs()) > 2.0)
    return h(torch.where(use_cam[..., None], cam, fb))

# ---------------------------------------------------------------------------------------------- pass151
def read_scratch(buf, off, sx, sy, C, W, H):
    """fp16 tensor at byte off + (x+1)*sx + (y+1)*sy + 2c (1-texel zero border) -> float32 [H, W, C]."""
    a = np.frombuffer(buf, np.uint8)
    rows = []
    for y in range(H):
        base = off + (y + 1) * sy + sx
        row = np.lib.stride_tricks.as_strided(a[base:], (W, C * 2), (sx, 1)).copy()
        rows.append(row.view(np.float16).reshape(W, C))
    return torch.from_numpy(np.stack(rows).astype(np.float32))

def pass151_flows(scratch, sizes):
    """Stage-1 flow head (4 ch at 1/8 res, pixels of that grid) -> two uv flow images (240x135 here):
    flow_0to05 = ch0..1 / size, flow_1to05 = ch2..3 / size: the uv offset from a t=0.5 position to where it
    is found in frame 0 / frame 1 (sample at uv + flow)."""
    W, H = (int(v) for v in sizes[70])
    t = read_scratch(scratch, 6150320, 8, 3856, 4, W, H)
    s = torch.tensor([float(W), float(H)])
    return h(t[..., 0:2] / s), h(t[..., 2:4] / s)

# ---------------------------------------------------------------------------------------------- transfer functions
def to_pq(c, cb):
    """backBufferTransferFunction 2 (scRGB) -> the PQ code value the network sees (literal transcription; the
    shader's mat3 columns are the rows of the usual BT.709->BT.2020 matrix)."""
    lo, hi = cb['minMaxLuminance']
    x = torch.clamp((c - lo * 0.0125) / ((hi - lo) * 0.0125), min=0.0)
    M = torch.tensor([[0.627403974533081, 0.329281985759735, 0.0433136001229286],
                      [0.0690969973802567, 0.919541001319885, 0.0113612003624439],
                      [0.0163915995508432, 0.0880132019519806, 0.895595014095306]])
    y = torch.pow((x @ M) * 300.0 * 1e-4, 0.1593019962310791)  # mat3(c0,c1,c2) * v with ci = rows above
    return torch.pow((0.835937976837158 + y * 18.8516006469727) / (1.0 + y * 18.6875), 78.8438034057617)

def net_space(c, cb):
    return to_pq(c, cb) if cb['backBufferTransferFunction'] == 2 else c  # 0 (sRGB) and 1 (PQ) pass through

# ---------------------------------------------------------------------------------------------- pass152
LUMA = torch.tensor([0.298999994993210, 0.587000012397766, 0.114000000059605])

def pass152_stage2_input(rgb0, rgb1, flow0, flow1, splat0, splat1, cb, sizes):
    """Stage-2 network input, 24 fp16 channels at 1/4 res (tensorSizes[72], 480x270 here), uv = texel centre:
      0-2  rgb0 at uv - s0       (s0 = splat0to05 flow at the texel, 0 if none: backward-warped frame 0)
      3-5  rgb1 at uv - s1       (s1 = splat1to05)
      6-8  rgb0 at uv + flow0    (flow0 = pass151 flow_0to05, bilinear at f16(uv))
      9-11 rgb1 at uv + flow1
      12   coverage of splat0to05 (bilinear over valid texels), 13 coverage of splat1to05
      14-16 rgb0 at uv, 17-19 rgb1 at uv (plain bilinear downsample)
      20 |Y(14-16) - Y(17-19)|, 21 |Y(0-2) - Y(3-5)|, 22 |Y(0-2) - Y(6-8)|, 23 |Y(6-8) - Y(9-11)|  (Y = BT.601 luma)
    All colour reads go through the binding-320 sampler (point on this GPU, see sample_point) at the generation-rect
    clamped uv, in the network colour space (transfer function 2 only is converted to PQ). NB the uv uses the
    tensor size 480x272 while the colour is 1920x1080, so rows are sampled at (y+0.5)/272*1080 (a 0.7% stretch).
    Texels outside the live size or the generation rect are 0. rgb0 = previous colour, rgb1 = current colour."""
    W, H = (int(v) for v in sizes[72])
    gx, gy = pix_grid(W, H)
    uv = texel_uv(torch.stack([gx, gy], -1), [float(W), float(H)])
    u, v = uv[..., 0], uv[..., 1]
    lo = torch.tensor(cb['generationRectMinUV']); hi = torch.tensor(cb['generationRectMaxUV'])
    out_rect = (uv > hi).any(-1) | (uv < lo).any(-1) | (gx >= rgb0.shape[1] // 4) | (gy >= rgb0.shape[0] // 4)
    s0 = splat_flow_at(splat0, u, v, W); s1 = splat_flow_at(splat1, u, v, W)
    uvh = h(uv) if FLOW_UV_F16 else uv
    rd = lambda img, q: net_space(sample_g(img, q[..., 0], q[..., 1])[..., :3], cb)
    cl = lambda q: torch.minimum(torch.maximum(q, lo), hi)
    c14 = rd(rgb0, uv); c17 = rd(rgb1, uv)
    c3 = rd(rgb1, cl(uv - s1)); c0 = rd(rgb0, cl(uv - s0))
    f0 = h(sample_g(flow0, uvh[..., 0], uvh[..., 1])); f1 = h(sample_g(flow1, uvh[..., 0], uvh[..., 1]))
    c6 = rd(rgb0, cl(uv + f0)); c9 = rd(rgb1, cl(uv + f1))
    m0 = splat_valid_weight(splat0, u, v); m1 = splat_valid_weight(splat1, u, v)
    Y = lambda c: (c * LUMA).sum(-1)
    y0, y6 = Y(c0), Y(c6)
    ch = torch.cat([c0, c3, c6, c9, m0[..., None], m1[..., None], c14, c17,
                    (Y(c14) - Y(c17)).abs()[..., None], (y0 - Y(c3)).abs()[..., None],
                    (y0 - y6).abs()[..., None], (y6 - Y(c9)).abs()[..., None]], -1)
    ch = torch.where(out_rect[..., None], torch.zeros(24), ch)
    return h(ch)

def read_pass152(scratch, sizes):
    W, H = (int(v) for v in sizes[72])
    return read_scratch(scratch, 0, 48, 46176, 24, W, H)

# ---------------------------------------------------------------------------------------------- pass190
def srgb_to_linear_ffx(c):
    m = c < 0.00313080498017371  # AMD tests the encoded value against the linear-side threshold
    return torch.where(m, c * 0.0773993805050850, torch.pow(c * 0.947867274284363 + 0.0521326996386051, 2.4))

def to_linear(c, cb):
    tf = cb['backBufferTransferFunction']
    if tf == 0: return srgb_to_linear_ffx(c)
    if tf == 1:
        p = torch.pow(c, 0.0126833003014326)
        return torch.pow(torch.clamp(p - 0.835937976837158, 0, 1) / (18.8516006469727 - 18.6875 * p), 6.27739000320435) * (10000.0 / cb['minMaxLuminance'][1])
    lo, hi = cb['minMaxLuminance']
    return (c - lo * 0.0125) / ((hi - lo) * 0.0125)

def pass190_composite(weights6, rgb0, rgb1, flow0, flow1, splat0, splat1, cb, sizes, hudless=None):
    """Full-res composite (output RGB in the backbuffer's encoding; the RGBA8 view stores round(clamp*255), a=0).
    weights6: pass188's 6-channel tensor at 1/4 res (tensorSizes[90]); channels 0-4 are used, upsampled with an
    fp16 bilinear (align-corners=False, edge-clamped). With s0/s1/flow0/flow1/m0/m1 as in pass152 but at the
    full-res pixel centre (splat flows read from texel ivec2(uv*640 - 1/6)):
      A = rgb0(uv - s0) * (1 - m1) + rgb1(uv - s1) * m1      (frame-0 backward warp, holes filled from frame 1)
      B = rgb1(uv - s1) * (1 - m0) + rgb0(uv - s0) * m0      (note: m0/m1 swapped relative to A's sources)
      out = A*w0 + B*w1 + rgb0(uv + flow0)*w2 + rgb1(uv + flow1)*w3 + (rgb0[px] + rgb1[px]) * f16(0.5*w4)
    Colours are raw (no transfer function). Reset or outside the generation rect: out = rgb1[px].
    w5 is never read. With pass188's softmax the five weights sum to 1 (w4 is shared equally by the two un-warped
    frames, so it is the 'plain blend of both inputs' candidate).
    Then optional HUD-less UI restore and the debug bars (dispatchFlags bits 0/1)."""
    W, H = rgb0.shape[1], rgb0.shape[0]
    gx, gy = pix_grid(W, H)
    uv = texel_uv(torch.stack([gx, gy], -1), [float(W), float(H)])
    u, v = uv[..., 0], uv[..., 1]
    lo = torch.tensor(cb['generationRectMinUV']); hi = torch.tensor(cb['generationRectMaxUV'])
    out_rect = (uv > hi).any(-1) | (uv < lo).any(-1)
    p0 = rgb0[..., :3]; p1 = rgb1[..., :3]
    # fp16 bilinear upsample of the weights
    TW, TH = (int(x) for x in sizes[90])
    xs = torch.clamp((gx.float() + 0.5) * 0.25 - 0.5, min=0.0); ys = torch.clamp((gy.float() + 0.5) * 0.25 - 0.5, min=0.0)
    x0 = torch.floor(xs).long().clamp(0, TW - 1); y0 = torch.floor(ys).long().clamp(0, TH - 1)
    fx = (xs - x0.float()).clamp(0, 1)[..., None]; fy = (ys - y0.float()).clamp(0, 1)[..., None]
    x1 = (x0 + 1).clamp(0, TW - 1); y1 = (y0 + 1).clamp(0, TH - 1)
    t = weights6[..., :6]
    w = WH(mix(WH(mix(t[y0, x0], t[y0, x1], fx)), WH(mix(t[y1, x0], t[y1, x1], fx)), fy))
    m1 = splat_valid_weight(splat1, u, v)[..., None]; m0 = splat_valid_weight(splat0, u, v)[..., None]
    s0 = splat_flow_at(splat0, u, v, W); s1 = splat_flow_at(splat1, u, v, W)
    cl = lambda q: torch.minimum(torch.maximum(q, lo), hi)
    rd = lambda img, q: sample_g(img, q[..., 0], q[..., 1])[..., :3]
    c1 = rd(rgb1, cl(uv - s1)); c0 = rd(rgb0, cl(uv - s0))
    uvh = h(uv) if FLOW_UV_F16 else uv
    f0 = h(sample_g(flow0, uvh[..., 0], uvh[..., 1])); f1 = h(sample_g(flow1, uvh[..., 0], uvh[..., 1]))
    w4 = WH(0.5 * w[..., 4:5])
    out = ((((c0 * (1 - m1) + c1 * m1) * w[..., 0:1] + (c1 * (1 - m0) + c0 * m0) * w[..., 1:2])
            + rd(rgb0, cl(uv + f0)) * w[..., 2:3]) + rd(rgb1, cl(uv + f1)) * w[..., 3:4]) + p0 * w4 + p1 * w4
    gen = (~out_rect) & (cb['reset'] == 0)
    out = torch.where(gen[..., None], out, p1)
    if cb['hudlessAttachedFactor'] != 0 and hudless is not None:
        hl = hudless[..., :3]
        a, b = to_linear(p1, cb), to_linear(hl, cb)
        mx = torch.maximum(a, b); mn = torch.minimum(a, b)
        ratio = torch.where(mx != 0, mn / torch.where(mx != 0, mx, torch.ones_like(mx)), torch.zeros_like(mx))
        k = torch.clamp((1 - ratio) * 10, 0, 1).amax(-1, keepdim=True)
        ui = torch.where(k > 0.001, mix(out, hl, k), out)
        ui = torch.where(((p1 - hl).abs() > 0).any(-1, keepdim=True), ui, out)
        out = torch.where(out_rect[..., None], hl, ui)
    if cb['dispatchFlags'] & 1:
        out = out.clone()
        out[:, :16, 1] = 1.0
        out[:, cb['displaySize'][0] - 15:, :] += torch.tensor(cb['debugBarColor'])
    if cb['dispatchFlags'] & 2 and cb['reset'] == 1:
        out = out.clone()
        out[:int(max(32.0, cb['displaySize'][0] * 0.015)), :, 0] = 1.0
    return out

WH = h  # fp16 rounding used by pass190's weight upsample

def read_pass188_weights(scratch, sizes):
    W, H = (int(v) for v in sizes[90])
    return read_scratch(scratch, 0, 12, 11544, 6, W, H)

def to_unorm8(x):
    return torch.floor(torch.clamp(x, 0, 1) * 255.0 + 0.5)

# ---------------------------------------------------------------------------------------------- capture I/O
CB_LAYOUTS = {
    # name: [(field, kind, count)] in std140 order; kind f = float, i = int32, u = uint32, '_' = padding words
    'CsDilate': [('motionVectorJitterCancellation', 'f', 2), ('renderSize', 'i', 2), ('displaySize', 'i', 2),
                 ('dispatchSize', 'i', 2), ('motionVectorScale', 'f', 2), ('generationRectMinUV', 'f', 2),
                 ('generationRectMaxUV', 'f', 2), ('depthInverted', 'u', 1), ('_', 'u', 1),
                 ('colorRectMinUV', 'f', 2), ('colorRectMaxUV', 'f', 2)],
    'CsAMDSplatting64Constants': [('dstSize', 'i', 2), ('renderSize', 'i', 2), ('projAmount', 'f', 1), ('flowIsUv', 'i', 1),
                                  ('generationRectMinUV', 'f', 2), ('generationRectMaxUV', 'f', 2), ('depthInverted', 'u', 1)],
    'CsComputeFwdMVUsingMatConstants': [('prevPstoPs', 'f', 16), ('prevPsToPrev2Ps', 'f', 16), ('displaySize', 'u', 2)],
    'CsFrameInterpolationConstants': [('renderSize', 'u', 2), ('displaySize', 'u', 2), ('reset', 'i', 1),
                                      ('backBufferTransferFunction', 'u', 1), ('hudlessAttachedFactor', 'i', 1),
                                      ('dispatchFlags', 'i', 1), ('minMaxLuminance', 'f', 2), ('generationRectMinUV', 'f', 2),
                                      ('generationRectMaxUV', 'f', 2), ('_', 'u', 2), ('debugBarColor', 'f', 3), ('_', 'u', 1)],
}
PIPE_CB = {191: 'CsDilate', 192: 'CsAMDSplatting64Constants', 193: 'CsComputeFwdMVUsingMatConstants',
           152: 'CsFrameInterpolationConstants', 190: 'CsFrameInterpolationConstants'}

def decode_cb(raw, layout):
    w = np.frombuffer(raw, np.uint32)
    out, k = {}, 0
    for name, kind, n in CB_LAYOUTS[layout]:
        vals = w[k:k + n]; k += n
        if name == '_' : continue
        vals = vals.view(np.float32).tolist() if kind == 'f' else (vals.view(np.int32).tolist() if kind == 'i' else vals.tolist())
        out[name] = vals[0] if n == 1 else vals
    return out

def frame_dispatches(cap, frame):
    """[(pipe id, entry, {binding: desc text}, cbuffer dict or None)] for the dispatches of a frame, in order."""
    L = open(os.path.join(cap, 'log.txt')).read().splitlines()
    ends = [i for i, l in enumerate(L) if l.startswith('FRAME ')]
    lines = L[(ends[frame - 1] + 1 if frame else 0):ends[frame]]
    out = []
    for l in lines:
        m = re.match(r'\d+ s\d+ D \S+ pipe(\d+) (\S+)', l)
        if not m: continue
        pid = int(m.group(1))
        binds = dict(re.findall(r's0b(\d+)\[0\]=(\S+)', l.split(' | ')[0]))
        snaps = dict(re.findall(r'\| buf\d+\+(\d+)=([0-9a-f]+)', l))
        cb = None
        b = re.match(r't6:buf\d+\+(\d+),(\d+)', binds.get('256', ''))
        key = 192 if m.group(2) == 'main' and pid not in PIPE_CB else pid
        num = int(re.search(r'pass(\d+)', m.group(2)).group(1)) if 'pass' in m.group(2) else pid
        if b and b.group(1) in snaps and num in PIPE_CB:
            cb = decode_cb(bytes.fromhex(snaps[b.group(1)]), PIPE_CB[num])
        out.append((pid, m.group(2), {int(k): v for k, v in binds.items()}, cb))
    return out

FMT = {83: (np.float16, 2), 100: (np.float32, 1), 101: (np.uint32, 2), 37: (np.uint8, 4)}

def load_images(cap):
    """{(seq, binding): tensor} from images/index.txt (unorm8 -> [0,1] float, R32G32_UINT -> int64)."""
    d = os.path.join(cap, 'images')
    out, meta = {}, {}
    for l in open(os.path.join(d, 'index.txt')):
        f = l.split()
        W, H = (int(x) for x in f[1].split('x'))
        fmt = int(f[3])
        dt, c = FMT[fmt]
        a = np.fromfile(os.path.join(d, f[0]), dt).reshape(H, W, c)
        t = torch.from_numpy(a.astype(np.int64)) if dt == np.uint32 else torch.from_numpy(a.astype(np.float32))
        if fmt == 37: t = t / 255.0
        seq, b = int(f[0].split('_')[1]), int(f[0].rsplit('_b', 1)[1].split('.')[0])
        out[(seq, b)] = t
        meta[(seq, b)] = l.strip()
    return out, meta

def err(a, b):
    return (a.float() - b.float()).abs().max().item()

def stats(name, a, b, extra=''):
    e = (a - b).abs().double() if a.dtype == torch.int64 else (a.float() - b.float()).abs()
    print(f'{name:34s} max abs err {e.max().item():.6g}  mismatching {(e > 0).float().mean().item() * 100:.4f}%  '
          f'ref range [{b.float().min().item():.4g}, {b.float().max().item():.4g}] {extra}')

# ---------------------------------------------------------------------------------------------- checker
def main():
    cap_a = sys.argv[1] if len(sys.argv) > 1 else r'D:\nfrux_ffx\capture\io_b'
    cap_b = sys.argv[2] if len(sys.argv) > 2 else r'D:\nfrux_ffx\capture\io_c'
    from mlfg_ref import sizes_table
    frame = 2
    sizes = sizes_table(cap_a)
    disp = frame_dispatches(cap_a, frame)
    img, meta = load_images(cap_a)
    seqs = [i for i, d in enumerate(disp) if d[1] == 'main' or re.search(r'_pass(151|152|190)$', d[1])]
    seq_of = {}
    for s, i in enumerate(seqs): seq_of.setdefault(disp[i][0], []).append((s, disp[i][3]))
    for k in sorted(meta): pass
    print('cbuffers:')
    for pid in sorted(seq_of):
        for s, cb in seq_of[pid]:
            if cb: print(f'  seq {s} pipe{pid}: {cb}')

    # pipe191
    s, cb = seq_of[191][0]
    mv, dp = pipe191_dilate(img[(s, 0)], img[(s, 1)], cb)
    stats('pipe191 dilated mv (uv)', mv, img[(s, 128)])
    stats('pipe191 downscaled depth', dp, img[(s, 129)])
    # pipe192 x3 (frame order: splat1to0, splat0to05, splat1to05)
    for s, cb in seq_of[192]:
        sp = pipe192_splat(img[(s, 0)], img[(s, 1)], cb)
        ref = img[(s, 128)]
        stats(f'pipe192 splat (seq {s}, proj {cb["projAmount"]}, uv {cb["flowIsUv"]})', sp, ref,
              f'exact {bool((sp == ref).all())}')
    # pipe193
    s, cb = seq_of[193][0]
    fw = pipe193_forward_mv(img[(s, 0)], img[(s, 1)], img[(s, 2)], cb)
    stats('pipe193 forward mv', fw, img[(s, 128)])
    # pass151
    pas = os.path.join(cap_a, 'passes')
    s, _ = seq_of[151][0]
    f0, f1 = pass151_flows(open(os.path.join(pas, f'{frame}_150.bin'), 'rb').read(), sizes)
    stats('pass151 flow_0to05', f0, img[(s, 129)]); stats('pass151 flow_1to05', f1, img[(s, 130)])
    # pass152
    s, cb = seq_of[152][0]
    x = pass152_stage2_input(img[(s, 1)], img[(s, 2)], img[(s, 3)], img[(s, 4)], img[(s, 5)], img[(s, 6)], cb, sizes)
    ref = read_pass152(open(os.path.join(pas, f'{frame}_152.bin'), 'rb').read(), sizes)
    stats('pass152 stage-2 input (24 ch)', x, ref)
    e = (x - ref).abs().amax((0, 1))
    print('    per-channel max err', [round(v, 5) for v in e.tolist()])
    # pass190 (weights from the other capture's pass-189 dump; inputs from its own image dump)
    img_b, _ = load_images(cap_b)
    disp_b = frame_dispatches(cap_b, frame)
    cb = [d[3] for d in disp_b if d[1].endswith('_pass190')][0]
    s = max(k[0] for k in img_b)
    wts = read_pass188_weights(open(os.path.join(cap_b, 'passes', f'{frame}_189.bin'), 'rb').read(), sizes)
    out = pass190_composite(wts, img_b[(s, 1)], img_b[(s, 2)], img_b[(s, 3)], img_b[(s, 4)], img_b[(s, 5)], img_b[(s, 6)],
                            cb, sizes, img_b.get((s, 7)))
    o8 = to_unorm8(out)
    stats('pass190 output (unorm8 levels)', o8, img_b[(s, 129)][..., :3] * 255)
    full = torch.from_numpy(np.fromfile(os.path.join(cap_b, f'output_{frame}.rgba'), np.uint8).reshape(1080, 1920, 4).astype(np.float32))
    stats('pass190 vs output_2.rgba', o8, full[..., :3])
    print('    weights (ch 0-5) mean', [round(v, 4) for v in wts.mean((0, 1)).tolist()], 'sum0-4 range',
          (wts[..., :5].sum(-1).min().item(), wts[..., :5].sum(-1).max().item()))

if __name__ == '__main__':
    main()
