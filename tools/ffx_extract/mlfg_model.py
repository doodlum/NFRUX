"""Byte-exact emulator of AMD FSR Frame Generation 4's network (the "mlfi" passes), in PyTorch.

The scratch buffer is emulated as the same bytes AMD uses; every pass reads and writes it through the decoded tensor
descriptors (graph.json), so the wiring between passes (channel windows, concatenation by layout, ping-pong buffers)
is exactly AMD's. Convolution passes use the exported weights (export_weights.py); glue passes come from
mlfg_glue.py / mlfg_io.py.

AMD semantics, not NVIDIA's: channels >= C of a tensor whose C % 16 == 8 are zero (AMD's lane layout); FP8 passes
compute fp8 x fp8 -> fp32 exactly (on NVIDIA AMD's 16x16x16 FP8 shapes are unsupported).

    from mlfg_model import Emulator
    em = Emulator(capture_dir, weights_npz, device='cuda')
    em.run_passes(scratch, first, last)          # scratch: torch.uint8 tensor (58 MB) on device
"""
import json, os, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlfg_ref import sizes_table, elem

F8 = torch.float8_e4m3fn

# ---------------------------------------------------------------- scratch tensor access (device, vectorised)
def _view(scratch, t, W, H, C):
    """Index tensor of byte offsets [C, H, W] (first byte of each element) for descriptor t."""
    dev = scratch.device
    e = elem(t); s = t['strides']; pb = t['padBegin']
    x = torch.arange(W, device=dev).view(1, 1, W) + pb[0]
    y = torch.arange(H, device=dev).view(1, H, 1) + pb[1]
    c = torch.arange(C, device=dev).view(C, 1, 1)
    if t['kind'] == 'Tensor4':
        off = t['offset'] + x * s[0] + y * s[1] + c * s[2]
    else:
        B = t['blocks'][2]
        off = t['offset'] + x * s[0] + y * s[1] + (c // B) * s[2] + (c % B) * e
    return off, e

def read(scratch, t, W, H, C):
    off, e = _view(scratch, t, W, H, C)
    if e == 2:
        b = scratch[off].to(torch.int32) | (scratch[off + 1].to(torch.int32) << 8)
        return b.to(torch.int16).view(torch.float16).float()
    return scratch[off].view(F8).float()

def write(scratch, t, v, W, H):
    C = v.shape[0]
    off, e = _view(scratch, t, W, H, C)
    if e == 2:
        b = v.to(torch.float16).view(torch.int16).to(torch.int32)
        scratch[off] = (b & 255).to(torch.uint8); scratch[off + 1] = ((b >> 8) & 255).to(torch.uint8)
    else:
        scratch[off] = v.to(F8).view(torch.uint8)

def q16(v): return v.to(torch.float16).float()
def q8(v): return v.to(F8).float()
def leaky(v): return torch.where(v > 0, v, 0.2 * v)

NV_LANES = False   # True reproduces what AMD's shaders do on NVIDIA (for checking against NVIDIA captures only)

def amd_zero(v, C):
    """AMD lanes >= 16 zeroing: for C % 16 == 8 the last 8 channels of the last 16-block are zero."""
    if C % 16 == 8 and v.shape[0] > C:
        v = v.clone(); base = C // 16 * 16
        rows = [4, 5, 6, 7, 12, 13, 14, 15] if NV_LANES else list(range(8, 16))
        for r in rows:
            if base + r < v.shape[0]: v[base + r] = 0
    return v

class Emulator:
    def __init__(self, capture, weights_npz, device='cuda'):
        self.graph = {g['pass']: g for g in json.load(open(os.path.join(capture, 'graph.json')))}
        self.man = {int(k): v for k, v in json.load(open(os.path.splitext(weights_npz)[0] + '.json')).items()}
        z = np.load(weights_npz)
        self.w = {k: torch.from_numpy(z[k]).to(device) for k in z.files}
        self.sizes = torch.from_numpy(sizes_table(capture).astype(np.int64))
        self.dev = device
        self.glue = {}

    def size(self, idx):
        return int(self.sizes[idx][0]), int(self.sizes[idx][1])

    # ------------------------------------------------------------ conv passes
    def conv_pass(self, scratch, p):
        g, m = self.graph[p], self.man[p]
        ts = g['tensors']
        acts = [t for t in ts if t['kind'] in ('Tensor4', 'Blocked5')]
        out_t = acts[-1]
        ow, oh = self.size(g['tensorSizeIdx'][0])
        W = self.w[m['layer']['w']]; b = self.w[m['layer']['b']]
        O, I = W.shape[0], W.shape[1]
        if m['op'] == 'conv_transpose':
            iw, ih = ow // 2, oh // 2
        elif m['stride'] == 2:
            iw, ih = ow * 2, oh * 2
        else:
            iw, ih = ow, oh
        x = read(scratch, acts[0], iw, ih, I)            # the shader reads whole 16-channel windows
        if m['op'] == 'conv_transpose':
            y = F.conv_transpose2d(x[None], W.permute(1, 0, 2, 3), b, stride=2)[0]
        else:
            pad = (m['k'] - 1) // 2 if m['stride'] == 1 else 0
            y = F.conv2d(x[None], W, b, stride=m['stride'], padding=pad)[0]
        C = out_t['logical'][2]
        if m['residual']: y = y + x[:y.shape[0]]
        if m['act'] == 'leaky': y = leaky(y)
        elif m['act'] == 'relu': y = torch.relu(y)
        elif m['act'] == 'softmax_max0':
            y = q16(y[:C])
            mx = torch.clamp(y.amax(0, keepdim=True), min=0.0)
            e = torch.exp(y - mx); y = e / e.sum(0, keepdim=True)
        y = amd_zero(y, C)
        Cw = C if out_t['kind'] == 'Tensor4' and out_t['storage'][2] < (C + 15) // 16 * 16 else min(y.shape[0], (C + 15) // 16 * 16)
        write(scratch, out_t, y[:Cw], ow, oh)

    def fasternet_pass(self, scratch, p):
        g, m = self.graph[p], self.man[p]
        ts = g['tensors']
        acts = [t for t in ts if t['kind'] in ('Blocked5',)]
        ow, oh = self.size(g['tensorSizeIdx'][0])
        Wp, bp = self.w[m['pconv']['w']], self.w[m['pconv']['b']]
        We, be = self.w[m['expand']['w']], self.w[m['expand']['b']]
        Wj, bj = self.w[m['project']['w']], self.w[m['project']['b']]
        Cn, P = m['channels'], m['pconv_channels']
        if m['op'] == 'fasternet':
            x = read(scratch, acts[0], ow, oh, Cn)
        else:  # 176: block at low res, then up-conv to the skip's resolution
            x = read(scratch, acts[0], ow // 2, oh // 2, Cn)
        pc = q8(F.conv2d(x[None, :P], Wp, bp, padding=1)[0])
        h = torch.cat([pc, x[P:]], 0)
        h = q8(torch.relu(F.conv2d(h[None], We, be)[0]))
        y = q8(F.conv2d(h[None], Wj, bj)[0] + x)
        if m['op'] != 'fasternet':
            Wu, bu = self.w[m['up']['w']], self.w[m['up']['b']]   # [O=48, I=64, 2, 2]
            skip = read(scratch, acts[1], ow, oh, Wu.shape[0])
            y = q8(F.conv_transpose2d(y[None], Wu.permute(1, 0, 2, 3), bu, stride=2)[0] + skip)
        write(scratch, acts[-1], y, ow, oh)

    def run_passes(self, scratch, first, last, ctx=None):
        for p in range(first, last + 1):
            if p not in self.graph: continue
            m = self.man.get(p)
            if m and m.get('op', '').startswith('fasternet'):
                self.fasternet_pass(scratch, p)
            elif m:
                self.conv_pass(scratch, p)
            elif p in self.glue:
                self.glue[p](scratch, ctx)
            # halo / no-op passes: the emulator keeps halos zero by never writing them
