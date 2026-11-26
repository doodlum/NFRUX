"""Writes the weights of AMD FSR Frame Generation 4's ML frame interpolation network (FidelityFX SDK v2.4.0, MIT; see
docs/FFX_MLFG.md) as NFRUX embeds them: one blob, layers in the order core/mlfg/mlfg_network.cpp runs them.

    python tools/mlfg_weights.py [NPZ] [--xz core/model/amd_mlfg_weights.xz] [--bin FILE]

NPZ defaults to mlfg_net's (D:\\nfrux_ffx\\work\\amd_mlfg_4k.npz). The weights are read through
tools/ffx_extract/mlfg_net.AmdMlfgNet.load_amd, so they are in the orientation that module runs them (passes 186 / 188
included). A retrained npz of the same format gives a drop-in file: --bin writes the same blob as a file, which the DLL
loads instead of its own when nfrux.ini's [General] AmdWeights names it.

Blob (little endian, every item 8-byte aligned):
  header  u32 magic 'MLFG', u32 version 1, u32 layer count, u32 total bytes
  layer   u32 kind (0 conv, 1 transposed conv 2x2/2), k, stride, cin, cout, fp8, weight offset, bias offset (bytes)
  weights conv [cout][cin][k][k], transposed [cin][cout][2][2] (torch's layouts); fp16, or fp8 e4m3 bytes when fp8
  biases  [cout] fp32
SPDX-License-Identifier: MIT
"""
import argparse, os, struct, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ffx_extract'))
import mlfg_net  # noqa: E402


def layers(net):
    """(name, module, transposed, fp8) in the runtime's order."""
    out = []
    for enc, n in ((net.encA, 'encA'), (net.encB, 'encB')):
        out += [(f'{n}.down', enc.down, False), (f'{n}.c1', enc.c1, False), (f'{n}.c2', enc.c2, False), (f'{n}.up', enc.up, True)]
    for i, b in enumerate((net.block1, net.block2, net.block3, net.block4)):
        n = f'block{i + 1}'
        out += [(f'{n}.d1', b.d1, False), (f'{n}.c1', b.c1, False), (f'{n}.d2', b.d2, False)]
        out += [(f'{n}.res{j}', r, False) for j, r in enumerate(b.res)]
        out += [(f'{n}.u1', b.u1, True), (f'{n}.u2', b.u2, True)]
    s = net.stage2
    out += [('s2.inp', s.inp, False), ('s2.d1', s.d1, False)]

    def fn(prefix, blocks):
        r = []
        for j, m in enumerate(blocks):
            r += [(f'{prefix}{j}.pconv', m.pconv, False), (f'{prefix}{j}.expand', m.expand, False), (f'{prefix}{j}.project', m.project, False)]
            if m.up is not None:
                r.append((f'{prefix}{j}.up', m.up, True))
        return r
    out += fn('s2.b1.', s.b1) + [('s2.d2', s.d2, False)] + fn('s2.b2.', s.b2) + fn('s2.b3.', s.b3)
    out += [('s2.up', s.up, True), ('s2.head', s.head, False)]
    return out


def blob(npz):
    net = mlfg_net.AmdMlfgNet(weights=npz, precision='amd')
    L = layers(net)
    F8 = torch.float8_e4m3fn
    head = struct.pack('<4I', 0x47464C4D, 1, len(L), 0)
    table, data = b'', b''
    base = 16 + 32 * len(L)
    pad = lambda b: b + b'\0' * (-len(b) % 8)
    names = []
    for name, m, tr in L:
        w = m.weight.detach().float().contiguous()
        b = m.bias.detach().float().contiguous()
        if not tr:
            cout, cin, k, _ = w.shape
        else:
            cin, cout, k, _ = w.shape
        stride = m.stride[0]
        assert bool((w.half().float() == w).all()), name
        fp8 = name.startswith('s2.') and bool((w.to(F8).float() == w).all()) and name not in ('s2.inp', 's2.d1', 's2.head')
        wb = w.to(F8).view(torch.uint8).numpy().tobytes() if fp8 else w.half().numpy().tobytes()
        woff = base + len(data)
        data = pad(data + wb)
        boff = base + len(data)
        data = pad(data + b.numpy().astype('<f4').tobytes())
        table += struct.pack('<8I', 1 if tr else 0, k, stride, cin, cout, 1 if fp8 else 0, woff, boff)
        names.append((name, tr, k, stride, cin, cout, fp8))
    total = base + len(data)
    return head[:12] + struct.pack('<I', total) + table + data, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='?', default=mlfg_net.DEFAULT_WEIGHTS)
    ap.add_argument('--xz', default=os.path.join(HERE, '..', 'core', 'model', 'amd_mlfg_weights.xz'))
    ap.add_argument('--bin')
    a = ap.parse_args()
    b, names = blob(a.npz)
    if a.bin:
        open(a.bin, 'wb').write(b)
        print(f'{a.bin}: {len(b)} bytes, {len(names)} layers')
        return
    # The repository keeps the blob compressed (byte planes of the 16-bit words, then xz); tools/embed_weights.py (standard
    # library only) expands it into the header the build includes.
    import lzma
    raw = b + b'\0' * (len(b) & 1)
    plane = raw[0::2] + raw[1::2]
    with open(a.xz, 'wb') as f:
        f.write(len(b).to_bytes(4, 'little') + lzma.compress(plane, preset=9 | lzma.PRESET_EXTREME))
    print(f'{a.xz}: {len(b)} bytes, {len(names)} layers, {os.path.getsize(a.xz)} compressed')

if __name__ == '__main__':
    main()
