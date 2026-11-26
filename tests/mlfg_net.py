"""Checks NFRUX's FSR Frame Generation 4 interpolator (core/mlfg, the DLL's code run by build-*/Release/mlfg_run.exe) against
the PyTorch specification tools/ffx_extract/mlfg_net.py (AmdMlfgNet, precision 'amd', sampler 'point', flow_t 'scale';
run on the CPU in fp32, no TF32) on real frames of the NFRU-MFG test sets, per tier and API:
  * the generated frames (8-bit): PSNR and the largest difference in codes against mlfg_net,
  * the network's tensors after the last frame (net_in, f54 .. f149, flowout, s2in, logits, weights): max error, fp16 ulps,
  * D3D12 against Vulkan: identical bytes for the same tier.

    python tests/mlfg_net.py [--exe build-ship/Release/mlfg_run.exe] [--work D:/nfrux_ffx/work/mlfg_net] [--tiers generic]
                             [--apis d3d12,vulkan] [--n 2,4] [--sample test_new:0] [--bench 0]
SPDX-License-Identifier: MIT
"""
import argparse, os, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'ffx_extract'))
sys.path.insert(0, r'C:\dev\modelgym\ng-model-gym\src')
import mlfg_net as N  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--exe', default=str(HERE.parent / 'build-ship' / 'Release' / 'mlfg_run.exe'))
ap.add_argument('--work', default=r'D:\nfrux_ffx\work\mlfg_net')
ap.add_argument('--tiers', default='generic')
ap.add_argument('--apis', default='d3d12,vulkan')
ap.add_argument('--n', default='2,4', help='multipliers: n - 1 frames at t = i / n')
ap.add_argument('--sample', default='test_new:0', help='set:index of the evaluation sample (eval_mlfg_net sampling)')
ap.add_argument('--bench', type=int, default=0)
ap.add_argument('--reuse', action='store_true', help='keep the reference outputs of an earlier run')
A = ap.parse_args()
torch.set_num_threads(4)
torch.backends.cudnn.allow_tf32 = False
W = Path(A.work)
W.mkdir(parents=True, exist_ok=True)
RESULTS = []


def report(name, msg, ok):
    RESULTS.append((name, ok))
    print(f'  [{"ok" if ok else "!!"}] {name:62s} {msg}', flush=True)


def ulp16(a, b):
    ia = a.astype(np.float16).view(np.int16).astype(np.int32)
    ib = b.astype(np.float16).view(np.int16).astype(np.int32)
    ia = np.where(ia < 0, -32768 - ia, ia); ib = np.where(ib < 0, -32768 - ib, ib)
    return np.abs(ia - ib)


def psnr8(a, b):
    e = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if e == 0 else 10 * np.log10(255.0 ** 2 / e)


def load_sample():
    """One evaluation sample as eval_mlfg_net samples them (crop 1080x1920, exposure 2.0, linear colour)."""
    from ng_model_gym.usecases.nfru_mfg.train_mfg import PairDataset, load_sample as ls
    sname, idx = A.sample.split(':')
    seqs = sorted(Path(rf'C:\dev\modelgym\data\nfru6x\{sname}').rglob('*.safetensors'))
    cfg = PairDataset(seqs[:1], 1, train=False, crop=(1080, 1920))
    info = cfg.infos[0]
    s = info.step
    c = [p for p in range(2 * s, info.length) if p % info.base_stride == 0]
    p1 = sorted(random.Random(1).sample(c, min(4, len(c))))[int(idx)]
    return ls(info, p1, [s // 2], random.Random(0), cfg), f'{info.path.stem}_{p1}'


def prepare():
    d, key = load_sample()
    f = lambda k: d[k].float()
    q8 = lambda x: torch.round(x.clamp(0, 1) * 255.0)
    rgb0, rgb1 = q8(f('rgb_m1')), q8(f('rgb_p1'))
    mv = lambda k: torch.stack([f(k)[1], f(k)[0]]).half()          # (x, y) render px, fp16 as the DLL reads them
    dep = lambda k: (1.0 - f(k)).float()                            # inverted device depth
    mv0, mv1, d0, d1 = mv('mv_m1'), mv('mv_p1'), dep('depth_m1'), dep('depth_p1')
    vps = {k: f(k).float() for k in ('vp_m3', 'vp_m1', 'vp_p1')}
    H, Wd = rgb1.shape[-2:]
    rh, rw = d1.shape[-2:]

    def save(name, arr):
        np.ascontiguousarray(arr).tofile(W / name)
    for n, img in (('c0', rgb0), ('c1', rgb1)):
        a = np.concatenate([img.permute(1, 2, 0).numpy().astype(np.uint8), np.full((H, Wd, 1), 255, np.uint8)], -1)
        save(f'{n}.bin', a)
    for n, m in (('m0', mv0), ('m1', mv1)):
        save(f'{n}.bin', m.permute(1, 2, 0).numpy().astype(np.float16))
    for n, x in (('d0', d0), ('d1', d1)):
        save(f'{n}.bin', x[0].numpy().astype(np.float32))
    inputs = dict(prev=(rgb0 / 255.0)[None], cur=(rgb1 / 255.0)[None], mv0=mv0.float()[None], mv1=mv1.float()[None],
                  d0=d0[None], d1=d1[None],
                  A=(vps['vp_p1'].double() @ torch.linalg.inv(vps['vp_m1'].double())).T.float()[None],
                  B=(vps['vp_m3'].double() @ torch.linalg.inv(vps['vp_m1'].double())).T.float()[None])
    vpl = {k: ' '.join(f'{x:.9g}' for x in v.T.contiguous().view(-1).tolist()) for k, v in vps.items()}  # column-major
    return inputs, vpl, (Wd, H, rw, rh), key


def frames_txt(n, vpl, size):
    """Frame Z (reset, VP_-1), frame A (frame 0, VP_0), frame B (frame 1, VP_1) with n - 1 generated frames."""
    Wd, H, rw, rh = size
    return (f'size {Wd} {H} {rw} {rh}\n'
            f'frame c0.bin d0.bin m0.bin 1 1 1 {vpl["vp_m3"]}\n'
            f'frame c0.bin d0.bin m0.bin 0 1 1 {vpl["vp_m1"]}\n'
            f'frame c1.bin d1.bin m1.bin 0 {n - 1} {n - 1} {vpl["vp_p1"]}\n')


class Double:
    """Every convolution of mlfg_net summed in fp64 (the 'amd' roundings unchanged): how far fp32 summation order alone
    moves this network's output, the yardstick for the GPU tiers."""
    def __enter__(self):
        F = torch.nn.functional
        self.c, self.t = torch.nn.Conv2d.forward, torch.nn.ConvTranspose2d.forward
        torch.nn.Conv2d.forward = lambda m, x: F.conv2d(x.double(), m.weight.double(), m.bias.double(), m.stride, m.padding).float()
        torch.nn.ConvTranspose2d.forward = lambda m, x, output_size=None: F.conv_transpose2d(x.double(), m.weight.double(), m.bias.double(), m.stride).float()

    def __exit__(self, *a):
        torch.nn.Conv2d.forward, torch.nn.ConvTranspose2d.forward = self.c, self.t


def reference(inputs, n, double=False):
    if double:
        with Double():
            return reference(inputs, n)
    net = N.AmdMlfgNet(precision='amd', sampler='point')
    net.flow_t = 'scale'
    outs, aux = [], None
    with torch.no_grad():
        Wd, H = inputs['cur'].shape[-1], inputs['cur'].shape[-2]
        hist = net.history(inputs['mv0'], inputs['d0'], (Wd, H))
        for i in range(1, n):
            o, aux = net(inputs['prev'], inputs['cur'], inputs['d1'], inputs['mv1'], hist, inputs['A'], inputs['B'], i / n)
            outs.append(torch.floor(o[0].clamp(0, 1) * 255.0 + 0.5).permute(1, 2, 0).numpy().astype(np.uint8))
    return outs, aux


def nhwc(t, pitch):
    C, H, Wd = t.shape
    a = np.zeros((H, Wd, pitch), np.float32)
    a[..., :C] = t.permute(1, 2, 0).numpy()
    return a


def main():
    inputs, vpl, size, key = prepare()
    Wd, H = size[0], size[1]
    print(f'sample {key}: display {Wd}x{H}, render {size[2]}x{size[3]}')
    for n in [int(x) for x in A.n.split(',')]:
        ref, aux = reference(inputs, n)
        ref64, _ = reference(inputs, n, double=True)
        base = []
        for k, (a, b) in enumerate(zip(ref, ref64)):
            d = np.abs(a.astype(np.int32) - b.astype(np.int32))
            base.append(psnr8(a, b))
            print(f'  {n}x t={(k + 1) / n:.3f}: mlfg_net fp32 sums vs fp64 sums: PSNR {base[-1]:.2f} dB, max {d.max()} codes (the yardstick)')
        (W / 'frames.txt').write_text(frames_txt(n, vpl, size))
        got = {}
        for tier in A.tiers.split(','):
            for api in A.apis.split(','):
                if (tier, api) == ('coopmat', 'd3d12'):
                    continue  # coopmat is Vulkan's
                env = dict(os.environ, NFRUX_MLFG=tier)
                r = subprocess.run([A.exe, api, str(W)] + (['--bench', str(A.bench)] if A.bench else []), capture_output=True, text=True, env=env)
                print('   ', r.stdout.strip().replace('\n', '\n    '))
                if r.returncode:
                    report(f'{n}x {tier} {api}: run', r.stderr[-300:], False)
                    continue
                outs = [np.fromfile(W / f'out_{api}_f2_k{k}.bin', np.uint8).reshape(H, Wd, 4)[..., :3] for k in range(n - 1)]
                got[(tier, api)] = outs
                for k, (o, rf) in enumerate(zip(outs, ref)):
                    d = np.abs(o.astype(np.int32) - rf.astype(np.int32))
                    p, p64 = psnr8(o, rf), psnr8(o, ref64[k])
                    # Pass: no further from the fp64-summed reference than fp32 torch is, within 3 dB.
                    report(f'{n}x t={(k + 1) / n:.3f} {tier} {api} vs mlfg_net', f'PSNR {p:.2f} dB (vs fp64 sums {p64:.2f}), max {d.max()} codes, '
                           f'{np.mean(d > 0) * 100:.3f}% values differ, {np.mean(d > 1) * 100:.4f}% by >1', p64 > base[k] - 3)
                if n == 2:  # the tensors (t = 0.5)
                    names = dict(net_in=('net_in', 32, 18), f54=('f54', 16, 13), f83=('f83', 16, 13), f112=('f112', 16, 13), f149=('f149', 16, 13),
                                 flowout=('flow', 4, 4), s2in=('stage2_in', 32, 24), weights=('weights', 8, 5))
                    for tn, (an, pitch, C) in names.items():
                        rt = aux[an][0]
                        g = np.fromfile(W / f't_{api}_{tn}.bin', np.float16).astype(np.float32)
                        Ht, Wt = rt.shape[-2:]
                        g = g[:Ht * Wt * pitch].reshape(Ht, Wt, pitch)[..., :C]
                        r_ = nhwc(rt, C)
                        e = np.abs(g - r_)
                        u = ulp16(g, r_)
                        report(f'   {tier} {api} {tn}', f'max err {e.max():.4g}, max ulp {u.max()}, {np.mean(u > 0) * 100:.3f}% differ', True)
        tiers = A.tiers.split(',')
        apis = A.apis.split(',')
        for tier in tiers:
            if len(apis) > 1 and all((tier, a) in got for a in apis):
                same = all(np.array_equal(got[(tier, apis[0])][k], got[(tier, a)][k]) for a in apis[1:] for k in range(n - 1))
                # Identical bytes are the strict build's promise; the shipped build lets compilers fuse the generic convolutions.
                strict = 'strict' in A.exe.lower()
                report(f'{n}x {tier}: {" vs ".join(apis)}', ('bytes identical' if same else 'BYTES DIFFER') + ('' if strict else ' (shipped build: informative)'),
                       same or tier != 'generic' or not strict)
        if 'generic' in tiers:
            for tier in tiers:
                if tier == 'generic': continue
                for api in apis:
                    if (tier, api) in got and ('generic', api) in got:
                        for k in range(n - 1):
                            a, b = got[(tier, api)][k], got[('generic', api)][k]
                            d = np.abs(a.astype(np.int32) - b.astype(np.int32))
                            report(f'{n}x t={(k + 1) / n:.3f} {tier} vs generic ({api})', f'PSNR {psnr8(a, b):.2f} dB, max {d.max()} codes', True)
    bad = [n for n, ok in RESULTS if not ok]
    print(f'{len(RESULTS) - len(bad)}/{len(RESULTS)} checks ok' + ('' if not bad else '; failing: ' + '; '.join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
