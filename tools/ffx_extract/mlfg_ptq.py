"""Post-training int8 quantisation of AMD FSR FG4's network (or a fine-tuned AMD-format npz) with ng_model_gym's QAT
scheme (amd_mlfg/qat.py): enable_qat, EMA-calibrate the activation scales on N training samples (no weight updates),
save a QAT checkpoint and export it with export_amd (float npz + OUT_int8.npz / .json). The stand-in for the int8 QAT
models until those exist; the int8 runtime (core/mlfg) and mlfg_int8.py read the same export.

    python tools/ffx_extract/mlfg_ptq.py OUT_DIR [--weights amd_mlfg_4k.npz] [--samples 48] [--cpu]
SPDX-License-Identifier: MIT
"""
import argparse, os, sys, time
from pathlib import Path
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MLFG_NET_PATH', HERE)
sys.path.insert(0, r'C:\dev\modelgym\ng-model-gym\src')

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--weights', default=r'D:\nfrux_ffx\work\amd_mlfg_4k.npz')
ap.add_argument('--samples', type=int, default=48)
ap.add_argument('--crop', type=int, nargs=2, default=[432, 768])
ap.add_argument('--data', default=r'C:\dev\modelgym\data')
ap.add_argument('--cpu', action='store_true')
a = ap.parse_args()

from ng_model_gym.usecases.amd_mlfg import qat as Q  # noqa: E402
from ng_model_gym.usecases.amd_mlfg.data import AmdPairDataset, files, to_dev  # noqa: E402
from ng_model_gym.usecases.amd_mlfg.model import TrainableMlfgNet, run  # noqa: E402
from ng_model_gym.usecases.amd_mlfg.export_amd import export_float, export_int8  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device('cpu' if a.cpu else 'cuda')
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
net = TrainableMlfgNet(weights=a.weights, precision='train', sampler='point').to(dev)
net.flow_t = 'scale'
Q.enable_qat(net)
Q.set_qat_mode(net, 'calib')
net.train()
d = a.data
ds = AmdPairDataset(files([rf'{d}\arm-nfru\nfru\train', rf'{d}\nfru6x\train']), items_per_epoch=a.samples, k_max=2,
                    crop=tuple(a.crop), train=True, srgb_prob=0.5, arm_share=0.3, render_scales=(1.0, 0.75), seed=7)
t0 = time.time()
with torch.no_grad():
    for i in range(a.samples):
        s = ds[i]
        x = to_dev(s, dev)
        for t in s['t'].tolist():
            run(net, x, float(t))
print(f'calibrated {a.samples} samples in {time.time() - t0:.0f} s', flush=True)
Q.set_qat_mode(net, 'frozen')
net.eval()
ck = out / 'ptq.pt'
torch.save({'net': net.state_dict(), 'step': 0, 'qat': True, 'qat_fp_heads': False, 'flow_t': 'scale', 'sampler': 'point',
            'args': dict(ptq=True, weights=a.weights, samples=a.samples, crop=a.crop)}, ck)
p = export_float(net.cpu(), a.weights, out / 'amd_mlfg_ptq.npz', meta=dict(ckpt=str(ck), ptq=True, samples=a.samples))
q = export_int8(net, a.weights, out / 'amd_mlfg_ptq_int8.npz')
print('checkpoint', ck, '\nfloat', p, '\nint8', q)
