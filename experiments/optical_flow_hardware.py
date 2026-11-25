import sys, random, subprocess, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\dev\modelgym\ng-model-gym\src")
from ng_model_gym.usecases.nfru_mfg import pipeline as P
from ng_model_gym.usecases.nfru_mfg.eval_mfg import load_net
from ng_model_gym.usecases.nfru_mfg.flow import FlowEstimator
from ng_model_gym.usecases.nfru_mfg.train_mfg import PairDataset, build_state, load_sample, psnr
D = Path(r"C:\dev\NFRUX\build\nvof"); D.mkdir(exist_ok=True)
seqs = [p for d in sys.argv[1:] for p in sorted(Path(d).rglob("*.safetensors"))]
cfg = PairDataset(seqs, 1, train=False); dev = torch.device("cuda"); fe = FlowEstimator().to(dev)
net, tc = load_net(r"C:\dev\modelgym\output\nfru_mfg_v14\qat15\last.pt", dev)
items = []
for info in cfg.infos:
    s = info.step
    c = [p for p in range(2 * s, info.length) if p % info.base_stride == 0]
    for p1 in sorted(random.Random(1).sample(c, 2)):
        n = {6: 6, 5: 5, 4: 4, 3: 3}[s]
        items.append((info, p1, n))
lum = torch.tensor((0.2126, 0.7152, 0.0722)).view(3, 1, 1)
for i, (info, p1, n) in enumerate(items):
    d = load_sample(info, p1, [1], random.Random(0), cfg)
    for k, key in (("a", "rgb_m1"), ("b", "rgb_p1")):
        ((d[key].clamp(0, 1) * lum).sum(0) * 255 + 0.5).clamp(0, 255).byte().numpy().tofile(D / f"{k}{i}.bin")
for lv in ("1", "2"):
    print(subprocess.run([r"C:\dev\NFRUX\build\Release\optical_flow_hardware.exe", "1280", "720", str(len(items)), str(D)], capture_output=True, text=True, env=dict(os.environ, NVOF_LEVEL=lv)).stdout)
    acc = {}
    for i, (info, p1, n) in enumerate(items):
        d = load_sample(info, p1, [k * info.step // n for k in range(1, n)], random.Random(0), cfg)
        crop = tuple(d["display_crop"].tolist())
        f = np.fromfile(D / f"f{i}.bin", np.int16).reshape(180, 320, 2).astype(np.float32) / 32
        hw = torch.from_numpy(np.stack((f[..., 1] / 720, f[..., 0] / 1280)))
        with torch.no_grad():
            for who in ("block match", "nvof", "none"):
                e = dict(d)
                e.pop("flow", None)
                if who == "nvof": e["flow"] = hw
                if who == "none": e["flow"] = torch.zeros_like(hw)
                st = build_state(e, fe, dev)
                for j, t in enumerate(d["t"].tolist()):
                    o, _ = P.run_timestep(st, t, net, tc, display_crop=crop, logit_prior=True)
                    acc.setdefault(who, {}).setdefault(f"{n}x", []).append(psnr(o.cpu(), d["gt"][j].unsqueeze(0)))
    for n in sorted(acc["nvof"]): print("level", lv, n, {w: round(float(np.mean(v[n])), 2) for w, v in acc.items()}, len(acc["nvof"][n]), flush=True)