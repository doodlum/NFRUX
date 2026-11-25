"""A3 / A14: NFRUX (the port, D3D12, shipped build) against the PyTorch pipeline with the same network and against Arm's NFRU v2, on
held-out clips with true in-between frames. Runs in C:\\dev\\modelgym's venv; gym code is imported, not copied.
    python reference.py OUT_JSON [test dirs...]     SPDX-License-Identifier: MIT"""
import json, os, random, subprocess, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\dev\modelgym\ng-model-gym\src")
from ng_model_gym.usecases.nfru_mfg import pipeline as P
from ng_model_gym.usecases.nfru_mfg.data_utils import load_pair
from ng_model_gym.usecases.nfru_mfg.eval_mfg import load_net
from ng_model_gym.usecases.nfru_mfg.flow import FlowEstimator
from ng_model_gym.usecases.nfru_mfg.train_mfg import PairDataset, build_state, load_sample, psnr

PARITY = os.environ.get("PARITY", r"C:\dev\NFRUX\build\Release\parity.exe")
API = os.environ.get("API", "d3d12")
CKPT = {"torch": r"C:\dev\modelgym\output\nfru_mfg_v14\qat15\last.pt", "arm": "arm"}
PAIRS = int(os.environ.get("PAIRS", "4"))
out = Path(sys.argv[1]); tmp = out.parent / "rec"; gen = out.parent / "gen"
for d in (tmp, gen): d.mkdir(parents=True, exist_ok=True)
seqs = [p for d in (sys.argv[2:] or [r"C:\dev\modelgym\data\nfru6x\test_new"]) for p in sorted(Path(d).rglob("*.safetensors")) if os.environ.get("ONLY", "") in p.name]
dev = torch.device("cuda"); cfg = PairDataset(seqs, 1, train=False); fe = FlowEstimator().to(dev)
tone = lambda x: (lambda y: y / (1 + y))(x * float(np.exp(2.0)))


def export(info, p1):
    """Three base frames (p1 - 2s, p1 - s, p1) as parity --replay reads them."""
    s, prev = info.step, None
    import safetensors
    with safetensors.safe_open(str(info.path), framework="numpy") as f:
        for i in range(3):
            t = p1 - (2 - i) * s; b = t // info.base_stride
            g = lambda k, j: np.ascontiguousarray(f.get_slice(k)[j]).astype(np.float32)
            rgb = tone(g("rgb_linear", t)); d = g("depth", b)[0]; mv = g("mv_{}_f30_m1", b); vp = g("ViewProj", b).reshape(4, 4).astype(np.float64)
            fov = float(np.ravel(g("FovY", b))[0])
            h, w = rgb.shape[-2:]; lh, lw = d.shape
            c = np.full((h, w, 4), 255, np.uint8); c[..., :3] = np.clip(rgb[::-1].transpose(1, 2, 0) * 255 + 0.5, 0, 255).astype(np.uint8)
            n = f"{w}x{h}_f{i}_k1.bin"
            c.tofile(tmp / f"hudless_{n}"); (1.0 - d).astype(np.float32).tofile(tmp / f"depth_{n}")
            np.stack((mv[1], mv[0]), -1).astype(np.float16).tofile(tmp / f"motion_{n}")
            clip = np.eye(4) if prev is None else prev @ np.linalg.inv(vp)
            prev = vp
            sx, sy = SIGN
            with open(tmp / f"params_f{i}.txt", "w") as o:
                o.write(f"{w} {h} {lw} {lh} {lw} {lh} 87\n{sx} {sy} 0 0 10 10000000 {fov} 1 {int(i == 0)} 0 1\n")
                o.write(" ".join(f"{x:.9g}" for x in clip.T.ravel()))
    return w, h


SIGN = (1, 1)
def ours(w, h, n):
    for f in gen.glob("*.bin"): f.unlink()
    r = subprocess.run([PARITY, "--replay", str(tmp), "--frames", "3", "--gen", str(n - 1), "--dump", str(gen), API], capture_output=True, text=True)
    frames = []
    for k in range(n - 1):
        p = gen / f"{API}_frame2_gen{k}.bin"
        if not p.exists(): raise RuntimeError(r.stdout + r.stderr)
        a = np.fromfile(p, np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1).float() / 255)
    return frames


acc, nets = {}, {}
def add(who, n, seq, v): acc.setdefault(who, {}).setdefault(f"{n}x", []).append(v)
first = True
for info in cfg.infos:
    s = info.step
    cands = [p for p in range(2 * s, info.length) if p % info.base_stride == 0]
    for p1 in sorted(random.Random(1).sample(cands, min(PAIRS, len(cands)))):
        w, h = export(info, p1)
        for n in range(2, 7):
            if s % n: continue
            d = load_sample(info, p1, [k * s // n for k in range(1, n)], random.Random(0), cfg)
            y0, x0, ch, cw = crop = tuple(d["display_crop"].tolist())
            if os.environ.get("GAME_INPUTS", "1") == "1":  # what a game passes: 8-bit colour, half-float vectors
                for k in ("rgb_m1", "rgb_p1"): d[k] = (d[k].clamp(0, 1) * 255).round() / 255
                for k in ("mv_p1", "mv_m1"): d[k] = d[k].half().float()
            gt = [d["gt"][j].unsqueeze(0) for j in range(n - 1)]
            if first:  # the motion vectors' sign, once
                best = []
                for sg in ((1, 1), (-1, -1)):
                    SIGN = sg; export(info, p1)
                    best.append((np.mean([psnr(f[None, :, y0:y0 + ch, x0:x0 + cw], g) for f, g in zip(ours(w, h, n), gt)]), sg))
                print("motion vector sign", best, flush=True); SIGN = max(best)[1]; export(info, p1); first = False
            for f, g in zip(ours(w, h, n), gt): add("nfrux", n, info, psnr(f[None, :, y0:y0 + ch, x0:x0 + cw], g))
            with torch.no_grad():
                st = build_state(d, fe, dev)
                for who, ck in CKPT.items():
                    net, tc = load_net(ck, dev)
                    for j, t in enumerate(d["t"].tolist()):
                        o, _ = P.run_timestep(st, t, net, tc, display_crop=crop, logit_prior=True)
                        add(who, n, info, psnr((o.cpu().clamp(0, 1) * 255).round() / 255, gt[j]))
            for j, t in enumerate(d["t"].tolist()):
                add("blend", n, info, psnr(((1 - t) * d["rgb_m1"] + t * d["rgb_p1"])[None, :, y0:y0 + ch, x0:x0 + cw], gt[j]))
    print(info.path.stem, {a: {k: round(float(np.mean(v)), 2) for k, v in b.items()} for a, b in acc.items()}, flush=True)
hard = {}
for k in acc["nfrux"]:
    m = np.array(acc["torch"][k]) < 45
    for a in acc: hard.setdefault(a, {})[k] = float(np.mean(np.array(acc[a][k])[m])) if m.any() else None
print("pairs under 45 dB in torch:", {k: (int((np.array(acc["torch"][k]) < 45).sum()), {a: round(hard[a][k], 2) for a in hard if hard[a][k]}) for k in sorted(hard["torch"])})
res = {a: {k: [float(np.mean(v)), len(v)] for k, v in sorted(b.items())} for a, b in acc.items()}
out.write_text(json.dumps(res, indent=1))
for k in sorted(acc["nfrux"]):
    print(k, {a: round(res[a][k][0], 2) for a in res}, "n", res["nfrux"][k][1])