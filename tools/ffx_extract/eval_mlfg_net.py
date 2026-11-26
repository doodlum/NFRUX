"""Scores AMD FSR FG 4's ML interpolation (mlfg_net.AmdMlfgNet) on the NFRU-MFG test sets, with the same samples,
colour space and metrics as C:/dev/modelgym/scripts/nfru_mfg_perceptual.py (PSNR, LPIPS-alex, sharpness, gradient error).

    python eval_mlfg_net.py OUT_DIR [--sets test_new test_bal] [--n 2] [--pairs 4] [--variants amd,amd-point,...] [--save]

Variants: amd (precision 'amd', linear sampler), amd-point (point sampler, as the NVIDIA captures), train (fp32),
amd-srgb (inputs sRGB-encoded before the network, output decoded back; scored in the same space as everything else).
Also reports flow-magnitude ratios of stage 1 at t = 0.25 / 0.75 vs 0.5 (t generalisation sanity) with --tcheck.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\dev\modelgym\ng-model-gym\src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lpips
from PIL import Image
from ng_model_gym.usecases.nfru_mfg.train_mfg import PairDataset, load_sample, psnr, srgb_oetf
import mlfg_net as N

DATA = Path(r"C:\dev\modelgym\data")
SETS = {"test_new": DATA / "nfru6x/test_new", "test_bal": DATA / "nfru6x/test_bal", "test": DATA / "nfru6x/test",
        "val": DATA / "nfru6x/val"}
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def set_frames(name, pairs, n):
    """Identical sampling to nfru_mfg_perceptual.set_frames."""
    seqs = sorted(Path(SETS[name]).rglob("*.safetensors"))
    cfg = PairDataset(seqs, 1, train=False, crop=(1080, 1920))
    for info in cfg.infos:
        s = info.step
        c = [p for p in range(2 * s, info.length) if p % info.base_stride == 0]
        for p1 in sorted(random.Random(1).sample(c, min(pairs, len(c)))):
            for K in (range(1, n) if s % n == 0 else ()):
                yield f"{info.path.stem}_{p1}_{K}", load_sample(info, p1, [s * K // n], random.Random(0), cfg)


def grad(x):
    y = x.mean(1, keepdim=True)
    return (y[..., 1:, :-1] - y[..., :-1, :-1]).abs() + (y[..., :-1, 1:] - y[..., :-1, :-1]).abs()


def srgb_eotf(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def amd_inputs(d, dev):
    """Dataset sample -> AMD's inputs. Dataset: depth 0 near..1 far (device z = 1 - depth), mv (y,x) lane px p1->m1,
    column-vector ViewProj with clip z = 1 - depth. AMD: device depth inverted, mv (x,y) render px (FFX: prev - cur),
    row-vector matrices."""
    f = lambda k: d[k].float()
    vp_m3, vp_m1, vp_p1 = (f(k).double() for k in ("vp_m3", "vp_m1", "vp_p1"))
    A = (vp_p1 @ torch.linalg.inv(vp_m1)).T.float()[None].to(dev)      # prevPstoPs: frame0 clip -> frame1 clip
    B = (vp_m3 @ torch.linalg.inv(vp_m1)).T.float()[None].to(dev)      # prevPsToPrev2Ps: frame0 -> frame -1
    mv = lambda k: torch.stack([f(k)[1], f(k)[0]])[None].to(dev)
    dep = lambda k: (1.0 - f(k))[None].to(dev)
    return dict(prev=f("rgb_m1")[None].to(dev), cur=f("rgb_p1")[None].to(dev), mv1=mv("mv_p1"), d1=dep("depth_p1"),
                mv0=mv("mv_m1"), d0=dep("depth_m1"), A=A, B=B)


def run_amd(net, x, t, srgb=False):
    prev, cur = x["prev"], x["cur"]
    if srgb:
        prev, cur = srgb_oetf(prev.clamp(0, 1)), srgb_oetf(cur.clamp(0, 1))
    H, W = cur.shape[-2:]
    hist = net.history(x["mv0"], x["d0"], (W, H))
    out, aux = net(prev, cur, x["d1"], x["mv1"], hist, x["A"], x["B"], t)
    if srgb:
        out = srgb_eotf(out.clamp(0, 1))
    return out, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--sets", nargs="+", default=["test_new", "test_bal"])
    ap.add_argument("--pairs", type=int, default=4); ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--variants", default="amd,amd-point,amd-srgb,train"); ap.add_argument("--save", action="store_true")
    ap.add_argument("--tcheck", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
    base = N.AmdMlfgNet(precision="amd").to(dev).eval()
    variants = {}
    for v in a.variants.split(","):
        b = v.replace("-fscale", "")
        variants[v] = dict(precision="train" if b == "train" else "amd",
                           sampler="point" if b in ("amd-point", "amd-ptnet") else "linear",
                           comp={"amd-ptnet": "linear", "amd-ptcomp": "point"}.get(b), srgb=b == "amd-srgb",
                           flow_t="scale" if v.endswith("-fscale") else "channel")
    res, tstats = {}, {}
    with torch.no_grad():
        for sname in a.sets:
            acc = {}
            for key, d in set_frames(sname, a.pairs, a.n):
                gt = d["gt"][:1].to(dev); ggt = grad(gt)
                t = d["t"][0].item()
                x = amd_inputs(d, dev)

                def score(who, o):
                    o = (o.clamp(0, 1) * 255).round() / 255; g = grad(o)
                    s = acc.setdefault(who, {})
                    for k, v in (("psnr", psnr(o, gt)), ("lpips", lp(o, gt, normalize=True).item()),
                                 ("sharp", (g.sum() / ggt.sum()).item()), ("graderr", (g - ggt).abs().mean().item() * 255)):
                        s.setdefault(k, []).append(v)
                    if a.save and who in ("amd", "amd-point"):
                        p = Path(a.out) / sname / who; p.mkdir(parents=True, exist_ok=True)
                        Image.fromarray((o[0].permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)).save(p / f"{key}.png")
                for who, cfg in variants.items():
                    base.precision, base.sampler, base.composite_sampler = cfg["precision"], cfg["sampler"], cfg["comp"]
                    base.flow_t = cfg["flow_t"]
                    o, aux = run_amd(base, x, t, cfg["srgb"])
                    score(who, o)
                    if who == "amd":
                        w = aux["weights"][0]
                        s = acc[who]
                        for i, nm in enumerate(("w_splat0", "w_splat1", "w_flow0", "w_flow1", "w_blend")):
                            s.setdefault(nm, []).append(w[i].mean().item())
                score("linear", (x["prev"] + x["cur"]) / 2)
                if a.tcheck and abs(t - 0.5) < 1e-6:
                    base.precision, base.sampler, base.composite_sampler, base.flow_t = "amd", "linear", None, "channel"
                    sz = N.mlfg_sizes(x["cur"].shape[-1], x["cur"].shape[-2])
                    fl = {tt: base.stage1(x["prev"], x["cur"], tt, sz)[0][0] for tt in (0.25, 0.5, 0.75)}
                    lw, lh = sz["l0_live"]
                    mag = lambda f, c: f[c:c + 2, :lh, :lw].norm(dim=0)
                    ts = tstats.setdefault(sname, {})
                    for tt in (0.25, 0.75):
                        for c, nm in ((0, "A"), (2, "B")):
                            m5 = mag(fl[0.5], c); mt = mag(fl[tt], c); sel = m5 > 0.25
                            if sel.any():
                                ts.setdefault(f"{nm}({tt})/{nm}(0.5)", []).append((mt[sel].mean() / m5[sel].mean()).item())
                        # direction agreement of A(t) with A(0.5)
                        ca = torch.nn.functional.cosine_similarity(fl[tt][0:2, :lh, :lw], fl[0.5][0:2, :lh, :lw], dim=0)
                        ts.setdefault(f"cos A({tt}),A(0.5)", []).append(ca[mag(fl[0.5], 0) > 0.25].mean().item())
                    # symmetry at t=0.5: A ~ -B for linear motion
                    cab = torch.nn.functional.cosine_similarity(fl[0.5][0:2, :lh, :lw], -fl[0.5][2:4, :lh, :lw], dim=0)
                    ts.setdefault("cos A(0.5),-B(0.5)", []).append(cab[mag(fl[0.5], 0) > 0.25].mean().item())
                print(key, f"t={t:.3f}", " ".join(f"{w}:{s['lpips'][-1]:.4f}/{s['psnr'][-1]:.2f}" for w, s in acc.items()), flush=True)
                del d, x
                torch.cuda.empty_cache()
            res[sname] = {w: {k: float(np.mean(v)) for k, v in s.items()} for w, s in acc.items()}
            print(f"== {sname} n={len(next(iter(acc.values()))['psnr'])}")
            for w_, s in res[sname].items():
                print(f"{w_:12s} " + "  ".join(f"{k} {v:.4f}" if k not in ("psnr", "graderr") else f"{k} {v:.2f}" for k, v in s.items()), flush=True)
            if sname in tstats:
                print("t-sanity:", {k: round(float(np.mean(v)), 3) for k, v in tstats[sname].items()})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "scores.json").write_text(json.dumps(dict(scores=res, tsanity={s: {k: float(np.mean(v)) for k, v in d.items()} for s, d in tstats.items()}), indent=1))


if __name__ == "__main__":
    main()
