"""Crop sheets truth | Arm | AMD (linear) | AMD (point) from the PNGs saved by nfru_mfg_perceptual.py --save (Arm,
truth) and eval_mlfg_net.py --save (AMD).

    python mlfg_sheets.py ARM_DIR AMD_DIR OUT_DIR [--per-set 3] [--crop 256]
"""
import argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB'))).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm'); ap.add_argument('amd'); ap.add_argument('out')
    ap.add_argument('--per-set', type=int, default=3); ap.add_argument('--crop', type=int, default=256)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = a.crop
    for sdir in sorted(Path(a.amd).iterdir()):
        if not (sdir / 'amd').is_dir(): continue
        keys = sorted(p.stem for p in (sdir / 'amd').glob('*.png'))
        errs = []
        for k in keys:
            tp = Path(a.arm) / sdir.name / 'truth' / f'{k}.png'
            if not tp.exists(): continue
            errs.append(((load(sdir / 'amd' / f'{k}.png') - load(tp)).abs().mean().item(), k))
        errs.sort()
        pick = [errs[-1][1], errs[-2][1], errs[len(errs) // 2][1]][:a.per_set] if len(errs) >= 3 else [e[1] for e in errs]
        for k in pick:
            tr = load(Path(a.arm) / sdir.name / 'truth' / f'{k}.png')
            ims = [('truth', tr), ('Arm', load(Path(a.arm) / sdir.name / 'arm' / f'{k}.png')),
                   ('AMD linear', load(sdir / 'amd' / f'{k}.png'))]
            if (sdir / 'amd-point' / f'{k}.png').exists():
                ims.append(('AMD point', load(sdir / 'amd-point' / f'{k}.png')))
            e = (ims[2][1] - tr).abs().mean(-1)[None, None]
            m = F.avg_pool2d(e, c, stride=16)[0, 0]
            iy, ix = divmod(int(m.argmax()), m.shape[1])
            y0, x0 = iy * 16, ix * 16
            tiles = []
            for name, im in ims:
                t = Image.fromarray((im[y0:y0 + c, x0:x0 + c].numpy() * 255 + .5).astype(np.uint8)).resize((2 * c, 2 * c), Image.NEAREST)
                ImageDraw.Draw(t).text((6, 4), name, fill=(255, 255, 0))
                tiles.append(t)
            sheet = Image.new('RGB', (2 * c * len(tiles), 2 * c))
            for i, t in enumerate(tiles): sheet.paste(t, (2 * c * i, 0))
            sheet.save(out / f'sheet_{sdir.name}_{k}_y{y0}x{x0}.png')
            print('saved', out / f'sheet_{sdir.name}_{k}_y{y0}x{x0}.png')


if __name__ == '__main__':
    main()
