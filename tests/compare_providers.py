"""NFRUX and other frame generators (e.g. DLSS-G) on the same recorded inputs (ngx_harness --replay), against the true skipped frames, and a picture.
   python compare_providers.py HALF OUT_PNG NAME=DIR...    SPDX-License-Identifier: MIT"""
import sys
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
half, png = sys.argv[1:3]
dirs = dict(a.split("=", 1) for a in sys.argv[3:])
w, h = 1280, 701
L = lambda p: np.fromfile(p, np.uint8).reshape(h, w, 4)[..., 2::-1].astype(np.float32)
P = lambda a, b: 10 * np.log10(255**2 / max(1e-9, float(((a - b) ** 2).mean())))
X = 450  # the sample's UI is left of here
S = {n: [] for n in dirs} | {"blend": [], "previous": []}
err = {}
for f in range(2, 30):
    t = L(f"{half}/truthback_f{f}.bin"); a = L(f"{half}/back_{w}x{h}_f{f-1}_k1.bin"); b = L(f"{half}/back_{w}x{h}_f{f}_k1.bin")
    g = {n: L(f"{d}/f{f}_k1.bin") for n, d in dirs.items()} | {"blend": (a + b) / 2, "previous": a}
    for n, x in g.items():
        ui = abs(b - L(f"{half}/hudless_{w}x{h}_f{f}_k1.bin")).max(2) > 90  # opaque or nearly: far from the picture under it
        S[n].append((P(x[:, X:], t[:, X:]), ssim(x[:, X:], t[:, X:], channel_axis=2, data_range=255), abs(x - b)[ui].mean(), P(x[:, X:], b[:, X:]), P(x[:, X:], a[:, X:])))
    err[f] = abs(g["blend"] - t)[:, X:].mean()
for n, s in S.items():
    m = np.mean(s, 0); print(f"{n:9s} {m[0]:.2f} dB  SSIM {m[1]:.4f}  (opaque UI {m[2]:.2f} codes from the new frame; against new {m[3]:.1f}, old {m[4]:.1f})  worst {min(x[0] for x in s):.2f}")
f = max(err, key=err.get)
t = L(f"{half}/truthback_f{f}.bin"); g = {"true frame": t} | {n: L(f"{d}/f{f}_k1.bin") for n, d in dirs.items()}
d = abs((L(f"{half}/back_{w}x{h}_f{f-1}_k1.bin") + L(f"{half}/back_{w}x{h}_f{f}_k1.bin")) / 2 - t).mean(2)
best = max(((d[y:y+200, x:x+300].mean(), x, y) for y in range(0, h - 200, 40) for x in range(X, w - 300, 40)))
x, y = best[1:]
sheet = Image.new("RGB", (640 * len(g), 351 + 400 + 400), "black")
for i, (n, im) in enumerate(g.items()):
    I = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
    sheet.paste(I.resize((636, 348)), (640 * i, 0)); sheet.paste(I.crop((x, y, x + 300, y + 187)).resize((636, 396), 0), (640 * i, 351))
    e = np.clip(abs(im - t) * 3, 0, 255).astype(np.uint8); sheet.paste(Image.fromarray(e).crop((x, y, x + 300, y + 187)).resize((636, 396), 0), (640 * i, 751))
    ImageDraw.Draw(sheet).text((640 * i + 8, 8), n, fill="yellow", font_size=26)
sheet.save(png); print("frame", f, "crop", x, y)
