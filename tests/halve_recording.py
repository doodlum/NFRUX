"""Every second frame of a sample recording, vectors and matrices chained over two frames; the frames between are the truth.
   python halve_recording.py IN FIRST COUNT OUT    SPDX-License-Identifier: MIT"""
import os, shutil, sys
import numpy as np
src, first, count, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(dst, exist_ok=True)
def P(f):
    L = open(f"{src}/params_f{f}.txt").read().split("\n")
    return L[0], L[1], np.array(L[2].split(), np.float64).reshape(4, 4)
for n, f in enumerate(range(first + 1, first + count, 2)):
    h, l1, c = P(f); _, _, cp = P(f - 1)
    w, ht, *_ = map(int, h.split())
    name = lambda k, x: f"{src}/{k}_{w}x{ht}_f{x}_k1.bin"
    out = lambda k: f"{dst}/{k}_{w}x{ht}_f{n}_k1.bin"
    m = lambda x: np.fromfile(name("motion", x), np.float16).reshape(ht, w, 2).astype(np.float32)
    s = np.array(list(map(float, l1.split()[:2])), np.float32)
    a, b = m(f), m(f - 1)
    yy, xx = np.mgrid[:ht, :w]
    px = np.clip(np.rint(xx + a[..., 0] * s[0]), 0, w - 1).astype(int); py = np.clip(np.rint(yy + a[..., 1] * s[1]), 0, ht - 1).astype(int)
    (a + b[py, px]).astype(np.float16).tofile(out("motion"))
    for k in ("back", "hudless", "depth"): shutil.copy(name(k, f), out(k))
    if n: shutil.copy(name("hudless", f - 1), f"{dst}/truth_f{n}.bin"); shutil.copy(name("back", f - 1), f"{dst}/truthback_f{n}.bin")
    open(f"{dst}/params_f{n}.txt", "w").write(h + "\n" + l1 + "\n" + " ".join("%.9g" % x for x in (c @ cp).ravel()))
print(n + 1, "frames")
