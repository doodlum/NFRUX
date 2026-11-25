"""Compares NFRUX_DUMP output of the Streamline sample on D3D12 and Vulkan: python compare_sample.py DIR_D3D12 DIR_VK OUT_PNG"""
import os, sys
import numpy as np
from PIL import Image
A, B, OUT = sys.argv[1:4]
def L(d, n):
    p = os.path.join(d, n)
    return np.fromfile(p, np.uint8).reshape(701, 1280, 4)[..., 2::-1].astype(int) if os.path.exists(p) else None
P = lambda a, b: 10 * np.log10(255**2 / max(1e-9, ((a - b) ** 2).mean()))
best = []
for f in range(1, 8):
    d = L(A, f"hudless_1280x701_f{f}_k1.bin")
    c = [(P(d, v), j) for j in range(1, 8) if d is not None and (v := L(B, f"hudless_1280x701_f{j}_k1.bin")) is not None]
    if not c:
        continue
    p, j = max(c)
    g = [P(L(A, f"gen_1280x701_f{f}_k{k}.bin")[:, 450:], L(B, f"gen_1280x701_f{j}_k{k}.bin")[:, 450:]) for k in (1, 2, 3)]
    prev = [x for x in (L(A, f"hudless_1280x701_f{f-1}_k1.bin"), L(B, f"hudless_1280x701_f{j-1}_k1.bin")) if x is not None]
    print(f"d3d12 f{f} ~ vk f{j}: rendered {p:.1f} dB, previous rendered {P(*prev) if len(prev) == 2 else 0:.1f} dB, generated beside the UI", ["%.1f" % x for x in g])
    best.append((p, f, j))
if best:
    p, f, j = max(best)
    a, b = L(A, f"gen_1280x701_f{f}_k2.bin"), L(B, f"gen_1280x701_f{j}_k2.bin")
    Image.fromarray(np.concatenate([a, b, np.clip(np.abs(a - b) * 8, 0, 255)], 1).astype(np.uint8)).save(OUT)