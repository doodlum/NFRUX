"""A4: are the sample's dumped generated frames between their neighbours?  python between_neighbours.py DIR..."""
import glob, os, re, sys
import numpy as np
for d in sys.argv[1:]:
    g = sorted({tuple(map(int, re.search(r"f(\d+)_k(\d+)", p).groups())) for p in glob.glob(os.path.join(d, "gen_*.bin"))})
    size = re.search(r"_(\d+)x(\d+)_", glob.glob(os.path.join(d, "gen_*.bin"))[0]).groups()
    w, h = map(int, size)
    def load(n, f, k):
        p = os.path.join(d, f"{n}_{w}x{h}_f{f}_k{k}.bin")
        return np.fromfile(p, np.uint8).reshape(h, w, 4)[..., :3] @ np.array([.114, .587, .299]) if os.path.exists(p) else None
    n = ok = 0; worst = -1e9; move = []
    for f, k in g:
        a, b, m = load("back", f - 1, 1), load("back", f, 1), load("gen", f, k)
        if a is None or b is None: continue
        whole = abs(a - b).mean(); e = max(abs(m - a).mean(), abs(m - b).mean()) - whole
        n += 1; ok += e <= 2; worst = max(worst, e); move.append(whole)
    print(d, f"{ok} of {n} between (worst excess {worst:.2f} codes; neighbours {np.mean(move):.2f} apart on average)")
