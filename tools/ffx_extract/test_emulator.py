"""Runs the emulator over whole blocks of passes, starting from AMD's scratch state, and compares block outputs.

    python test_emulator.py CAPTURE_DIR DUMP_DIR WEIGHTS_NPZ
"""
import json, os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlfg_model import Emulator, read

def main(cap, dumps, npz):
    em = Emulator(cap, npz)
    load = lambda n: torch.from_numpy(np.fromfile(os.path.join(dumps, f'2_{n:03d}.bin'), np.uint8)).cuda()
    # (start dump, first pass, last pass): blocks with no glue pass inside
    for start, a, b in [(1, 2, 8), (9, 10, 16), (19, 20, 52), (56, 57, 81), (85, 86, 110), (114, 115, 147), (153, 154, 154)]:
        s = load(start)
        em.run_passes(s, a, b)
        ref = load(b)
        g = em.graph[b]; out = [t for t in g['tensors'] if t['kind'] in ('Tensor4', 'Blocked5')][-1]
        W, H = em.size(g['tensorSizeIdx'][0]); C = out['logical'][2]
        mine, theirs = read(s, out, W, H, C), read(ref, out, W, H, C)
        err = (mine - theirs).abs()
        print(f'passes {a:3d}-{b:3d}: output {C} ch {W}x{H}: max err {err.max().item():.4f}, mean {err.mean().item():.5f}, absmax {theirs.abs().max().item():.3f}')

if __name__ == '__main__':
    main(*sys.argv[1:4])
