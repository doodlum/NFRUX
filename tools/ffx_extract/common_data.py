"""Finds the data the D3D12 and Vulkan frame-generation DLLs share outside their shaders (the network's weights,
which both APIs upload to the same InitializerBuffer).

    python common_data.py DLL_A SCAN_A DLL_B SCAN_B OUT_DIR

SCAN_* are scan.py output folders (their dxbc.txt / spirv.txt mark the shader ranges to skip).
Writes OUT_DIR/common_NN_<offA>_<offB>.bin for each shared run of 4 KiB or more, and common.txt.
"""
import sys, os, hashlib

def ranges(scan):
    r = []
    for n in ('dxbc.txt', 'spirv.txt'):
        p = os.path.join(scan, n)
        if os.path.exists(p):
            for l in open(p):
                a = l.split(); r.append((int(a[1], 16), int(a[2])))
    return sorted(r)

def masked(b, r):
    b = bytearray(b)
    for o, s in r: b[o:o + s] = b'\0' * s
    return bytes(b)

def main(da, sa, db, sb, out):
    A = masked(open(da, 'rb').read(), ranges(sa)); B = masked(open(db, 'rb').read(), ranges(sb))
    K = 256
    idx = {}
    for o in range(0, len(B) - K, 16):
        w = B[o:o + K]
        if w.count(0) > K // 2: continue
        idx.setdefault(hashlib.blake2b(w, digest_size=8).digest(), o)
    runs = []; o = 0
    while o < len(A) - K:
        w = A[o:o + K]
        j = idx.get(hashlib.blake2b(w, digest_size=8).digest()) if w.count(0) <= K // 2 else None
        if j is None or B[j:j + K] != w:
            o += 16; continue
        s = o; t = j
        while s > 0 and t > 0 and A[s - 1] == B[t - 1]: s -= 1; t -= 1
        e = o + K; f = j + K
        while e < len(A) and f < len(B) and A[e] == B[f]: e += 1; f += 1
        runs.append((s, t, e - s)); o = e
    os.makedirs(out, exist_ok=True)
    with open(f'{out}/common.txt', 'w') as f:
        for i, (s, t, n) in enumerate(r for r in runs if r[2] >= 4096):
            open(f'{out}/common_{i:02d}_{s:08x}_{t:08x}.bin', 'wb').write(A[s:s + n])
            f.write(f'{i:02d} A {s:08x} B {t:08x} {n}\n')
    big = [r for r in runs if r[2] >= 4096]
    print(len(runs), 'shared runs;', len(big), 'of 4 KiB+, total', sum(r[2] for r in big), 'bytes')

if __name__ == '__main__':
    main(*sys.argv[1:6])
