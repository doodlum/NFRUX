"""Scans an AMD FidelityFX signed DLL for embedded GPU code and data.

    python scan.py DLL OUT_DIR

Writes, per DLL:
  dxbc/NNNN_<offset>.dxbc   every DXBC/DXIL container (header 'DXBC', size from the header)
  spirv/NNNN_<offset>.spv   every SPIR-V module (magic 0x07230203, walked instruction by instruction to its end)
  sections.txt              PE sections
  strings.txt               printable ASCII strings of 6+ characters that look like names (shader, pass, network)
  blobs.txt                 large runs that are neither code nor shader (candidate weight tables), with entropy

The FidelityFX licence (Kits/FidelityFX/docs/license.md, v2.4.0) lists the signedbin DLLs under its MIT section.
"""
import os, re, struct, sys, math, collections

def sections(b):
    pe = struct.unpack_from('<I', b, 0x3c)[0]
    n = struct.unpack_from('<H', b, pe + 6)[0]
    opt = struct.unpack_from('<H', b, pe + 20)[0]
    s = pe + 24 + opt
    out = []
    for i in range(n):
        name, vsz, va, rsz, raw = struct.unpack_from('<8sIIII', b, s + 40 * i)
        out.append((name.rstrip(b'\0').decode(), va, vsz, raw, rsz))
    return out

def dxbc(b):
    for m in re.finditer(b'DXBC', b):
        o = m.start()
        if o + 32 > len(b): continue
        size, chunks = struct.unpack_from('<II', b, o + 24)
        if 32 < size <= len(b) - o and 0 < chunks < 64:
            offs = struct.unpack_from(f'<{chunks}I', b, o + 32)
            if all(32 <= c < size for c in offs):
                kinds = [b[o + c:o + c + 4].decode('latin1') for c in offs]
                yield o, size, kinds

def spirv(b):
    for m in re.finditer(re.escape(struct.pack('<I', 0x07230203)), b):
        o = m.start()
        if o % 4 or o + 20 > len(b): continue
        ver, gen, bound, schema = struct.unpack_from('<IIII', b, o + 4)
        if schema != 0 or not (0x10000 <= ver <= 0x10600) or not (0 < bound < 1 << 22): continue
        p = o + 20
        while p + 4 <= len(b):
            w = struct.unpack_from('<I', b, p)[0]; wc, op = w >> 16, w & 0xffff
            if wc == 0 or op > 6000: break
            p += 4 * wc
        if p - o > 64: yield o, p - o, ver, bound

def entropy(x):
    c = collections.Counter(x); n = len(x)
    return -sum(v / n * math.log2(v / n) for v in c.values())

def main(dll, out):
    b = open(dll, 'rb').read(); os.makedirs(out, exist_ok=True)
    secs = sections(b)
    with open(f'{out}/sections.txt', 'w') as f:
        for s in secs: f.write('%-8s va %08x vsz %9d raw %08x rsz %9d\n' % s)
    used = []
    os.makedirs(f'{out}/dxbc', exist_ok=True); os.makedirs(f'{out}/spirv', exist_ok=True)
    nd = 0
    with open(f'{out}/dxbc.txt', 'w') as f:
        for o, size, kinds in dxbc(b):
            open(f'{out}/dxbc/{nd:04d}_{o:08x}.dxbc', 'wb').write(b[o:o + size]); used.append((o, size))
            f.write(f'{nd:04d} {o:08x} {size:8d} {" ".join(kinds)}\n'); nd += 1
    ns = 0
    with open(f'{out}/spirv.txt', 'w') as f:
        for o, size, ver, bound in spirv(b):
            open(f'{out}/spirv/{ns:04d}_{o:08x}.spv', 'wb').write(b[o:o + size]); used.append((o, size))
            f.write(f'{ns:04d} {o:08x} {size:8d} v{ver >> 16}.{(ver >> 8) & 255} bound {bound}\n'); ns += 1
    pat = re.compile(rb'[\x20-\x7e]{6,}')
    with open(f'{out}/strings.txt', 'w') as f:
        for m in pat.finditer(b):
            s = m.group().decode()
            if re.search(r'(?i)shader|pass|net|layer|conv|weight|bias|tensor|ml|wmma|onnx|frame|interp|flow|fsr|\.hlsl|\.h\b|main', s):
                f.write(f'{m.start():08x} {s}\n')
    # candidate data blobs: 64 KiB windows in data sections outside shaders, with entropy
    used.sort()
    def inside(o): return any(a <= o < a + s for a, s in used)
    with open(f'{out}/blobs.txt', 'w') as f:
        for name, va, vsz, raw, rsz in secs:
            if name in ('.text', '.reloc', '.pdata'): continue
            for o in range(raw, raw + rsz, 65536):
                if inside(o): continue
                w = b[o:o + 65536]
                if len(w) < 4096: continue
                f.write(f'{name:8s} {o:08x} entropy {entropy(w):.2f} zero {w.count(0) / len(w):.2f}\n')
    print(f'{dll}: {len(secs)} sections, {nd} DXBC, {ns} SPIR-V')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
