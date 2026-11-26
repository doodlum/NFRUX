"""Catalogues the shaders scan.py extracted: per module its entry point, local size, bindings, capabilities and
cooperative-matrix use (Vulkan, from spirv-dis output) and whether the D3D12 twin uses AMD driver intrinsics.

    python catalogue.py SCAN_DIR OUT_TXT

SCAN_DIR holds vk/ (with dis/*.spvasm) and dx12/ (with dxbc/*.dxbc) from scan.py / fetch.ps1.
"""
import os, re, sys, glob

AMD_SPACE = b'\xde\x0a\xff\x7f'  # 2147420894, the register space of the AMD extension UAV (little-endian)

def vk(scan):
    rows = []
    for p in sorted(glob.glob(os.path.join(scan, 'vk', 'dis', '*.spvasm'))):
        t = open(p, encoding='utf-8', errors='replace').read()
        ep = re.search(r'OpEntryPoint (\w+) %(\w+) "(\w+)"', t)
        ls = re.search(r'OpExecutionMode %\w+ LocalSize (\d+) (\d+) (\d+)', t)
        names = re.findall(r'OpName %\w+ "(\w+Buffer|\w+Sizes|tempLDS|preloadedWeights|g_\w+|r_\w+|rw_\w+)"', t)
        caps = sorted(set(re.findall(r'OpCapability (\w+)', t)) & {'CooperativeMatrixKHR', 'Float16', 'Int8', 'Float8EXT',
                                                                    'Float8CooperativeMatrixEXT', 'CooperativeMatrixTensorAddressingNV'})
        rows.append((os.path.basename(p)[:-7], ep.group(3) if ep else '?', ep.group(1) if ep else '?',
                     'x'.join(ls.groups()) if ls else '-', t.count('OpCooperativeMatrixMulAddKHR'),
                     t.count('OpCooperativeMatrixLoad'), ','.join(caps), ','.join(sorted(set(names)))))
    return rows

def main(scan, out):
    rows = vk(scan)
    amd = sum(open(p, 'rb').read().count(AMD_SPACE) > 0 for p in glob.glob(os.path.join(scan, 'dx12', 'dxbc', '*.dxbc')))
    n = len(glob.glob(os.path.join(scan, 'dx12', 'dxbc', '*.dxbc')))
    ml = [r for r in rows if r[1].startswith('mlfi_')]
    with open(out, 'w') as f:
        f.write(f'Vulkan modules {len(rows)}, ML passes {len(ml)} (4k {sum(r[1].startswith("mlfi_4k") for r in ml)}, '
                f'8k {sum(r[1].startswith("mlfi_8k") for r in ml)}); D3D12 containers {n}, with the AMD intrinsic space {amd}\n')
        f.write('module entry stage local mulAdds loads caps bindings\n')
        for r in sorted(rows, key=lambda r: (not r[1].startswith('mlfi'), r[1][:8], int(re.sub(r'\D', '', r[1]) or 0))):
            f.write(' '.join(str(x) for x in r) + '\n')
    print(open(out).readline().strip())

if __name__ == '__main__':
    main(*sys.argv[1:3])
