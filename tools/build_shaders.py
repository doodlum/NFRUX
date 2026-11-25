"""Compiles NFRUX's shaders once from GLSL: SPIR-V for Vulkan, and that SPIR-V through SPIRV-Cross and DXC to DXIL for
D3D12, so both APIs run the same code. Writes one C++ header with every pass, its DXIL and its D3D12 bindings.

    python build_shaders.py CORE_DIR OUT_DIR [STRICT]   STRICT 1: no compiler may fuse or reorder (both APIs give the same bytes)
SPDX-License-Identifier: MIT
"""
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys

core, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(out, exist_ok=True)
sdk = os.environ.get("VULKAN_SDK", "")


def tool(name):
    p = os.path.join(sdk, "Bin", name + ".exe")
    return p if os.path.exists(p) else (shutil.which(name) or sys.exit(f"{name} not found (Vulkan SDK)"))


GLSLANG, XCROSS, DXC = tool("glslangValidator"), tool("spirv-cross"), tool("dxc")
OF = ["-DFFX_GPU=1", "-DFFX_GLSL=1", "-DMIN_SAD"]
P = "FFX_ARM_FRAMEINTERPOLATION_OPTION_"
FI = ["-DFFX_GPU=1", "-DFFX_GLSL=1", f"-D{P}JITTER_MOTION_VECTORS=1", f"-D{P}ENABLE_DATA_GRAPH_FI=1", f"-D{P}SUPPORT_TENSOR=0",
      f"-D{P}MANAGE_PREVIOUS_DEPTH=1"]
HIST, NOHIST = [f"-D{P}MANAGE_PREVIOUS_COLOR=1"], [f"-D{P}MANAGE_PREVIOUS_COLOR=0"]
fmt = lambda f: [f"-DOUTPUT_IMG_FORMAT={f}"]
of_ = "shaders/opticalflow/ffx_opticalflow"
fi_ = "shaders/frameinterpolation/ffx_frameinterpolation"
dk_ = "shaders/deck/ffx_"
# name, source, defines, inversion define, D3D12 shader model
passes = []
for n, s in [("kOfMvHints", of_ + "_mv_hints"), ("kOfArmPyramid", of_ + "_arm_pyramid"), ("kOfArmWarp", of_ + "_arm_warp"),
             ("kOfArmMatchShared", dk_ + "opticalflow_arm_match_shared"), ("kOfArmMedian", of_ + "_arm_median"),
             ("kOfArmBilateralShared", dk_ + "opticalflow_arm_bilateral_shared")]:
    passes.append((n, s, OF, "DEPTH_INVERTED", 60))
INV = P + "INVERTED_DEPTH"
D = FI + HIST
# Full precision in the source, for both APIs: the 16-bit float types are 32-bit (and no pass may declare 8- or 16-bit types).
F32 = ["-Dfloat16_t=float", "-Df16vec2=vec2", "-Df16vec3=vec3", "-Df16vec4=vec4", "-Dmediump=", "-Dlowp="]
BANNED = {9: "Float16", 22: "Int16", 39: "Int8", **{c: "a group operation" for c in (18, *range(61, 69), 4423, 4431)},
          **{c: "8/16-bit storage" for c in (4433, 4434, 4435, 4436, 4448, 4449, 4450)}}
for n, s, d in [("kFiInitWarp", fi_ + "_00_init_warp_pass", D + fmt("rgba16f")),
                ("kFiWarpFlow", fi_ + "_10_warp_flow_pass", D + fmt("rgba16f")),
                ("kFiWarpFlowTm1", fi_ + "_11_warp_flow_tm1_pass", D + fmt("rgba16f")),
                ("kFiFillHolesTm1", fi_ + "_21_fill_holes_tm1_pass", D + fmt("rgba16f")),
                ("kFiPreprocess", fi_ + "_30_preprocess_pass", D + fmt("rgba16f")),
                ("kFiInpaintSeed", fi_ + "_31_downsample_inpaint_seed_pass", D + fmt("rgba16f")),
                ("kFiInpaintMip", fi_ + "_31_downsample_inpaint_mip_pass", D + fmt("rgba16f")),
                ("kFiInpaint", fi_ + "_50_inpaint_pass", D + fmt("rgba16f"))]:
    passes.append((n, s, d, INV, 60))
for suffix, f in [("R11", "r11f_g11f_b10f"), ("Rgba16f", "rgba16f"), ("Rgb10a2", "rgb10_a2"), ("Rgba8", "rgba8")]:  # lanes in the colour's format
    for n, s in [("kFiDownsampleOf", "01_downsample_of_colour"), ("kFiDownsampleAll", "02_downsample_all_input"), ("kFiDownsampleColour", "03_downsample_colour")]:
        passes.append((n + suffix, dk_ + f"frameinterpolation_{s}_pass", D + fmt(f), INV, 60))
for n, f in [("kCropR32f", "r32f"), ("kCropRg16f", "rg16f")]:
    passes.append((n, "shaders/frontend/crop", fmt(f), "NFRUX_UNUSED", 60))
passes += [("kPrepare", "shaders/frontend/prepare", [], "NFRUX_UNUSED", 60), ("kMenu", "shaders/frontend/menu", [], "NFRUX_UNUSED", 60),
           ("kCheck", "shaders/frontend/check", [], "NFRUX_UNUSED", 60)]
for suffix, f in [("R11", "r11f_g11f_b10f"), ("Rgba16f", "rgba16f"), ("Rgb10a2", "rgb10_a2"), ("Rgba8", "rgba8")]:
    passes.append((f"kCompose{suffix}", "shaders/frontend/compose", fmt(f), "NFRUX_UNUSED", 60))
for suffix, f in [("R11", "r11f_g11f_b10f"), ("Rgba16f", "rgba16f"), ("Rgb10a2", "rgb10_a2"), ("Rgba8", "rgba8")]:
    passes.append((f"kFiPostprocessNoHistory{suffix}", fi_ + "_50_postprocess_pass", FI + NOHIST + fmt(f), INV, 60))


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(" ".join(cmd[:1] + cmd[-3:]) + "\n" + r.stdout[-2000:] + r.stderr[-2000:])


RELAXED = len(sys.argv) < 4 or sys.argv[3] not in ("1", "ON")
FLOAT_OPS = {127, 129, 131, 133, 136, 140, 142, 143, 144, 145, 146, 147, 148}  # OpFNegate FAdd FSub FMul FDiv FRem and the vector/matrix products


def precise(path, name):
    """Puts NoContraction on every float arithmetic result (as GLSL precise), so neither API's compiler fuses a * b + c."""
    w = list(memoryview(open(path, "rb").read()).cast("I"))
    has, ops, first, i = set(), [], None, 5
    while i < len(w):
        op, n = w[i] & 0xffff, w[i] >> 16
        if op == 17 and w[i + 1] in BANNED:
            raise RuntimeError(f"{name}: declares {BANNED[w[i + 1]]}")
        if 143 <= op <= 148 or (op == 12 and w[i + 4] in (46, 50, 66, 67, 69)):  # products, mix, fma, length, normalize: each API's compiler fuses them its own way
            raise RuntimeError(f"{name}: opaque operation {op, w[i + 4] if op == 12 else 0}; write it out")
        if op == 71 and w[i + 2] == 42:
            has.add(w[i + 1])
        if first is None and 19 <= op <= 39:  # the first type declaration: annotations end here
            first = i
        if op in FLOAT_OPS:
            ops.append(w[i + 2])
        i += n
    add = [] if RELAXED else [x for r in ops if r not in has for x in ((3 << 16) | 71, r, 42)]
    w[first:first] = add
    import struct
    open(path, "wb").write(struct.pack(f"<{len(w)}I", *w))


def build(job):
    name, src, defs, inv, sm, v = job
    n = name + ("Inv" if v else "")
    b = os.path.join(out, n)
    run([GLSLANG, "-V", "--target-env", "vulkan1.1", "-S", "comp", *F32, *defs, f"-D{inv}={v}", "-Igpu", "-Igpu/opticalflow",
         "-Igpu/frameinterpolation", "-Ishaders/opticalflow", "-o", b + ".spv", src + ".glsl"])
    run([tool("spirv-opt"), "--eliminate-dead-functions", b + ".spv", "-o", b + ".spv"])
    precise(b + ".spv", n)
    run([tool("spirv-val"), "--target-env", "vulkan1.1", b + ".spv"])
    run([XCROSS, b + ".spv", "--hlsl", "--shader-model", "62" if sm >= 62 else "60", *(["--hlsl-enable-16bit-types"] if sm >= 62 else []),
         "--output", b + ".hlsl"])
    if True:  # full precision everywhere, so D3D12 matches Vulkan
        h = open(b + ".hlsl").read()
        for k in ("float", "int", "uint"):
            h = h.replace("min16" + k, k)
        if "uint64_t" in h:  # SPIRV-Cross emits the 32-bit intrinsic for 64-bit buffer atomics
            h = h.replace(".InterlockedMin(", ".InterlockedMin64(")
        open(b + ".hlsl", "w").write(h)
    run([XCROSS, b + ".spv", "--reflect", "--output", b + ".json"])
    run([DXC, "-nologo", "-T", f"cs_6_{sm - 60}", "-E", "main", "-O3", "-Wno-conversion", "-Wno-ambig-lit-shift", *([] if RELAXED else ["-Gis"]),
         *(["-enable-16bit-types"] if sm >= 62 else []), b + ".hlsl", "-Fo", b + ".dxil"])
    refl = json.load(open(b + ".json"))
    hl = open(b + ".hlsl").read()
    binds = []
    for kind in ("ubos", "ssbos", "textures", "images", "separate_images", "separate_samplers"):
        for r in refl.get(kind, []):
            if kind == "separate_samplers":
                binds.append((r["binding"], "Sampler"))
                continue
            k = {"ubos": "Cbv", "textures": "SrvTexture", "separate_images": "SrvTexture", "images": "UavTexture"}.get(kind)
            if kind == "ssbos":
                k = "SrvBuffer" if f": register(t{r['binding']}" in hl else "UavBuffer"
            if kind == "images" and f": register(t{r['binding']}" in hl:
                k = "SrvTexture"
            binds.append((r["binding"], k))
    return n, open(b + ".spv", "rb").read(), open(b + ".dxil", "rb").read(), sorted(set(binds)), sm


jobs = [(n, s, d, i, sm, v) for n, s, d, i, sm in passes for v in (0, 1)]
os.chdir(core)
with cf.ThreadPoolExecutor() as ex:
    built = list(ex.map(build, jobs))

# The network's D3D12 kernels, one per layer shape: K, stride, input words of both sources, output channels, resize, per-channel
# rescale, table, output into an image.
LAYERS = [(3, 2, 5, 0, 16, 0, 1, 0, 0), (5, 1, 4, 0, 16, 0, 1, 0, 0), (3, 1, 4, 0, 16, 0, 1, 0, 0), (3, 2, 4, 0, 16, 0, 1, 0, 0),
          (5, 2, 4, 0, 16, 0, 1, 0, 0), (1, 1, 4, 0, 16, 0, 1, 0, 0), (7, 1, 4, 0, 16, 0, 1, 0, 0), (1, 1, 4, 4, 32, 0, 0, 0, 0),
          (1, 1, 4, 8, 48, 0, 0, 0, 0), (1, 1, 4, 12, 16, 0, 1, 0, 0), (3, 1, 4, 0, 16, 1, 1, 0, 0), (3, 1, 4, 4, 16, 0, 1, 0, 0),
          (3, 1, 4, 0, 1, 0, 0, 1, 0), (5, 1, 4, 0, 4, 0, 1, 0, 0)]


def conv(job):
    l, dot4 = job
    k, s, i0, i1, oc, rs, pc, tb, im = l
    n = "kConv_" + "_".join(map(str, l)) + ("_dot4" if dot4 else "")
    px = 4 if s == 1 else 2
    ocb = min(16, (oc + 3) & ~3)
    d = dict(K=k, STRIDE=s, IC0=i0, IC1=i1, OC=oc, OCB=ocb, RESIZE=rs, PERCH=pc, TABLE=tb, IMAGE=im, DOT4=dot4, PX=px, GX=8, GY=8)
    b = os.path.join(out, n + ".dxil")
    run([DXC, "-nologo", "-T", "cs_6_4" if dot4 else "cs_6_0", "-E", "main", "-O3", *[f"-D{a}={v}" for a, v in d.items()],
         os.path.join("ml", "nfrux_conv.hlsl"), "-Fo", b])
    return l, dot4, px, ocb, n, open(b, "rb").read()


with cf.ThreadPoolExecutor() as ex:
    convs = list(ex.map(conv, [(l, d) for l in LAYERS for d in (0, 1)]))

def arr(b):
    b = b + b"\0" * (-len(b) % 4)
    return ",".join(str(int.from_bytes(b[i:i + 4], "little")) for i in range(0, len(b), 4))


tmp = os.path.join(out, "nfrux_shaders.h.tmp")
with open(tmp, "w") as f:
    f.write("// Generated by tools/build_shaders.py.\n#pragma once\n#include \"rhi.h\"\nnamespace nfrux::shaders {\n")
    for n, spv, dxil, binds, sm in built:
        f.write(f"inline constexpr uint32_t {n}_spv[]={{{arr(spv)}}};\ninline constexpr uint32_t {n}_dxil[]={{{arr(dxil)}}};\n")
        f.write(f"inline constexpr ShaderBinding {n}_b[]={{" + ",".join(f"{{{b},BindKind::{k}}}" for b, k in binds) + "};\n")
        f.write(f'inline constexpr Shader {n}{{"{n}",{n}_spv,{len(spv)},{n}_dxil,{len(dxil)},{n}_b,{len(binds)},{sm}}};\n')
    f.write("struct ConvShader { int32_t key[9]; bool dot4; uint32_t px, ocb; const uint32_t* dxil; size_t bytes; };\n")
    for l, d, px, ocb, n, b in convs:
        f.write(f"inline constexpr uint32_t {n}[]={{{arr(b)}}};\n")
    f.write("inline constexpr ConvShader kConvs[]={" + ",".join(
        f"{{{{{',' if 0 else ','.join([])}" if 0 else "{{" + ",".join(map(str, l)) + f"}},{'true' if d else 'false'},{px},{ocb},{n},{len(b)}}}" for l, d, px, ocb, n, b in convs) + "};\n")
    f.write("}\n")
os.replace(tmp, os.path.join(out, "nfrux_shaders.h"))
KERNELS = {"scalar": ["-DML_PATH_SCALAR=1"], "dot4": ["-DML_PATH_DOT4=1"], "dot4_tiled": ["-DML_PATH_DOT4=1", "-DML_DOT4_TILED=1"],
           "dot4_tiled_img": ["-DML_PATH_DOT4=1", "-DML_DOT4_TILED=1", "-DML_DST_IMAGE=1"], "dot4_chain": ["-DML_PATH_DOT4=1", "-DML_DOT4_CHAIN=1"],
           "coopmat_int8": ["-DML_PATH_COOPMAT=1", "-DML_MAT_INT8=1"], "coopmat_fp16": ["-DML_PATH_COOPMAT=1", "-DML_MAT_FP16=1"],
           "coopmat_bf16": ["-DML_PATH_COOPMAT=1", "-DML_MAT_BF16=1"]}
for n, d in KERNELS.items():
    run([GLSLANG, "-V", "--target-env", "vulkan1.2", "-S", "comp", *d, "--vn", "ffx_ml_conv2d_" + n,
         "-o", os.path.join(out, f"ffx_ml_conv2d_{n}_spv.h"), os.path.join("ml", "ffx_ml_conv2d.comp")])
print(f"{len(built)} shaders, {os.path.getsize(os.path.join(out, 'nfrux_shaders.h')) >> 10} KB")
