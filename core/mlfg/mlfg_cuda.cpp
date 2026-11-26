// The "cuda" tier of NFRUX's FSR FG4 runtime on D3D12: the network's layers as CUDA kernels on fp16 tensor cores
// (mma.sync m16n8k8, fp16 x fp16 -> fp32, sm_75 and later), launched into the host's command list through NVAPI like the
// int8 network's (core/ml/cuda_network.cpp; no CUDA toolkit: the PTX is written here and the driver links it). Same
// layers, constants and epilogue as mlfg_conv.hlsl; the fp32 sums are the tensor cores'. One warp makes 16 output pixels
// of a row (transposed: 16 input pixels -> 32 x 2 outputs) by 64 channels (transposed: 32); input channels 8 at a time,
// A fragments loaded straight from the NHWC tensors (zero padding by predicate), B fragments from the weights in
// fragment order ([kc][tap][ntPad][lane] words). SPDX-License-Identifier: MIT
#include "mlfg_cuda.h"

#include "cuda_network.h"

#include <cstdarg>
#include <cstdio>

namespace nfrux::mlfg
{
namespace
{
struct Ptx
{
    std::string s;
    int         label = 0;
    void operator()(const char* f, ...)
    {
        char    b[512];
        va_list a;
        va_start(a, f);
        vsnprintf(b, sizeof(b), f, a);
        va_end(a);
        s += "    ", s += b, s += '\n';
    }
};

// One kernel: shape 0 = 3x3/1 (pad 1), 1 = 2x2/2, 2 = 1x1, 3 = transposed 2x2/2; narrow: one n tile (8 channels) a warp,
// for the layers with at most 16 outputs.
std::string Kernel(int shape, bool narrow)
{
    const bool T = shape == 3;
    const int  K = shape == 0 ? 3 : shape == 1 ? 2 : shape == 2 ? 1 : 1, S = shape == 1 ? 2 : 1, pad = shape == 0 ? 1 : 0;
    const int  taps = K * K, NB = narrow ? 1 : T ? 4 : 8, NA = T ? 4 : 1;  // n tiles a warp; accumulator sets (transposed: one per output phase)
    static const char* names[4] = {"conv3", "conv2", "conv1", "convT"};
    Ptx w;
    w.s = std::string(".visible .entry ") + names[shape] + (narrow ? "n" : "") + "(.param .align 8 .b8 P[128])\n{\n";
    w(".reg .b32 %%r<160>; .reg .b64 %%rd<160>; .reg .f32 %%f<200>; .reg .pred %%p<120>; .reg .b16 %%h<8>;");
    for (int i = 0; i < 6; ++i) w("ld.param.u64 %%rd%d, [P+%d];", 1 + i, i * 8);  // s0 s1 w b add out
    // r1 inW r2 inH r3 p0 r4 ch0 r5 p1 r6 ch1m r7 split r8 cin r9 outW r10 outH r11 po r12 cho r13 cout r14 writeEnd r15 act
    // r16 fp8 r17 pa r18 cha r19 kc r20 ntp
    for (int i = 0; i < 20; ++i) w("ld.param.u32 %%r%d, [P+%d];", 1 + i, 48 + i * 4);
    w("mov.u32 %%r21, %%tid.x; and.b32 %%r21, %%r21, 31; shr.u32 %%r22, %%r21, 2; and.b32 %%r23, %%r21, 3;");  // lane, g, q
    w("mov.u32 %%r24, %%ctaid.x; shl.b32 %%r24, %%r24, 4; mov.u32 %%r25, %%ctaid.y; mov.u32 %%r26, %%ctaid.z; mul.lo.u32 %%r26, %%r26, %d;", NB);  // x0, y, nt0
    w("add.u32 %%r27, %%r24, %%r22; add.u32 %%r28, %%r27, 8;");  // xa, xb (output pixels; transposed: input pixels)
    w("setp.eq.s32 %%p1, %%r15, 1; setp.eq.s32 %%p2, %%r15, 2; setp.ne.s32 %%p3, %%r16, 0; setp.ne.s32 %%p4, %%r17, 0;");  // leaky, relu, fp8, add
    w("or.b32 %%r34, %%r3, %%r4; or.b32 %%r34, %%r34, %%r5; or.b32 %%r34, %%r34, %%r6; and.b32 %%r34, %%r34, 1; setp.eq.u32 %%p30, %%r34, 0;");  // aligned
    // Per tap and row: in-range predicate and the byte addresses of channel 0 in src0 / src1 (rd 10 + 4t ..).
    for (int t = 0; t < (T ? 1 : taps); ++t)
    {
        const int ky = t / K, kx = t % K;
        w("mad.lo.s32 %%r30, %%r25, %d, %d;", S, ky - pad);  // iy
        w("setp.ge.s32 %%p5, %%r30, 0; setp.lt.s32 %%p6, %%r30, %%r2; and.pred %%p5, %%p5, %%p6;");
        for (int r = 0; r < 2; ++r)
        {
            const int pr = 10 + t * 2 + r;  // predicate in range
            w("mad.lo.s32 %%r31, %%r%d, %d, %d;", r ? 28 : 27, S, kx - pad);
            w("setp.ge.s32 %%p6, %%r31, 0; setp.lt.s32 %%p7, %%r31, %%r1; and.pred %%p6, %%p6, %%p7; and.pred %%p%d, %%p5, %%p6;", pr);
            w("mad.lo.s32 %%r32, %%r30, %%r1, %%r31; selp.b32 %%r32, %%r32, 0, %%p%d;", pr);  // pixel (0 when outside)
            w("mad.lo.s32 %%r33, %%r32, %%r3, %%r4; mul.wide.s32 %%rd7, %%r33, 2; add.s64 %%rd%d, %%rd1, %%rd7;", 10 + (t * 2 + r) * 2);
            w("mad.lo.s32 %%r33, %%r32, %%r5, %%r6; mul.wide.s32 %%rd7, %%r33, 2; add.s64 %%rd%d, %%rd2, %%rd7;", 11 + (t * 2 + r) * 2);
        }
    }
    for (int i = 0; i < NA * NB * 4; ++i) w("mov.f32 %%f%d, 0f00000000;", i);
    // Weights: this lane's word of n tile nt0: w + nt0 * 128 + lane * 4; a chunk of 8 channels is taps * ntp * 128 bytes.
    w("mul.wide.u32 %%rd8, %%r26, 128; add.s64 %%rd8, %%rd3, %%rd8; mul.wide.u32 %%rd7, %%r21, 4; add.s64 %%rd8, %%rd8, %%rd7;");
    w("mul.wide.u32 %%rd9, %%r20, 128;");  // a tap's stride (ntp * 128)
    w("mul.lo.u64 %%rd60, %%rd9, %d;", T ? 4 : taps);  // a chunk's stride
    w("mov.u32 %%r40, 0;");  // kc
    w("LOOP:");
    w("setp.ge.s32 %%p8, %%r40, %%r19; @%%p8 bra END;");
    w("shl.b32 %%r41, %%r40, 3; setp.lt.s32 %%p9, %%r41, %%r7;");  // chunk's first channel, from src0
    w("shl.b32 %%r42, %%r23, 1; add.s32 %%r42, %%r41, %%r42; add.s32 %%r43, %%r42, 1;");  // c, c + 1
    w("setp.lt.s32 %%p60, %%r42, %%r8; setp.lt.s32 %%p61, %%r43, %%r8; mul.wide.s32 %%rd61, %%r42, 2;");
    for (int t = 0; t < (T ? 1 : taps); ++t)
    {
        for (int r = 0; r < 2; ++r)
        {
            const int pr = 10 + t * 2 + r, ra = 10 + (t * 2 + r) * 2;
            w("selp.b64 %%rd62, %%rd%d, %%rd%d, %%p9; add.s64 %%rd62, %%rd62, %%rd61;", ra, ra + 1);
            // Both channels in range and the tensors' halves word-aligned: one 32-bit load; else two 16-bit ones.
            w("and.pred %%p62, %%p%d, %%p60; and.pred %%p63, %%p%d, %%p61; and.pred %%p64, %%p63, %%p30; not.pred %%p65, %%p64;", pr, pr);
            w("and.pred %%p66, %%p62, %%p65; and.pred %%p67, %%p63, %%p65; mov.b16 %%h1, 0; mov.b16 %%h2, 0;");
            w("@%%p64 ld.global.u32 %%r%d, [%%rd62]; @%%p66 ld.global.u16 %%h1, [%%rd62]; @%%p67 ld.global.u16 %%h2, [%%rd62+2];", 50 + r);
            w("@%%p65 mov.b32 %%r%d, {%%h1, %%h2};", 50 + r);
        }
        for (int a = 0; a < NA; ++a)
        {
            const int tap = T ? a : t;
            w("mul.lo.u64 %%rd63, %%rd9, %d; add.s64 %%rd63, %%rd63, %%rd8;", tap);
            for (int j = 0; j < NB; ++j)
            {
                const int f = (a * NB + j) * 4;
                w("ld.global.u32 %%r52, [%%rd63+%d];", j * 128);
                w("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%%f%d, %%f%d, %%f%d, %%f%d}, {%%r50, %%r51}, {%%r52}, {%%f%d, %%f%d, %%f%d, %%f%d};", f, f + 1, f + 2, f + 3, f, f + 1, f + 2,
                  f + 3);
            }
        }
    }
    w("add.s64 %%rd8, %%rd8, %%rd60; add.s32 %%r40, %%r40, 1; bra LOOP;");
    w("END:");
    // Epilogue: bias, add, activation, rounding (fp8 e4m3 as torch, then fp16 round to nearest even), two channels a word.
    for (int a = 0; a < NA; ++a)
        for (int j = 0; j < NB; ++j)
        {
            w("add.u32 %%r60, %%r26, %d; shl.b32 %%r60, %%r60, 3; shl.b32 %%r61, %%r23, 1; add.s32 %%r60, %%r60, %%r61;", j);  // c
            w("setp.lt.s32 %%p20, %%r60, %%r14; add.s32 %%r62, %%r60, 1; setp.lt.s32 %%p21, %%r60, %%r13; setp.lt.s32 %%p22, %%r62, %%r13;");
            w("mul.wide.s32 %%rd64, %%r60, 4; add.s64 %%rd64, %%rd4, %%rd64; ld.global.f32 %%f180, [%%rd64]; ld.global.f32 %%f181, [%%rd64+4];");
            for (int r = 0; r < 2; ++r)
            {
                const int f = (a * NB + j) * 4 + r * 2, lab = w.label++;
                if (T) w("shl.b32 %%r63, %%r%d, 1; add.s32 %%r63, %%r63, %d; shl.b32 %%r64, %%r25, 1; add.s32 %%r64, %%r64, %d;", r ? 28 : 27, a & 1, a >> 1);
                else w("mov.u32 %%r63, %%r%d; mov.u32 %%r64, %%r25;", r ? 28 : 27);
                w("setp.lt.s32 %%p23, %%r63, %%r9; setp.lt.s32 %%p24, %%r64, %%r10; and.pred %%p23, %%p23, %%p24; and.pred %%p23, %%p23, %%p20;");
                w("@!%%p23 bra SKIP%d;", lab);
                w("mad.lo.s32 %%r65, %%r64, %%r9, %%r63;");  // output pixel
                w("add.rn.f32 %%f182, %%f%d, %%f180; add.rn.f32 %%f183, %%f%d, %%f181;", f, f + 1);
                w("mad.lo.s32 %%r66, %%r65, %%r17, %%r18; add.s32 %%r66, %%r66, %%r60; mul.wide.s32 %%rd65, %%r66, 2; add.s64 %%rd65, %%rd5, %%rd65;");
                w("@%%p4 ld.global.u16 %%h3, [%%rd65]; @%%p4 ld.global.u16 %%h4, [%%rd65+2];");
                w("@%%p4 cvt.f32.f16 %%f184, %%h3; @%%p4 cvt.f32.f16 %%f185, %%h4; @%%p4 add.rn.f32 %%f182, %%f182, %%f184; @%%p4 add.rn.f32 %%f183, %%f183, %%f185;");
                for (int e = 0; e < 2; ++e)
                {
                    const int v = 182 + e;
                    w("setp.gt.f32 %%p25, %%f%d, 0f00000000; mul.rn.f32 %%f186, %%f%d, 0f3E4CCCCD; selp.f32 %%f186, %%f%d, %%f186, %%p25; @%%p1 mov.f32 %%f%d, %%f186;", v, v, v, v);
                    w("max.f32 %%f186, %%f%d, 0f00000000; @%%p2 mov.f32 %%f%d, %%f186;", v, v);
                    // fp8 e4m3: scale by 2^(130 - max(e, 121)), round to an integer (nearest even), scale back; above 448 or not finite: NaN.
                    w("mov.b32 %%r67, %%f%d; and.b32 %%r68, %%r67, 0x7FFFFFFF; shr.u32 %%r69, %%r68, 23; max.u32 %%r69, %%r69, 121;", v);
                    w("mov.u32 %%r74, 257; sub.u32 %%r70, %%r74, %%r69; shl.b32 %%r70, %%r70, 23; mov.b32 %%f187, %%r70; sub.u32 %%r71, %%r69, 3; shl.b32 %%r71, %%r71, 23; mov.b32 %%f188, %%r71;");
                    w("mul.rn.f32 %%f189, %%f%d, %%f187; cvt.rni.f32.f32 %%f189, %%f189; mul.rn.f32 %%f189, %%f189, %%f188;", v);
                    w("abs.f32 %%f190, %%f189; setp.gt.f32 %%p26, %%f190, 0f43E00000; setp.ge.u32 %%p27, %%r68, 0x7F800000; or.pred %%p26, %%p26, %%p27;");
                    w("selp.f32 %%f189, 0f7FC00000, %%f189, %%p26; @%%p3 mov.f32 %%f%d, %%f189;", v);
                }
                w("selp.f32 %%f182, %%f182, 0f00000000, %%p21; selp.f32 %%f183, %%f183, 0f00000000, %%p22;");
                w("cvt.rn.f16.f32 %%h5, %%f182; cvt.rn.f16.f32 %%h6, %%f183; mov.b32 %%r72, {%%h5, %%h6};");
                w("mad.lo.s32 %%r73, %%r65, %%r11, %%r12; add.s32 %%r73, %%r73, %%r60; mul.wide.s32 %%rd66, %%r73, 2; add.s64 %%rd66, %%rd6, %%rd66; st.global.u32 [%%rd66], %%r72;");
                w("SKIP%d:", lab);
            }
        }
    w("ret;");
    return w.s + "}\n";
}
}  // namespace

std::string CudaPtx()
{
    std::string p = ".version 7.0\n.target sm_75\n.address_size 64\n";
    for (int s = 0; s < 4; ++s) p += Kernel(s, false);
    p += Kernel(0, true), p += Kernel(3, true);
    return p;
}

std::unique_ptr<CudaKernels> CudaKernels::Create(Device& dev, std::string* error)
{
    auto fail = [&](const std::string& m) -> std::unique_ptr<CudaKernels> { if (error) *error = m; return nullptr; };
    auto* d3d = static_cast<ID3D12Device*>(dev.NativeDevice());
    if (!d3d || !cuda::Available()) return fail("no NVAPI CUDA launch");
    if (!cuda::ComputeCapability(d3d->GetAdapterLuid(), error)) return nullptr;
    const std::vector<uint8_t> cubin = cuda::Cubin(CudaPtx(), d3d->GetAdapterLuid(), error);
    if (cubin.empty()) return nullptr;
    std::unique_ptr<CudaKernels> k(new CudaKernels());
    k->m_device = d3d;
    if (!(k->m_module = cuda::CreateModule(d3d, cubin))) return fail("CreateCuModule");
    static const char* names[6] = {"conv3", "conv2", "conv1", "convT", "conv3n", "convTn"};
    for (int i = 0; i < 6; ++i)
        if (!(k->m_fn[i] = cuda::CreateFunction(d3d, k->m_module, names[i]))) return fail("CreateCuFunction");
    return k;
}
CudaKernels::~CudaKernels() { cuda::Destroy(m_device, m_module, {m_fn[0], m_fn[1], m_fn[2], m_fn[3], m_fn[4], m_fn[5]}); }

bool CudaKernels::Launch(void* list, int shape, const uint32_t grid[3], const CudaParams& p) const
{
    return cuda::Launch(static_cast<ID3D12GraphicsCommandList*>(list), m_fn[shape], grid, 32, &p, sizeof(p));
}
}  // namespace nfrux::mlfg
