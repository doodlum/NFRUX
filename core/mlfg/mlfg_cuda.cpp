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
constexpr int kNbs[6] = {1, 2, 3, 4, 6, 8};  // n tiles a warp the kernels are made for (transposed: up to 4)
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

// One kernel: shape 0 = 3x3/1 (pad 1), 1 = 2x2/2, 2 = 1x1, 3 = transposed 2x2/2; NB n tiles (8 channels each) a warp.
std::string Kernel(int shape, int NB)
{
    const bool T = shape == 3;
    const int  K = shape == 0 ? 3 : shape == 1 ? 2 : shape == 2 ? 1 : 1, S = shape == 1 ? 2 : 1, pad = shape == 0 ? 1 : 0;
    const int  taps = K * K, NA = T ? 4 : 1;  // accumulator sets (transposed: one per output phase)
    static const char* names[4] = {"conv3", "conv2", "conv1", "convT"};
    Ptx w;
    w.s = std::string(".visible .entry ") + names[shape] + "_" + std::to_string(NB) + "(.param .align 8 .b8 P[128])\n{\n";
    w(".reg .b32 %%r<160>; .reg .b64 %%rd<160>; .reg .f32 %%f<200>; .reg .pred %%p<120>; .reg .b16 %%h<8>;");
    for (int i = 0; i < 6; ++i) w("ld.param.u64 %%rd%d, [P+%d];", 1 + i, i * 8);  // s0 s1 w b add out
    // r1 inW r2 inH r3 p0 r4 ch0 r5 p1 r6 ch1m r7 split r8 cin r9 outW r10 outH r11 po r12 cho r13 cout r14 writeEnd r15 act
    // r16 fp8 r17 pa r18 cha r19 kc r20 ntp
    for (int i = 0; i < 20; ++i) w("ld.param.u32 %%r%d, [P+%d];", 1 + i, 48 + i * 4);
    w("mov.u32 %%r21, %%tid.x; and.b32 %%r21, %%r21, 31; shr.u32 %%r22, %%r21, 2; and.b32 %%r23, %%r21, 3;");  // lane, g, q
    w("mov.u32 %%r24, %%ctaid.x; shl.b32 %%r24, %%r24, 4; mov.u32 %%r25, %%ctaid.y; mov.u32 %%r35, %%tid.x; shr.u32 %%r35, %%r35, 5; mad.lo.u32 %%r25, %%r25, %d, %%r35; mov.u32 %%r26, %%ctaid.z; mul.lo.u32 %%r26, %%r26, %d;", kCudaWarps, NB);  // x0, y (a warp a row), nt0
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
void Fp8(Ptx& w, int v);

// A convolution (shape 0-2, no transposed) whose warp makes MT * 16 consecutive pixels of a row (rows of 8: g, g + 8,
// g + 16, ...) by NB n tiles: each B fragment serves MT products (the loads, not the tensor cores, limit the 16-pixel
// kernel). Addresses: a base per 8-row group plus a per-tap offset; zero padding by predicates (x per kx and group, y per
// ky). A second source only for 1x1 (the FasterNet concatenation). Same products per output, same order: same bits.
std::string KernelM(int shape, int NB, int MT)
{
    const int K = shape == 0 ? 3 : shape == 1 ? 2 : 1, S = shape == 1 ? 2 : 1, pad = shape == 0 ? 1 : 0, taps = K * K, R = 2 * MT;
    static const char* names[3] = {"conv3", "conv2", "conv1"};
    Ptx w;
    w.s = std::string(".visible .entry ") + names[shape] + "_" + std::to_string(NB) + "m" + std::to_string(MT) + "(.param .align 8 .b8 P[128])\n{\n";
    w(".reg .b32 %%r<200>; .reg .b64 %%rd<200>; .reg .f32 %%f<200>; .reg .pred %%p<160>; .reg .b16 %%h<8>;");
    for (int i = 0; i < 6; ++i) w("ld.param.u64 %%rd%d, [P+%d];", 1 + i, i * 8);
    for (int i = 0; i < 20; ++i) w("ld.param.u32 %%r%d, [P+%d];", 1 + i, 48 + i * 4);
    w("mov.u32 %%r21, %%tid.x; and.b32 %%r21, %%r21, 31; shr.u32 %%r22, %%r21, 2; and.b32 %%r23, %%r21, 3;");
    w("mov.u32 %%r24, %%ctaid.x; mul.lo.u32 %%r24, %%r24, %d; mov.u32 %%r25, %%ctaid.y; mov.u32 %%r35, %%tid.x; shr.u32 %%r35, %%r35, 5; mad.lo.u32 %%r25, %%r25, %d, %%r35;", 16 * MT, kCudaWarps);
    w("mov.u32 %%r26, %%ctaid.z; mul.lo.u32 %%r26, %%r26, %d;", NB);
    w("setp.eq.s32 %%p1, %%r15, 1; setp.eq.s32 %%p2, %%r15, 2; setp.ne.s32 %%p3, %%r16, 0; setp.ne.s32 %%p4, %%r17, 0;");
    w("or.b32 %%r34, %%r3, %%r4; or.b32 %%r34, %%r34, %%r5; or.b32 %%r34, %%r34, %%r6; and.b32 %%r34, %%r34, 1; setp.eq.u32 %%p100, %%r34, 0;");
    // Row groups: x (r 80 + r), base addresses rd(100 + 2r) (src0) / rd(101 + 2r) (src1) of input pixel (x S, y S), channel 0.
    w("mul.lo.s32 %%r36, %%r25, %d;", S);  // input row of ky = pad
    for (int r = 0; r < R; ++r)
    {
        w("add.u32 %%r%d, %%r24, %%r22; add.u32 %%r%d, %%r%d, %d;", 80 + r, 80 + r, 80 + r, 8 * r);
        w("mul.lo.s32 %%r37, %%r%d, %d; mad.lo.s32 %%r37, %%r36, %%r1, %%r37;", 80 + r, S);
        w("mad.lo.s32 %%r38, %%r37, %%r3, %%r4; mul.wide.s32 %%rd7, %%r38, 2; add.s64 %%rd%d, %%rd1, %%rd7;", 100 + 2 * r);
        w("mad.lo.s32 %%r38, %%r37, %%r5, %%r6; mul.wide.s32 %%rd7, %%r38, 2; add.s64 %%rd%d, %%rd2, %%rd7;", 101 + 2 * r);
        for (int kx = 0; kx < K; ++kx)  // x predicates p(40 + kx R + r)
            w("mad.lo.s32 %%r39, %%r%d, %d, %d; setp.ge.s32 %%p5, %%r39, 0; setp.lt.s32 %%p6, %%r39, %%r1; and.pred %%p%d, %%p5, %%p6;", 80 + r, S, kx - pad, 40 + kx * R + r);
    }
    for (int ky = 0; ky < K; ++ky)  // y predicates p(35 + ky)
        w("add.s32 %%r39, %%r36, %d; setp.ge.s32 %%p5, %%r39, 0; setp.lt.s32 %%p6, %%r39, %%r2; and.pred %%p%d, %%p5, %%p6;", ky - pad, 35 + ky);
    // Tap offsets rd(130 + t) (src0 pitch; taps > 1 never have a second source).
    for (int t = 0; t < taps; ++t)
        w("mov.s32 %%r39, %d; mul.lo.s32 %%r39, %%r39, %%r1; add.s32 %%r39, %%r39, %d; mul.lo.s32 %%r39, %%r39, %%r3; mul.wide.s32 %%rd%d, %%r39, 2;", t / K - pad, t % K - pad, 130 + t);
    for (int i = 0; i < MT * NB * 4; ++i) w("mov.f32 %%f%d, 0f00000000;", i);
    w("mul.wide.u32 %%rd8, %%r26, 128; add.s64 %%rd8, %%rd3, %%rd8; mul.wide.u32 %%rd7, %%r21, 4; add.s64 %%rd8, %%rd8, %%rd7;");
    w("mul.wide.u32 %%rd9, %%r20, 128; mul.lo.u64 %%rd60, %%rd9, %d;", taps);
    w("mov.u32 %%r40, 0;");
    w("LOOP:");
    w("setp.ge.s32 %%p8, %%r40, %%r19; @%%p8 bra END;");
    w("shl.b32 %%r41, %%r40, 3; setp.lt.s32 %%p9, %%r41, %%r7;");
    w("shl.b32 %%r42, %%r23, 1; add.s32 %%r42, %%r41, %%r42; add.s32 %%r43, %%r42, 1;");
    w("setp.lt.s32 %%p60, %%r42, %%r8; setp.lt.s32 %%p61, %%r43, %%r8; mul.wide.s32 %%rd61, %%r42, 2;");
    for (int t = 0; t < taps; ++t)
    {
        const int ky = t / K, kx = t % K;
        for (int r = 0; r < R; ++r)
        {
            w("and.pred %%p70, %%p%d, %%p%d;", 35 + ky, 40 + kx * R + r);  // tap in range
            if (taps == 1) w("selp.b64 %%rd62, %%rd%d, %%rd%d, %%p9; add.s64 %%rd62, %%rd62, %%rd61;", 100 + 2 * r, 101 + 2 * r);
            else w("add.s64 %%rd62, %%rd%d, %%rd%d; add.s64 %%rd62, %%rd62, %%rd61;", 100 + 2 * r, 130 + t);
            w("and.pred %%p62, %%p70, %%p60; and.pred %%p63, %%p70, %%p61; and.pred %%p64, %%p63, %%p100; not.pred %%p65, %%p64;");
            w("and.pred %%p66, %%p62, %%p65; and.pred %%p67, %%p63, %%p65; mov.b16 %%h1, 0; mov.b16 %%h2, 0;");
            w("@%%p64 ld.global.u32 %%r%d, [%%rd62]; @%%p66 ld.global.u16 %%h1, [%%rd62]; @%%p67 ld.global.u16 %%h2, [%%rd62+2];", 50 + r);
            w("@%%p65 mov.b32 %%r%d, {%%h1, %%h2};", 50 + r);
        }
        w("mul.lo.u64 %%rd63, %%rd9, %d; add.s64 %%rd63, %%rd63, %%rd8;", t);
        for (int j = 0; j < NB; ++j)
        {
            w("ld.global.u32 %%r49, [%%rd63+%d];", j * 128);
            for (int m = 0; m < MT; ++m)
            {
                const int f = (m * NB + j) * 4;
                w("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%%f%d, %%f%d, %%f%d, %%f%d}, {%%r%d, %%r%d}, {%%r49}, {%%f%d, %%f%d, %%f%d, %%f%d};", f, f + 1, f + 2, f + 3, 50 + 2 * m,
                  51 + 2 * m, f, f + 1, f + 2, f + 3);
            }
        }
    }
    w("add.s64 %%rd8, %%rd8, %%rd60; add.s32 %%r40, %%r40, 1; bra LOOP;");
    w("END:");
    for (int j = 0; j < NB; ++j)
    {
        w("add.u32 %%r60, %%r26, %d; shl.b32 %%r60, %%r60, 3; shl.b32 %%r61, %%r23, 1; add.s32 %%r60, %%r60, %%r61;", j);
        w("setp.lt.s32 %%p20, %%r60, %%r14; add.s32 %%r62, %%r60, 1; setp.lt.s32 %%p21, %%r60, %%r13; setp.lt.s32 %%p22, %%r62, %%r13;");
        w("mul.wide.s32 %%rd64, %%r60, 4; add.s64 %%rd64, %%rd4, %%rd64; ld.global.f32 %%f180, [%%rd64]; ld.global.f32 %%f181, [%%rd64+4];");
        for (int r = 0; r < R; ++r)
        {
            const int f = ((r / 2) * NB + j) * 4 + (r % 2) * 2, lab = w.label++;
            w("mov.u32 %%r63, %%r%d; mov.u32 %%r64, %%r25;", 80 + r);
            w("setp.lt.s32 %%p23, %%r63, %%r9; setp.lt.s32 %%p24, %%r64, %%r10; and.pred %%p23, %%p23, %%p24; and.pred %%p23, %%p23, %%p20;");
            w("@!%%p23 bra SKIP%d;", lab);
            w("mad.lo.s32 %%r65, %%r64, %%r9, %%r63;");
            w("add.rn.f32 %%f182, %%f%d, %%f180; add.rn.f32 %%f183, %%f%d, %%f181;", f, f + 1);
            w("mad.lo.s32 %%r66, %%r65, %%r17, %%r18; add.s32 %%r66, %%r66, %%r60; mul.wide.s32 %%rd65, %%r66, 2; add.s64 %%rd65, %%rd5, %%rd65;");
            w("@%%p4 ld.global.u16 %%h3, [%%rd65]; @%%p4 ld.global.u16 %%h4, [%%rd65+2];");
            w("@%%p4 cvt.f32.f16 %%f184, %%h3; @%%p4 cvt.f32.f16 %%f185, %%h4; @%%p4 add.rn.f32 %%f182, %%f182, %%f184; @%%p4 add.rn.f32 %%f183, %%f183, %%f185;");
            for (int e = 0; e < 2; ++e)
            {
                const int v = 182 + e;
                w("setp.gt.f32 %%p25, %%f%d, 0f00000000; mul.rn.f32 %%f186, %%f%d, 0f3E4CCCCD; selp.f32 %%f186, %%f%d, %%f186, %%p25; @%%p1 mov.f32 %%f%d, %%f186;", v, v, v, v);
                w("max.f32 %%f186, %%f%d, 0f00000000; @%%p2 mov.f32 %%f%d, %%f186;", v, v);
                w("mov.f32 %%f191, %%f%d;", v);
                Fp8(w, 191);
                w("@%%p3 mov.f32 %%f%d, %%f191;", v);
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

// fp32 %f<v> -> its fp8 e4m3 value (torch's rounding; scratch f186-f190, r67-r71, r74, p26-p27).
void Fp8(Ptx& w, int v)
{
    w("mov.b32 %%r67, %%f%d; and.b32 %%r68, %%r67, 0x7FFFFFFF; shr.u32 %%r69, %%r68, 23; max.u32 %%r69, %%r69, 121;", v);
    w("mov.u32 %%r74, 257; sub.u32 %%r70, %%r74, %%r69; shl.b32 %%r70, %%r70, 23; mov.b32 %%f187, %%r70; sub.u32 %%r71, %%r69, 3; shl.b32 %%r71, %%r71, 23; mov.b32 %%f188, %%r71;");
    w("mul.rn.f32 %%f189, %%f%d, %%f187; cvt.rni.f32.f32 %%f189, %%f189; mul.rn.f32 %%f189, %%f189, %%f188;", v);
    w("abs.f32 %%f190, %%f189; setp.gt.f32 %%p26, %%f190, 0f43E00000; setp.ge.u32 %%p27, %%r68, 0x7F800000; or.pred %%p26, %%p26, %%p27;");
    w("selp.f32 %%f%d, 0f7FC00000, %%f189, %%p26;", v);
}

// A whole FasterNet block of stage 2 in one warp: 16 pixels of a row, y = q8(project(q8(relu(expand(cat(q8(pconv(x[:P])),
// x[P:]))))) + x). An mma's accumulator fragment (rows g, g + 8; columns 2q, 2q + 1 of an n tile) is exactly the A
// fragment of the next product's k chunk, so each step's rounded result stays in registers. The same products in the
// same order as the three separate layers: the same bits. ntp*: the layers' padded n tiles (the weights' layout).
std::string FnKernel(int C, int P, int ntpP, int ntpE, int ntpJ)
{
    const int KP = P / 8, KE = C / 8, NE = 2 * C / 8, NJ = C / 8;
    Ptx w;
    w.s = ".visible .entry fn" + std::to_string(C) + "(.param .align 8 .b8 P[72])\n{\n";
    w(".reg .b32 %%r<200>; .reg .b64 %%rd<80>; .reg .f32 %%f<200>; .reg .pred %%p<60>; .reg .b16 %%h<8>;");
    for (int i = 0; i < 8; ++i) w("ld.param.u64 %%rd%d, [P+%d];", 1 + i, i * 8);  // x y wp we wj bp be bj
    w("ld.param.u32 %%r1, [P+64]; ld.param.u32 %%r2, [P+68];");                      // W H
    w("mov.u32 %%r21, %%tid.x; and.b32 %%r21, %%r21, 31; shr.u32 %%r22, %%r21, 2; and.b32 %%r23, %%r21, 3;");
    w("mov.u32 %%r24, %%ctaid.x; shl.b32 %%r24, %%r24, 4; mov.u32 %%r25, %%ctaid.y; mov.u32 %%r35, %%tid.x; shr.u32 %%r35, %%r35, 5; mad.lo.u32 %%r25, %%r25, %d, %%r35; add.u32 %%r27, %%r24, %%r22; add.u32 %%r28, %%r27, 8;", kCudaWarps);
    w("mul.wide.u32 %%rd9, %%r21, 4; mul.wide.u32 %%rd20, %%r23, 4; mul.wide.u32 %%rd21, %%r23, 8;");  // lane*4, q*4 (channel pair bytes), q*8
    for (int t = 0; t < 9; ++t)  // tap t's pixel of rows a / b: predicate p(10 + 2t + r), channel-0 address rd(30 + 2t + r)
    {
        w("add.s32 %%r30, %%r25, %d; setp.ge.s32 %%p5, %%r30, 0; setp.lt.s32 %%p6, %%r30, %%r2; and.pred %%p5, %%p5, %%p6;", t / 3 - 1);
        for (int r = 0; r < 2; ++r)
        {
            const int pr = 10 + t * 2 + r;
            w("add.s32 %%r31, %%r%d, %d; setp.ge.s32 %%p6, %%r31, 0; setp.lt.s32 %%p7, %%r31, %%r1; and.pred %%p6, %%p6, %%p7; and.pred %%p%d, %%p5, %%p6;", r ? 28 : 27, t % 3 - 1, pr);
            w("mad.lo.s32 %%r32, %%r30, %%r1, %%r31; selp.b32 %%r32, %%r32, 0, %%p%d; mul.lo.s32 %%r32, %%r32, %d; mul.wide.s32 %%rd10, %%r32, 2; add.s64 %%rd%d, %%rd1, %%rd10;", pr, C,
              30 + t * 2 + r);
            w("add.s64 %%rd%d, %%rd%d, %%rd20;", 30 + t * 2 + r, 30 + t * 2 + r);  // + this lane's channel pair
        }
    }
    // Loads two channels (word-aligned) of row r at tap t into %r<dst>, 0 outside.
    auto loadA = [&](int t, int r, int chunk, int dst) {
        w("mov.u32 %%r%d, 0; @%%p%d ld.global.u32 %%r%d, [%%rd%d+%d];", dst, 10 + t * 2 + r, dst, 30 + t * 2 + r, chunk * 16);
    };
    auto mma = [&](int f, int a0, int a1) {
        w("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%%f%d, %%f%d, %%f%d, %%f%d}, {%%r%d, %%r%d}, {%%r52}, {%%f%d, %%f%d, %%f%d, %%f%d};", f, f + 1, f + 2, f + 3, a0, a1, f, f + 1, f + 2, f + 3);
    };
    auto bload = [&](int wreg, int kc, int t, int taps, int ntp, int nt) {
        w("add.s64 %%rd11, %%rd%d, %%rd9; ld.global.u32 %%r52, [%%rd11+%d];", wreg, (((kc * taps + t) * ntp + nt) * 32) * 4);
    };
    // Epilogue of n tile n (accumulators f..f+3) into packed fp16 words r<dst> (row a), r<dst+1> (row b).
    auto finish = [&](int f, int breg, int n, bool relu, bool residual, int dst) {
        w("add.s64 %%rd12, %%rd%d, %%rd21; ld.global.f32 %%f180, [%%rd12+%d]; ld.global.f32 %%f181, [%%rd12+%d];", breg, n * 32, n * 32 + 4);
        for (int r = 0; r < 2; ++r)
        {
            w("add.rn.f32 %%f182, %%f%d, %%f180; add.rn.f32 %%f183, %%f%d, %%f181;", f + 2 * r, f + 2 * r + 1);
            if (residual)
            {
                loadA(4, r, n, 58);
                w("mov.b32 {%%h3, %%h4}, %%r58; cvt.f32.f16 %%f184, %%h3; cvt.f32.f16 %%f185, %%h4; add.rn.f32 %%f182, %%f182, %%f184; add.rn.f32 %%f183, %%f183, %%f185;");
            }
            if (relu) w("max.f32 %%f182, %%f182, 0f00000000; max.f32 %%f183, %%f183, 0f00000000;");
            Fp8(w, 182), Fp8(w, 183);
            w("cvt.rn.f16.f32 %%h5, %%f182; cvt.rn.f16.f32 %%h6, %%f183; mov.b32 %%r%d, {%%h5, %%h6};", dst + r);
        }
    };
    // pconv 3x3 P -> P on x[:P]: accumulators f0 .. f(4 KP - 1); results -> pA r100 + 2j (+1).
    for (int i = 0; i < KP * 4; ++i) w("mov.f32 %%f%d, 0f00000000;", i);
    for (int kc = 0; kc < KP; ++kc)
        for (int t = 0; t < 9; ++t)
        {
            loadA(t, 0, kc, 50), loadA(t, 1, kc, 51);
            for (int n = 0; n < KP; ++n) bload(3, kc, t, 9, ntpP, n), mma(n * 4, 50, 51);
        }
    for (int n = 0; n < KP; ++n) finish(n * 4, 6, n, false, false, 100 + 2 * n);
    // expand 1x1 C -> 2C on cat(pA, x[P:]), relu: accumulators f0 .. f(4 NE - 1); results -> hA r120 + 2j (+1).
    for (int i = 0; i < NE * 4; ++i) w("mov.f32 %%f%d, 0f00000000;", i);
    for (int kc = 0; kc < KE; ++kc)
    {
        int a0 = 100 + 2 * kc, a1 = a0 + 1;
        if (kc >= KP) loadA(4, 0, kc, 50), loadA(4, 1, kc, 51), a0 = 50, a1 = 51;
        for (int n = 0; n < NE; ++n) bload(4, kc, 0, 1, ntpE, n), mma(n * 4, a0, a1);
    }
    for (int n = 0; n < NE; ++n) finish(n * 4, 7, n, true, false, 120 + 2 * n);
    // project 1x1 2C -> C, + x: accumulators f0 .. f(4 NJ - 1); stored.
    for (int i = 0; i < NJ * 4; ++i) w("mov.f32 %%f%d, 0f00000000;", i);
    for (int kc = 0; kc < NE; ++kc)
        for (int n = 0; n < NJ; ++n) bload(5, kc, 0, 1, ntpJ, n), mma(n * 4, 120 + 2 * kc, 121 + 2 * kc);
    // Output addresses: y at the centre pixel of rows a / b (rd 30 + 8, 31 + 8 relative to x), stored where in range.
    w("sub.s64 %%rd13, %%rd2, %%rd1; add.s64 %%rd14, %%rd38, %%rd13; add.s64 %%rd15, %%rd39, %%rd13;");
    for (int n = 0; n < NJ; ++n)
    {
        finish(n * 4, 8, n, false, true, 160 + 2 * n);
        w("@%%p18 st.global.u32 [%%rd14+%d], %%r%d; @%%p19 st.global.u32 [%%rd15+%d], %%r%d;", n * 16, 160 + 2 * n, n * 16, 161 + 2 * n);
    }
    w("ret;");
    return w.s + "}\n";
}
}  // namespace

std::string CudaPtx()
{
    std::string p = ".version 7.0\n.target sm_75\n.address_size 64\n";
    for (int s = 0; s < 4; ++s)
        for (int nb : kNbs)
            if (s < 3 || nb <= 4) p += Kernel(s, nb);
    for (int s = 0; s < 3; ++s) p += KernelM(s, 8, 2);
    p += FnKernel(48, 16, 2, 16, 8), p += FnKernel(64, 32, 8, 16, 8);  // the layers' padded n tiles (CudaNb)
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
    static const char* names[4] = {"conv3", "conv2", "conv1", "convT"};
    for (int s = 0; s < 4; ++s)
        for (int i = 0; i < 6; ++i)
            if ((s < 3 || kNbs[i] <= 4) && !(k->m_conv[s][i] = cuda::CreateFunction(d3d, k->m_module, (std::string(names[s]) + "_" + std::to_string(kNbs[i])).c_str()))) return fail("CreateCuFunction");
    for (int s = 0; s < 3; ++s)
        if (!(k->m_convM[s] = cuda::CreateFunction(d3d, k->m_module, (std::string(names[s]) + "_8m2").c_str()))) return fail("CreateCuFunction");
    for (int i = 0; i < 2; ++i)
        if (!(k->m_fnk[i] = cuda::CreateFunction(d3d, k->m_module, i ? "fn64" : "fn48"))) return fail("CreateCuFunction");
    return k;
}
CudaKernels::~CudaKernels()
{
    std::vector<void*> f(m_fnk, m_fnk + 2);
    f.insert(f.end(), m_convM, m_convM + 3);
    for (auto& s : m_conv) f.insert(f.end(), s, s + 6);
    cuda::Destroy(m_device, m_module, f);
}

bool CudaKernels::LaunchFn(void* list, uint32_t C, const uint32_t grid[3], const FnParams& p) const
{
    return cuda::Launch(static_cast<ID3D12GraphicsCommandList*>(list), m_fnk[C == 48 ? 0 : 1], grid, 32 * kCudaWarps, &p, sizeof(p));
}

bool CudaKernels::LaunchM(void* list, int shape, const uint32_t grid[3], const CudaParams& p) const
{
    return cuda::Launch(static_cast<ID3D12GraphicsCommandList*>(list), m_convM[shape], grid, 32 * kCudaWarps, &p, sizeof(p));
}

bool CudaKernels::Launch(void* list, int shape, uint32_t nb, const uint32_t grid[3], const CudaParams& p) const
{
    int i = 0;
    while (i < 5 && kNbs[i] != int(nb)) ++i;
    return kNbs[i] == int(nb) && m_conv[shape][i] && cuda::Launch(static_cast<ID3D12GraphicsCommandList*>(list), m_conv[shape][i], grid, 32 * kCudaWarps, &p, sizeof(p));
}
}  // namespace nfrux::mlfg
