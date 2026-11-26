// The "cuda" tier of NFRUX's FSR FG4 runtime on D3D12: the network's layers as CUDA kernels on fp16 tensor cores
// (mma.sync m16n8k8, fp16 x fp16 -> fp32, sm_75 and later), launched into the host's command list through NVAPI like the
// int8 network's (core/ml/cuda_network.cpp; no CUDA toolkit: the PTX is written here and the driver links it). Same
// layers, constants and epilogue as mlfg_conv.hlsl; the fp32 sums are the tensor cores'. One warp makes 16 output pixels
// of a row (transposed: 16 input pixels -> 32 x 2 outputs) by 64 channels (transposed: 32); input channels 8 at a time,
// A fragments loaded straight from the NHWC tensors (zero padding by predicate), B fragments from the weights in
// fragment order ([kc][tap][ntPad][lane] words). SPDX-License-Identifier: MIT
#include "mlfg_cuda.h"

#include "cuda_network.h"
#include "log.h"

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

void Mma(Ptx& w, int f, int a0, int a1, int b)  // accumulator f .. f + 3 (fp32) += A (a0, a1) x B (b)
{
    w("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%%f%d, %%f%d, %%f%d, %%f%d}, {%%r%d, %%r%d}, {%%r%d}, {%%f%d, %%f%d, %%f%d, %%f%d};", f, f + 1, f + 2, f + 3, a0, a1, b, f, f + 1, f + 2, f + 3);
}
// One kernel: shape 0 = 3x3/1 (pad 1), 1 = 2x2/2, 2 = 1x1, 3 = transposed 2x2/2; NB n tiles (8 channels each) a warp.
std::string Kernel(int shape, int NB, bool split = false)
{
    const bool T = shape == 3;
    const int  K = shape == 0 ? 3 : shape == 1 ? 2 : shape == 2 ? 1 : 1, S = shape == 1 ? 2 : 1, pad = shape == 0 ? 1 : 0;
    const int  taps = K * K, NA = T ? 4 : 1;  // accumulator sets (transposed: one per output phase)
    static const char* names[4] = {"conv3", "conv2", "conv1", "convT"};
    Ptx w;
    w.s = std::string(".visible .entry ") + names[shape] + "_" + std::to_string(NB) + (split ? "k" : "") + (split ? "(.param .align 8 .b8 P[144])\n{\n" : "(.param .align 8 .b8 P[128])\n{\n");
    w(".reg .b32 %%r<200>; .reg .b64 %%rd<160>; .reg .f32 %%f<200>; .reg .pred %%p<120>; .reg .b16 %%h<8>;");
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
    if (split)  // grid z = n-tile groups x splits: this split's chunks [kc0, kc0 + per)
    {
        w("ld.param.u64 %%rd70, [P+128]; ld.param.u32 %%r110, [P+136];");  // partial sums, chunks a split
        w("mov.u32 %%r111, %%ctaid.z; div.u32 %%r112, %%r20, %d; rem.u32 %%r113, %%r111, %%r112; div.u32 %%r114, %%r111, %%r112;", NB);  // groups, group, split
        w("mul.lo.u32 %%r26, %%r113, %d; mul.lo.u32 %%r40, %%r114, %%r110; add.u32 %%r115, %%r40, %%r110; min.u32 %%r19, %%r19, %%r115;", NB);
        // the weight base again for this group, and the first chunk
        w("mul.wide.u32 %%rd8, %%r26, 128; add.s64 %%rd8, %%rd3, %%rd8; mul.wide.u32 %%rd7, %%r21, 4; add.s64 %%rd8, %%rd8, %%rd7; mul.wide.u32 %%rd7, %%r40, 1; mul.lo.u64 %%rd7, %%rd7, %%rd60; add.s64 %%rd8, %%rd8, %%rd7;");
    }
    else
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
                Mma(w, f, 50, 51, 52);
            }
        }
    }
    w("add.s64 %%rd8, %%rd8, %%rd60; add.s32 %%r40, %%r40, 1; bra LOOP;");
    w("END:");
    if (split)  // the raw sums: part[split][pixel][ntp * 8] fp32
    {
        w("mul.lo.u32 %%r116, %%r9, %%r10; mul.lo.u32 %%r116, %%r116, %%r114; shl.b32 %%r117, %%r20, 3;");  // split * pixels, cpad
        for (int j = 0; j < NB; ++j)
        {
            w("add.u32 %%r60, %%r26, %d; shl.b32 %%r60, %%r60, 3; shl.b32 %%r61, %%r23, 1; add.s32 %%r60, %%r60, %%r61;", j);
            for (int r = 0; r < 2; ++r)
            {
                const int f = j * 4 + r * 2, lab = w.label++;
                w("setp.lt.s32 %%p23, %%r%d, %%r9; setp.lt.s32 %%p24, %%r25, %%r10; and.pred %%p23, %%p23, %%p24; @!%%p23 bra SKIP%d;", r ? 28 : 27, lab);
                w("mad.lo.s32 %%r65, %%r25, %%r9, %%r%d; add.u32 %%r65, %%r65, %%r116; mad.lo.u32 %%r65, %%r65, %%r117, %%r60; mul.wide.u32 %%rd66, %%r65, 4; add.s64 %%rd66, %%rd70, %%rd66;", r ? 28 : 27);
                w("st.global.v2.f32 [%%rd66], {%%f%d, %%f%d};", f, f + 1);
                w("SKIP%d:", lab);
            }
        }
        w("ret;");
        return w.s + "}\n";
    }
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

// The 8-row-group convolutions' epilogue (KernelM): bias, add, activation, rounding, two channels a word.
void Epilogue(Ptx& w, int NB, int R)
{
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
}


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
        for (int kx = 0; kx < K; ++kx)  // x predicates p(120 + kx R + r)
            w("mad.lo.s32 %%r39, %%r%d, %d, %d; setp.ge.s32 %%p5, %%r39, 0; setp.lt.s32 %%p6, %%r39, %%r1; and.pred %%p%d, %%p5, %%p6;", 80 + r, S, kx - pad, 120 + kx * R + r);
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
            w("and.pred %%p70, %%p%d, %%p%d;", 35 + ky, 120 + kx * R + r);  // tap in range
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
                Mma(w, f, 50 + 2 * m, 51 + 2 * m, 49);
            }
        }
    }
    w("add.s64 %%rd8, %%rd8, %%rd60; add.s32 %%r40, %%r40, 1; bra LOOP;");
    w("END:");
    Epilogue(w, NB, R);
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
// The FasterNet block on FP8 tensor cores (sm_89+: mma.m16n8k32 e4m3 x e4m3 -> fp32). Every operand is an fp8 value
// already (x and each step's rounded output; the stage-2 weights), so the products are exact and only the fp32 sums'
// order differs from the fp16 kernel. K order inside a 32-chunk is permuted (kpos -> channel Fp8Channel) so that an
// accumulator fragment (columns 2q, 2q + 1 of four n tiles) is directly an A fragment; the weights are packed the same
// way (Network, Fp8Channel). Chunks: pconv P = 16 two taps a chunk (the 10th tap zero), P = 32 a tap a chunk; expand
// and project 32 input channels a chunk (past C / 2C zero).
std::string FnKernel8(int C, int P)
{
    const int TP = P / 8, TE = 2 * C / 8, TJ = C / 8, KCP = P == 16 ? 5 : 9, KCE = (C + 31) / 32, KCJ = 2 * C / 32;
    Ptx w;
    w.s = ".visible .entry fn" + std::to_string(C) + "f8(.param .align 8 .b8 P[72])\n{\n";
    w(".reg .b32 %%r<220>; .reg .b64 %%rd<80>; .reg .f32 %%f<200>; .reg .pred %%p<60>; .reg .b16 %%h<120>;");
    for (int i = 0; i < 8; ++i) w("ld.param.u64 %%rd%d, [P+%d];", 1 + i, i * 8);
    w("ld.param.u32 %%r1, [P+64]; ld.param.u32 %%r2, [P+68];");
    w("mov.u32 %%r21, %%tid.x; and.b32 %%r21, %%r21, 31; shr.u32 %%r22, %%r21, 2; and.b32 %%r23, %%r21, 3;");
    w("mov.u32 %%r24, %%ctaid.x; shl.b32 %%r24, %%r24, 4; mov.u32 %%r25, %%ctaid.y; mov.u32 %%r35, %%tid.x; shr.u32 %%r35, %%r35, 5; mad.lo.u32 %%r25, %%r25, %d, %%r35; add.u32 %%r27, %%r24, %%r22; add.u32 %%r28, %%r27, 8;", kCudaWarps);
    w("mul.wide.u32 %%rd9, %%r21, 8; mul.wide.u32 %%rd20, %%r23, 4; mul.wide.u32 %%rd21, %%r23, 8;");  // lane*8 (two B words), q*4, q*8
    for (int t = 0; t < 9; ++t)
    {
        w("add.s32 %%r30, %%r25, %d; setp.ge.s32 %%p5, %%r30, 0; setp.lt.s32 %%p6, %%r30, %%r2; and.pred %%p5, %%p5, %%p6;", t / 3 - 1);
        for (int r = 0; r < 2; ++r)
        {
            const int pr = 10 + t * 2 + r;
            w("add.s32 %%r31, %%r%d, %d; setp.ge.s32 %%p6, %%r31, 0; setp.lt.s32 %%p7, %%r31, %%r1; and.pred %%p6, %%p6, %%p7; and.pred %%p%d, %%p5, %%p6;", r ? 28 : 27, t % 3 - 1, pr);
            w("mad.lo.s32 %%r32, %%r30, %%r1, %%r31; selp.b32 %%r32, %%r32, 0, %%p%d; mul.lo.s32 %%r32, %%r32, %d; mul.wide.s32 %%rd10, %%r32, 2; add.s64 %%rd%d, %%rd1, %%rd10;", pr, C, 30 + t * 2 + r);
            w("add.s64 %%rd%d, %%rd%d, %%rd20;", 30 + t * 2 + r, 30 + t * 2 + r);
        }
    }
    int lab = 0;
    // Channels ch, ch + 1 of x at tap t, row r, as an e4m3 pair in %h<dst> (0 outside).
    auto x8 = [&](int t, int r, int ch, int dst) {
        w("mov.u32 %%r58, 0; @%%p%d ld.global.u32 %%r58, [%%rd%d+%d]; cvt.rn.satfinite.e4m3x2.f16x2 %%h%d, %%r58;", 10 + t * 2 + r, 30 + t * 2 + r, ch * 2, dst);
        (void)lab;
    };
    auto pack = [&](int dst, int lo, int hi) { w("mov.b32 %%r%d, {%%h%d, %%h%d};", dst, lo, hi); };
    auto mma = [&](int f) {
        w("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%%f%d, %%f%d, %%f%d, %%f%d}, {%%r50, %%r51, %%r52, %%r53}, {%%r54, %%r55}, {%%f%d, %%f%d, %%f%d, %%f%d};", f, f + 1, f + 2, f + 3, f, f + 1, f + 2, f + 3);
    };
    auto bload = [&](int wreg, int kc, int nt, int NT) { w("add.s64 %%rd11, %%rd%d, %%rd9; ld.global.v2.u32 {%%r54, %%r55}, [%%rd11+%d];", wreg, ((kc * NT + nt) * 32) * 8); };
    // Epilogue of n tile n: fp8 values; e4m3 pairs into h<dst> (row a), h<dst+1> (row b), or for the last step fp16 stores.
    auto finish = [&](int f, int breg, int n, bool relu, bool residual, int dst) {
        w("add.s64 %%rd12, %%rd%d, %%rd21; ld.global.f32 %%f180, [%%rd12+%d]; ld.global.f32 %%f181, [%%rd12+%d];", breg, n * 32, n * 32 + 4);
        for (int r = 0; r < 2; ++r)
        {
            w("add.rn.f32 %%f182, %%f%d, %%f180; add.rn.f32 %%f183, %%f%d, %%f181;", f + 2 * r, f + 2 * r + 1);
            if (residual)
                w("mov.u32 %%r59, 0; @%%p%d ld.global.u32 %%r59, [%%rd%d+%d]; mov.b32 {%%h3, %%h4}, %%r59; cvt.f32.f16 %%f184, %%h3; cvt.f32.f16 %%f185, %%h4; add.rn.f32 %%f182, %%f182, %%f184; add.rn.f32 %%f183, %%f183, %%f185;",
                  18 + r, 38 + r, n * 16);
            if (relu) w("max.f32 %%f182, %%f182, 0f00000000; max.f32 %%f183, %%f183, 0f00000000;");
            Fp8(w, 182), Fp8(w, 183);
            if (dst >= 0) w("cvt.rn.satfinite.e4m3x2.f32 %%h%d, %%f183, %%f182;", dst + r);  // (high, low)
            else
                w("cvt.rn.f16.f32 %%h5, %%f182; cvt.rn.f16.f32 %%h6, %%f183; mov.b32 %%r72, {%%h5, %%h6}; @%%p%d st.global.u32 [%%rd%d+%d], %%r72;", 18 + r, 14 + r, n * 16);
        }
    };
    // --- pconv: chunk c: kpos < 16 tap a, >= 16 tap b (P = 16: taps 2c, 2c + 1; P = 32: tap c, channels 0-15 | 16-31).
    for (int i = 0; i < TP * 4; ++i) w("mov.f32 %%f%d, 0f00000000;", i);
    for (int c = 0; c < KCP; ++c)
    {
        const int ta = P == 16 ? 2 * c : c, tb = P == 16 ? 2 * c + 1 : c, cb = P == 16 ? 0 : 16;
        for (int r = 0; r < 2; ++r)
        {
            x8(ta, r, 0, 10), x8(ta, r, 8, 11);
            pack(50 + r, 10, 11);
            if (tb < 9) x8(tb, r, cb, 12), x8(tb, r, cb + 8, 13), pack(52 + r, 12, 13);
            else w("mov.u32 %%r%d, 0;", 52 + r);
        }
        for (int n = 0; n < TP; ++n) bload(3, c, n, TP), mma(n * 4);
    }
    for (int n = 0; n < TP; ++n) finish(n * 4, 6, n, false, false, 20 + 2 * n);  // pc tile n: h(20 + 2n) row a, (21 + 2n) row b
    // --- expand: chunk c: channels 32c + lc: < P from pc tiles, < C from x (centre tap 4), else 0.
    auto pair = [&](int ci, int r, int dst) {  // the e4m3 pair of channels ci, ci + 1 (ci = 8 tile + 2q structure) into h<dst>
        if (ci < P) w("mov.b16 %%h%d, %%h%d;", dst, 20 + 2 * (ci / 8) + r);
        else if (ci < C) x8(4, r, ci, dst);
        else w("mov.b16 %%h%d, 0;", dst);
    };
    for (int i = 0; i < TE * 4; ++i) w("mov.f32 %%f%d, 0f00000000;", i);
    for (int c = 0; c < KCE; ++c)
    {
        for (int r = 0; r < 2; ++r)
        {
            pair(32 * c, r, 10), pair(32 * c + 8, r, 11), pack(50 + r, 10, 11);
            pair(32 * c + 16, r, 12), pair(32 * c + 24, r, 13), pack(52 + r, 12, 13);
        }
        for (int n = 0; n < TE; ++n) bload(4, c, n, TE), mma(n * 4);
    }
    for (int n = 0; n < TE; ++n) finish(n * 4, 7, n, true, false, 40 + 2 * n);  // h tile n: h(40 + 2n), (41 + 2n)
    // --- project + x, stored.
    for (int i = 0; i < TJ * 4; ++i) w("mov.f32 %%f%d, 0f00000000;", i);
    for (int c = 0; c < KCJ; ++c)
    {
        for (int r = 0; r < 2; ++r)
        {
            pack(50 + r, 40 + 2 * (4 * c) + r, 40 + 2 * (4 * c + 1) + r);
            pack(52 + r, 40 + 2 * (4 * c + 2) + r, 40 + 2 * (4 * c + 3) + r);
        }
        for (int n = 0; n < TJ; ++n) bload(5, c, n, TJ), mma(n * 4);
    }
    w("sub.s64 %%rd13, %%rd2, %%rd1; add.s64 %%rd14, %%rd38, %%rd13; add.s64 %%rd15, %%rd39, %%rd13;");
    for (int n = 0; n < TJ; ++n) finish(n * 4, 8, n, false, true, -1);
    w("ret;");
    return w.s + "}\n";
}
// Split-K's second pass: one thread a (pixel, channel pair): the splits' partial sums added in split order (the same bytes
// every run and on both APIs), then the epilogue of the other kernels. SplitParams; grid x = pixels * pairs / 128.
std::string ReduceKernel()
{
    Ptx w;
    w.s = ".visible .entry reducek(.param .align 8 .b8 P[144])\n{\n";
    w(".reg .b32 %%r<200>; .reg .b64 %%rd<100>; .reg .f32 %%f<200>; .reg .pred %%p<60>; .reg .b16 %%h<8>;");
    for (int i = 0; i < 6; ++i) w("ld.param.u64 %%rd%d, [P+%d];", 1 + i, i * 8);
    for (int i = 0; i < 20; ++i) w("ld.param.u32 %%r%d, [P+%d];", 1 + i, 48 + i * 4);
    w("ld.param.u64 %%rd70, [P+128]; ld.param.u32 %%r110, [P+140];");  // partials, splits
    w("setp.eq.s32 %%p1, %%r15, 1; setp.eq.s32 %%p2, %%r15, 2; setp.ne.s32 %%p3, %%r16, 0; setp.ne.s32 %%p4, %%r17, 0;");
    w("mov.u32 %%r21, %%ctaid.x; mov.u32 %%r22, %%tid.x; mad.lo.u32 %%r21, %%r21, 128, %%r22;");  // item
    w("shr.u32 %%r23, %%r14, 1; mul.lo.u32 %%r24, %%r9, %%r10; mul.lo.u32 %%r25, %%r24, %%r23; setp.ge.u32 %%p5, %%r21, %%r25; @%%p5 ret;");
    w("div.u32 %%r26, %%r21, %%r23; rem.u32 %%r60, %%r21, %%r23; shl.b32 %%r60, %%r60, 1;");  // pixel, channel
    w("add.s32 %%r62, %%r60, 1; setp.lt.s32 %%p21, %%r60, %%r13; setp.lt.s32 %%p22, %%r62, %%r13; shl.b32 %%r27, %%r20, 3;");  // cpad
    w("mov.f32 %%f182, 0f00000000; mov.f32 %%f183, 0f00000000; mov.u32 %%r30, 0;");
    w("RED:");
    w("setp.ge.u32 %%p6, %%r30, %%r110; @%%p6 bra REDDONE;");
    w("mad.lo.u32 %%r31, %%r30, %%r24, %%r26; mad.lo.u32 %%r31, %%r31, %%r27, %%r60; mul.wide.u32 %%rd7, %%r31, 4; add.s64 %%rd7, %%rd70, %%rd7; ld.global.v2.f32 {%%f184, %%f185}, [%%rd7];");
    w("setp.eq.u32 %%p7, %%r30, 0; @%%p7 mov.f32 %%f182, %%f184; @%%p7 mov.f32 %%f183, %%f185; @!%%p7 add.rn.f32 %%f182, %%f182, %%f184; @!%%p7 add.rn.f32 %%f183, %%f183, %%f185;");
    w("add.u32 %%r30, %%r30, 1; bra RED;");
    w("REDDONE:");
    w("mul.wide.s32 %%rd64, %%r60, 4; add.s64 %%rd64, %%rd4, %%rd64; ld.global.f32 %%f180, [%%rd64]; ld.global.f32 %%f181, [%%rd64+4];");
    w("add.rn.f32 %%f182, %%f182, %%f180; add.rn.f32 %%f183, %%f183, %%f181;");
    w("mad.lo.s32 %%r66, %%r26, %%r17, %%r18; add.s32 %%r66, %%r66, %%r60; mul.wide.s32 %%rd65, %%r66, 2; add.s64 %%rd65, %%rd5, %%rd65;");
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
    w("mad.lo.s32 %%r73, %%r26, %%r11, %%r12; add.s32 %%r73, %%r73, %%r60; mul.wide.s32 %%rd66, %%r73, 2; add.s64 %%rd66, %%rd6, %%rd66; st.global.u32 [%%rd66], %%r72;");
    w("ret;");
    return w.s + "}\n";
}

}  // namespace

// kpos (0-31) of an fp8 k chunk -> its input channel within the chunk's 32 (see FnKernel8).
int Fp8Channel(int kp) { return (kp >= 16 ? 16 : 0) + 8 * ((kp % 4) / 2) + 2 * ((kp % 16) / 4) + (kp % 2); }

std::string CudaPtx(bool fp8)
{
    std::string p = fp8 ? ".version 8.4\n.target sm_89\n.address_size 64\n" : ".version 7.0\n.target sm_75\n.address_size 64\n";
    if (fp8) p += FnKernel8(48, 16), p += FnKernel8(64, 32);
    for (int s = 0; s < 4; ++s)
        for (int nb : kNbs)
            if (s < 3 || nb <= 4) p += Kernel(s, nb);
    for (int s = 0; s < 3; ++s) p += KernelM(s, 8, 2);
    p += Kernel(0, 8, true), p += ReduceKernel();
    p += FnKernel(48, 16, 2, 16, 8), p += FnKernel(64, 32, 8, 16, 8);  // the layers' padded n tiles (CudaNb)
    return p;
}

std::unique_ptr<CudaKernels> CudaKernels::Create(Device& dev, std::string* error)
{
    auto fail = [&](const std::string& m) -> std::unique_ptr<CudaKernels> { if (error) *error = m; return nullptr; };
    LUID luid;
    if (!dev.CudaAvailable() || !dev.CudaLuid(reinterpret_cast<uint8_t*>(&luid))) return fail("no CUDA launch on this device");
    const int cc = cuda::ComputeCapability(luid, error);
    if (!cc) return nullptr;
    // FP8 tensor cores from sm_89 (RTX 40): the module for that target with the e4m3 FasterNet kernels.
    const bool fp8 = cc >= 89 && !EnvFlag("NFRUX_MLFG_NO_FP8", false);
    const std::vector<uint8_t> cubin = cuda::Cubin(CudaPtx(fp8), luid, error);
    if (cubin.empty()) return nullptr;
    std::unique_ptr<CudaKernels> k(new CudaKernels(dev));
    if (!(k->m_module = dev.CudaModule(cubin))) return fail("CUDA module");
    auto fn = [&](void*& f, const std::string& name) { return (f = dev.CudaFunction(k->m_module, name.c_str())) != nullptr; };
    static const char* names[4] = {"conv3", "conv2", "conv1", "convT"};
    for (int s = 0; s < 4; ++s)
        for (int i = 0; i < 6; ++i)
            if ((s < 3 || kNbs[i] <= 4) && !fn(k->m_conv[s][i], std::string(names[s]) + "_" + std::to_string(kNbs[i]))) return fail("CUDA function");
    for (int s = 0; s < 3; ++s)
        if (!fn(k->m_convM[s], std::string(names[s]) + "_8m2")) return fail("CUDA function");
    if (!fn(k->m_splitK, "conv3_8k") || !fn(k->m_reduce, "reducek")) return fail("CUDA function");
    for (int i = 0; i < 2; ++i)
        if (!fn(k->m_fnk[i], i ? "fn64" : "fn48")) return fail("CUDA function");
    for (int i = 0; i < 2 && fp8; ++i)
        if (!fn(k->m_fn8[i], i ? "fn64f8" : "fn48f8")) return fail("CUDA function");
    return k;
}
CudaKernels::~CudaKernels()
{
    std::vector<void*> f(m_fnk, m_fnk + 2);
    f.insert(f.end(), m_fn8, m_fn8 + 2);
    f.insert(f.end(), m_convM, m_convM + 3);
    f.push_back(m_splitK), f.push_back(m_reduce);
    for (auto& s : m_conv) f.insert(f.end(), s, s + 6);
    m_dev.CudaDestroy(m_module, f);
}

bool CudaKernels::LaunchFn(const Buffer* const* bufs, uint32_t n, uint32_t C, bool fp8, const uint32_t grid[3], const FnParams& p) const
{
    return m_dev.CudaLaunch(fp8 ? m_fn8[C == 48 ? 0 : 1] : m_fnk[C == 48 ? 0 : 1], grid, 32 * kCudaWarps, &p, sizeof(p), bufs, n);
}

bool CudaKernels::LaunchM(const Buffer* const* bufs, uint32_t n, int shape, const uint32_t grid[3], const CudaParams& p) const
{
    return m_dev.CudaLaunch(m_convM[shape], grid, 32 * kCudaWarps, &p, sizeof(p), bufs, n);
}

bool CudaKernels::Launch(const Buffer* const* bufs, uint32_t n, int shape, uint32_t nb, const uint32_t grid[3], const CudaParams& p) const
{
    int i = 0;
    while (i < 5 && kNbs[i] != int(nb)) ++i;
    return kNbs[i] == int(nb) && m_dev.CudaLaunch(m_conv[shape][i], grid, 32 * kCudaWarps, &p, sizeof(p), bufs, n);
}

bool CudaKernels::LaunchSplit(const Buffer* const* bufs, uint32_t n, const uint32_t grid[3], const SplitParams& p, uint32_t pixels) const
{
    if (!m_dev.CudaLaunch(m_splitK, grid, 32 * kCudaWarps, &p, sizeof(p), bufs, n)) return false;
    const uint32_t rg[3] = {(pixels * uint32_t(p.c.writeEnd / 2) + 127) / 128, 1, 1};
    return m_dev.CudaLaunch(m_reduce, rg, 128, &p, sizeof(p), bufs, n);
}
}  // namespace nfrux::mlfg
