// Host side of NFRUX's port of AMD FSR Frame Generation 4's hand-written passes (core/shaders/mlfg/*.hlsl; AMD
// FidelityFX SDK v2.4.0, MIT). Constant layouts (every member a 16-byte vector, so std140 and HLSL packing agree), the
// timestep rules and the tensor table. Not used by the shipped interpolator yet.
//
// Per-frame order (AMD's, with t the generated frame's time, 0 = previous frame, 1 = current frame):
//   kMlfgDilate        mv / depth (render res) -> dilated mv (uv) + depth at outSize (AMD: 1/3 of the display)
//   kMlfgSplatKey+Flow (dilated mv, depth, proj 1)               -> splat1to0  (keys and flow words cleared to 0 first)
//   kMlfgForwardMv     (last frame's dilated mv / depth, splat1to0, matrices) -> forward mv of frame 0
//   kMlfgSplatKey+Flow (forward mv, last depth, proj t, sign -1) -> splat0toT
//   kMlfgSplatKey+Flow (dilated mv, depth, proj 1 - t, sign 1)   -> splat1toT
//   kMlfgNetInput -> network (encoder) -> kMlfgDownsample -> network -> kMlfgRefine(54) -> network -> kMlfgRefine(83)
//   -> network -> kMlfgRefine(112) -> network -> kMlfgFlowOut -> kMlfgStage2Input -> network (stage 2, up to the 3x3
//   conv of pass 188) -> kMlfgSoftmax5 -> kMlfgComposite*
// The dilated mv / depth must be kept for the next frame (kMlfgForwardMv reads them as frame 0's).
//
// Tensors (NHWC fp16, one buffer each, half index (y * width + x) * pitch + c; channels >= C written 0):
//   name     pitch  C   size (AMD 4K network, 1080p)   writer -> readers
//   net_in   32     18  tensorSizes[0]  480x288        NetInput (0-5), network (6-17) -> Downsample, Refine, network
//                       (windows 0-15 and 2-17: contiguous in a texel)
//   in_q     32     18  tensorSizes[10] 120x72         Downsample -> network
//   f54      16     13  tensorSizes[23] 120x72         network -> Refine(54)
//   x54      32     32  tensorSizes[24] 240x144        Refine(54) -> network
//   flow54    4      4  2 * tensorSizes[24]            Refine(54) -> Refine(83)
//   f83      16     13  tensorSizes[38] 240x144        network -> Refine(83)
//   x83      32     32  tensorSizes[39] 480x288        Refine(83) -> network
//   flow83    4      4  tensorSizes[39]                Refine(83) -> Refine(112)
//   f112     16     13  tensorSizes[54] 480x288        network -> Refine(112)
//   x112     32     32  tensorSizes[54]                Refine(112) -> network
//   flow112   4      4  tensorSizes[54]                Refine(112) -> FlowOut
//   f149     16     13  tensorSizes[69] 480x288        network -> FlowOut (ch 0-3)
//   flowout   4      4  tensorSizes[69]                FlowOut (kept for inspection)
//   s2in     32     24  tensorSizes[72] live, buffer 960x540   Stage2Input -> network
//   logits   16      5  tensorSizes[90] 960x540        network (pass 188's conv) -> Softmax5
//   weights   8      5  tensorSizes[90]                Softmax5 -> Composite
// Images: dilated mv (RG16F), depth (R32F), splat key + flow word (2x R32_UINT each), forward mv (RG16F),
// flow0toT / flow1toT (RG16F, tensorSizes[70] = display / 8).
// AMD's own layout differs in one way the runtime must reproduce if it matters: AMD stores 18- and 24-channel
// tensors with an 18 / 24 channel stride and its convolutions read whole 32-channel windows, so channels 18-31
// (24-31) of a texel read the next texel's first channels; here they are 0. If the corresponding weights are not
// zero the runtime has to feed what AMD feeds.
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>

namespace nfrux::mlfg
{
// The fraction of its motion a splat pushes source pixels along (kMlfgSplat*, c_proj.x), for a frame at t.
// AMD: 1 / 0.5 / 0.5 (t = 0.5).
enum class Splat { OneToZero, ZeroToT, OneToT };
inline float SplatProjAmount(Splat s, float t) { return s == Splat::OneToZero ? 1.0f : s == Splat::ZeroToT ? t : 1.0f - t; }
inline float SplatSign(Splat s) { return s == Splat::ZeroToT ? -1.0f : 1.0f; }  // AMD flowIsUv: the forward mv is uv0 - uv1

struct DilateConstants { int32_t sizes[4]; float mvScale[4]; float genRect[4]; float colorRect[4]; };
struct SplatConstants { int32_t sizes[4]; float proj[4]; float genRect[4]; };
struct ForwardMvConstants { float prevPstoPs[16]; float prevPsToPrev2Ps[16]; int32_t sizes[4]; int32_t depthSize[4]; };
struct NetInputConstants { int32_t tensor[4]; int32_t tex[4]; float genRect[4]; float lum[4]; };
struct DownsampleConstants { int32_t src[4]; int32_t dst[4]; float scale[4]; };
struct RefineConstants { int32_t mode[4]; int32_t net[4]; int32_t head[4]; int32_t prev[4]; int32_t out[4]; int32_t flow[4]; float params[4]; };
struct FlowOutConstants { int32_t head[4]; int32_t out[4]; int32_t image[4]; };
struct Stage2InputConstants { int32_t out[4]; int32_t colour[4]; int32_t splat[4]; float genRect[4]; float lum[4]; };
struct Softmax5Constants { int32_t size[4]; };
struct CompositeConstants { int32_t colour[4]; int32_t splat[4]; int32_t weights[4]; int32_t misc[4]; float genRect[4]; float lum[4]; float debug[4]; };
static_assert(sizeof(DilateConstants) == 64 && sizeof(ForwardMvConstants) == 160 && sizeof(RefineConstants) == 112 &&
                  sizeof(CompositeConstants) == 112, "16-byte rows");
}  // namespace nfrux::mlfg
