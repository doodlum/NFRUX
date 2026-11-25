// SPDX-License-Identifier: MIT
#include "interpolator.h"

#include "log.h"
#include "nfrux_shaders.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace nfrux
{
namespace
{
constexpr uint32_t kMvDepthLaneMaxHeight = 540, kGroup = 8, kFiGroup = 16;
struct FiConstants
{
    float    _MotionTM1ToTP1[16];
    float    _MotionTP1ToTM1[16];
    float    _QuantParamsSINT[4];
    float    _QuantParamsSNORM[4];
    float    _DeviceToViewDepth[4];
    int32_t  _RenderSize[2];
    int32_t  _DisplaySize[2];
    float    _DisplaySizeRcp[2];
    float    _Jitter[2];
    float    _MotionVectorScale[2];
    float    _JitterCancellation[2];
    float    _MvSimilarityThresholds[3];
    float    _MvSimilarityThresholdsPadding;
    int32_t  _InputTensorSize[2];
    int32_t  _Reset;
    float    _Timestep;
    uint32_t _RandomSeed;
    uint32_t _UseReducedMotionInput;
    int32_t  _MvDepthLaneSize[2];
    float    _MvDepthLaneSizeRcp[2];
    int32_t  _FlowLaneSize[2];
    float    _FlowLaneSizeRcp[2];
    float    _ParamsLaneSizeRcp[2];
    float    _QuantParamsGateSNORM[4];
    float    _OfInputSizeRcp[2];
    float    _DepthClipScale;
    float    _DepthClipPower;
    uint32_t _Options;
};
static_assert(offsetof(FiConstants, _MvSimilarityThresholds) == 224 && offsetof(FiConstants, _InputTensorSize) == 240 &&
                  offsetof(FiConstants, _QuantParamsGateSNORM) == 304 && offsetof(FiConstants, _Options) == 336,
              "constants must match the std140 block");

// ffxNfruRoundPositive (ffx_frameinterpolation_resolution.h): to nearest, ties to even.
uint32_t RoundPositive(double v)
{
    const double whole = floor(v), f = v - whole;
    return uint32_t(whole + ((f > 0.5 || (f == 0.5 && (uint64_t(whole) & 1u))) ? 1.0 : 0.0));
}

// ffx_opticalflow_private.h
struct MvHintsConstants
{
    float   motion_matrix_m1p1[16];
    int32_t output_dims[2];
    int32_t depth_size[2];
    float   depth_size_rcp[2];
    int32_t color_size[2];
    int32_t mv_depth_dims[2];  // SDK 1.1.1: the motion lane, whose pixels the warp (pass 10) represents each hint by
    float   inv_mv_depth_dims[2];
};
struct ImageConstants
{
    int32_t  level_dims[2];
    int32_t  src_dims[2];
    int32_t  cell_dims[2];
    int32_t  coarse_dims[2];
    int32_t  hint_dims[2];
    float    out_scale[2];
    int32_t  level_row;
    int32_t  flow_row;
    int32_t  coarse_row;
    int32_t  level;
    uint32_t lambda_hint;
    uint32_t lambda_seed;
    int32_t  level_count;
    uint32_t settled_cost;
    int32_t  final;
    int32_t  fsr3;
    int32_t  unorm8;
    int32_t  pad;
    int32_t  levels[5][4];
};

struct Dims
{
    uint32_t width = 0, height = 0;
};

// Column-major 4x4 (m[col * 4 + row]), as the SDK's matrixInvert / matrixMul (MESA GLU's inverse).
void Invert(const float* m, float* out)
{
    float inv[16];
    inv[0]  = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8]  = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5]  = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9]  = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2]  = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6]  = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3]  = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7]  = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det == 0)
    {
        memset(out, 0, 16 * sizeof(float));
        return;
    }
    det = 1.0f / det;
    for (int i = 0; i < 16; i++)
        out[i] = inv[i] * det;
}

void Multiply(const float* a, const float* b, float* out)
{
    for (uint32_t col = 0; col < 4; ++col)
        for (uint32_t row = 0; row < 4; ++row)
        {
            float sum = 0.0f;
            for (uint32_t k = 0; k < 4; ++k)
                sum += a[k * 4 + row] * b[col * 4 + k];
            out[col * 4 + row] = sum;
        }
}


uint32_t Groups(uint32_t n, uint32_t g = kGroup) { return (n + g - 1) / g; }
using Tex = std::unique_ptr<Texture>;
using PassPtr = std::unique_ptr<Pass>;
}  // namespace

struct Out
{
    std::unique_ptr<Network> net;
    Tex qMotionTp1, qFlowTm1, filledFlowTm1, level[5], inpaint, seed, output, dynScratch, depthScratch;
};

struct Interpolator::Impl
{
    Device&  dev;
    Settings s;
    Dims     display, render, ofInput, of, mvLane, flowLane, graph;
    bool     allInput = false, colourDown = false, bytes = false;
    FiConstants c = {};
    float    lastVP[16] = {};
    PassPtr  initWarp, downOf, downAll, downColour, warp, warpTm1, fillTm1, preprocess, inpaintSeed, inpaintMip, inpaint, post;
    PassPtr  mvHints, pyramidP, flowWarp, match, median, bilateral;
    Tex      dynMask[2], depthTm1[2], colorTm1[2], prep[2], reducedMotion, ofColour;
    Tex      computedMv, flow, flowHint, pyramid[2], fWarp, fMatch, fSub, fMed, fFlow;
    bool     pyramidValid = false;
    uint32_t dynCur = 0, depthCur = 0, prepCur = 0, pyrCur = 0;
    uint64_t frame = 0;
    Dims     level[6];
    int32_t  row[6] = {};
    Out      outs[kMaxGenerated];
    uint32_t made = 0;

    Impl(Device& d) : dev(d) {}
    bool MakePass(PassPtr& p, const Shader& a, const Shader& b) { p = dev.CreatePass(s.depthInverted ? b : a); return p != nullptr; }
    bool Create(std::string* error);
    bool MakeOuts(uint32_t n, std::string* error);
    bool Flow(const FrameInputs& in, const Texture& color, const Texture& depth, const Texture& mv, bool reset);
    bool Frame(const FrameInputs& in, uint32_t count, uint32_t total);
};

#define NFRUX_PASS(var, name) if (!MakePass(var, shaders::name, shaders::name##Inv)) return fail("pass " #name);

bool Interpolator::Impl::Create(std::string* error)
{
    auto fail = [&](const char* m) { if (error) *error = m; return false; };
    display = {s.displayWidth, s.displayHeight};
    render  = {s.renderWidth, s.renderHeight};
    if (!display.width || !display.height || !render.width || !render.height || display.width > 3840 || display.height > 2160)
        return fail("unsupported size");
    const double aspect = double(display.width) / double(display.height);
    ofInput = display.height > 1080 ? Dims{RoundPositive(1080 * aspect), 1080} : display;  // the SDK's resolution plan
    of      = {(ofInput.width + 3) / 4, (ofInput.height + 3) / 4};
    const bool   fixed  = render.height >= kMvDepthLaneMaxHeight;
    mvLane              = fixed ? Dims{RoundPositive(kMvDepthLaneMaxHeight * aspect), kMvDepthLaneMaxHeight} : render;
    const float depthPerLane = float(render.height) / float(kMvDepthLaneMaxHeight), colourScale = float(display.height) / float(mvLane.height);
    allInput   = fixed && depthPerLane >= 1.5f;
    colourDown = depthPerLane < 1.5f && colourScale >= 1.5f;
    const uint32_t flowRows = std::max(ofInput.height / 4u, 1u);
    flowLane = {RoundPositive(double(flowRows) * aspect), flowRows};
    graph    = {(mvLane.width + 7) / 8 * 8, (mvLane.height + 7) / 8 * 8};
    bytes    = s.outputFormat == Format::RGBA8Unorm || s.outputFormat == Format::BGRA8Unorm || s.outputFormat == Format::RGB10A2Unorm;

    NFRUX_PASS(initWarp, kFiInitWarp)
    Format lane = s.outputFormat == Format::BGRA8Unorm ? Format::RGBA8Unorm : s.outputFormat;
#define NFRUX_LANE(suffix) { NFRUX_PASS(downOf, kFiDownsampleOf##suffix) NFRUX_PASS(downAll, kFiDownsampleAll##suffix) NFRUX_PASS(downColour, kFiDownsampleColour##suffix) }
    switch (lane)
    {
    case Format::RGBA8Unorm: NFRUX_LANE(Rgba8) break;
    case Format::RGB10A2Unorm: NFRUX_LANE(Rgb10a2) break;
    case Format::RGBA16Float: NFRUX_LANE(Rgba16f) break;
    case Format::RG11B10Float: NFRUX_LANE(R11) break;
    default: return fail("unsupported output format");
    }
    NFRUX_PASS(warp, kFiWarpFlow) NFRUX_PASS(warpTm1, kFiWarpFlowTm1) NFRUX_PASS(fillTm1, kFiFillHolesTm1) NFRUX_PASS(preprocess, kFiPreprocess)
    NFRUX_PASS(inpaintSeed, kFiInpaintSeed) NFRUX_PASS(inpaintMip, kFiInpaintMip) NFRUX_PASS(inpaint, kFiInpaint)
    switch (s.outputFormat)
    {
    case Format::RGBA8Unorm: case Format::BGRA8Unorm: NFRUX_PASS(post, kFiPostprocessNoHistoryRgba8) break;
    case Format::RGB10A2Unorm: NFRUX_PASS(post, kFiPostprocessNoHistoryRgb10a2) break;
    case Format::RGBA16Float: NFRUX_PASS(post, kFiPostprocessNoHistoryRgba16f) break;
    case Format::RG11B10Float: NFRUX_PASS(post, kFiPostprocessNoHistoryR11) break;
    default: return fail("unsupported output format");
    }
    NFRUX_PASS(mvHints, kOfMvHints) NFRUX_PASS(pyramidP, kOfArmPyramid) NFRUX_PASS(flowWarp, kOfArmWarp)
    NFRUX_PASS(match, kOfArmMatchShared) NFRUX_PASS(median, kOfArmMedian) NFRUX_PASS(bilateral, kOfArmBilateralShared)
    level[0] = ofInput;
    int32_t rows = 0;
    for (uint32_t l = 1; l < 6; ++l)
    {
        level[l] = {((level[l - 1].width + 1) / 2 + 1) & ~1u, ((level[l - 1].height + 1) / 2 + 1) & ~1u};
        row[l]   = rows;
        rows += int32_t(level[l].height);
    }
    bool ok  = true;
    auto tex = [&](Tex& t, Format f, Dims d, bool uint = false) { t = dev.CreateTexture(f, d.width, d.height, uint); ok = ok && t; };
    for (auto& t : dynMask) tex(t, Format::R8Unorm, mvLane);
    for (auto& t : depthTm1) tex(t, Format::R32Float, mvLane);
    for (auto& t : colorTm1) ok = ok && (t = dev.CreateTextureLike(s.colorLike, display.width, display.height));
    if (allInput || colourDown)
        for (auto& t : prep) tex(t, lane, mvLane);
    if (display.height > 1080) tex(ofColour, lane, ofInput);
    if (allInput) tex(reducedMotion, Format::RG16Float, mvLane);
    tex(computedMv, Format::RG16Float, of), tex(flow, Format::RG16Float, of), tex(flowHint, Format::RG16Float, of);
    for (auto& t : pyramid) tex(t, Format::R8Unorm, {level[1].width, uint32_t(rows)});
    tex(fWarp, Format::R8Unorm, level[2]), tex(fMatch, Format::R32Uint, level[2]);
    tex(fSub, Format::RG16Float, level[2]), tex(fMed, Format::RG16Float, level[2]), tex(fFlow, Format::RG16Float, level[2]);
    if (!ok)
        return fail("out of memory");

    FiConstants& k = c;
    k._RenderSize[0] = int32_t(render.width), k._RenderSize[1] = int32_t(render.height);
    k._DisplaySize[0] = int32_t(display.width), k._DisplaySize[1] = int32_t(display.height);
    k._DisplaySizeRcp[0] = 1.0f / display.width, k._DisplaySizeRcp[1] = 1.0f / display.height;
    k._MvDepthLaneSize[0] = int32_t(mvLane.width), k._MvDepthLaneSize[1] = int32_t(mvLane.height);
    k._MvDepthLaneSizeRcp[0] = 1.0f / mvLane.width, k._MvDepthLaneSizeRcp[1] = 1.0f / mvLane.height;
    k._FlowLaneSize[0] = int32_t(flowLane.width), k._FlowLaneSize[1] = int32_t(flowLane.height);
    k._FlowLaneSizeRcp[0] = 1.0f / flowLane.width, k._FlowLaneSizeRcp[1] = 1.0f / flowLane.height;
    k._OfInputSizeRcp[0] = 1.0f / ofInput.width, k._OfInputSizeRcp[1] = 1.0f / ofInput.height;
    k._InputTensorSize[0] = int32_t(graph.width), k._InputTensorSize[1] = int32_t(graph.height);
    k._ParamsLaneSizeRcp[0] = 2.0f / graph.width, k._ParamsLaneSizeRcp[1] = 2.0f / graph.height;
    // The model's quantisation (nfru_mfg_v14_int8_metadata.json).
    k._QuantParamsSINT[0] = float(1 / 0.003921568859368563), k._QuantParamsSINT[1] = -128;
    k._QuantParamsSINT[2] = float(0.062118448317050934), k._QuantParamsSINT[3] = 54;
    k._QuantParamsSNORM[0] = float(1 / 0.49803924513980746), k._QuantParamsSNORM[1] = float(-1.0078740157480315);
    k._QuantParamsSNORM[2] = float(7.889042936265469), k._QuantParamsSNORM[3] = float(0.4251968503937008);
    k._QuantParamsGateSNORM[0] = float(1 / 0.49803924513980746), k._QuantParamsGateSNORM[1] = float(-1.0078740157480315);
    k._QuantParamsGateSNORM[2] = float(0.4980375892482698), k._QuantParamsGateSNORM[3] = float(-1.0078740157480315);
    k._MvSimilarityThresholds[0] = 0.25f, k._MvSimilarityThresholds[1] = 0.125f, k._MvSimilarityThresholds[2] = 0.002f;
    k._RandomSeed            = 2357136044u;
    k._UseReducedMotionInput = allInput ? 1u : 0u;
    memcpy(lastVP, s.initialViewProjection, sizeof(lastVP));
    return MakeOuts(kMaxGenerated, error);
}

bool Interpolator::Impl::MakeOuts(uint32_t n, std::string* error)
{
    for (; made < std::min(n, kMaxGenerated); ++made)
    {
        Out& o   = outs[made];
        bool ok  = true;
        auto tex = [&](Tex& t, Format f, Dims d, bool uint = false) { t = dev.CreateTexture(f, d.width, d.height, uint); ok = ok && t; };
        o.net = dev.CreateNetwork(graph.width, graph.height, error);
        tex(o.qMotionTp1, Format::R32Uint, mvLane), tex(o.qFlowTm1, Format::R32Uint, flowLane), tex(o.filledFlowTm1, Format::RG16Float, flowLane);
        tex(o.inpaint, Format::RGBA8Unorm, flowLane), tex(o.seed, Format::RGBA8Unorm, mvLane);
        for (uint32_t l = 0; l < 5; ++l)
            tex(o.level[l], Format::RGBA8Unorm, {std::max(1u, flowLane.width >> l), std::max(1u, flowLane.height >> l)});
        tex(o.output, s.outputFormat, display);
        if (made) tex(o.dynScratch, Format::R8Unorm, mvLane), tex(o.depthScratch, Format::R32Float, mvLane);
        if (!ok || !o.net)
            return false;
    }
    return true;
}

bool Interpolator::Impl::Flow(const FrameInputs& in, const Texture& color, const Texture& depth, const Texture& mv, bool reset)
{
    const Texture& depthPrev = *depthTm1[depthCur];
    MvHintsConstants h = {};
    h.output_dims[0] = int32_t(of.width), h.output_dims[1] = int32_t(of.height);
    h.depth_size[0] = int32_t(depthPrev.width), h.depth_size[1] = int32_t(depthPrev.height);
    h.depth_size_rcp[0] = 1.0f / depthPrev.width, h.depth_size_rcp[1] = 1.0f / depthPrev.height;
    h.color_size[0] = int32_t(ofInput.width), h.color_size[1] = int32_t(ofInput.height);
    h.mv_depth_dims[0] = int32_t(mvLane.width), h.mv_depth_dims[1] = int32_t(mvLane.height);
    h.inv_mv_depth_dims[0] = 1.0f / mvLane.width, h.inv_mv_depth_dims[1] = 1.0f / mvLane.height;
    float invPrev[16];
    Invert(lastVP, invPrev);
    Multiply(in.viewProjection, invPrev, h.motion_matrix_m1p1);
    {
        const Bind b[] = {Bind::Data(0, &h, sizeof(h)), Bind::Tex(1, depthPrev), Bind::Tex(2, *computedMv)};
        if (!mvHints->Dispatch(b, 3, Groups(of.width, 16), Groups(of.height, 16))) return false;
    }
    dev.CopyTexture(*computedMv, *flow);
    const Texture &p1 = *pyramid[pyrCur], &m1 = *pyramid[pyrCur ^ 1];
    ImageConstants ic = {};
    ic.src_dims[0] = int32_t(ofInput.width), ic.src_dims[1] = int32_t(ofInput.height);
    ic.hint_dims[0] = int32_t(of.width), ic.hint_dims[1] = int32_t(of.height);
    ic.unorm8 = s.outputFormat == Format::RGBA8Unorm || s.outputFormat == Format::BGRA8Unorm;
    for (int32_t l = 1; l < 6; ++l)
    {
        const Dims d = level[l], p = level[l - 1];
        ic.level = l, ic.level_row = row[l], ic.coarse_row = row[l - 1];
        ic.level_dims[0] = int32_t(d.width), ic.level_dims[1] = int32_t(d.height);
        ic.coarse_dims[0] = int32_t(p.width), ic.coarse_dims[1] = int32_t(p.height);
        const Bind b[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, color), Bind::Tex(2, p1)};
        if (!pyramidP->Dispatch(b, 3, Groups(d.width), Groups(d.height))) return false;
    }
    if (pyramidValid && !reset)
    {
        dev.CopyTexture(*flow, *flowHint);
        for (int32_t l = 5; l >= 2; --l)
        {
            const Dims d = level[l], cc = l == 5 ? Dims{0, 0} : level[l + 1];
            ic.level = l, ic.level_row = row[l], ic.coarse_row = 0, ic.final = l == 2;
            ic.level_dims[0] = int32_t(d.width), ic.level_dims[1] = int32_t(d.height);
            ic.coarse_dims[0] = int32_t(cc.width), ic.coarse_dims[1] = int32_t(cc.height);
            const uint32_t gx = Groups(d.width), gy = Groups(d.height);
            const Bind w[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, p1), Bind::Tex(2, *fFlow), Bind::Tex(3, *fWarp)};
            const Bind mb[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, m1), Bind::Tex(2, *fWarp), Bind::Tex(3, *fFlow), Bind::Tex(4, *fSub), Bind::Tex(5, *fMatch)};
            const Bind md[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, *fSub), Bind::Tex(2, *fMed)};
            const Bind bl[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, m1), Bind::Tex(2, p1), Bind::Tex(3, *fMed), Bind::Tex(4, *fMatch),
                               Bind::Tex(5, *flowHint), Bind::Tex(6, *fFlow), Bind::Tex(7, *flow)};
            if (!flowWarp->Dispatch(w, 4, gx, gy) || !match->Dispatch(mb, 6, gx, gy) || !median->Dispatch(md, 3, gx, gy) || !bilateral->Dispatch(bl, 8, gx, gy))
                return false;
        }
    }
    pyramidValid = true;
    pyrCur ^= 1;
    return true;
}

bool Interpolator::Impl::Frame(const FrameInputs& in, uint32_t count, uint32_t total)
{
    if (!MakeOuts(count, nullptr))
        return false;
    const bool needsAlpha = in.uiKind >= 2 && in.uiKind <= 5;
    const Texture *pc = in.colorTexture ? in.colorTexture : dev.Wrap(in.color, false), *pd = in.depthTexture ? in.depthTexture : dev.Wrap(in.depth, false), *pm = in.motionTexture ? in.motionTexture : dev.Wrap(in.motion, false);
    const Texture *ps = in.uiSource ? dev.Wrap(in.uiSource, false) : pc, *pa = needsAlpha && in.uiAlpha ? dev.Wrap(in.uiAlpha, false) : ps;
    const Texture *pn = in.noWarp ? dev.Wrap(in.noWarp, false) : ps;
    if (!pc || !pd || !pm || !ps || !pa || !pn)
        return false;
    const Texture &color = *pc, &depth = *pd, &mv = *pm, &uiSource = *ps, &uiAlpha = *pa, &noWarp = *pn;
    const bool     reset   = in.reset || frame == 0;
    const Texture &prev = *colorTm1[frame % 2], &backup = *colorTm1[(frame + 1) % 2];
    ++frame;

    FiConstants& k = c;
    float inv[16];
    Invert(lastVP, inv), Multiply(in.viewProjection, inv, k._MotionTM1ToTP1);
    Invert(in.viewProjection, inv), Multiply(lastVP, inv, k._MotionTP1ToTM1);
    k._Jitter[0] = in.jitter[0], k._Jitter[1] = in.jitter[1];
    k._MotionVectorScale[0] = in.mvScale[0] / float(display.width), k._MotionVectorScale[1] = in.mvScale[1] / float(display.height);
    {
        float fMin = std::min(in.cameraNear, in.cameraFar), fMax = std::max(in.cameraNear, in.cameraFar);
        if (s.depthInverted) std::swap(fMin, fMax);
        const float q = fMax / (fMin - fMax);
        k._DeviceToViewDepth[0] = -1.0f * q, k._DeviceToViewDepth[1] = q * fMin;
        if (in.orthographic) k._DeviceToViewDepth[0] = fMin, k._DeviceToViewDepth[1] = fMax - fMin;
        const float w = float(display.width), h = float(display.height), cot = cosf(0.5f * in.fovY) / sinf(0.5f * in.fovY);
        k._DeviceToViewDepth[2] = 1.0f / (cot / (w / h)), k._DeviceToViewDepth[3] = 1.0f / cot;
        const float diagonal = sqrtf(w * w + h * h), reference = sqrtf(1080.0f * 1080.0f + 1920.0f * 1920.0f);
        const float cx = float(uint32_t(w * 0.5f)), cy = float(uint32_t(h * 0.5f)), ndcX = cy / h * 2.0f - 1.0f, ndcY = cx / w * -2.0f + 1.0f;
        const float px = k._DeviceToViewDepth[2], py = k._DeviceToViewDepth[3];
        const float corner = sqrtf(px * px + py * py + 1.0f), center = sqrtf(px * px * ndcX * ndcX + py * py * ndcY * ndcY + 1.0f);
        k._DepthClipScale = 1.37e-05f * (corner / center) * diagonal;
        k._DepthClipPower = 1.0f + 2.0f * std::min(diagonal / reference, 1.0f);
    }
    k._Reset = reset ? 1 : 0;
    const bool down = allInput || colourDown;
    const uint32_t options = (!in.uiSource || (needsAlpha && !in.uiAlpha) ? 0u : in.uiKind) | (in.noWarp ? 0x100u : 0u) | (in.orthographic ? 0x200u : 0u) | (s.swapRB ? 0x1000u : 0u) |
                             (in.scrgb ? 0x2000u : 0u);

    if (ofColour)
    {
        FiConstants o = k;  // its own: RenderSize is the output's there
        o._RenderSize[0] = int32_t(ofInput.width), o._RenderSize[1] = int32_t(ofInput.height);
        const Bind b[] = {Bind::Tex(0, color), Bind::Tex(1, *ofColour), Bind::Data(2, &o, sizeof(o))};
        if (!downOf->Dispatch(b, 3, Groups(ofInput.width, kFiGroup), Groups(ofInput.height, kFiGroup))) return false;
    }
    const Texture& flowColour = ofColour ? *ofColour : color;
    if (!Flow(in, flowColour, depth, mv, reset))
        return false;
    memcpy(lastVP, in.viewProjection, sizeof(lastVP));

    const Texture &dynTm1 = *dynMask[dynCur ^ 1], &dynTp1 = *dynMask[dynCur], &dTm1 = *depthTm1[depthCur], &dNext = *depthTm1[depthCur ^ 1];
    const Texture &prepP1 = down ? *prep[prepCur] : color, &prepM1 = down ? *prep[prepCur ^ 1] : prev;
    const Texture &warpDepth = allInput ? dNext : depth, &warpMotion = allInput ? *reducedMotion : mv;
    const uint32_t lx = Groups(mvLane.width, kFiGroup), ly = Groups(mvLane.height, kFiGroup);
    const uint32_t gx = Groups(std::max(mvLane.width, flowLane.width), kFiGroup), gy = Groups(std::max(mvLane.height, flowLane.height), kFiGroup);
    if (reset)
        count = 1, total = 1;
    auto set = [&](uint32_t i) { k._Timestep = float(i + 1) / float(total + 1), k._Options = options | (i ? 0x400u : 0u); };
    const Bind downB[] = {Bind::Tex(0, flowColour), Bind::Tex(1, depth), Bind::Tex(2, mv), Bind::Tex(3, down ? prepP1 : color), Bind::Tex(4, dNext),
                          Bind::Tex(5, allInput ? *reducedMotion : dNext), Bind::Data(6, &k, sizeof(k))};
    auto warpRun = [&](uint32_t i) {
        const Out& o = outs[i];
        const Bind b[] = {Bind::Tex(0, warpMotion), Bind::Tex(1, warpDepth), Bind::Tex(2, dTm1), Bind::Tex(3, dynTm1), Bind::Tex(4, i ? *o.dynScratch : dynTp1),
                          Bind::Tex(5, *o.qMotionTp1), Bind::Tex(8, i ? *o.depthScratch : dNext), Bind::Data(9, &k, sizeof(k))};
        return warp->Dispatch(b, 8, lx, ly);
    };
    for (uint32_t i = 0; i < count; ++i)
    {
        set(i);
        const Bind b[] = {Bind::Tex(0, *outs[i].qMotionTp1), Bind::Tex(1, *outs[i].qFlowTm1), Bind::Data(4, &k, sizeof(k))};
        if (!initWarp->Dispatch(b, 3, gx, gy)) return false;
    }
    set(0);
    if (allInput && !downAll->Dispatch(downB, 7, lx, ly)) return false;
    if (colourDown && !downColour->Dispatch(downB, 7, lx, ly)) return false;
    if (reset)
    {
        if (!warpRun(0)) return false;
        dev.Clear(dynTp1, 1.0f);
        dev.CopyTexture(color, backup);
        return true;
    }
    for (uint32_t i = 0; i < count; ++i) { set(i); if (!warpRun(i)) return false; }
    for (uint32_t i = 0; i < count; ++i)
    {
        set(i);
        const Out& o = outs[i];
        const Bind b[] = {Bind::Tex(0, *flow), Bind::Tex(1, dTm1), Bind::Tex(2, *o.qFlowTm1), Bind::Data(3, &k, sizeof(k))};
        if (!warpTm1->Dispatch(b, 4, Groups(flowLane.width, 32), Groups(flowLane.height, 4))) return false;
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        set(i);
        const Out& o = outs[i];
        const Bind b[] = {Bind::Tex(0, *o.qFlowTm1), Bind::Tex(1, *o.filledFlowTm1), Bind::Data(2, &k, sizeof(k))};
        if (!fillTm1->Dispatch(b, 3, Groups(flowLane.width, kFiGroup), Groups(flowLane.height, kFiGroup))) return false;
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        set(i);
        const Out& o = outs[i];
        const Bind b[] = {Bind::Tex(0, *o.qMotionTp1), Bind::Tex(1, *o.filledFlowTm1), Bind::Tex(2, prepM1), Bind::Tex(3, prepP1), Bind::Tex(4, dNext), Bind::Tex(5, dTm1),
                          Bind::Buf(8, *o.net->input), Bind::Tex(10, *o.seed), Bind::Data(9, &k, sizeof(k))};
        if (!preprocess->Dispatch(b, 9, Groups(graph.width, 32), Groups(graph.height, 16))) return false;
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        set(i);
        const Out& o = outs[i];
        const Bind b[] = {Bind::Tex(0, *o.seed), Bind::Tex(1, *o.level[0]), Bind::Data(2, &k, sizeof(k))};
        if (!inpaintSeed->Dispatch(b, 3, Groups(flowLane.width, 16), Groups(flowLane.height, 8))) return false;
    }
    for (uint32_t l = 1; l < 5; ++l)
        for (uint32_t i = 0; i < count; ++i)
        {
            set(i);
            const Out& o = outs[i];
            const Bind b[] = {Bind::Tex(0, *o.level[l - 1]), Bind::Tex(1, *o.level[l]), Bind::Data(2, &k, sizeof(k))};
            if (!inpaintMip->Dispatch(b, 3, Groups(o.level[l]->width, 16), Groups(o.level[l]->height, 8))) return false;
        }
    Network* nets[kMaxGenerated];
    for (uint32_t i = 0; i < count; ++i) nets[i] = outs[i].net.get();
    dev.RunNetworks(nets, count);
    dev.Mark("network");
    for (uint32_t i = 0; i < count; ++i)
    {
        set(i);
        const Out& o = outs[i];
        const Bind b[] = {Bind::Tex(0, *o.level[0]), Bind::Tex(1, *o.level[1]), Bind::Tex(2, *o.level[2]), Bind::Tex(3, *o.level[3]), Bind::Tex(4, *o.level[4]),
                          Bind::Tex(5, *o.inpaint), Bind::Data(6, &k, sizeof(k))};
        if (!inpaint->Dispatch(b, 7, Groups(flowLane.width, 16), Groups(flowLane.height, 8))) return false;
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        set(i);
        const Out& o = outs[i];
        const Bind b[] = {Bind::Tex(0, *o.net->params), Bind::Tex(2, *o.filledFlowTm1), Bind::Tex(4, prev), Bind::Tex(5, color), Bind::Tex(6, *o.output),
                          Bind::Data(7, &k, sizeof(k)), Bind::Tex(9, *o.net->gate), Bind::Tex(10, *o.inpaint), Bind::Tex(11, *o.qMotionTp1), Bind::Tex(12, uiSource),
                          Bind::Tex(13, uiAlpha), Bind::Tex(14, noWarp)};
        if (!post->Dispatch(b, 12, Groups(display.width, kFiGroup), Groups(display.height, kFiGroup))) return false;
    }
    dev.CopyTexture(color, backup);  // the history, in the colour's own format
    return true;
}

std::unique_ptr<Interpolator> Interpolator::Create(Device& device, const Settings& settings, std::string* error)
{
    std::unique_ptr<Interpolator> it(new Interpolator());
    it->m.reset(new Impl(device));
    it->m->s = settings;
    if (!it->m->Create(error))
        return nullptr;
    const Impl& m = *it->m;
    Log("interpolator %ux%u, render %ux%u, motion lane %ux%u, network %ux%u", m.display.width, m.display.height, m.render.width, m.render.height,
        m.mvLane.width, m.mvLane.height, m.graph.width, m.graph.height);
    return it;
}
Interpolator::~Interpolator() = default;
const Texture&  Interpolator::Output(uint32_t i) const { return *m->outs[i].output; }
const Settings& Interpolator::settings() const { return m->s; }

bool Interpolator::Dispatch(const FrameInputs& in, uint32_t count, uint32_t total)
{
    if (!count || !m->Frame(in, std::min(count, kMaxGenerated), total))
        return false;
    m->dynCur ^= 1, m->depthCur ^= 1, m->prepCur ^= 1;
    return true;
}

std::vector<std::pair<std::string, const Texture*>> Interpolator::Intermediates() const
{
    const Impl& i = *m;
    const Out&  o = i.outs[0];
    return {{"pyramid", i.pyramid[i.pyrCur ^ 1].get()}, {"hints", i.computedMv.get()}, {"flow_hint", i.flowHint.get()}, {"flow", i.flow.get()},
            {"depth", i.depthTm1[i.depthCur].get()}, {"motion_t", o.qMotionTp1.get()}, {"flow_t", o.qFlowTm1.get()}, {"flow_filled", o.filledFlowTm1.get()},
            {"seed", o.seed.get()}, {"params", o.net->params}, {"gate", o.net->gate}, {"inpaint", o.inpaint.get()}, {"output", o.output.get()}};
}
}  // namespace nfrux