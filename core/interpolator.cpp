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
    uint32_t _OptionsPadding[3];
    float    _TimestepsAfterFirst[4];  // the batch's timesteps 1-4 (each pass runs them all in one dispatch, z = timestep)
};
static_assert(offsetof(FiConstants, _MvSimilarityThresholds) == 224 && offsetof(FiConstants, _InputTensorSize) == 240 &&
                  offsetof(FiConstants, _QuantParamsGateSNORM) == 304 && offsetof(FiConstants, _Options) == 336 &&
                  offsetof(FiConstants, _TimestepsAfterFirst) == 352,
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
    Tex qMotionTp1, qFlowTm1, filledFlowTm1, level[5], inpaint, seed, output;
};

struct Interpolator::Impl
{
    Device&  dev;
    Settings s;
    Dims     display, render, ofInput, of, mvLane, flowLane, graph;
    bool     allInput = false, colourDown = false;
    FiConstants c = {};
    float    lastVP[16] = {};
    PassPtr  downOf, downAll, downColour;
    // [0] one timestep, [1] a batch's timesteps in one dispatch (the shaders' ...Batch variants)
    PassPtr  initWarp[2], warp[2], warpTm1[2], fillTm1[2], preprocess[2], inpaintSeed[2], inpaintMip[2], inpaint[2], post[2], postHistory[2];
    PassPtr  mvHints, pyramidP, flowWarp, match, bilateral;
    Tex      dynMask[2], depthTm1[2], colorTm1[2], prep[2], reducedMotion, ofColour;
    Tex      computedMv, flow, pyramid[2], fWarp, fMatch, fSub, fFlow;
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
    // The flow in parts, so that the motion vectors' passes, which do not need it, are recorded between them (and run beside them).
    bool Hints(const FrameInputs& in);
    ImageConstants Image() const;
    bool Pyramid(const Texture& color, int32_t l);
    bool Match(bool reset);
    bool Frame(const FrameInputs& in, uint32_t count, uint32_t total);
};

#define NFRUX_PASS(var, name) if (!MakePass(var, shaders::name, shaders::name##Inv)) return fail("pass " #name);
#define NFRUX_PASS2(var, name) NFRUX_PASS(var[0], name) NFRUX_PASS(var[1], name##Batch)

bool Interpolator::Impl::Create(std::string* error)
{
    auto fail = [&](const char* m) { if (error) *error = m; return false; };
    display = {s.displayWidth, s.displayHeight};
    render  = {s.renderWidth, s.renderHeight};
    if (!display.width || !display.height || !render.width || !render.height || display.width > 8192 || display.height > 4320)
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

    NFRUX_PASS2(initWarp, kFiInitWarp)
    Format lane = s.outputFormat == Format::BGRA8Unorm ? Format::RGBA8Unorm : s.outputFormat;
#define NFRUX_LANE(suffix) { NFRUX_PASS(downOf, kFiDownsampleOf##suffix) NFRUX_PASS(downAll, kFiDownsampleAll##suffix) NFRUX_PASS(downColour, kFiDownsampleColour##suffix) }
    switch (lane)
    {
    case Format::RGBA8Unorm: NFRUX_LANE(Rgba8) break;
    case Format::RGB10A2Unorm: NFRUX_LANE(Rgb10a2) break;
    case Format::RGBA16Float: NFRUX_LANE(Rgba16f) break;
    default: return fail("unsupported output format");
    }
    NFRUX_PASS2(warp, kFiWarpFlow) NFRUX_PASS2(warpTm1, kFiWarpFlowTm1) NFRUX_PASS2(fillTm1, kFiFillHolesTm1) NFRUX_PASS2(preprocess, kFiPreprocess)
    NFRUX_PASS2(inpaintSeed, kFiInpaintSeed) NFRUX_PASS2(inpaintMip, kFiInpaintMip) NFRUX_PASS2(inpaint, kFiInpaint)
    switch (s.outputFormat)
    {
    case Format::RGBA8Unorm: case Format::BGRA8Unorm: NFRUX_PASS2(post, kFiPostprocessNoHistoryRgba8) NFRUX_PASS2(postHistory, kFiPostprocessHistoryRgba8) break;
    case Format::RGB10A2Unorm: NFRUX_PASS2(post, kFiPostprocessNoHistoryRgb10a2) NFRUX_PASS2(postHistory, kFiPostprocessHistoryRgb10a2) break;
    case Format::RGBA16Float: NFRUX_PASS2(post, kFiPostprocessNoHistoryRgba16f) NFRUX_PASS2(postHistory, kFiPostprocessHistoryRgba16f) break;
    default: return fail("unsupported output format");
    }
    NFRUX_PASS(mvHints, kOfMvHints) NFRUX_PASS(pyramidP, kOfArmPyramid) NFRUX_PASS(flowWarp, kOfArmWarp)
    NFRUX_PASS(match, kOfArmMatch) NFRUX_PASS(bilateral, kOfArmBilateral)
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
    for (auto& t : colorTm1) ok = ok && (t = dev.CreateTextureLike(s.colorLike, display.width, display.height, true));
    if (allInput || colourDown)
        for (auto& t : prep) tex(t, lane, mvLane);
    if (display.height > 1080) tex(ofColour, lane, ofInput);
    if (allInput) tex(reducedMotion, Format::RG16Float, mvLane);
    tex(computedMv, Format::RG16Float, of), tex(flow, Format::RG16Float, of);
    for (auto& t : pyramid) tex(t, Format::R8Unorm, {level[1].width, uint32_t(rows)});
    tex(fWarp, Format::R8Unorm, level[2]), tex(fMatch, Format::R32Uint, level[2]);
    tex(fSub, Format::RG16Float, level[2]), tex(fFlow, Format::RG16Float, level[2]);
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
    // The model's quantisation (nfru_mfg_v15_int8_metadata.json: inputs.x, outputs.getitem; the gate is outputs.getitem_1).
    k._QuantParamsSINT[0] = float(1 / 0.003921568859368563), k._QuantParamsSINT[1] = -128;
    k._QuantParamsSINT[2] = float(0.3125910758972168), k._QuantParamsSINT[3] = 86;
    k._QuantParamsSNORM[0] = float(1 / 0.49803924513980746), k._QuantParamsSNORM[1] = float(-1.0078740157480315);
    k._QuantParamsSNORM[2] = float(39.69906663894653), k._QuantParamsSNORM[3] = float(0.6771653543307087);
    k._QuantParamsGateSNORM[0] = float(1 / 0.4980377075262368), k._QuantParamsGateSNORM[1] = float(-1.0078740157480315);
    k._QuantParamsGateSNORM[2] = float(0.4980377075262368), k._QuantParamsGateSNORM[3] = float(-1.0078740157480315);
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
        if (!ok || !o.net)
            return false;
    }
    return true;
}

bool Interpolator::Impl::Hints(const FrameInputs& in)
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
    const Bind b[] = {Bind::Data(0, &h, sizeof(h)), Bind::Tex(1, depthPrev), Bind::Tex(2, *computedMv)};
    return mvHints->Dispatch(b, 3, Groups(of.width, 16), Groups(of.height, 16));
}

ImageConstants Interpolator::Impl::Image() const
{
    ImageConstants ic = {};
    ic.src_dims[0] = int32_t(ofInput.width), ic.src_dims[1] = int32_t(ofInput.height);
    ic.hint_dims[0] = int32_t(of.width), ic.hint_dims[1] = int32_t(of.height);
    ic.unorm8 = s.outputFormat == Format::RGBA8Unorm || s.outputFormat == Format::BGRA8Unorm;
    return ic;
}

bool Interpolator::Impl::Pyramid(const Texture& color, int32_t l)  // TP1's level l
{
    ImageConstants ic = Image();
    const Dims     d = level[l], p = level[l - 1];
    ic.level = l, ic.level_row = row[l], ic.coarse_row = row[l - 1];
    ic.level_dims[0] = int32_t(d.width), ic.level_dims[1] = int32_t(d.height);
    ic.coarse_dims[0] = int32_t(p.width), ic.coarse_dims[1] = int32_t(p.height);
    const Bind b[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, color), Bind::Tex(2, *pyramid[pyrCur])};
    return pyramidP->Dispatch(b, 3, Groups(d.width), Groups(d.height));
}

// After the hints and all of TP1's pyramid: the flow (levels 5 to 2), or the hints as the flow.
bool Interpolator::Impl::Match(bool reset)
{
    const Texture &p1 = *pyramid[pyrCur], &m1 = *pyramid[pyrCur ^ 1];
    ImageConstants ic = Image();
    static const bool noFlow = EnvFlag("NFRUX_TEST_NO_FLOW", false);  // cost measurement: the hints as the flow
    if (!pyramidValid || reset || noFlow)
        dev.CopyTexture(*computedMv, *flow);
    else
    {
        // The last level writes every texel of the flow (level 2 covers the hints' size): the hints are read where they are.
        for (int32_t l = 5; l >= 2; --l)
        {
            const Dims d = level[l], cc = l == 5 ? Dims{0, 0} : level[l + 1];
            ic.level = l, ic.level_row = row[l], ic.coarse_row = 0, ic.final = l == 2;
            ic.level_dims[0] = int32_t(d.width), ic.level_dims[1] = int32_t(d.height);
            ic.coarse_dims[0] = int32_t(cc.width), ic.coarse_dims[1] = int32_t(cc.height);
            const uint32_t gx = Groups(d.width), gy = Groups(d.height);
            const Bind w[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, p1), Bind::Tex(2, *fFlow), Bind::Tex(3, *fWarp)};
            const Bind mb[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, m1), Bind::Tex(2, *fWarp), Bind::Tex(3, *fFlow), Bind::Tex(4, *fSub), Bind::Tex(5, *fMatch)};
            const Bind bl[] = {Bind::Data(0, &ic, sizeof(ic)), Bind::Tex(1, m1), Bind::Tex(2, p1), Bind::Tex(3, *fSub), Bind::Tex(4, *fMatch),
                               Bind::Tex(5, *computedMv), Bind::Tex(6, *fFlow), Bind::Tex(7, *flow)};
            if (!flowWarp->Dispatch(w, 4, gx, gy) || !match->Dispatch(mb, 6, gx, gy) || !bilateral->Dispatch(bl, 8, gx, gy))  // the median in the filter
                return false;
        }
    }
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
    // Vectors that contain the sample positions' jitter (render pixels): its change taken out, as the SDK's jitter cancellation does.
    const bool jittered = in.motionJittered && !reset;
    k._JitterCancellation[0] = jittered ? (in.jitter[0] - k._Jitter[0]) / float(render.width) : 0.0f;
    k._JitterCancellation[1] = jittered ? (in.jitter[1] - k._Jitter[1]) / float(render.height) : 0.0f;
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
    // The history written by the first composite where the colour's format is the composite's (else copied at the end).
    const bool storeHistory = backup.storeFormat == s.outputFormat;
    const uint32_t options = (!in.uiSource || (needsAlpha && !in.uiAlpha) ? 0u : in.uiKind) | (in.noWarp ? 0x100u : 0u) | (in.orthographic ? 0x200u : 0u) | (s.swapRB ? 0x1000u : 0u) |
                           (in.uiSource ? 0u : 0x2000u);  // no UI source: the composite skips its UI test (the colour against itself)

    if (ofColour)
    {
        FiConstants o = k;  // its own: RenderSize is the output's there
        o._RenderSize[0] = int32_t(ofInput.width), o._RenderSize[1] = int32_t(ofInput.height);
        const Bind b[] = {Bind::Tex(0, color), Bind::Tex(1, *ofColour), Bind::Data(2, &o, sizeof(o))};
        if (!downOf->Dispatch(b, 3, Groups(ofInput.width, kFiGroup), Groups(ofInput.height, kFiGroup))) return false;
    }
    const Texture& flowColour = ofColour ? *ofColour : color;
    if (!Hints(in))
        return false;
    memcpy(lastVP, in.viewProjection, sizeof(lastVP));

    const Texture &dynTm1 = *dynMask[dynCur ^ 1], &dynTp1 = *dynMask[dynCur], &dTm1 = *depthTm1[depthCur], &dNext = *depthTm1[depthCur ^ 1];
    const Texture &prepP1 = down ? *prep[prepCur] : color, &prepM1 = down ? *prep[prepCur ^ 1] : prev;
    const Texture &warpDepth = allInput ? dNext : depth, &warpMotion = allInput ? *reducedMotion : mv;
    const uint32_t lx = Groups(mvLane.width, kFiGroup), ly = Groups(mvLane.height, kFiGroup);
    const uint32_t gx = Groups(std::max(mvLane.width, flowLane.width), kFiGroup), gy = Groups(std::max(mvLane.height, flowLane.height), kFiGroup);
    if (reset)
        count = 1, total = 1;
    // Each pass after the flow runs every timestep of the batch in one dispatch (z = timestep): the timesteps are in the constants,
    // and a resource of each timestep's own is bound once a timestep, at its binding + 16 x timestep (the shaders pick by z).
    // Slots past the batch are given the first timestep's (bound, never used). One timestep: the passes without the other slots.
    const uint32_t batch = count > 1 ? 1 : 0, slotCount = batch ? kMaxGenerated : 1;
    k._Timestep = 1.0f / float(total + 1), k._Options = options;
    for (uint32_t i = 1; i < kMaxGenerated; ++i) k._TimestepsAfterFirst[i - 1] = i < count ? float(i + 1) / float(total + 1) : 0.0f;
    struct Binds
    {
        Bind     b[96];
        uint32_t n = 0;
        Binds&   operator<<(const Bind& x) { b[n++] = x; return *this; }
    };
    auto slots = [&](Binds& bs, uint32_t binding, auto&& of) {
        for (uint32_t i = 0; i < slotCount; ++i)
        {
            Bind x = of(i < count ? i : 0u);
            x.binding = binding + 16 * i;
            bs << x;
        }
    };
    auto tex = [this](Tex Out::*t) { return [this, t](uint32_t i) { return Bind::Tex(0, *(outs[i].*t)); }; };
    auto level = [this](uint32_t l) { return [this, l](uint32_t i) { return Bind::Tex(0, *outs[i].level[l]); }; };
    const Bind downB[] = {Bind::Tex(0, flowColour), Bind::Tex(1, depth), Bind::Tex(2, mv), Bind::Tex(3, down ? prepP1 : color), Bind::Tex(4, dNext),
                          Bind::Tex(5, allInput ? *reducedMotion : dNext), Bind::Data(6, &k, sizeof(k))};
    {
        Binds b;
        b << Bind::Data(4, &k, sizeof(k));
        slots(b, 0, tex(&Out::qMotionTp1)), slots(b, 1, tex(&Out::qFlowTm1));
        if (!initWarp[batch]->Dispatch(b.b, b.n, gx, gy, count)) return false;
    }
    if (allInput && !downAll->Dispatch(downB, 7, lx, ly)) return false;
    if (colourDown && !downColour->Dispatch(downB, 7, lx, ly)) return false;
    // The warp beside the pyramid's first level (each level reads the one before). Its timestep-free outputs (the next depth, the TP1
    // mask) are written by timestep 0.
    if (!Pyramid(flowColour, 1)) return false;
    {
        Binds b;
        b << Bind::Tex(0, warpMotion) << Bind::Tex(1, warpDepth) << Bind::Tex(2, dTm1) << Bind::Tex(3, dynTm1) << Bind::Tex(4, dynTp1) << Bind::Tex(8, dNext)
          << Bind::Data(9, &k, sizeof(k));
        slots(b, 5, tex(&Out::qMotionTp1));
        if (!warp[batch]->Dispatch(b.b, b.n, lx, ly, count)) return false;
    }
    for (int32_t l = 2; l < 6; ++l) if (!Pyramid(flowColour, l)) return false;
    if (!Match(reset)) return false;
    pyramidValid = true;
    pyrCur ^= 1;
    if (reset)
    {
        dev.Clear(dynTp1, 1.0f);
        dev.CopyTexture(color, backup);
        return true;
    }
    {
        Binds b;
        b << Bind::Tex(0, *flow) << Bind::Tex(1, dTm1) << Bind::Data(3, &k, sizeof(k));
        slots(b, 2, tex(&Out::qFlowTm1));
        if (!warpTm1[batch]->Dispatch(b.b, b.n, Groups(flowLane.width, 32), Groups(flowLane.height, 4), count)) return false;
    }
    {
        Binds b;
        b << Bind::Data(2, &k, sizeof(k));
        slots(b, 0, tex(&Out::qFlowTm1)), slots(b, 1, tex(&Out::filledFlowTm1));
        if (!fillTm1[batch]->Dispatch(b.b, b.n, Groups(flowLane.width, kFiGroup), Groups(flowLane.height, kFiGroup), count)) return false;
    }
    {
        Binds b;
        b << Bind::Tex(2, prepM1) << Bind::Tex(3, prepP1) << Bind::Tex(4, dNext) << Bind::Tex(5, dTm1) << Bind::Data(9, &k, sizeof(k));
        slots(b, 0, tex(&Out::qMotionTp1)), slots(b, 1, tex(&Out::filledFlowTm1)), slots(b, 10, tex(&Out::seed));
        slots(b, 8, [this](uint32_t i) { return Bind::Buf(0, *outs[i].net->input); });
        if (!preprocess[batch]->Dispatch(b.b, b.n, Groups(graph.width, 16), Groups(graph.height, 16), count)) return false;
    }
    {
        Binds b;
        b << Bind::Data(2, &k, sizeof(k));
        slots(b, 0, tex(&Out::seed)), slots(b, 1, level(0));
        if (!inpaintSeed[batch]->Dispatch(b.b, b.n, Groups(flowLane.width, 16), Groups(flowLane.height, 8), count)) return false;
    }
    for (uint32_t l = 1; l < 5; ++l)
    {
        Binds b;
        b << Bind::Data(2, &k, sizeof(k));
        slots(b, 0, level(l - 1)), slots(b, 1, level(l));
        if (!inpaintMip[batch]->Dispatch(b.b, b.n, Groups(outs[0].level[l]->width, 16), Groups(outs[0].level[l]->height, 8), count)) return false;
    }
    Network* nets[kMaxGenerated];
    for (uint32_t i = 0; i < count; ++i) nets[i] = outs[i].net.get();
    dev.RunNetworks(nets, count);
    dev.Mark("network");
    {
        Binds b;
        b << Bind::Data(6, &k, sizeof(k));
        for (uint32_t l = 0; l < 5; ++l) slots(b, l, level(l));
        slots(b, 5, tex(&Out::inpaint));
        if (!inpaint[batch]->Dispatch(b.b, b.n, Groups(flowLane.width, 16), Groups(flowLane.height, 8), count)) return false;
    }
    {
        // Frame 1 may go straight into the host's output; the history is stored by timestep 0.
        Binds b;
        b << Bind::Tex(4, prev) << Bind::Tex(5, color) << Bind::Data(7, &k, sizeof(k)) << Bind::Tex(12, uiSource) << Bind::Tex(13, uiAlpha) << Bind::Tex(14, noWarp);
        if (storeHistory) b << Bind::Tex(8, backup);
        slots(b, 0, [this](uint32_t i) { return Bind::Tex(0, *outs[i].net->params); }), slots(b, 9, [this](uint32_t i) { return Bind::Tex(0, *outs[i].net->gate); });
        slots(b, 2, tex(&Out::filledFlowTm1)), slots(b, 10, tex(&Out::inpaint)), slots(b, 11, tex(&Out::qMotionTp1));
        slots(b, 6, [&](uint32_t i) { return Bind::Tex(0, i == 0 && in.output0 ? *in.output0 : *outs[i].output); });
        if (!(storeHistory ? postHistory : post)[batch]->Dispatch(b.b, b.n, Groups(display.width, kFiGroup), Groups(display.height, kFiGroup), count)) return false;
    }
    dev.Mark("(composite)");
    if (!storeHistory) dev.CopyTexture(color, backup);  // the history, in the colour's own format
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
    return {{"pyramid", i.pyramid[i.pyrCur ^ 1].get()}, {"hints", i.computedMv.get()}, {"flow_hint", i.computedMv.get()}, {"flow", i.flow.get()},
            {"depth", i.depthTm1[i.depthCur].get()}, {"motion_t", o.qMotionTp1.get()}, {"flow_t", o.qFlowTm1.get()}, {"flow_filled", o.filledFlowTm1.get()},
            {"seed", o.seed.get()}, {"params", o.net->params}, {"gate", o.net->gate}, {"inpaint", o.inpaint.get()}, {"output", o.output.get()}};
}
}  // namespace nfrux