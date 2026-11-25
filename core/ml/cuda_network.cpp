// SPDX-License-Identifier: MIT
#include "cuda_network.h"

#include "../log.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

using Microsoft::WRL::ComPtr;

namespace nfrux
{
namespace
{
using CUresult = int;
struct Dim { uint32_t x, y, z; };
struct Launch { void* fn; Dim grid, block; uint32_t shared; const void* params; uint32_t size; };
struct Api
{
    int (*createModule)(ID3D12Device*, const void*, uint32_t, void**) = nullptr;
    int (*createFunction)(ID3D12Device*, void*, const char*, void**) = nullptr;
    int (*launch)(ID3D12GraphicsCommandList*, const Launch*, uint32_t) = nullptr;
    int (*destroyModule)(ID3D12Device*, void*) = nullptr;
    int (*destroyFunction)(ID3D12Device*, void*) = nullptr;
    HMODULE cuda = nullptr;
    bool ok = false;
};
const Api& Nv()
{
    static const Api api = [] {
        Api a;
        HMODULE nvapi = LoadLibraryExA("nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        a.cuda        = LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        auto query    = nvapi ? reinterpret_cast<void* (*)(uint32_t)>(GetProcAddress(nvapi, "nvapi_QueryInterface")) : nullptr;
        auto init     = query ? reinterpret_cast<int (*)()>(query(0x0150e828)) : nullptr;
        if (!a.cuda || !init || init() != 0) return a;
        a.createModule    = reinterpret_cast<decltype(a.createModule)>(query(0xad1a677d));
        a.createFunction  = reinterpret_cast<decltype(a.createFunction)>(query(0xe2436e22));
        a.launch          = reinterpret_cast<decltype(a.launch)>(query(0x24973538));
        a.destroyModule   = reinterpret_cast<decltype(a.destroyModule)>(query(0x41c65285));
        a.destroyFunction = reinterpret_cast<decltype(a.destroyFunction)>(query(0xdf295ea6));
        a.ok = a.createModule && a.createFunction && a.launch;
        return a;
    }();
    return api;
}

// PTX -> cubin for the CUDA device that is this D3D12 adapter (by LUID).
std::vector<uint8_t> Link(const std::string& ptx, LUID luid, std::string* error)
{
    HMODULE m = Nv().cuda;
    auto f = [&](const char* n) { return GetProcAddress(m, n); };
    auto init = reinterpret_cast<CUresult (*)(unsigned)>(f("cuInit"));
    auto count = reinterpret_cast<CUresult (*)(int*)>(f("cuDeviceGetCount"));
    auto get = reinterpret_cast<CUresult (*)(int*, int)>(f("cuDeviceGet"));
    auto getLuid = reinterpret_cast<CUresult (*)(char*, unsigned*, int)>(f("cuDeviceGetLuid"));
    auto create = reinterpret_cast<CUresult (*)(void**, unsigned, int)>(f("cuCtxCreate_v2"));
    auto destroy = reinterpret_cast<CUresult (*)(void*)>(f("cuCtxDestroy_v2"));
    auto linkCreate = reinterpret_cast<CUresult (*)(unsigned, int*, void**, void**)>(f("cuLinkCreate_v2"));
    auto linkAdd = reinterpret_cast<CUresult (*)(void*, int, void*, size_t, const char*, unsigned, int*, void**)>(f("cuLinkAddData_v2"));
    auto linkComplete = reinterpret_cast<CUresult (*)(void*, void**, size_t*)>(f("cuLinkComplete"));
    auto linkDestroy = reinterpret_cast<CUresult (*)(void*)>(f("cuLinkDestroy"));
    std::vector<uint8_t> out;
    auto fail = [&](const std::string& s) { if (error) *error = s; return out; };
    int n = 0, dev = -1;
    if (!init || !count || !get || !getLuid || !create || !linkCreate || !linkAdd || !linkComplete || init(0) || count(&n)) return fail("CUDA not usable");
    for (int i = 0; i < n && dev < 0; ++i)
    {
        int d; char l[8]; unsigned mask;
        if (!get(&d, i) && !getLuid(l, &mask, d) && !memcmp(l, &luid, 8)) dev = d;
    }
    void* ctx = nullptr;
    if (dev < 0 || create(&ctx, 0, dev)) return fail("no CUDA device for this adapter");
    std::vector<char> log(8192, 0);
    int   opts[] = {5, 6};  // the error log's buffer and size
    void* vals[] = {log.data(), reinterpret_cast<void*>(size_t(log.size()))};
    void *state = nullptr, *bin = nullptr;
    size_t size = 0;
    CUresult r = linkCreate(2, opts, vals, &state);
    if (!r) r = linkAdd(state, 1, const_cast<char*>(ptx.c_str()), ptx.size() + 1, "nfrux", 0, nullptr, nullptr);
    if (!r) r = linkComplete(state, &bin, &size);
    if (!r) out.assign(static_cast<uint8_t*>(bin), static_cast<uint8_t*>(bin) + size);
    if (state && linkDestroy) linkDestroy(state);
    destroy(ctx);
    if (r && error) *error = "PTX link " + std::to_string(r) + ": " + std::string(log.data()).substr(0, 300);
    return out;
}

// A cubin by the PTX's hash, device and driver, in LOCALAPPDATA\NFRUX.
std::string CachePath(const std::string& ptx, LUID luid)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) { for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull; };
    mix(ptx.data(), ptx.size()), mix(&luid, 8);
    if (auto v = reinterpret_cast<CUresult (*)(int*)>(GetProcAddress(Nv().cuda, "cuDriverGetVersion"))) { int d = 0; v(&d), mix(&d, 4); }
    char path[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("LOCALAPPDATA", path, MAX_PATH);
    if (!n || n > MAX_PATH - 60) return {};
    strcat_s(path, "\\NFRUX");
    CreateDirectoryA(path, nullptr);
    char name[32];
    snprintf(name, sizeof(name), "\\%016llx.cubin", static_cast<unsigned long long>(h));
    return std::string(path) + name;
}

struct Writer
{
    std::string s;
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

// A source coordinate: the convolution's input coordinate (register v, in range) through the nearest resize, if any.
void NearestPtx(Writer& w, const char* dst, const char* v, int n, int d, int off, int size)
{
    // y = v * d + off; iy = floor(y / n); ry = y - iy * n; 2 ry >= n ? min(iy + 1, size - 1) : max(iy, 0). y >= -n here: shifted by n first.
    w("mul.lo.s32 %%t1, %s, %d;", v, d);
    w("add.s32 %%t1, %%t1, %d;", off + n);
    w("div.u32 %%t2, %%t1, %d;", n);
    w("mul.lo.s32 %%t3, %%t2, %d;", n);
    w("sub.s32 %%t3, %%t1, %%t3;");
    w("shl.b32 %%t3, %%t3, 1;");
    w("sub.s32 %%t2, %%t2, 1;");
    w("add.s32 %%t4, %%t2, 1;");
    w("min.s32 %%t4, %%t4, %d;", size - 1);
    w("max.s32 %%t2, %%t2, 0;");
    w("setp.ge.s32 %%pt, %%t3, %d;", n);
    w("selp.b32 %s, %%t4, %%t2, %%pt;", dst);
}
}  // namespace

static std::string LayerPtx(uint32_t index, const ffx_ml::FusedConv& fc, const ffx_ml::LoweredGraph& g, bool external)
{
    const auto& c  = *fc.conv;
    const auto& r  = *fc.rescale;
    const auto& s0 = fc.sources[0];
    const auto& t0 = g.tensors[s0.tensor];
    const int   ic0 = t0.channelStride / 4, ic1 = fc.sources.size() > 1 ? g.tensors[fc.sources[1].tensor].channelStride / 4 : 0, icw = ic0 + ic1;
    const int   chunks = (icw + 3) / 4, oc = fc.outC, nt = (oc + 7) / 8, stride = c.stride[0], k = c.kernelH;
    const int   ocPad = (oc + 3) & ~3, outStride = g.tensors[fc.output].channelStride;
    Writer w;
    char   h[256];
    snprintf(h, sizeof(h), ".visible .entry layer%u(.param .u64 p0, .param .u64 p1, .param .u64 p2, .param .u64 p3, .param .u64 p4, .param .u32 p5)\n{\n", index);
    std::string head = h;
    w(".reg .u64 %%a<64>; .reg .u32 %%r<64>; .reg .s32 %%t<8>; .reg .s32 %%x<32>; .reg .s32 %%acc<32>; .reg .u32 %%f<8>; .reg .pred %%p<32>; .reg .pred %%pt; .reg .s64 %%l<16>; .reg .s32 %%q<32>;");
    w("ld.param.u64 %%a1, [p0]; ld.param.u64 %%a2, [p1]; ld.param.u64 %%a3, [p2]; ld.param.u64 %%a4, [p3]; ld.param.u64 %%a5, [p4]; ld.param.u32 %%r50, [p5];");
    w("mov.u32 %%r1, %%tid.x; mov.u32 %%r2, %%ctaid.x; mov.u32 %%r3, %%ctaid.y;");
    w("shr.u32 %%r4, %%r1, 5; mad.lo.u32 %%r5, %%r2, 4, %%r4; shl.b32 %%r5, %%r5, 4;");         // first output column of this warp
    w("setp.ge.u32 %%p1, %%r5, %d; setp.ge.u32 %%p2, %%r3, %d; or.pred %%p1, %%p1, %%p2;", fc.outW, fc.outH);
    w("@%%p1 ret;");
    w("and.b32 %%r6, %%r1, 31; shr.u32 %%r7, %%r6, 2; and.b32 %%r8, %%r6, 3;");                   // fragment row and column
    w("add.s32 %%x1, %%r5, %%r7; add.s32 %%x2, %%x1, 8;");                                        // this lane's two output columns
    w("mov.u32 %%f6, 0x%08x;", uint32_t(c.inputZp & 0xff) * 0x01010101u);
    for (int i = 0; i < nt * 4; ++i) w("mov.s32 %%acc%d, 0;", i);
    w("cvt.u64.u32 %%a8, %%r8; shl.b64 %%a8, %%a8, 2;");                                          // fragment column in bytes
    w("cvt.u64.u32 %%a9, %%r7; shl.b64 %%a9, %%a9, 4; add.u64 %%a9, %%a9, %%a8; add.u64 %%a9, %%a3, %%a9;");  // weights: n (row) * 16 + column bytes
    if (icw % 4) w("setp.ge.u32 %%p9, %%r8, %d; min.u32 %%r9, %%r8, %d; cvt.u64.u32 %%a10, %%r9; shl.b64 %%a10, %%a10, 2;", icw % 4, icw % 4 - 1);
    // Input columns: both output columns, for every kx (the taps of a row share them).
    for (int kx = 0; kx < k; ++kx)
        for (int j = 0; j < 2; ++j)
        {
            w("mad.lo.s32 %%x3, %%x%d, %d, %d;", 1 + j, stride, kx - c.pad[2]);
            w("setp.ge.s32 %%p3, %%x3, 0; setp.lt.s32 %%p4, %%x3, %d; and.pred %%p%d, %%p3, %%p4;", fc.inW, 10 + kx * 2 + j);
            w("max.s32 %%x3, %%x3, 0; min.s32 %%x3, %%x3, %d;", fc.inW - 1);
            if (s0.resize) { NearestPtx(w, "%x4", "%x3", s0.scale[2], s0.scale[3], s0.offset[1], t0.w); w("mov.s32 %%x3, %%x4;"); }
            w("mov.s32 %%q%d, %%x3;", kx * 2 + j);
        }
    for (int ky = 0; ky < k; ++ky)
    {
        w("mad.lo.s32 %%x5, %%r3, %d, %d;", stride, ky - c.pad[0]);
        w("setp.ge.s32 %%p3, %%x5, 0; setp.lt.s32 %%p4, %%x5, %d; and.pred %%p5, %%p3, %%p4;", fc.inH);
        w("max.s32 %%x5, %%x5, 0; min.s32 %%x5, %%x5, %d;", fc.inH - 1);
        if (s0.resize) { NearestPtx(w, "%x6", "%x5", s0.scale[0], s0.scale[1], s0.offset[0], t0.h); w("mov.s32 %%x5, %%x6;"); }
        w("mul.lo.s32 %%x6, %%x5, %d;", t0.w);
        for (int kx = 0; kx < k; ++kx)
        {
            for (int j = 0; j < 2; ++j)
            {
                w("add.s32 %%x7, %%x6, %%q%d;", kx * 2 + j);
                w("cvt.u64.s32 %%a%d, %%x7; mul.lo.u64 %%a%d, %%a%d, %d; add.u64 %%a%d, %%a1, %%a%d;", 12 + j, 14 + j, 12 + j, ic0 * 4, 14 + j, 14 + j);
                if (ic1) w("mul.lo.u64 %%a%d, %%a%d, %d; add.u64 %%a%d, %%a2, %%a%d;", 16 + j, 12 + j, ic1 * 4, 16 + j, 16 + j);
                w("and.pred %%p%d, %%p5, %%p%d;", 6 + j, 10 + kx * 2 + j);
            }
            for (int ch = 0; ch < chunks; ++ch)
            {
                const bool one = ch * 4 < ic0;
                const int  base = (one ? ch * 4 : ch * 4 - ic0) * 4;
                const bool part = icw % 4 && ch == chunks - 1;  // the last chunk is not four whole words
                for (int j = 0; j < 2; ++j)
                {
                    w("add.u64 %%a20, %%a%d, %%a%d; ld.global.u32 %%f%d, [%%a20 + %d];", (one ? 14 : 16) + j, part ? 10 : 8, j, base);
                    w("selp.b32 %%f%d, %%f%d, %%f6, %%p%d;", j, j, 6 + j);
                    if (part) w("selp.b32 %%f%d, 0, %%f%d, %%p9;", j, j);
                }
                for (int n = 0; n < nt; ++n)
                {
                    const uint32_t off = uint32_t(((n * k * k + ky * k + kx) * chunks + ch) * 128);
                    w("ld.global.u32 %%f2, [%%a9 + %u];", off);
                    w("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%%acc%d, %%acc%d, %%acc%d, %%acc%d}, {%%f0, %%f1}, {%%f2}, {%%acc%d, %%acc%d, %%acc%d, %%acc%d};",
                      n * 4, n * 4 + 1, n * 4 + 2, n * 4 + 3, n * 4, n * 4 + 1, n * 4 + 2, n * 4 + 3);
                }
            }
        }
    }
    // Each lane: channels 8 n + 2 col and + 1, of both columns.
    if (external) w("cvt.u64.u32 %%a30, %%r3; cvt.u64.u32 %%a31, %%r50; mul.lo.u64 %%a30, %%a30, %%a31; add.u64 %%a30, %%a5, %%a30;");
    else w("cvt.u64.u32 %%a30, %%r3; mul.lo.u64 %%a30, %%a30, %d; add.u64 %%a30, %%a5, %%a30;", fc.outW * outStride);
    for (int j = 0; j < 2; ++j)
    {
        w("setp.lt.s32 %%p%d, %%x%d, %d;", 26 + j, 1 + j, fc.outW);
        w("cvt.u64.s32 %%a%d, %%x%d; mul.lo.u64 %%a%d, %%a%d, %d; add.u64 %%a%d, %%a30, %%a%d;", 32 + j, 1 + j, 32 + j, 32 + j, outStride, 32 + j, 32 + j);
    }
    w("shl.b32 %%r10, %%r8, 1; cvt.u64.u32 %%a34, %%r10;");
    for (int n = 0; n < nt; ++n)
        for (int e = 0; e < 2; ++e)
        {
            const int ch = n * 8 + e;
            w("add.u32 %%r11, %%r10, %d; setp.lt.u32 %%p20, %%r11, %d;", ch, oc);
            w("min.u32 %%r12, %%r11, %d; cvt.u64.u32 %%a35, %%r12; shl.b64 %%a36, %%a35, 2; add.u64 %%a36, %%a4, %%a36;", oc - 1);
            w("ld.global.s32 %%x10, [%%a36];");
            if (r.perChannel) w("ld.global.s32 %%x11, [%%a36 + %d]; ld.global.u32 %%r13, [%%a36 + %d];", ocPad * 4, ocPad * 8);
            else w("ld.global.s32 %%x11, [%%a4 + %d]; ld.global.u32 %%r13, [%%a4 + %d];", ocPad * 4, ocPad * 8);
            w("sub.u32 %%r14, %%r13, 1; mov.s64 %%l1, 1; shl.b64 %%l1, %%l1, %%r14;");
            for (int j = 0; j < 2; ++j)
            {
                w("add.s32 %%x12, %%acc%d, %%x10; mul.wide.s32 %%l2, %%x12, %%x11; add.s64 %%l2, %%l2, %%l1; shr.s64 %%l2, %%l2, %%r13; cvt.s32.s64 %%x13, %%l2;", n * 4 + j * 2 + e);
                w("add.s32 %%x13, %%x13, %d; max.s32 %%x13, %%x13, -128; min.s32 %%x13, %%x13, 127;", r.rescaleOutZp);
                if (fc.table) w("add.s32 %%x13, %%x13, 128; cvt.u64.s32 %%a37, %%x13; add.u64 %%a37, %%a4, %%a37; ld.global.s8 %%x13, [%%a37 + %d];", ocPad * 12);
                w("and.pred %%p21, %%p20, %%p%d; add.u64 %%a38, %%a%d, %%a35;", 26 + j, 32 + j);
                w("@%%p21 st.global.u8 [%%a38], %%x13;");
            }
        }
    w("ret;");
    return head + w.s + "}\n";
}

std::unique_ptr<CudaGraph> CudaGraph::Create(ID3D12Device* device, const ffx_ml::GraphDesc& desc, std::string* error)
{
    auto fail = [&](const std::string& m) -> std::unique_ptr<CudaGraph> { if (error) *error = m; return nullptr; };
    if (!Nv().ok) return fail("no NVAPI CUDA launch");
    ffx_ml::tosa::Graph  parsed;
    ffx_ml::LoweredGraph g;
    if (!ffx_ml::tosa::Parse(desc, parsed, error) || !ffx_ml::Lower(parsed, g, error)) return nullptr;
    auto out = std::unique_ptr<CudaGraph>(new CudaGraph());
    out->m_device = device;
    out->m_outputBindings = parsed.outputBindings;
    auto buffer = [&](uint64_t size, D3D12_HEAP_TYPE heap) {
        D3D12_HEAP_PROPERTIES hp = {heap};
        D3D12_RESOURCE_DESC   d  = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, (size + 255) & ~255ull, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                    heap == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
        ComPtr<ID3D12Resource> r;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r));
        return r;
    };
    std::map<uint32_t, int32_t> slot;
    for (uint32_t i = 0; i < g.tensors.size(); ++i)
    {
        const auto& t = g.tensors[i];
        if (!t.external)
        {
            slot[i] = int32_t(out->m_tensors.size());
            out->m_tensors.push_back(buffer(uint64_t(t.h) * t.w * t.channelStride, D3D12_HEAP_TYPE_DEFAULT));
            if (!out->m_tensors.back()) return fail("out of memory");
            continue;
        }
        const auto it = std::find(parsed.outputBindings.begin(), parsed.outputBindings.end(), t.binding);
        slot[i] = std::find(parsed.inputBindings.begin(), parsed.inputBindings.end(), t.binding) != parsed.inputBindings.end() ? -1 : -2 - int32_t(it - parsed.outputBindings.begin());
    }
    std::string          ptx = ".version 7.1\n.target sm_80\n.address_size 64\n";
    std::vector<uint8_t> constants;
    for (size_t li = 0; li < g.convs.size(); ++li)
    {
        const auto& fc = g.convs[li];
        const auto& c  = *fc.conv;
        const auto& r  = *fc.rescale;
        const auto& s0 = fc.sources[0];
        const auto& t0 = g.tensors[s0.tensor];
        const bool  two = fc.sources.size() == 2;
        const int   ic0 = t0.channelStride / 4, ic1 = two ? g.tensors[fc.sources[1].tensor].channelStride / 4 : 0;
        bool ok = fc.sources.size() <= 2 && c.kernelH == c.kernelW && c.kernelH <= 7 && c.stride[0] == c.stride[1] && c.dilation[0] == 1 && c.dilation[1] == 1 && c.weightZp == 0 &&
                  r.scale32 && !r.doubleRound && r.rescaleInZp == 0 && t0.c == t0.channelStride && fc.inC == (ic0 + ic1) * 4 && (!two || (ic0 % 4 == 0 && ic1 % 4 == 0 && !fc.sources[1].resize && !s0.resize)) &&
                  fc.outC <= 64 && (s0.resize ? s0.offset[0] >= -s0.scale[0] && s0.offset[1] >= -s0.scale[2] : t0.h == fc.inH && t0.w == fc.inW);
        for (int32_t s : r.shift) ok = ok && s >= 2 && s <= 62;
        if (!ok) return fail("unsupported layer " + std::to_string(li));
        Layer l;
        l.dst = slot[fc.output], l.external = l.dst < -1;
        ptx += LayerPtx(uint32_t(li), fc, g, l.external);
        const int chunks = (ic0 + ic1 + 3) / 4, nt = (fc.outC + 7) / 8, taps = c.kernelH * c.kernelW, ic = fc.inC, pad = (fc.outC + 3) & ~3;
        auto align = [&] { constants.resize((constants.size() + 255) & ~size_t(255)); };
        align();
        l.weights = constants.size();
        constants.resize(constants.size() + size_t(nt) * taps * chunks * 128);
        for (int o = 0; o < fc.outC; ++o)
            for (int tap = 0; tap < taps; ++tap)
                for (int i = 0; i < ic; ++i)
                    constants[l.weights + (size_t(o / 8 * taps + tap) * chunks + i / 16) * 128 + size_t(o % 8) * 16 + i % 16] = uint8_t(c.weights[(size_t(o) * taps + tap) * ic + i]);
        align();
        l.params = constants.size();
        std::vector<int32_t> p(size_t(pad) * 3 + 64, 0);
        for (int o = 0; o < fc.outC; ++o)
        {
            int64_t sum = 0;
            for (int i = 0; i < taps * ic; ++i) sum += c.weights[size_t(o) * taps * ic + i];
            p[o] = int32_t(int64_t(c.bias[o]) - int64_t(c.inputZp) * sum);
            const int k = r.perChannel ? o : 0;
            p[pad + o] = r.multiplier[k], p[2 * pad + o] = r.shift[k];
        }
        if (fc.table) memcpy(&p[3 * pad], fc.table->table.data(), 256);
        constants.insert(constants.end(), reinterpret_cast<uint8_t*>(p.data()), reinterpret_cast<uint8_t*>(p.data() + p.size()));
        l.src[0] = slot[s0.tensor], l.src[1] = two ? slot[fc.sources[1].tensor] : l.src[0];
        l.grid[0] = uint32_t(fc.outW + 63) / 64, l.grid[1] = uint32_t(fc.outH);
        out->m_layers.push_back(l);
    }
    const LUID        luid = device->GetAdapterLuid();
    const std::string cache = CachePath(ptx, luid);
    std::vector<uint8_t> cubin;
    if (FILE* f = cache.empty() ? nullptr : fopen(cache.c_str(), "rb"))
    {
        fseek(f, 0, SEEK_END), cubin.resize(size_t(ftell(f))), fseek(f, 0, SEEK_SET);
        if (fread(cubin.data(), 1, cubin.size(), f) != cubin.size()) cubin.clear();
        fclose(f);
    }
    if (cubin.empty() || Nv().createModule(device, cubin.data(), uint32_t(cubin.size()), &out->m_module))
    {
        cubin = Link(ptx, luid, error);
        if (cubin.empty()) return nullptr;
        if (FILE* f = cache.empty() ? nullptr : fopen(cache.c_str(), "wb")) fwrite(cubin.data(), 1, cubin.size(), f), fclose(f);
        if (Nv().createModule(device, cubin.data(), uint32_t(cubin.size()), &out->m_module)) return fail("CreateCuModule");
    }
    for (size_t i = 0; i < out->m_layers.size(); ++i)
        if (Nv().createFunction(device, out->m_module, ("layer" + std::to_string(i)).c_str(), &out->m_layers[i].fn)) return fail("CreateCuFunction");
    out->m_constants = buffer(constants.size(), D3D12_HEAP_TYPE_DEFAULT), out->m_upload = buffer(constants.size(), D3D12_HEAP_TYPE_UPLOAD);
    void* map = nullptr;
    if (!out->m_constants || !out->m_upload || FAILED(out->m_upload->Map(0, nullptr, &map))) return fail("out of memory");
    memcpy(map, constants.data(), constants.size());
    out->m_upload->Unmap(0, nullptr);
    return out;
}

CudaGraph::~CudaGraph()
{
    if (Nv().destroyFunction) for (auto& l : m_layers) if (l.fn) Nv().destroyFunction(m_device.Get(), l.fn);
    if (Nv().destroyModule && m_module) Nv().destroyModule(m_device.Get(), m_module);
}

void CudaGraph::Record(ID3D12GraphicsCommandList* list, D3D12_GPU_VIRTUAL_ADDRESS input, ID3D12Resource* const* outputs, uint32_t outputPitch)
{
    auto transition = [](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
        D3D12_RESOURCE_BARRIER x = {};
        x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
        return x;
    };
    std::vector<D3D12_RESOURCE_BARRIER> b;
    if (!m_uploaded)
    {
        list->CopyResource(m_constants.Get(), m_upload.Get());
        b.push_back(transition(m_constants.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        m_uploaded = true;
    }
    for (auto& t : m_tensors) b.push_back(transition(t.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    list->ResourceBarrier(UINT(b.size()), b.data());
    const uint64_t    base = m_constants->GetGPUVirtualAddress();
    std::vector<bool> dirty(m_tensors.size(), false);
    auto va = [&](int32_t t) { return t >= 0 ? m_tensors[t]->GetGPUVirtualAddress() : t == -1 ? input : outputs[-2 - t]->GetGPUVirtualAddress(); };
    for (const Layer& l : m_layers)
    {
        b.clear();
        for (int32_t t : {l.src[0], l.src[1], l.dst})
            if (t >= 0 && dirty[t])
            {
                D3D12_RESOURCE_BARRIER u = {D3D12_RESOURCE_BARRIER_TYPE_UAV};
                u.UAV.pResource = m_tensors[t].Get();
                b.push_back(u), dirty[t] = false;
            }
        if (!b.empty()) list->ResourceBarrier(UINT(b.size()), b.data());
#pragma pack(push, 4)
        struct { uint64_t s0, s1, w, p, d; uint32_t row; } params = {va(l.src[0]), va(l.src[1]), base + l.weights, base + l.params, va(l.dst), outputPitch};
#pragma pack(pop)
        const Launch k = {l.fn, {l.grid[0], l.grid[1], 1}, {128, 1, 1}, 0, &params, 44};
        Nv().launch(list, &k, 1);
        if (l.dst >= 0) dirty[l.dst] = true;
    }
    b.clear();
    for (auto& t : m_tensors) b.push_back(transition(t.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON));
    list->ResourceBarrier(UINT(b.size()), b.data());
}
}  // namespace nfrux
