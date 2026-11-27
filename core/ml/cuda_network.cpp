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
        const char* absent = getenv("NFRUX_TEST_ABSENT");  // "nvapi64.dll" or "nvcuda.dll": as if it were not installed
    auto load = [&](const char* n) { return absent && !strcmp(absent, n) ? nullptr : LoadLibraryExA(n, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); };
    HMODULE nvapi = load("nvapi64.dll");
        a.cuda        = load("nvcuda.dll");
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
    // The error log's buffer and size. The target is the context's device: the driver compiles for the GPU it is (tests: CU_JIT_TARGET).
    const char* target = Env("NFRUX_TEST_CUDA_TARGET");
    int   opts[] = {5, 6, 9};
    void* vals[] = {log.data(), reinterpret_cast<void*>(size_t(log.size())), reinterpret_cast<void*>(size_t(target ? atoi(target) : 0))};
    void *state = nullptr, *bin = nullptr;
    size_t size = 0;
    CUresult r = linkCreate(target ? 3 : 2, opts, vals, &state);
    if (!r) r = linkAdd(state, 1, const_cast<char*>(ptx.c_str()), ptx.size() + 1, "nfrux", 0, nullptr, nullptr);
    if (!r) r = linkComplete(state, &bin, &size);
    if (!r) out.assign(static_cast<uint8_t*>(bin), static_cast<uint8_t*>(bin) + size);
    if (state && linkDestroy) linkDestroy(state);
    destroy(ctx);
    if (target && error) *error = "linked " + std::to_string(out.size()) + " bytes for sm_" + target;
    if (r && error) *error = "PTX link " + std::to_string(r) + ": " + std::string(log.data()).substr(0, 300);
    return out;
}

// At sm_75: whole words only. A word that begins with RTX, or is T4 (not T400).
}  // namespace
bool HasTensorCores(const char* name)
{
    for (const char* w = name; *w;)
    {
        const size_t n = strcspn(w, " ");
        if (!strncmp(w, "RTX", 3) || (n == 2 && !strncmp(w, "T4", 2)) || (n == 3 && !strncmp(w, "T4G", 3))) return true;
        w += n + (w[n] != 0);
    }
    return false;
}
namespace
{
// The adapter's compute capability (75, 80, 89 ...); 0 without int8 tensor cores. GTX 16 reports 75 as RTX 20 does, and has none.
int ComputeCapability(LUID luid, std::string* error)
{
    if (const char* t = Env("NFRUX_TEST_CUDA_SM")) return atoi(t);
    HMODULE m = Nv().cuda;
    auto f = [&](const char* n) { return GetProcAddress(m, n); };
    auto init = reinterpret_cast<CUresult (*)(unsigned)>(f("cuInit"));
    auto count = reinterpret_cast<CUresult (*)(int*)>(f("cuDeviceGetCount"));
    auto get = reinterpret_cast<CUresult (*)(int*, int)>(f("cuDeviceGet"));
    auto getLuid = reinterpret_cast<CUresult (*)(char*, unsigned*, int)>(f("cuDeviceGetLuid"));
    auto attribute = reinterpret_cast<CUresult (*)(int*, int, int)>(f("cuDeviceGetAttribute"));
    auto getName = reinterpret_cast<CUresult (*)(char*, int, int)>(f("cuDeviceGetName"));
    int n = 0;
    if (!init || !count || !get || !getLuid || !attribute || !getName || init(0) || count(&n)) { if (error) *error = "CUDA not usable"; return 0; }
    for (int i = 0; i < n; ++i)
    {
        int d, major = 0, minor = 0;
        char l[8], name[128] = "";
        unsigned mask;
        if (get(&d, i) || getLuid(l, &mask, d) || memcmp(l, &luid, 8)) continue;
        attribute(&major, 75, d), attribute(&minor, 76, d), getName(name, sizeof(name), d);  // COMPUTE_CAPABILITY_MAJOR, _MINOR
        const int sm = major * 10 + minor;
        if (sm >= 80 || (sm == 75 && HasTensorCores(name))) return sm;
        if (error) *error = std::string(name) + " (sm_" + std::to_string(sm) + ") has no int8 tensor cores";
        return 0;
    }
    if (error) *error = "no CUDA device for this adapter";
    return 0;
}
// Linked once a process and kernel text (a batch's networks are the same size); nothing on disk.
std::vector<uint8_t> Cubin(const std::string& ptx, LUID luid, std::string* error)
{
    static std::mutex mutex;
    static std::map<std::pair<uint64_t, std::string>, std::vector<uint8_t>> made;
    uint64_t adapter;
    memcpy(&adapter, &luid, 8);
    std::lock_guard<std::mutex> lock(mutex);
    auto& c = made[{adapter, ptx}];
    if (c.empty()) c = Link(ptx, luid, error);
    return c;
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
// A layer's entry: weights, layer constants, the output pitch, then for each network of the launch (grid z: network x zt) its
// two sources and output. Sets %a1, %a2 (sources), %a3 (weights), %a4 (constants), %a5 (output), %r50 (pitch), %r40 (z within the network).
std::string EntryPtx(uint32_t index, int shared)
{
    char h[320];
    snprintf(h, sizeof(h), ".visible .entry layer%u(.param .u64 pw, .param .u64 pp, .param .u32 pitch, .param .align 8 .b8 io[%d])\n{\n", index, kCudaBatch * 24);
    std::string s = h;
    if (shared) snprintf(h, sizeof(h), "    .shared .align 16 .b8 s[%d];\n", shared), s += h;
    return s;
}
void ParamsPtx(Writer& w, int zt)
{
    w(".reg .u64 %%io<2>; .reg .u32 %%nz<2>;");
    w("mov.u32 %%nz0, %%ctaid.z; div.u32 %%nz1, %%nz0, %d; mul.lo.u32 %%r40, %%nz1, %d; sub.u32 %%r40, %%nz0, %%r40;", zt, zt);
    w("mov.u64 %%io0, io; mul.wide.u32 %%io1, %%nz1, 24; add.u64 %%io0, %%io0, %%io1;");
    w("ld.param.u64 %%a1, [%%io0]; ld.param.u64 %%a2, [%%io0 + 8]; ld.param.u64 %%a5, [%%io0 + 16]; ld.param.u64 %%a3, [pw]; ld.param.u64 %%a4, [pp]; ld.param.u32 %%r50, [pitch];");
}
}  // namespace

// The previous kernels (variant 1, timed against the tiled ones at the first start): a warp computes cols x 8 output pixels of a row,
// ns channel tiles, reading its inputs and weights from global memory.
static std::string RowPtx(uint32_t index, const ffx_ml::FusedConv& fc, const ffx_ml::LoweredGraph& g, bool external, int cols, int ns, int zt)
{
    const auto& c  = *fc.conv;
    const auto& r  = *fc.rescale;
    const auto& s0 = fc.sources[0];
    const auto& t0 = g.tensors[s0.tensor];
    const int   ic0 = t0.channelStride / 4, ic1 = fc.sources.size() > 1 ? g.tensors[fc.sources[1].tensor].channelStride / 4 : 0, icw = ic0 + ic1;
    const int   chunks = (icw + 3) / 4, oc = fc.outC, nt = (oc + 7) / 8, stride = c.stride[0], k = c.kernelH;
    const int   ocPad = (oc + 3) & ~3, outStride = g.tensors[fc.output].channelStride;
    Writer w;
    std::string head = EntryPtx(index, 0);
    w(".reg .u64 %%a<64>; .reg .u32 %%r<64>; .reg .s32 %%t<8>; .reg .s32 %%x<32>; .reg .s32 %%acc<64>; .reg .u32 %%f<8>; .reg .pred %%p<32>; .reg .pred %%pi<32>; .reg .pred %%pv<4>; .reg .pred %%po<4>; .reg .u64 %%ah<4>; .reg .u64 %%aj<4>; .reg .u64 %%ak<4>; .reg .u64 %%ao<4>; .reg .u32 %%fi<4>; .reg .s32 %%xo<4>; .reg .pred %%pt; .reg .s64 %%l<16>; .reg .s32 %%q<32>;");
    ParamsPtx(w, zt);
    w("mov.u32 %%r1, %%tid.x; mov.u32 %%r2, %%ctaid.x; mov.u32 %%r3, %%ctaid.y;");
    w("shr.u32 %%r4, %%r1, 5; mad.lo.u32 %%r5, %%r2, 4, %%r4; mul.lo.u32 %%r5, %%r5, %d;", cols * 8);         // first output column of this warp
    w("setp.ge.u32 %%p1, %%r5, %d; setp.ge.u32 %%p2, %%r3, %d; or.pred %%p1, %%p1, %%p2;", fc.outW, fc.outH);
    w("@%%p1 ret;");
    w("and.b32 %%r6, %%r1, 31; shr.u32 %%r7, %%r6, 2; and.b32 %%r8, %%r6, 3;");                   // fragment row and column
    for (int j = 0; j < cols; ++j) w("add.s32 %%xo%d, %%r5, %%r7; add.s32 %%xo%d, %%xo%d, %d;", j, j, j, j * 8);  // this lane's output columns
    w("mov.u32 %%f6, 0x%08x;", uint32_t(c.inputZp & 0xff) * 0x01010101u);
    for (int i = 0; i < ns * cols * 2; ++i) w("mov.s32 %%acc%d, 0;", i);
    w("cvt.u64.u32 %%a8, %%r8; shl.b64 %%a8, %%a8, 2;");                                          // fragment column in bytes
    w("cvt.u64.u32 %%a9, %%r7; shl.b64 %%a9, %%a9, 4; add.u64 %%a9, %%a9, %%a8; add.u64 %%a9, %%a3, %%a9;");  // weights: n (row) * 16 + column bytes
    w("mul.wide.u32 %%a40, %%r40, %u; add.u64 %%a9, %%a9, %%a40;", uint32_t(ns * k * k * chunks * 128));  // this warp's channel tiles
    if (icw % 4) w("setp.ge.u32 %%p9, %%r8, %d; min.u32 %%r9, %%r8, %d; cvt.u64.u32 %%a10, %%r9; shl.b64 %%a10, %%a10, 2;", icw % 4, icw % 4 - 1);
    // Input columns: both output columns, for every kx (the taps of a row share them).
    for (int kx = 0; kx < k; ++kx)
        for (int j = 0; j < cols; ++j)
        {
            w("mad.lo.s32 %%x3, %%xo%d, %d, %d;", j, stride, kx - c.pad[2]);
            w("setp.ge.s32 %%p3, %%x3, 0; setp.lt.s32 %%p4, %%x3, %d; and.pred %%pi%d, %%p3, %%p4;", fc.inW, kx * cols + j);
            w("max.s32 %%x3, %%x3, 0; min.s32 %%x3, %%x3, %d;", fc.inW - 1);
            if (s0.resize) { NearestPtx(w, "%x4", "%x3", s0.scale[2], s0.scale[3], s0.offset[1], t0.w); w("mov.s32 %%x3, %%x4;"); }
            w("mov.s32 %%q%d, %%x3;", kx * cols + j);
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
            for (int j = 0; j < cols; ++j)
            {
                w("add.s32 %%x7, %%x6, %%q%d;", kx * cols + j);
                w("cvt.u64.s32 %%ah%d, %%x7; mul.lo.u64 %%aj%d, %%ah%d, %d; add.u64 %%aj%d, %%a1, %%aj%d;", j, j, j, ic0 * 4, j, j);
                if (ic1) w("mul.lo.u64 %%ak%d, %%ah%d, %d; add.u64 %%ak%d, %%a2, %%ak%d;", j, j, ic1 * 4, j, j);
                w("and.pred %%pv%d, %%p5, %%pi%d;", j, kx * cols + j);
            }
            for (int ch = 0; ch < chunks; ++ch)
            {
                const bool one = ch * 4 < ic0;
                const int  base = (one ? ch * 4 : ch * 4 - ic0) * 4;
                const bool part = icw % 4 && ch == chunks - 1;  // the last chunk is not four whole words
                for (int j = 0; j < cols; ++j)
                {
                    w("add.u64 %%a20, %%%s%d, %%a%d; ld.global.u32 %%fi%d, [%%a20 + %d];", one ? "aj" : "ak", j, part ? 10 : 8, j, base);
                    w("selp.b32 %%fi%d, %%fi%d, %%f6, %%pv%d;", j, j, j);
                    if (part) w("selp.b32 %%fi%d, 0, %%fi%d, %%p9;", j, j);
                }
                for (int n = 0; n < ns; ++n)
                {
                    const uint32_t off = uint32_t(((n * k * k + ky * k + kx) * chunks + ch) * 128);
                    w("ld.global.u32 %%f2, [%%a9 + %u];", off);
                    // Two 8 x 8 x 16 products (tensor cores from sm_75, RTX 20): as fast here as sm_80's 16 x 8 x 16.
                    for (int j = 0; j < cols; ++j)
                        w("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%%acc%d, %%acc%d}, {%%fi%d}, {%%f2}, {%%acc%d, %%acc%d};", (n * cols + j) * 2, (n * cols + j) * 2 + 1, j,
                          (n * cols + j) * 2, (n * cols + j) * 2 + 1);
                }
            }
        }
    }
    // Each lane: channels 8 n + 2 col and + 1, of both columns.
    if (external) w("cvt.u64.u32 %%a30, %%r3; cvt.u64.u32 %%a31, %%r50; mul.lo.u64 %%a30, %%a30, %%a31; add.u64 %%a30, %%a5, %%a30;");
    else w("cvt.u64.u32 %%a30, %%r3; mul.lo.u64 %%a30, %%a30, %d; add.u64 %%a30, %%a5, %%a30;", fc.outW * outStride);
    for (int j = 0; j < cols; ++j)
    {
        w("setp.lt.s32 %%po%d, %%xo%d, %d;", j, j, fc.outW);
        w("cvt.u64.s32 %%ao%d, %%xo%d; mul.lo.u64 %%ao%d, %%ao%d, %d; add.u64 %%ao%d, %%a30, %%ao%d;", j, j, j, j, outStride, j, j);
    }
    w("shl.b32 %%r10, %%r8, 1; mad.lo.u32 %%r10, %%r40, %d, %%r10; cvt.u64.u32 %%a34, %%r10;", ns * 8);
    for (int n = 0; n < ns; ++n)
        for (int e = 0; e < 2; ++e)
        {
            const int ch = n * 8 + e;
            w("add.u32 %%r11, %%r10, %d; setp.lt.u32 %%p20, %%r11, %d;", ch, oc);
            w("min.u32 %%r12, %%r11, %d; cvt.u64.u32 %%a35, %%r12; shl.b64 %%a36, %%a35, 2; add.u64 %%a36, %%a4, %%a36;", oc - 1);
            w("ld.global.s32 %%x10, [%%a36];");
            if (r.perChannel) w("ld.global.s32 %%x11, [%%a36 + %d]; ld.global.u32 %%r13, [%%a36 + %d];", ocPad * 4, ocPad * 8);
            else w("ld.global.s32 %%x11, [%%a4 + %d]; ld.global.u32 %%r13, [%%a4 + %d];", ocPad * 4, ocPad * 8);
            w("sub.u32 %%r14, %%r13, 1; mov.s64 %%l1, 1; shl.b64 %%l1, %%l1, %%r14;");
            for (int j = 0; j < cols; ++j)
            {
                w("add.s32 %%x12, %%acc%d, %%x10; mul.wide.s32 %%l2, %%x12, %%x11; add.s64 %%l2, %%l2, %%l1; shr.s64 %%l2, %%l2, %%r13; cvt.s32.s64 %%x13, %%l2;", (n * cols + j) * 2 + e);
                w("add.s32 %%x13, %%x13, %d; max.s32 %%x13, %%x13, -128; min.s32 %%x13, %%x13, 127;", r.rescaleOutZp);
                if (fc.table) w("add.s32 %%x13, %%x13, 128; cvt.u64.s32 %%a37, %%x13; add.u64 %%a37, %%a4, %%a37; ld.global.s8 %%x13, [%%a37 + %d];", ocPad * 12);
                w("and.pred %%p21, %%p20, %%po%d; add.u64 %%a38, %%ao%d, %%a35;", j, j);
                w("@%%p21 st.global.u8 [%%a38], %%x13;");
            }
        }
    w("ret;");
    return head + w.s + "}\n";
}

// Variant 0, one layer: a block of 4 warps stages its input tile (the zero point outside the image; sources side by side, words a pixel as stored) and
// NT channel tiles' weights in shared memory, then each warp computes R rows x P groups of 8 output pixels by 8 x 8 x 16 products.
static std::string TiledPtx(uint32_t index, const ffx_ml::FusedConv& fc, const ffx_ml::LoweredGraph& g, bool external, const int R, const int P, const int NT, const bool k32, const int zt)
{
    const auto& c = *fc.conv;
    const auto& r = *fc.rescale;
    const int   S = c.stride[0], k = c.kernelH, taps = k * k, oc = fc.outC, ocPad = (oc + 3) & ~3, outStride = g.tensors[fc.output].channelStride;
    int         icw[2] = {}, words = 0;
    for (size_t s = 0; s < fc.sources.size(); ++s) words += icw[s] = g.tensors[fc.sources[s].tensor].channelStride / 4;
    const int chunks = (words + 3) / 4, PB = words * 4, TH = 4 * R, TW = 8 * P, IH = (TH - 1) * S + k, IW = (TW - 1) * S + k, F = R * P;
    const int inBytes = (IH * IW * PB + 16 + 15) & ~15, wBytes = NT * taps * chunks * 128;  // a fragment reads whole chunks: up to 12 bytes past the last pixel
    if (inBytes + wBytes > 48 * 1024) return {};  // static shared memory
    Writer    w;
    std::string head = EntryPtx(index, inBytes + wBytes);
    w(".reg .u64 %%a<16>; .reg .u32 %%r<64>; .reg .s32 %%t<8>; .reg .s32 %%x<16>; .reg .s32 %%acc<128>; .reg .u32 %%v<4>; .reg .u32 %%fa<32>; .reg .u32 %%fb<16>; .reg .pred %%p<8>; .reg .pred %%pt;");
    w(".reg .s64 %%l<4>; .reg .s32 %%bi<16>; .reg .s32 %%mu<16>; .reg .u32 %%sh<16>; .reg .s64 %%rn<16>;");
    ParamsPtx(w, zt);
    w("mov.u32 %%r1, %%tid.x; mov.u32 %%r2, %%ctaid.x; mov.u32 %%r3, %%ctaid.y; mov.u32 %%r60, s;");
    w("mul.lo.u32 %%r20, %%r2, %d; mul.lo.u32 %%r21, %%r3, %d;", TW, TH);  // the tile's first output column and row
    w("mad.lo.s32 %%r22, %%r20, %d, %d; mad.lo.s32 %%r23, %%r21, %d, %d;", S, -c.pad[2], S, -c.pad[0]);  // its first input column and row
    const uint32_t zp = uint32_t(c.inputZp & 0xff) * 0x01010101u;
    for (size_t si = 0; si < fc.sources.size(); ++si)
    {
        const auto& src = fc.sources[si];
        const auto& t = g.tensors[src.tensor];
        const bool  vec = icw[si] % 4 == 0;
        const int   ub = vec ? 16 : 4, U = icw[si] * 4 / ub, n = IH * IW;
        for (int it = 0; it * 128 < n; ++it)
        {
            w("add.u32 %%r5, %%r1, %d; setp.lt.u32 %%p1, %%r5, %d;", it * 128, n);  // tile pixel
            w("div.u32 %%r7, %%r5, %d; mul.lo.u32 %%r8, %%r7, %d; sub.u32 %%r8, %%r5, %%r8;", IW, IW);  // tile row, column
            w("add.s32 %%x1, %%r7, %%r23; add.s32 %%x2, %%r8, %%r22;");
            w("setp.ge.s32 %%p2, %%x1, 0; setp.lt.s32 %%p3, %%x1, %d; and.pred %%p2, %%p2, %%p3; setp.ge.s32 %%p3, %%x2, 0; and.pred %%p2, %%p2, %%p3; setp.lt.s32 %%p3, %%x2, %d; and.pred %%p2, %%p2, %%p3;",
              fc.inH, fc.inW);
            w("max.s32 %%x1, %%x1, 0; min.s32 %%x1, %%x1, %d; max.s32 %%x2, %%x2, 0; min.s32 %%x2, %%x2, %d;", fc.inH - 1, fc.inW - 1);
            if (src.resize)
            {
                NearestPtx(w, "%x3", "%x1", src.scale[0], src.scale[1], src.offset[0], t.h), w("mov.s32 %%x1, %%x3;");
                NearestPtx(w, "%x3", "%x2", src.scale[2], src.scale[3], src.offset[1], t.w), w("mov.s32 %%x2, %%x3;");
            }
            w("mad.lo.s32 %%x3, %%x1, %d, %%x2; mul.wide.s32 %%a6, %%x3, %d; add.u64 %%a6, %%a%d, %%a6;", t.w, icw[si] * 4, si ? 2 : 1);
            w("mad.lo.u32 %%r9, %%r5, %d, %%r60; and.pred %%p2, %%p2, %%p1;", PB);
            for (int u = 0; u < U; ++u)
            {
                const int o = (si ? icw[0] * 4 : 0) + u * ub;
                if (vec)
                {
                    w("mov.b32 %%v0, 0x%08x; mov.b32 %%v1, 0x%08x; mov.b32 %%v2, 0x%08x; mov.b32 %%v3, 0x%08x;", zp, zp, zp, zp);
                    w("@%%p2 ld.global.v4.u32 {%%v0, %%v1, %%v2, %%v3}, [%%a6 + %d]; @%%p1 st.shared.v4.u32 [%%r9 + %d], {%%v0, %%v1, %%v2, %%v3};", u * ub, o);
                }
                else w("mov.b32 %%v0, 0x%08x; @%%p2 ld.global.u32 %%v0, [%%a6 + %d]; @%%p1 st.shared.u32 [%%r9 + %d], %%v0;", zp, u * ub, o);
            }
        }
    }
    // This block's channel tiles of the weights, ((tile * taps + tap) * chunks + chunk) * 128 bytes, 8 output channels x 16 input channels.
    const int wTotal = (oc + 7) / 8 * taps * chunks * 8, wv = wBytes / 16;
    w("mul.lo.u32 %%r10, %%r40, %d;", wv);
    for (int it = 0; it * 128 < wv; ++it)
    {
        w("add.u32 %%r4, %%r1, %d; add.u32 %%r5, %%r4, %%r10; setp.lt.u32 %%p1, %%r4, %d; setp.lt.u32 %%p2, %%r5, %d; and.pred %%p2, %%p2, %%p1;", it * 128, wv, wTotal);
        w("mul.wide.u32 %%a6, %%r5, 16; add.u64 %%a6, %%a3, %%a6; @%%p2 ld.global.v4.u32 {%%v0, %%v1, %%v2, %%v3}, [%%a6];");
        w("mad.lo.u32 %%r9, %%r4, 16, %%r60; @%%p2 st.shared.v4.u32 [%%r9 + %d], {%%v0, %%v1, %%v2, %%v3};", inBytes);
    }
    w("bar.sync 0;");
    w("shr.u32 %%r11, %%r1, 5; and.b32 %%r12, %%r1, 31; shr.u32 %%r13, %%r12, 2; and.b32 %%r14, %%r12, 3;");  // warp, fragment row, fragment column
    w("mul.lo.u32 %%r15, %%r11, %d; mad.lo.u32 %%r15, %%r13, %d, %%r15; mul.lo.u32 %%r15, %%r15, %d; mad.lo.u32 %%r15, %%r14, 4, %%r15; add.u32 %%r15, %%r15, %%r60;", R * S * IW, S, PB);  // A
    w("mad.lo.u32 %%r16, %%r13, 16, %%r60; mad.lo.u32 %%r16, %%r14, 4, %%r16;");  // B
    for (int i = 0; i < F * NT * 2; ++i) w("mov.s32 %%acc%d, 0;", i);
    // K steps (tap, chunk); from sm_80 two at a time with two pixel groups (16 x 8 x 32 products), a missing second step with zero weights.
    const int steps = taps * chunks, ks = k32 ? 2 : 1;
    for (int s0 = 0; s0 < steps; s0 += ks)
    {
        for (int j = 0; j < ks; ++j)
        {
            const int st = std::min(s0 + j, steps - 1), tap = st / chunks, ch = st % chunks, ky = tap / k, kx = tap % k;
            for (int f = 0; f < F; ++f) w("ld.shared.u32 %%fa%d, [%%r15 + %d];", f + j * F, ((f / P * S + ky) * IW + f % P * 8 * S + kx) * PB + ch * 16);
            for (int n = 0; n < NT; ++n)
                if (s0 + j < steps) w("ld.shared.u32 %%fb%d, [%%r16 + %d];", n + j * NT, inBytes + ((n * taps + tap) * chunks + ch) * 128);
                else w("mov.b32 %%fb%d, 0;", n + j * NT);
        }
        for (int f = 0; f < F; f += ks)
            for (int n = 0; n < NT; ++n)
            {
                const int a = (f * NT + n) * 2, b = ((f + 1) * NT + n) * 2;
                if (k32)
                    w("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%%acc%d, %%acc%d, %%acc%d, %%acc%d}, {%%fa%d, %%fa%d, %%fa%d, %%fa%d}, {%%fb%d, %%fb%d}, {%%acc%d, %%acc%d, %%acc%d, %%acc%d};", a, a + 1, b,
                      b + 1, f, f + 1, f + F, f + 1 + F, n, n + NT, a, a + 1, b, b + 1);
                else w("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%%acc%d, %%acc%d}, {%%fa%d}, {%%fb%d}, {%%acc%d, %%acc%d};", a, a + 1, f, n, a, a + 1);
            }
    }
    // Each lane: channels 8 n + 2 column and + 1 of its pixel (fragment row) in every fragment. Shifts of 33 or more: from the high word
    // of the product alone, exactly ((x m + 2^(s-1)) >> s = (hi + 2^(s-33)) >> (s-32), the low word is below one step).
    bool high = true;
    for (int32_t s : r.shift) high = high && s >= 33;
    w(".reg .s32 %%ro<16>;");
    w("mul.lo.u32 %%r17, %%r40, %d; mad.lo.u32 %%r17, %%r14, 2, %%r17;", NT * 8);  // the lane's first channel
    for (int n = 0; n < NT; ++n)
        for (int e = 0; e < 2; ++e)
        {
            const int q = n * 2 + e;
            w("add.u32 %%r18, %%r17, %d; min.u32 %%r18, %%r18, %d; mul.wide.u32 %%a7, %%r18, 4; add.u64 %%a7, %%a4, %%a7;", n * 8 + e, oc - 1);
            w("ld.global.s32 %%bi%d, [%%a7];", q);
            if (r.perChannel) w("ld.global.s32 %%mu%d, [%%a7 + %d]; ld.global.u32 %%sh%d, [%%a7 + %d];", q, ocPad * 4, q, ocPad * 8);
            else w("ld.global.s32 %%mu%d, [%%a4 + %d]; ld.global.u32 %%sh%d, [%%a4 + %d];", q, ocPad * 4, q, ocPad * 8);
            if (high) w("sub.u32 %%sh%d, %%sh%d, 32; sub.u32 %%r18, %%sh%d, 1; shl.b32 %%ro%d, 1, %%r18;", q, q, q, q);  // the rounding term's high word
            else w("sub.u32 %%r18, %%sh%d, 1; mov.s64 %%rn%d, 1; shl.b64 %%rn%d, %%rn%d, %%r18;", q, q, q, q);
        }
    w("mad.lo.u32 %%r19, %%r11, %d, %%r21; add.u32 %%r24, %%r20, %%r13;", R);  // the warp's first row, the lane's first column
    for (int f = 0; f < F; ++f)
    {
        w("add.u32 %%r25, %%r19, %d; add.u32 %%r26, %%r24, %d;", f / P, f % P * 8);  // y, x
        w("setp.lt.u32 %%p4, %%r25, %d; setp.lt.u32 %%p5, %%r26, %d; and.pred %%p4, %%p4, %%p5;", fc.outH, fc.outW);
        if (external) w("mul.wide.u32 %%a8, %%r25, %%r50;");
        else w("mul.wide.u32 %%a8, %%r25, %d;", fc.outW * outStride);
        w("mul.wide.u32 %%a9, %%r26, %d; add.u64 %%a8, %%a8, %%a9; add.u64 %%a8, %%a5, %%a8; cvt.u64.u32 %%a9, %%r17; add.u64 %%a8, %%a8, %%a9;", outStride);
        for (int n = 0; n < NT; ++n)
            for (int e = 0; e < 2; ++e)
            {
                const int q = n * 2 + e;
                const int a = (f * NT + n) * 2 + e, v = 13 + e;
                if (high) w("add.s32 %%x12, %%acc%d, %%bi%d; mul.hi.s32 %%x%d, %%x12, %%mu%d; add.s32 %%x%d, %%x%d, %%ro%d; shr.s32 %%x%d, %%x%d, %%sh%d;", a, q, v, q, v, v, q, v, v, q);
                else w("add.s32 %%x12, %%acc%d, %%bi%d; mul.wide.s32 %%l2, %%x12, %%mu%d; add.s64 %%l2, %%l2, %%rn%d; shr.s64 %%l2, %%l2, %%sh%d; cvt.s32.s64 %%x%d, %%l2;", a, q, q, q, q, v);
                w("add.s32 %%x%d, %%x%d, %d; max.s32 %%x%d, %%x%d, -128; min.s32 %%x%d, %%x%d, 127;", v, v, r.rescaleOutZp, v, v, v, v);
                if (fc.table) w("add.s32 %%x%d, %%x%d, 128; cvt.u64.s32 %%a10, %%x%d; add.u64 %%a10, %%a4, %%a10; ld.global.s8 %%x%d, [%%a10 + %d];", v, v, v, v, ocPad * 12);
                if (oc % 2 == 0 && e == 0) continue;
                if (oc % 2) w("add.u32 %%r27, %%r17, %d; setp.lt.u32 %%p6, %%r27, %d; and.pred %%p6, %%p6, %%p4; @%%p6 st.global.u8 [%%a8 + %d], %%x%d;", n * 8 + e, oc, n * 8 + e, v);
                else w("prmt.b32 %%x13, %%x13, %%x14, 0x40; add.u32 %%r27, %%r17, %d; setp.lt.u32 %%p6, %%r27, %d; and.pred %%p6, %%p6, %%p4; @%%p6 st.global.u16 [%%a8 + %d], %%x13;", n * 8, oc, n * 8);  // both channels
            }
    }
    w("ret;");
    return head + w.s + "}\n";
}

bool MakeCudaPlan(const ffx_ml::GraphDesc& desc, LUID luid, int variant, CudaPlan& out, std::string* error)
{
    auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
    if (!Nv().cuda) return fail("no nvcuda.dll");
    ffx_ml::tosa::Graph  parsed;
    ffx_ml::LoweredGraph g;
    if (!ffx_ml::tosa::Parse(desc, parsed, error) || !ffx_ml::Lower(parsed, g, error)) return false;
    out.outputBindings = parsed.outputBindings;
    std::map<uint32_t, int32_t> slot;
    for (uint32_t i = 0; i < g.tensors.size(); ++i)
    {
        const auto& t = g.tensors[i];
        if (!t.external)
        {
            slot[i] = int32_t(out.tensors.size());
            out.tensors.push_back(uint64_t(t.h) * t.w * t.channelStride);
            continue;
        }
        const auto it = std::find(parsed.outputBindings.begin(), parsed.outputBindings.end(), t.binding);
        slot[i] = std::find(parsed.inputBindings.begin(), parsed.inputBindings.end(), t.binding) != parsed.inputBindings.end() ? -1 : -2 - int32_t(it - parsed.outputBindings.begin());
    }
    // Without tensor cores (GTX) dot4 uses the same hardware as CUDA would.
    const int sm = ComputeCapability(luid, error);
    if (!sm) return false;
    const bool  tiled = variant == 0, k32 = tiled && sm >= 80;  // 16 x 8 x 32 products from sm_80 (RTX 30)
    std::string ptx = k32 ? ".version 7.1\n.target sm_80\n.address_size 64\n" : ".version 7.1\n.target sm_75\n.address_size 64\n";
    std::vector<uint8_t>& constants = out.constants;
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
        CudaPlan::Layer l;
        l.dst = slot[fc.output], l.external = l.dst < -1;
        // Tiled, two channel tiles a block: 16 x 16 pixels at 100k pixels or more, 4 x 32 at stride 2 and for 1 x 1 kernels, else 8 x 32
        // (chosen on an RTX 4070). Rows: a layer of fewer than 20k pixels (the low-resolution end of the U-Net) is too small to fill the
        // GPU with 2 columns x 8 channel tiles per warp: 1 x 1 there gives 8x the warps.
        const bool big = fc.outW * fc.outH >= 100000 && c.stride[0] == 1 && c.kernelH > 1;
        int        nt = (fc.outC + 7) / 8, NT = std::min(nt, 2), R = big ? 4 : c.stride[0] == 2 || c.kernelH == 1 ? 1 : 2, P = big ? 2 : 4;
        const bool narrow = fc.outW * fc.outH < 20000;
        const int  lc = narrow ? 1 : 2, ln = std::min(narrow ? 1 : 8, nt);
        const int  zt = tiled ? (nt + NT - 1) / NT : (nt + ln - 1) / ln;  // grid z of one network
        const std::string kernel = tiled ? TiledPtx(uint32_t(li), fc, g, l.external, R, P, NT, k32, zt) : RowPtx(uint32_t(li), fc, g, l.external, lc, ln, zt);
        if (kernel.empty()) return fail("unsupported layer " + std::to_string(li) + " (shared memory)");
        ptx += kernel;
        const int chunks = (ic0 + ic1 + 3) / 4, taps = c.kernelH * c.kernelW, ic = fc.inC, pad = (fc.outC + 3) & ~3;
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
        if (tiled) l.grid[0] = uint32_t(fc.outW + P * 8 - 1) / (P * 8), l.grid[1] = uint32_t(fc.outH + R * 4 - 1) / (R * 4);
        else l.grid[0] = uint32_t(fc.outW + lc * 32 - 1) / (lc * 32), l.grid[1] = uint32_t(fc.outH);
        l.grid[2] = uint32_t(zt);
        out.layers.push_back(l);
    }
    out.cubin = Cubin(ptx, luid, error);
    return !out.cubin.empty();
}

std::unique_ptr<CudaGraph> CudaGraph::Create(ID3D12Device* device, const ffx_ml::GraphDesc& desc, std::string* error, int variant)
{
    auto fail = [&](const std::string& m) -> std::unique_ptr<CudaGraph> { if (error) *error = m; return nullptr; };
    if (!Nv().ok) return fail("no NVAPI CUDA launch");
    CudaPlan plan;
    if (!MakeCudaPlan(desc, device->GetAdapterLuid(), variant, plan, error)) return nullptr;
    auto out = std::unique_ptr<CudaGraph>(new CudaGraph());
    out->m_device = device;
    out->m_outputBindings = plan.outputBindings;
    auto buffer = [&](uint64_t size, D3D12_HEAP_TYPE heap) {
        D3D12_HEAP_PROPERTIES hp = {heap};
        D3D12_RESOURCE_DESC   d  = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, (size + 255) & ~255ull, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                    heap == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
        ComPtr<ID3D12Resource> r;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r));
        return r;
    };
    for (uint64_t s : plan.tensors)
    {
        out->m_tensors.push_back(buffer(s, D3D12_HEAP_TYPE_DEFAULT));
        if (!out->m_tensors.back()) return fail("out of memory");
    }
    if (Nv().createModule(device, plan.cubin.data(), uint32_t(plan.cubin.size()), &out->m_module)) return fail(error && error->rfind("linked", 0) == 0 ? *error + ", CreateCuModule" : std::string("CreateCuModule"));
    for (size_t i = 0; i < plan.layers.size(); ++i)
    {
        Layer l;
        static_cast<CudaPlan::Layer&>(l) = plan.layers[i];
        if (Nv().createFunction(device, out->m_module, ("layer" + std::to_string(i)).c_str(), &l.fn)) return fail("CreateCuFunction");
        out->m_layers.push_back(l);
    }
    out->m_constants = buffer(plan.constants.size(), D3D12_HEAP_TYPE_DEFAULT), out->m_upload = buffer(plan.constants.size(), D3D12_HEAP_TYPE_UPLOAD);
    void* map = nullptr;
    if (!out->m_constants || !out->m_upload || FAILED(out->m_upload->Map(0, nullptr, &map))) return fail("out of memory");
    memcpy(map, plan.constants.data(), plan.constants.size());
    out->m_upload->Unmap(0, nullptr);
    return out;
}
CudaGraph::~CudaGraph()
{
    if (Nv().destroyFunction) for (auto& l : m_layers) if (l.fn) Nv().destroyFunction(m_device.Get(), l.fn);
    if (Nv().destroyModule && m_module) Nv().destroyModule(m_device.Get(), m_module);
}

void CudaGraph::Record(ID3D12GraphicsCommandList* list, CudaGraph* const* graphs, const D3D12_GPU_VIRTUAL_ADDRESS* inputs, ID3D12Resource* const* const* outputs, uint32_t count, uint32_t outputPitch)
{
    auto transition = [](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
        D3D12_RESOURCE_BARRIER x = {};
        x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
        return x;
    };
    std::vector<D3D12_RESOURCE_BARRIER> b;
    for (uint32_t i = 0; i < count; ++i)
    {
        CudaGraph& g = *graphs[i];
        if (!g.m_uploaded)
        {
            list->CopyResource(g.m_constants.Get(), g.m_upload.Get());
            b.push_back(transition(g.m_constants.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            g.m_uploaded = true;
        }
        for (auto& t : g.m_tensors) b.push_back(transition(t.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    }
    list->ResourceBarrier(UINT(b.size()), b.data());
    std::vector<bool> dirty(graphs[0]->m_tensors.size(), false);
    // The networks of a batch are one model: graph 0's constants for all.
    const uint32_t        batches = (count + kCudaBatch - 1) / kCudaBatch;
    std::vector<CudaParams> params(batches);
    std::vector<Launch>     launches(batches);
    for (size_t li = 0; li < graphs[0]->m_layers.size(); ++li)
    {
        b.clear();
        const Layer& first = graphs[0]->m_layers[li];
        for (int32_t t : {first.src[0], first.src[1], first.dst})
            if (t >= 0 && dirty[t])
            {
                dirty[t] = false;
                for (uint32_t i = 0; i < count; ++i)
                {
                    D3D12_RESOURCE_BARRIER u = {D3D12_RESOURCE_BARRIER_TYPE_UAV};
                    u.UAV.pResource = graphs[i]->m_tensors[t].Get();
                    b.push_back(u);
                }
            }
        if (!b.empty()) list->ResourceBarrier(UINT(b.size()), b.data());
        const uint64_t base = graphs[0]->m_constants->GetGPUVirtualAddress();
        for (uint32_t j = 0; j < batches; ++j)
        {
            const uint32_t n = std::min<uint32_t>(kCudaBatch, count - j * kCudaBatch);
            params[j] = {base + first.weights, base + first.params, outputPitch};
            for (uint32_t k = 0; k < n; ++k)
            {
                const uint32_t i = j * kCudaBatch + k;
                CudaGraph& g = *graphs[i];
                auto va = [&](int32_t t) { return t >= 0 ? g.m_tensors[t]->GetGPUVirtualAddress() : t == -1 ? inputs[i] : outputs[i][-2 - t]->GetGPUVirtualAddress(); };
                params[j].io[k][0] = va(first.src[0]), params[j].io[k][1] = va(first.src[1]), params[j].io[k][2] = va(first.dst);
            }
            launches[j] = {first.fn, {first.grid[0], first.grid[1], first.grid[2] * n}, {128, 1, 1}, 0, &params[j], uint32_t(sizeof(CudaParams))};
        }
        // Not as one chain: the launches of a chain run one after another.
        for (auto& l : launches) Nv().launch(list, &l, 1);
        if (first.dst >= 0) dirty[first.dst] = true;
        if (s_afterLayer) s_afterLayer(list, li);
    }
    b.clear();
    for (uint32_t i = 0; i < count; ++i)
        for (auto& t : graphs[i]->m_tensors) b.push_back(transition(t.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON));
    list->ResourceBarrier(UINT(b.size()), b.data());
}
}  // namespace nfrux
