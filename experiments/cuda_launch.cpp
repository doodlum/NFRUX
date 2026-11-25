// M0b: a CUDA kernel inside a D3D12 compute list, without a CUDA toolkit. PTX -> cubin by the driver's nvcuda.dll (cuLink*), NVAPI launches it.
// Two kernels: out[i] = in[i] * 3 + 1 (does it run, do barriers work), and an int8 m16n8k16 tensor-core product (does mma run and is it exact).
// SPDX-License-Identifier: MIT
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

static const char* kPtx = R"(
.version 7.4
.target sm_80
.address_size 64
.visible .entry muladd(.param .u64 pin, .param .u64 pout, .param .u32 n)
{
    .reg .u64 %rd<8>; .reg .u32 %r<12>; .reg .pred %p<2>;
    ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout]; ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra END;
    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd1, %rd3; add.u64 %rd5, %rd2, %rd3;
    ld.global.u32 %r6, [%rd4];
    mad.lo.u32 %r7, %r6, 3, 1;
    st.global.u32 [%rd5], %r7;
END:
    ret;
}
// One warp: A 16 x 16 int8 (row major), B 8 x 16 int8 (B transposed: row = column of B), C 16 x 8 int32 = A . B.
.visible .entry mma(.param .u64 pa, .param .u64 pb, .param .u64 pc)
{
    .reg .u64 %rd<24>; .reg .u32 %r<64>; .reg .s32 %c<8>; .reg .pred %p<4>;
    ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pc];
    mov.u32 %r1, %tid.x;
    // lane = row * 4 + col: A fragments a0..a1 hold bytes [4 col, 4 col + 4) of rows (row) and (row + 8)
    shr.u32 %r2, %r1, 2; and.b32 %r3, %r1, 3;
    mul.lo.u32 %r4, %r2, 16; mad.lo.u32 %r4, %r3, 4, %r4;
    cvt.u64.u32 %rd4, %r4; add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r10, [%rd5]; ld.global.u32 %r11, [%rd5 + 128];
    // m16n8k16 s8: a = { a0 (row, k = 4 col), a1 (row + 8, same k) }, b = { b0 (k = 4 row', n = col' * ...), ... }
    ld.global.u32 %r12, [%rd5];
    // B: lane holds column n = row (0..7), k bytes [4 col, 4 col + 4)
    cvt.u64.u32 %rd6, %r4; add.u64 %rd7, %rd2, %rd6;
    ld.global.u32 %r13, [%rd7];
    mov.s32 %c1, 0; mov.s32 %c2, 0; mov.s32 %c3, 0; mov.s32 %c4, 0;
    mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%c1, %c2, %c3, %c4}, {%r10, %r11}, {%r13}, {%c1, %c2, %c3, %c4};
    // d0 = (row, 2 col), d1 = (row, 2 col + 1), d2 = (row + 8, 2 col), d3 = (row + 8, 2 col + 1)
    mul.lo.u32 %r20, %r2, 8; mad.lo.u32 %r20, %r3, 2, %r20; shl.b32 %r20, %r20, 2;
    cvt.u64.u32 %rd8, %r20; add.u64 %rd9, %rd3, %rd8;
    st.global.s32 [%rd9], %c1; st.global.s32 [%rd9 + 4], %c2; st.global.s32 [%rd9 + 256], %c3; st.global.s32 [%rd9 + 260], %c4;
    ret;
}
)";

using CUresult = int; using CUdevice = int; using CUcontext = void*; using CUlinkState = void*;
template <class F> static F Fn(HMODULE m, const char* n) { return reinterpret_cast<F>(GetProcAddress(m, n)); }

static std::vector<uint8_t> Link(const char* ptx)
{
    HMODULE m = LoadLibraryA("nvcuda.dll");
    if (!m) { puts("no nvcuda.dll"); return {}; }
    auto init = Fn<CUresult (*)(unsigned)>(m, "cuInit");
    auto get = Fn<CUresult (*)(CUdevice*, int)>(m, "cuDeviceGet");
    auto create = Fn<CUresult (*)(CUcontext*, unsigned, CUdevice)>(m, "cuCtxCreate_v2");
    auto destroy = Fn<CUresult (*)(CUcontext)>(m, "cuCtxDestroy_v2");
    auto linkCreate = Fn<CUresult (*)(unsigned, int*, void**, CUlinkState*)>(m, "cuLinkCreate_v2");
    auto linkAdd = Fn<CUresult (*)(CUlinkState, int, void*, size_t, const char*, unsigned, int*, void**)>(m, "cuLinkAddData_v2");
    auto linkComplete = Fn<CUresult (*)(CUlinkState, void**, size_t*)>(m, "cuLinkComplete");
    CUdevice d; CUcontext c;
    if (init(0) || get(&d, 0) || create(&c, 0, d)) { puts("cuda context failed"); return {}; }
    static char err[4096];
    int opts[] = {5, 6};  // CU_JIT_ERROR_LOG_BUFFER, ..._SIZE_BYTES
    void* vals[] = {err, reinterpret_cast<void*>(size_t(sizeof(err)))};
    CUlinkState s;
    std::vector<uint8_t> out;
    CUresult r = linkCreate(2, opts, vals, &s);
    if (!r) r = linkAdd(s, 1 /*PTX*/, const_cast<char*>(ptx), strlen(ptx) + 1, "nfrux", 0, nullptr, nullptr);
    void* p; size_t n;
    if (!r) r = linkComplete(s, &p, &n);
    if (r) printf("link failed %d: %s\n", r, err);
    else out.assign(static_cast<uint8_t*>(p), static_cast<uint8_t*>(p) + n);
    destroy(c);
    return out;
}

struct Dim { uint32_t x, y, z; };
struct Launch { void* fn; Dim grid, block; uint32_t shared; const void* params; uint32_t size; };

int main()
{
    const auto cubin = Link(kPtx);
    printf("cubin %zu bytes\n", cubin.size());
    if (cubin.empty()) return 1;
    auto query = Fn<void* (*)(uint32_t)>(LoadLibraryA("nvapi64.dll"), "nvapi_QueryInterface");
    auto nvInit = reinterpret_cast<int (*)()>(query(0x0150e828));
    auto createModule = reinterpret_cast<int (*)(ID3D12Device*, const void*, uint32_t, void**)>(query(0xad1a677d));
    auto createFunction = reinterpret_cast<int (*)(ID3D12Device*, void*, const char*, void**)>(query(0xe2436e22));
    auto launch = reinterpret_cast<int (*)(ID3D12GraphicsCommandList*, const Launch*, uint32_t)>(query(0x24973538));
    printf("nvapi init %d, entry points %p %p %p\n", nvInit ? nvInit() : -1, createModule, createFunction, launch);
    ComPtr<ID3D12Device> d;
    D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d));
    void *mod = nullptr, *f1 = nullptr, *f2 = nullptr;
    printf("module %d\n", createModule(d.Get(), cubin.data(), uint32_t(cubin.size()), &mod));
    printf("functions %d %d\n", createFunction(d.Get(), mod, "muladd", &f1), createFunction(d.Get(), mod, "mma", &f2));
    if (!f1 || !f2) return 1;
    ComPtr<ID3D12CommandQueue> q;
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_COMPUTE};
    d->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
    ComPtr<ID3D12CommandAllocator> al; ComPtr<ID3D12GraphicsCommandList> cl; ComPtr<ID3D12Fence> fence;
    d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&al));
    d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, al.Get(), nullptr, IID_PPV_ARGS(&cl));
    d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    auto buf = [&](uint64_t n, D3D12_HEAP_TYPE t, bool uav, D3D12_RESOURCE_STATES s) {
        D3D12_HEAP_PROPERTIES hp = {t};
        D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, n, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
        ComPtr<ID3D12Resource> o;
        d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, s, nullptr, IID_PPV_ARGS(&o));
        return o;
    };
    const uint32_t n = 4096;
    auto up = buf(65536, D3D12_HEAP_TYPE_UPLOAD, false, D3D12_RESOURCE_STATE_GENERIC_READ), in = buf(65536, D3D12_HEAP_TYPE_DEFAULT, true, D3D12_RESOURCE_STATE_COMMON);
    auto mid = buf(65536, D3D12_HEAP_TYPE_DEFAULT, true, D3D12_RESOURCE_STATE_COMMON), out = buf(65536, D3D12_HEAP_TYPE_DEFAULT, true, D3D12_RESOURCE_STATE_COMMON), rb = buf(65536, D3D12_HEAP_TYPE_READBACK, false, D3D12_RESOURCE_STATE_COPY_DEST);
    std::vector<uint32_t> x(n);
    int8_t A[16][16], B[8][16];
    std::mt19937 rng(1);
    for (auto& v : x) v = rng();
    for (auto& r : A) for (auto& v : r) v = int8_t(rng());
    for (auto& r : B) for (auto& v : r) v = int8_t(rng());
    uint8_t* p;
    up->Map(0, nullptr, reinterpret_cast<void**>(&p));
    memcpy(p, x.data(), n * 4), memcpy(p + 32768, A, 256), memcpy(p + 33024, B, 128), up->Unmap(0, nullptr);
    cl->CopyResource(in.Get(), up.Get());
    auto tr = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) { D3D12_RESOURCE_BARRIER t = {}; t.Transition = {r, 0, a, b}; cl->ResourceBarrier(1, &t); };
    auto uav = [&](ID3D12Resource* r) { D3D12_RESOURCE_BARRIER t = {D3D12_RESOURCE_BARRIER_TYPE_UAV}; t.UAV.pResource = r; cl->ResourceBarrier(1, &t); };
    tr(in.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    tr(mid.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS), tr(out.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
#pragma pack(push, 8)
    struct P1 { uint64_t a, b; uint32_t n; } p1 = {in->GetGPUVirtualAddress(), mid->GetGPUVirtualAddress(), n}, p2 = {mid->GetGPUVirtualAddress(), out->GetGPUVirtualAddress(), n};
    struct P3 { uint64_t a, b, c; } p3 = {in->GetGPUVirtualAddress() + 32768, in->GetGPUVirtualAddress() + 33024, out->GetGPUVirtualAddress() + 32768};
#pragma pack(pop)
    Launch l1 = {f1, {64, 1, 1}, {64, 1, 1}, 0, &p1, 20}, l2 = {f1, {64, 1, 1}, {64, 1, 1}, 0, &p2, 20}, l3 = {f2, {1, 1, 1}, {32, 1, 1}, 0, &p3, 24};
    printf("launch %d", launch(cl.Get(), &l1, 1));
    uav(mid.Get());
    printf(" %d", launch(cl.Get(), &l2, 1));
    printf(" %d\n", launch(cl.Get(), &l3, 1));
    tr(out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->CopyResource(rb.Get(), out.Get());
    const HRESULT hc = cl->Close();
    ID3D12CommandList* ls = cl.Get();
    q->ExecuteCommandLists(1, &ls), q->Signal(fence.Get(), 1);
    for (int i = 0; i < 5000 && fence->GetCompletedValue() < 1; ++i) Sleep(1);
    printf("close %lx, completed %llu, removed %lx\n", (unsigned long)hc, fence->GetCompletedValue(), (unsigned long)d->GetDeviceRemovedReason());
    rb->Map(0, nullptr, reinterpret_cast<void**>(&p));
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) { uint32_t v; memcpy(&v, p + i * 4, 4); bad += v != (x[i] * 3 + 1) * 3 + 1; }
    uint32_t badM = 0;
    for (int m = 0; m < 16; ++m)
        for (int k = 0; k < 8; ++k)
        {
            int s = 0, v;
            for (int i = 0; i < 16; ++i) s += A[m][i] * B[k][i];
            memcpy(&v, p + 32768 + (m * 8 + k) * 4, 4);
            badM += v != s;
        }
    printf("chained kernels: %u of %u wrong; tensor core int8 product: %u of 128 wrong\n", bad, n, badM);
    return bad || badM;
}
