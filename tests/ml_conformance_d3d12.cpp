// A1 on D3D12: the kernels against the CPU reference, byte for byte. SPDX-License-Identifier: MIT
#include "../core/ml/d3d12_network.h"
#include "../core/ml/cuda_network.h"
#include "../core/model.h"

#include <dxgi1_6.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

using Microsoft::WRL::ComPtr;

static ComPtr<ID3D12Resource> Buf(ID3D12Device* d, uint64_t size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES s)
{
    D3D12_HEAP_PROPERTIES hp = {heap};
    D3D12_RESOURCE_DESC   r  = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, (size + 255) & ~255ull, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags};
    ComPtr<ID3D12Resource> out;
    d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, s, nullptr, IID_PPV_ARGS(&out));
    return out;
}

int main(int argc, char** argv)
{
    for (auto [name, yes] : std::initializer_list<std::pair<const char*, bool>>{
             {"NVIDIA GeForce RTX 2060", true}, {"NVIDIA GeForce RTX 2080 SUPER", true}, {"Quadro RTX 4000", true}, {"NVIDIA TITAN RTX", true}, {"Tesla T4", true}, {"T4G", true}, {"NVIDIA T4", true},
             {"NVIDIA GeForce GTX 1650", false}, {"NVIDIA GeForce GTX 1660 Ti", false}, {"NVIDIA T400", false}, {"NVIDIA T400 4GB", false}, {"NVIDIA T600", false}, {"NVIDIA T1000", false},
             {"NVIDIA T1200 Laptop GPU", false}, {"NVIDIA GeForce MX450", false}})
        if (nfrux::HasTensorCores(name) != yes) return printf("tensor cores by name: %s wrong\n", name), 1;
    puts("tensor cores by name: 15 names right");
    const bool timing = argc > 1 && !strcmp(argv[1], "time");
    ComPtr<ID3D12Debug> dbg;
    if (!timing && SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg))))
        dbg->EnableDebugLayer();
    ComPtr<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev))))
        return puts("no device"), 1;
    ComPtr<ID3D12InfoQueue> iq;
    dev.As(&iq);
    ComPtr<ID3D12CommandQueue> q;
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_COMPUTE};
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
    ComPtr<ID3D12CommandAllocator> al;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&al));
    ComPtr<ID3D12GraphicsCommandList> cl;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, al.Get(), nullptr, IID_PPV_ARGS(&cl));
    cl->Close();
    ComPtr<ID3D12Fence> fence;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    uint64_t fv = 0;
    auto run = [&] { cl->Close(); ID3D12CommandList* l = cl.Get(); q->ExecuteCommandLists(1, &l); q->Signal(fence.Get(), ++fv); fence->SetEventOnCompletion(fv, nullptr); };
    D3D12_QUERY_HEAP_DESC qh = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2};
    ComPtr<ID3D12QueryHeap> heap;
    dev->CreateQueryHeap(&qh, IID_PPV_ARGS(&heap));

    const auto b = nfrux::GetModelBindings();
    if (argc > 1 && !strcmp(argv[1], "layers"))  // CUDA GPU time by layer (timestamps between layers), 1 and 5 networks interleaved
    {
        const uint32_t w = argc > 2 ? atoi(argv[2]) : 992, h = argc > 3 ? atoi(argv[3]) : 544;
        static ComPtr<ID3D12QueryHeap> lh;
        D3D12_QUERY_HEAP_DESC lq = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 64};
        dev->CreateQueryHeap(&lq, IID_PPV_ARGS(&lh));
        nfrux::CudaGraph::s_afterLayer = [](ID3D12GraphicsCommandList* l, size_t i) {
            D3D12_RESOURCE_BARRIER u = {D3D12_RESOURCE_BARRIER_TYPE_UAV};
            l->ResourceBarrier(1, &u);
            l->EndQuery(lh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, UINT(i + 1));
        };
        std::string err;
        std::unique_ptr<nfrux::CudaGraph> gs[5];
        for (auto& g : gs) if (!(g = nfrux::CudaGraph::Create(dev.Get(), nfrux::DescribeModel(w, h), &err, argc > 4 ? atoi(argv[4]) : 0))) return printf("CUDA: %s\n", err.c_str()), 1;
        const uint64_t inSize = uint64_t(w) * h * nfrux::kInputChannels, outSize = uint64_t(w / 2) * (h / 2) * 4;
        ComPtr<ID3D12Resource> in[5];  // separate inputs, as the timesteps have
        for (auto& x : in) x = Buf(dev.Get(), inSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto rb = Buf(dev.Get(), 64 * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        ComPtr<ID3D12Resource> o[10];
        for (auto& x : o) x = Buf(dev.Get(), outSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        uint64_t f;
        q->GetTimestampFrequency(&f);
        for (uint32_t count : {1u, 5u})
        {
            nfrux::CudaGraph* cg[5];
            D3D12_GPU_VIRTUAL_ADDRESS ins[5];
            ID3D12Resource* outs[5][2];
            ID3D12Resource* const* op[5];
            for (int i = 0; i < 5; ++i) cg[i] = gs[i].get(), ins[i] = in[i]->GetGPUVirtualAddress(), outs[i][0] = o[2 * i].Get(), outs[i][1] = o[2 * i + 1].Get(), op[i] = outs[i];
            const size_t n = gs[0] ? 22 : 0;
            std::vector<std::vector<double>> t(n + 1);
            for (int rep = 0; rep < 200; ++rep)
            {
                al->Reset(); cl->Reset(al.Get(), nullptr);
                for (int k = 0; k < 4; ++k)  // the last one's stamps are kept
                {
                    if (k == 3) { D3D12_RESOURCE_BARRIER u = {D3D12_RESOURCE_BARRIER_TYPE_UAV}; cl->ResourceBarrier(1, &u); cl->EndQuery(lh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0); }
                    nfrux::CudaGraph::Record(cl.Get(), cg, ins, op, count, w / 2 * 4);
                }
                cl->ResolveQueryData(lh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, UINT(n + 1), rb.Get(), 0);
                run();
                uint64_t* s; rb->Map(0, nullptr, reinterpret_cast<void**>(&s));
                if (rep >= 20) for (size_t i = 0; i < n; ++i) t[i].push_back(double(s[i + 1] - s[i]) * 1000 / double(f));
                rb->Unmap(0, nullptr);
            }
            double total = 0;
            printf("%ux%u, %u network(s), ms (median of 180, a barrier before each stamp):\n", w, h, count);
            for (size_t i = 0; i < n; ++i)
            {
                std::sort(t[i].begin(), t[i].end());
                total += t[i][t[i].size() / 2];
                printf("  layer %2zu %.4f\n", i, t[i][t[i].size() / 2]);
            }
            printf("  sum %.4f\n", total);
        }
        nfrux::CudaGraph::s_afterLayer = nullptr;
        return 0;
    }
    int bad = 0;
    const uint32_t sizes[][2] = {{64, 64}, {744, 464}, {960, 544}};
    for (auto& wh : sizes)
        for (int dot4 = 2 + nfrux::kCudaVariants - 1; dot4 >= 0; --dot4)  // 2 and up: CUDA, variant dot4 - 2
        {
            const uint32_t w = wh[0], h = wh[1];
            const int before = bad;
            const ffx_ml::GraphDesc desc = nfrux::DescribeModel(w, h);
            std::string err;
            std::unique_ptr<nfrux::D3D12Graph> g;
            std::unique_ptr<nfrux::CudaGraph> cu;
            const ULONGLONG t0 = GetTickCount64();
            if (dot4 >= 2) { if (!(cu = nfrux::CudaGraph::Create(dev.Get(), desc, &err, dot4 - 2))) { printf("%ux%u CUDA: not available: %s\n", w, h, err.c_str()); continue; } printf("CUDA created in %llu ms\n", GetTickCount64() - t0); }
            else g = nfrux::D3D12Graph::Create(dev.Get(), desc, dot4 != 0, &err);
            if (!g && !cu) { printf("%ux%u dot4=%d: create failed: %s\n", w, h, dot4, err.c_str()); return 1; }
            const uint64_t inSize = uint64_t(w) * h * nfrux::kInputChannels, outSize = uint64_t(w / 2) * (h / 2) * 4;
            auto up = Buf(dev.Get(), inSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            auto in = Buf(dev.Get(), inSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            ComPtr<ID3D12Resource> o[2], rb;
            for (auto& x : o) x = Buf(dev.Get(), outSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            rb = Buf(dev.Get(), outSize * 2 + 512, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            const int cases = timing ? 60 : (w == 64 ? 23 : 5);
            std::vector<double> ms;
            for (int cs = 0; cs < cases; ++cs)
            {
                std::vector<int8_t> x(inSize);
                std::mt19937 rng(cs * 7 + 1);
                for (size_t i = 0; i < x.size(); ++i)
                    x[i] = cs == 0 ? -128 : cs == 1 ? 127 : cs == 2 ? (i & 1 ? 127 : -128) : int8_t(rng());
                void* m; up->Map(0, nullptr, &m); memcpy(m, x.data(), x.size()); up->Unmap(0, nullptr);
                al->Reset(); cl->Reset(al.Get(), nullptr);
                cl->CopyResource(in.Get(), up.Get());
                D3D12_RESOURCE_BARRIER t[3] = {};
                t[0].Transition = {in.Get(), 0, D3D12_RESOURCE_STATE_COPY_DEST, cu ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
                for (int i = 0; i < 2; ++i) t[1 + i].Transition = {o[i].Get(), 0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
                cl->ResourceBarrier(3, t);
                cl->EndQuery(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                ID3D12Resource* outs[2] = {o[0].Get(), o[1].Get()};
                for (int rep = timing ? 39 : 0; rep > 0; --rep)  // keeps the clocks up; the time is per network
                {
                    if (cu) cu->Record(cl.Get(), in->GetGPUVirtualAddress(), outs, w / 2 * 4); else g->Record(cl.Get(), in->GetGPUVirtualAddress(), outs);
                    D3D12_RESOURCE_BARRIER u[2] = {{D3D12_RESOURCE_BARRIER_TYPE_UAV}, {D3D12_RESOURCE_BARRIER_TYPE_UAV}};
                    u[0].UAV.pResource = outs[0], u[1].UAV.pResource = outs[1];
                    cl->ResourceBarrier(2, u);
                }
                if (cu) cu->Record(cl.Get(), in->GetGPUVirtualAddress(), outs, w / 2 * 4);
                else g->Record(cl.Get(), in->GetGPUVirtualAddress(), outs);
                cl->EndQuery(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                for (int i = 0; i < 2; ++i) t[i].Transition = {o[i].Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
                cl->ResourceBarrier(2, t);
                for (int i = 0; i < 2; ++i) cl->CopyBufferRegion(rb.Get(), outSize * i, o[i].Get(), 0, outSize);
                cl->ResolveQueryData(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, rb.Get(), (outSize * 2 + 255) & ~255ull);
                run();
                uint8_t* r; rb->Map(0, nullptr, reinterpret_cast<void**>(&r));
                uint64_t ts[2], f; memcpy(ts, r + ((outSize * 2 + 255) & ~255ull), 16); q->GetTimestampFrequency(&f);
                ms.push_back(double(ts[1] - ts[0]) * 1000 / double(f) / (timing ? 40 : 1));
                if (!timing)
                {
                    std::map<uint32_t, std::vector<int8_t>> ri{{b.input, x}}, ro;
                    if (!ffx_ml::RunReference(desc, ri, ro, &err)) { printf("reference: %s\n", err.c_str()); return 1; }
                    size_t diff = 0, total = 0;
                    for (int i = 0; i < 2; ++i)
                    {
                        const auto& ref = ro[cu ? cu->outputBinding(i) : g->outputBinding(i)];
                        const size_t px = size_t(w / 2) * (h / 2), ch = ref.size() / px;
                        for (size_t p = 0; p < px; ++p)
                            for (size_t c = 0; c < ch; ++c)
                                ++total, diff += ref[p * ch + c] != int8_t(r[outSize * i + p * 4 + c]);
                    }
                    if (diff) printf("%ux%u dot4=%d case %d: %zu of %zu bytes differ\n", w, h, dot4, cs, diff, total), ++bad;
                }
                rb->Unmap(0, nullptr);
            }
            std::sort(ms.begin(), ms.end());
            printf("%ux%u %s: %d cases, %s, median %.3f ms\n", w, h, dot4 == 3 ? "CUDA rows" : dot4 == 2 ? "CUDA tiled" : dot4 ? "dot4" : "int32", cases, timing ? "timed" : bad > before ? "DIFFERENCES" : "exact", ms[ms.size() / 2]);
        }
    uint64_t msgs = iq ? iq->GetNumStoredMessages() : 0;
    for (uint64_t i = 0; i < msgs && i < 8; ++i)
    {
        size_t n = 0; iq->GetMessage(i, nullptr, &n); std::vector<char> v(n); auto* mm = reinterpret_cast<D3D12_MESSAGE*>(v.data()); iq->GetMessage(i, mm, &n);
        printf("debug layer: %s\n", mm->pDescription);
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}