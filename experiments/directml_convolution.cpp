// M0d: DirectML's integer convolution (int8 in, int32 sums) for one of the network's layers: exact, and how fast?
// SPDX-License-Identifier: MIT
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#define DML_TARGET_VERSION 0x3000
#include <DirectML.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using Microsoft::WRL::ComPtr;
#define HR(x) do { const HRESULT h_ = (x); if (FAILED(h_)) { printf("%s = %08lx\n", #x, h_); return 1; } } while (0)
static ComPtr<ID3D12Resource> Buf(ID3D12Device* d, uint64_t size, D3D12_HEAP_TYPE heap, bool uav)
{
    D3D12_HEAP_PROPERTIES hp = {heap};
    D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, (size + 255) & ~255ull, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
    ComPtr<ID3D12Resource> o;
    d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : heap == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&o));
    return o;
}
int main(int argc, char** argv)
{
    const UINT W = argc > 2 ? atoi(argv[1]) : 372, H = argc > 2 ? atoi(argv[2]) : 232, C = 16, K = 3, OC = 16;
    ComPtr<ID3D12Device> d;
    HR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d)));
    ComPtr<ID3D12CommandQueue> q;
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_COMPUTE};
    HR(d->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    ComPtr<ID3D12CommandAllocator> al;
    HR(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&al)));
    ComPtr<ID3D12GraphicsCommandList> cl;
    HR(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, al.Get(), nullptr, IID_PPV_ARGS(&cl)));
    ComPtr<ID3D12Fence> fe;
    HR(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fe)));
    uint64_t fv = 0;
    auto run = [&] { cl->Close(); ID3D12CommandList* l = cl.Get(); q->ExecuteCommandLists(1, &l), q->Signal(fe.Get(), ++fv), fe->SetEventOnCompletion(fv, nullptr), al->Reset(), cl->Reset(al.Get(), nullptr); };
    ComPtr<IDMLDevice> dml;
    HR(DMLCreateDevice1(d.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_2_0, IID_PPV_ARGS(&dml)));
    auto align = [](uint64_t n) { return (n + 3) & ~3ull; };
    const uint64_t inB = uint64_t(W) * H * C, wB = OC * C * K * K, outB = uint64_t(W) * H * OC * 4;
    std::vector<int8_t> x(inB), w(wB);
    std::mt19937 rng(1);
    for (auto& v : x) v = int8_t(rng());
    for (auto& v : w) v = int8_t(rng());
    for (int nhwc = 0; nhwc < 2; ++nhwc)
        for (int owned = 0; owned < 2; ++owned)  // weights owned by DirectML: it may lay them out for its metacommand
        {
            const UINT si[4] = {1, C, H, W}, sw[4] = {OC, C, K, K}, so[4] = {1, OC, H, W}, sti[4] = {H * W * C, 1, W * C, C}, sto[4] = {H * W * OC, 1, W * OC, OC};
            DML_BUFFER_TENSOR_DESC bi = {DML_TENSOR_DATA_TYPE_INT8, DML_TENSOR_FLAG_NONE, 4, si, nhwc ? sti : nullptr, align(inB)};
            DML_BUFFER_TENSOR_DESC bw = {DML_TENSOR_DATA_TYPE_INT8, owned ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE, 4, sw, nullptr, align(wB)};
            DML_BUFFER_TENSOR_DESC bo = {DML_TENSOR_DATA_TYPE_INT32, DML_TENSOR_FLAG_NONE, 4, so, nhwc ? sto : nullptr, outB};
            DML_TENSOR_DESC ti = {DML_TENSOR_TYPE_BUFFER, &bi}, tw = {DML_TENSOR_TYPE_BUFFER, &bw}, to = {DML_TENSOR_TYPE_BUFFER, &bo};
            const UINT one[2] = {1, 1};
            DML_CONVOLUTION_INTEGER_OPERATOR_DESC cd = {&ti, nullptr, &tw, nullptr, &to, 2, one, one, one, one, 1};
            DML_OPERATOR_DESC od = {DML_OPERATOR_CONVOLUTION_INTEGER, &cd};
            ComPtr<IDMLOperator> op;
            HR(dml->CreateOperator(&od, IID_PPV_ARGS(&op)));
            ComPtr<IDMLCompiledOperator> co;
            HR(dml->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&co)));
            ComPtr<IDMLOperatorInitializer> init;
            IDMLCompiledOperator* cop = co.Get();
            HR(dml->CreateOperatorInitializer(1, &cop, IID_PPV_ARGS(&init)));
            const auto pi = init->GetBindingProperties(), pe = co->GetBindingProperties();
            ComPtr<ID3D12DescriptorHeap> heap;
            D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, pi.RequiredDescriptorCount + pe.RequiredDescriptorCount, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
            HR(d->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
            const UINT inc = d->GetDescriptorHandleIncrementSize(hd.Type);
            auto temp = Buf(d.Get(), std::max<uint64_t>(4, std::max(pi.TemporaryResourceSize, pe.TemporaryResourceSize)), D3D12_HEAP_TYPE_DEFAULT, true);
            auto pers = Buf(d.Get(), std::max<uint64_t>(4, pe.PersistentResourceSize), D3D12_HEAP_TYPE_DEFAULT, true);
            auto in = Buf(d.Get(), inB, D3D12_HEAP_TYPE_DEFAULT, true), wt = Buf(d.Get(), wB, D3D12_HEAP_TYPE_DEFAULT, true), out = Buf(d.Get(), outB, D3D12_HEAP_TYPE_DEFAULT, true);
            auto up = Buf(d.Get(), inB + wB + 512, D3D12_HEAP_TYPE_UPLOAD, false), rb = Buf(d.Get(), outB + 256, D3D12_HEAP_TYPE_READBACK, false);
            uint8_t* m;
            up->Map(0, nullptr, reinterpret_cast<void**>(&m));
            const uint64_t wOff = (inB + 255) & ~255ull;
            for (uint64_t i = 0; i < inB; ++i)
            {
                const UINT c = UINT(i % C), px = UINT(i / C);  // x is NHWC
                m[nhwc ? i : uint64_t(c) * W * H + px] = uint8_t(x[i]);
            }
            memcpy(m + wOff, w.data(), wB), up->Unmap(0, nullptr);
            cl->CopyBufferRegion(in.Get(), 0, up.Get(), 0, inB), cl->CopyBufferRegion(wt.Get(), 0, up.Get(), wOff, wB);
            run();
            ComPtr<IDMLCommandRecorder> rec;
            HR(dml->CreateCommandRecorder(IID_PPV_ARGS(&rec)));
            DML_BINDING_TABLE_DESC bt = {init.Get(), heap->GetCPUDescriptorHandleForHeapStart(), heap->GetGPUDescriptorHandleForHeapStart(), pi.RequiredDescriptorCount};
            ComPtr<IDMLBindingTable> ti2, te;
            if (pi.RequiredDescriptorCount == 0) bt.SizeInDescriptors = 1;
            HR(dml->CreateBindingTable(&bt, IID_PPV_ARGS(&ti2)));
            DML_BUFFER_BINDING bIn = {in.Get(), 0, align(inB)}, bW = {wt.Get(), 0, align(wB)}, bOut = {out.Get(), 0, outB}, bP = {pers.Get(), 0, pe.PersistentResourceSize}, none = {};
            DML_BUFFER_BINDING ia[5] = {none, none, owned ? bW : none, none, none};
            DML_BUFFER_ARRAY_BINDING arr = {4, ia};
            DML_BINDING_DESC bda = {DML_BINDING_TYPE_BUFFER_ARRAY, &arr}, bdp = {DML_BINDING_TYPE_BUFFER, &bP};
            if (owned) ti2->BindInputs(1, &bda);
            if (pe.PersistentResourceSize) ti2->BindOutputs(1, &bdp);
            DML_BUFFER_BINDING bT = {temp.Get(), 0, std::max<uint64_t>(4, pi.TemporaryResourceSize)};
            DML_BINDING_DESC bdt = {DML_BINDING_TYPE_BUFFER, &bT};
            if (pi.TemporaryResourceSize) ti2->BindTemporaryResource(&bdt);
            ID3D12DescriptorHeap* hp = heap.Get();
            cl->SetDescriptorHeaps(1, &hp);
            rec->RecordDispatch(cl.Get(), init.Get(), ti2.Get());
            run();
            bt = {co.Get(), {heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(inc) * pi.RequiredDescriptorCount}, {heap->GetGPUDescriptorHandleForHeapStart().ptr + UINT64(inc) * pi.RequiredDescriptorCount}, pe.RequiredDescriptorCount};
            HR(dml->CreateBindingTable(&bt, IID_PPV_ARGS(&te)));
            DML_BINDING_DESC bd[4] = {{DML_BINDING_TYPE_BUFFER, &bIn}, {DML_BINDING_TYPE_NONE}, {owned ? DML_BINDING_TYPE_NONE : DML_BINDING_TYPE_BUFFER, owned ? nullptr : &bW}, {DML_BINDING_TYPE_NONE}};
            te->BindInputs(4, bd);
            DML_BINDING_DESC bdo = {DML_BINDING_TYPE_BUFFER, &bOut};
            te->BindOutputs(1, &bdo);
            bT.SizeInBytes = std::max<uint64_t>(4, pe.TemporaryResourceSize);
            if (pe.TemporaryResourceSize) te->BindTemporaryResource(&bdt);
            if (pe.PersistentResourceSize) te->BindPersistentResource(&bdp);
            ComPtr<ID3D12QueryHeap> qh;
            D3D12_QUERY_HEAP_DESC qhd = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2};
            HR(d->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qh)));
            auto qr = Buf(d.Get(), 16, D3D12_HEAP_TYPE_READBACK, false);
            double best = 1e9;
            for (int r = 0; r < 20; ++r)
            {
                cl->SetDescriptorHeaps(1, &hp);
                cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                for (int j = 0; j < 200; ++j)
                {
                    rec->RecordDispatch(cl.Get(), co.Get(), te.Get());
                    D3D12_RESOURCE_BARRIER u = {D3D12_RESOURCE_BARRIER_TYPE_UAV};
                    cl->ResourceBarrier(1, &u);
                }
                cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                cl->ResolveQueryData(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, qr.Get(), 0);
                run();
                uint64_t* t, f;
                qr->Map(0, nullptr, reinterpret_cast<void**>(&t)), q->GetTimestampFrequency(&f);
                best = std::min(best, double(t[1] - t[0]) * 1e6 / double(f) / 200), qr->Unmap(0, nullptr);
            }
            D3D12_RESOURCE_BARRIER b = {};
            b.Transition = {out.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
            cl->ResourceBarrier(1, &b), cl->CopyBufferRegion(rb.Get(), 0, out.Get(), 0, outB);
            run();
            const int32_t* o;
            rb->Map(0, nullptr, reinterpret_cast<void**>(const_cast<int32_t**>(&o)));
            size_t bad = 0;
            for (UINT y = 0; y < H; ++y)
                for (UINT xx = 0; xx < W; ++xx)
                    for (UINT c = 0; c < OC; ++c)
                    {
                        int32_t s = 0;
                        for (UINT ky = 0; ky < K; ++ky)
                            for (UINT kx = 0; kx < K; ++kx)
                            {
                                const int py = int(y + ky) - 1, px = int(xx + kx) - 1;
                                if (py < 0 || px < 0 || py >= int(H) || px >= int(W)) continue;
                                for (UINT i = 0; i < C; ++i) s += int32_t(x[(size_t(py) * W + px) * C + i]) * w[((c * C + i) * K + ky) * K + kx];
                            }
                        bad += s != o[nhwc ? (size_t(y) * W + xx) * OC + c : (size_t(c) * H + y) * W + xx];
                    }
            rb->Unmap(0, nullptr);
            printf("%ux%ux%u 3x3 -> %u, %s, weights %s: %.1f us, %zu of %llu sums wrong\n", W, H, C, OC, nhwc ? "NHWC" : "NCHW", owned ? "owned" : "plain", best, bad, outB / 4);
        }
    return 0;
}