// Prints the D3D12 features NFRUX's rungs depend on. SPDX-License-Identifier: MIT
#include <initializer_list>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <wrl/client.h>
#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxguid.lib")
using Microsoft::WRL::ComPtr;
int main(){ ComPtr<ID3D12Device> d; D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d));
 for(int m : {0x6a,0x69,0x68,0x67,0x66,0x65}){ D3D12_FEATURE_DATA_SHADER_MODEL s={(D3D_SHADER_MODEL)m}; HRESULT h=d->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,&s,sizeof s); printf("SM query %x -> hr %lx highest %x\n",m,(unsigned long)h,(unsigned)s.HighestShaderModel); if(SUCCEEDED(h)) break; }
 struct { UINT a[16]; } x={}; for(int f=60; f<80; ++f){ for(UINT n=4;n<=48;n+=4){ x={}; if(SUCCEEDED(d->CheckFeatureSupport((D3D12_FEATURE)f,&x,n))) { printf("feature %d size %u:",f,n); for(UINT i=0;i<n/4;i++) printf(" %u",x.a[i]); puts(""); break; } } }
 D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1={}; d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1,&o1,sizeof o1); printf("int64 %d wave %u-%u\n",o1.Int64ShaderOps,o1.WaveLaneCountMin,o1.WaveLaneCountMax);
 ComPtr<ID3D12Device5> d5; d.As(&d5); if(d5){ UINT n=0; d5->EnumerateMetaCommands(&n,nullptr); D3D12_META_COMMAND_DESC*m=new D3D12_META_COMMAND_DESC[n]; d5->EnumerateMetaCommands(&n,m); printf("%u metacommands\n",n); for(UINT i=0;i<n;i++) printf("  %ls\n",m[i].Name); } 
 ComPtr<ID3D12Debug> g; printf("debug layer hr %lx\n",(unsigned long)D3D12GetDebugInterface(IID_PPV_ARGS(&g))); }