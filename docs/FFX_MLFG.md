# AMD FSR Frame Generation 4 (ML) — extraction notes

Status: extraction done, port not started (2026-11-26).

## Source and licence

- FidelityFX SDK v2.4.0 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag `v2.4.0`, 2026-10-28), FSR Frame Generation 4.0.2.
- `Kits/FidelityFX/docs/license.md`: line 1 gives a default binary-only licence that forbids reverse engineering, but the
  list that starts at "These source files" (line 37) and ends at "are subject to the following license" (line 983) puts
  `Kits\FidelityFX\signedbin\amd_fidelityfx_framegeneration_dx12.dll` and `..._vk.dll` (lines 650-651) under the MIT
  licence, which allows use, copying and modification. We rely on that listing. The DLLs are not stored in this repo;
  `tools/ffx_extract/fetch.ps1` downloads them from the tag. Keep AMD's copyright and the MIT notice with anything
  derived from them (`licenses/`).

## Tools (`tools/ffx_extract/`)

| File | What it does |
|---|---|
| `fetch.ps1` | Downloads both DLLs from the tag, runs `scan.py` on each, disassembles the SPIR-V with `spirv-dis`, runs `catalogue.py`. |
| `scan.py` | Carves every DXBC/DXIL container and every SPIR-V module out of a DLL (sizes from the container header / by walking the instruction stream); writes PE sections, name-like strings and an entropy map of the rest. |
| `catalogue.py` | One line per Vulkan module: entry point, local size, cooperative-matrix multiply-adds and loads, capabilities, bindings; counts D3D12 containers that use AMD driver intrinsics. |
| `common_data.py` | Looks for byte runs shared by the two DLLs outside their shaders (found none: the weight blocks differ per API). |

## Findings

| | D3D12 DLL (29.7 MB) | Vulkan DLL (36.8 MB) |
|---|---|---|
| Shaders | 487 DXIL containers, 17.6 MB | 487 SPIR-V modules (472 unique), 25.0 MB |
| ML passes | same graph | 369: `mlfi_4k_pass0..184` and `mlfi_8k_pass0..183` (two resolution tiers) |
| Other shaders | classic FSR 3 frame interpolation / optical flow / UI composition | same (`CS`, `main`, `mainVS`/`mainPS`) |
| Weights | fp16 block, about 10 MB, `.rdata` 0xdf4800-0x17b4800 | fp16 block, about 10 MB, `.rdata` 0xe18400-0x17b8400 |
| Vendor dependence | 156 containers use the AMD driver-intrinsic UAV (register space 2147420894); they run only on AMD drivers | standard extensions only: `SPV_KHR_cooperative_matrix`, `SPV_NV_cooperative_matrix2`, `SPV_EXT_float8`, Int8, Float16 |

- The network is a compiled graph: about 185 passes per tier, alternating cooperative-matrix passes (convolutions /
  matmuls; `InitializerBuffer` = weights, `ScratchBuffer` = activations, `CsTensorSizes` = shapes) with element-wise
  passes. Pass 0 samples `r_input_rgb0` / `r_input_rgb1`. FP8 cooperative matrices are declared.
- Roughly 5 M parameters (10 MB of fp16), about 40x NFRUX's 124 KB network. AMD quotes 2.2 ms at 4K on an RX 9070 XT,
  410 MB of memory at 4K (360 MB aliasable).
- Inputs (from the API docs): two colour frames, depth, motion vectors, and camera position / up / right / forward
  (`ffxDispatchDescFrameGenerationPrepareV2`). DLSS-G gives us camera matrices, from which these follow.
- AMD documents the feature for RX 9000 GPUs only. The RTX 4070 (driver 610.88) exposes every Vulkan feature the
  SPIR-V needs (`VK_EXT_shader_float8` with `shaderFloat8CooperativeMatrix`, `VK_NV_cooperative_matrix2`,
  `VK_KHR_cooperative_matrix`).

## What a port needs

1. The dispatch graph: pass order, workgroup counts, `CsTensorSizes` contents, scratch offsets, and where each weight
   tensor sits in the fp16 block. This lives in the DLL's host code. Plan: drive the Vulkan DLL headlessly through the
   FFX API on the RTX 4070 and record every dispatch and buffer upload with a small Vulkan layer; fall back to static
   analysis of the host code if the DLL refuses non-AMD GPUs.
2. Vulkan: replay AMD's SPIR-V passes in NFRUX with the recorded graph (runs on any GPU with the features above).
3. D3D12: AMD's DXIL cannot run off AMD drivers, and the design requires matching output on both APIs. Each pass
   type has to be re-implemented in NFRUX's own kernels (CUDA, coopmat/dot4/scalar), with AMD's SPIR-V on Vulkan as the
   reference for exact comparison.
4. Quality and cost comparison against NFRUX v15/v16 on the same test sets (LPIPS, PSNR, crop sheets), and timing on the
   RTX 4070.
