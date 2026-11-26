# AMD FSR Frame Generation 4 (ML) — extraction notes and NFRUX's port

Status: extraction done; ported as NFRUX's second interpolator (`Model=amd`), section "The port" (2026-11-26).

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

## The port (core/mlfg, Model=amd)

The network is re-implemented, not replayed: AMD's SPIR-V / DXIL are not used. Spec: `tools/ffx_extract/mlfg_net.py`
(`AmdMlfgNet`, precision `'amd'`, sampler `'point'` = AMD's nearest sampler, `flow_t='scale'`).

### Architecture

| Part | Files | What |
|---|---|---|
| Weights | `tools/mlfg_weights.py` -> `core/model/amd_mlfg_weights.h` (10.6 MB blob as `uint64_t`, 25 MB source) | the 105 layers read through `mlfg_net.load_amd` (passes 186 / 188 in its orientation); fp16 weights, fp8 e4m3 bytes for the stage-2 FasterNet layers, fp32 biases. `--bin FILE` writes the same blob as a file: nfrux.ini `[General] AmdWeights=path` (or `NFRUX_AMD_WEIGHTS`) loads a retrained npz of the same shape; the layer list is checked |
| Hand-written passes | `core/shaders/mlfg/*.hlsl` (cf2393b) | prepass, glue, stage-2 input, softmax, composite; `kMlfgFlowOut` gained `c_scale` (flows times 2t / 2(1-t)), `kMlfgDilate` the jitter cancellation in `c_mvScale.zw` (0 = AMD); composite and stage-2 input do their uniform divisions once a group (same bits; `tests/mlfg_passes.py` 68/68) |
| Network runtime | `core/mlfg/mlfg_network.*` | encoders, 4 flow blocks, stage 2 over NHWC fp16 tensors; a layer = conv kxk/s or transposed 2x2/2 with bias, optional second source (the FasterNet concatenation), add (residual / skip), LeakyReLU 0.2 / ReLU, fp16 or fp8 rounding of the result (torch's e4m3: round to nearest even, above 448 NaN) |
| Interpolator | `core/mlfg/mlfg_interpolator.cpp` | `CreateMlfgInterpolator`, behind `Interpolator` (now an interface; NFRU is the other implementation) |
| UI rules | `core/shaders/mlfg/mlfg_finish.hlsl` | NFRUX's DLSS-G rules (no-warp, HUD-less difference test, UI alpha / layer, 2x2 blocks) and the BGRA8 byte order over the composite, as the NFRU composite applies them; alpha 0 as AMD's composite |

Per rendered frame: dilation of the current vectors (kept as the next frame's history), the splat of frame 1's vectors
onto frame 0, frame 0's forward motion from the camera (`prevPstoPs` = VP1 inv(VP0), `prevPsToPrev2Ps` = VP-1 inv(VP0),
column-major numbers = AMD's row-vector rows; VP as the front end chains it), then stage 1 once at t = 0.5. Per generated
frame (t = i / (n + 1)): flow export scaled by 2t and 2(1-t), splats 0->t and 1->t, stage-2 input, stage 2, softmax,
composite, finish. Stage-2 tensors are shared by the generated frames (they run one after another). Reset or the first
frame: only the history is made (the front end shows the frame). The int8 network's self-check (D7) is not run for
Model=amd; the tiers below have crash markers only (one attempt counted a process; cleared after 120 frames or at an orderly
end after a frame, as the int8 runtime's).

### Tiers and how they are chosen

| Tier | API | Kernels | Chosen |
|---|---|---|---|
| generic | both | `mlfg_conv.hlsl` (kMlfgConv3/2/1/T + small-tile variants `s`): fp16 values in fp32 registers, fp32 sums per 8-channel chunk and tap, strict like every kMlfg pass; 64 threads make 16x16 (16x8) pixels x 8 channels, 4 / 2 pixels a thread; layers that would give under 768 groups use 4x4-pixel tiles with the input channels in 4 slices (fixed-order reduction) | the floor; NFRUX_MLFG=generic or AmdNetwork=generic |
| coopmat | Vulkan | `mlfg_conv_coopmat.comp` (kMlfgMat*, per subgroup size 16 / 32 / 64): one subgroup a group, 16 pixels x 64 channels (16 for layers with at most 16 outputs), fp16 16x16x16 -> fp32, A from a shared fp16 patch (1x1: straight from the tensor), epilogue as generic | Vulkan with fp16 16x16x16 cooperative matrices, 16-bit storage, the memory model and a 16/32/64 subgroup (rhi `Caps.coopMatF16`, `subgroupSize`); crash marker off NVIDIA (cleared after 120 frames) |
| cuda | D3D12 | `mlfg_cuda.cpp`: PTX written here (sm_75, `mma.sync.m16n8k8` f16 -> f32), linked by the driver, launched by NVAPI into the host's list as the int8 network's; a warp makes 16 pixels x 64 channels (8 for <= 16 outputs), weights in fragment order, epilogue with `cvt.rn.f16` and a scaled `cvt.rni` for e4m3 | D3D12 on NVIDIA; crash marker `d3d12_mlfg_cuda_<device>` |

`NFRUX_MLFG` (tests) forces a tier strictly; nfrux.ini `[D3D12]` / `[Vulkan]` `AmdNetwork` forces it with fallback to
the automatic choice (logged). The log says `FSR FG4 interpolator WxH ..., network: <tier>`. Not done: CUDA on Vulkan
(VK_NVX_binary_import; on NVIDIA Vulkan the coopmat tier runs the same tensor cores), FP8 tensor cores for stage 2,
fused FasterNet blocks.

### Accuracy (tests/mlfg_net.py, mlfg_run.exe)

A test-set sample (`test_new` ph_moon_0_000 p1=30, 1280x720, render 960x540, real depth / vectors / matrices) through the
DLL's code against mlfg_net run on the CPU (fp32, no TF32), outputs quantised as the composite does. The network is very
sensitive to summation order (fp16 / fp8 roundings after every layer): mlfg_net with fp32 sums against mlfg_net with
fp64 sums is the yardstick, 50.2 dB at t = 0.5 (49.8 / 50.2 at t = 0.25 / 0.75), max 53-84 codes.

| | vs mlfg_net fp32 (2x / 4x worst) | vs mlfg_net fp64 sums | vs generic (same API) |
|---|---|---|---|
| generic D3D12 and Vulkan | 50.3 / 50.2 dB, max 46-53 codes | 49.8-50.1 dB | - |
| coopmat Vulkan | 49.5 / 49.3 dB, max 48-66 | 49.6-49.8 dB | 49.6-49.8 dB, max 57 |
| cuda D3D12 | 50.0 / 49.6 dB, max 41-65 | 49.9-50.7 dB | 49.9-50.6 dB, max 80 |

Every tier is as close to the fp64-summed network as torch's fp32 is (pass: within 3 dB of the yardstick); 58/58 checks.
Tensors after the last frame (net_in, f54 .. f149, flowout, s2in, softmax weights) are reported per tier as well.

### Parity

- Generic tier, D3D12 against Vulkan: identical bytes (mlfg_net.py 2x and 4x; `parity --strict` with NFRUX_MODEL=amd,
  NFRUX_MLFG=generic in the strict build: 1080p 4x RGBA8, 720p BGRA8 / RGB10A2 / RGBA16F with inverted depth, 1366x767
  render 683x384 at 6x: 0 differing bytes in every output and intermediate; the shipped build's kMlfg passes are strict
  too, ngx_harness: 63 of 63 outputs byte-identical between the APIs).
- coopmat and cuda run on one API each, so there is no same-kernel comparison across APIs; across APIs the automatic
  choice (cuda on D3D12, coopmat on Vulkan) differs by fp32 summation order: 56-80 dB.

### ngx_harness with Model=amd (both APIs, the shipped DLL)

77 of 78 checks pass on each API (UI rules, subrects, distortion, protocol, formats, menu, reset, release, soak parts).
Fails, explained:
- "RGB10A2 PQ, dB" 38.2 / 38.3 (D3D12 cuda / Vulkan coopmat; generic 38.3) against the 40 the test sets (NFRU: 46.8): AMD's network on that synthetic HDR scene (a moving
  square over a bright fine background in PQ); a quality result, not a defect (its outputs match mlfg_net's).
- Between the APIs ("both" run): two outputs (depth subrect) at 56.3 / 57.1 dB under the 60 dB A2b line, because D3D12
  runs cuda and Vulkan coopmat. With NFRUX_MLFG=generic on both: 158 checks, only the two PQ failures, and every output
  byte-identical between the APIs.
Quality on the harness's synthetic pan: 44.3 dB against the truth (NFRU 57.7).

### Cost (RTX 4070, `parity --bench`: 16 frames a submission, median of 30, ms a rendered frame)

Measured with another process (a training job) on the GPU; NFRU measured at the same time gave 0.87 / 0.89 ms at 1080p
2x against 0.80 idle, so these are about 10% high.

| | 1080p 2x | 1080p 4x | 4K 2x | 4K 4x |
|---|---|---|---|---|
| cuda (D3D12) | 5.0 | 9.5 | 13.2 | 29.4 |
| coopmat (Vulkan) | 6.2 | 11.8 | 16.4 | 37.6 |
| generic D3D12 | 27.7 | 59.1 | 100.7 | - |
| generic Vulkan | 19.0 | 40.2 | 61.4 | - |

Stage 1 and the prepass are about half of the 1080p 2x time; each further generated frame costs about 2.3 ms (cuda) at
1080p and 8 ms at 4K. Vulkan timestamps at 4K (coopmat) put the 1x1 FasterNet convolutions first, then the 3x3 layers.
AMD quotes 2.2 ms at 4K on an RX 9070 XT (FP8, fused kernels): the fast tiers here are 6x slower, so Model=amd is for
quality comparison, not the default.