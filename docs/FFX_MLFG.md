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
| Weights | `tools/mlfg_weights.py` -> `core/model/amd_mlfg_weights.xz` (the 10.6 MB blob, byte planes + xz: 9.0 MB, in the repository); the build expands it into `amd_mlfg_weights.h` in the build tree (`tools/embed_weights.py`, standard library) | the 105 layers read through `mlfg_net.load_amd` (passes 186 / 188 in its orientation); fp16 weights, fp8 e4m3 bytes for the stage-2 FasterNet layers, fp32 biases. `--bin FILE` writes the same blob as a file: nfrux.ini `[General] AmdWeights=path` (or `NFRUX_AMD_WEIGHTS`) loads a retrained npz of the same shape; the layer list is checked |
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
| cuda | both | `mlfg_cuda.cpp`: PTX written here (sm_75, `mma.sync.m16n8k8` f16 -> f32; sm_89: FP8 fused FasterNet), linked by the driver, launched by NVAPI (D3D12) or VK_NVX_binary_import (Vulkan) into the host's list as the int8 network's; a warp makes 16-64 pixels x 8-64 channels, weights in fragment order, epilogue with `cvt.rn.f16` and a scaled `cvt.rni` for e4m3; the same kernels give identical bytes on both APIs | NVIDIA RTX; crash marker `d3d12_mlfg_cuda_<device>` / `vulkan_mlfg_cuda_<device>`, coopmat (Vulkan) as the fallback |

`NFRUX_MLFG` (tests) forces a tier strictly; nfrux.ini `[D3D12]` / `[Vulkan]` `AmdNetwork` forces it with fallback to
the automatic choice (logged). The log says `FSR FG4 interpolator WxH ..., network: <tier>`. (The speed sections below say what changed since: CUDA on Vulkan, FP8, fused blocks, fp16 accumulation.)

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
A second sample (`test_bal` tb_moon_000 p1=120): yardstick 44.5-45.4 dB; generic 45.4-45.5, coopmat 44.9-45.6, cuda 44.4-45.3 dB
against mlfg_net, 58/58 (`--sample test_bal:1`).
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
### Speed work (2026-11-26, second phase)

Same RTX 4070, `parity --bench` (16 frames a submission, median of 30, ms a rendered frame), a training job on the GPU
throughout (NFRU measured alongside: 0.84-0.88 ms at 1080p 2x against 0.80 idle, so about 5-10% high). Per-pass GPU
times: NFRUX_TEST_TIMES=1 (each network layer now has its own label; D3D12 timestamps work: the frequency is asked of a
queue of the device's own).

| Tier | 1080p 2x | 1080p 4x | 4K 2x | 4K 4x |
|---|---|---|---|---|
| cuda (D3D12), before | 5.0 | 9.5 | 13.2 | 29.4 |
| cuda, after | **3.96** | **6.15** | **8.29** | **16.3** |
| coopmat (Vulkan), before | 6.2 | 11.8 | 16.4 | 37.6 |
| coopmat, after | 6.2 | 11.5 | 15.9 | 35.7 |
| generic D3D12, before / after | 27.7 / 21.6 | 59.1 / 43.1 | 100.7 / 67.1 | - |
| generic Vulkan, before / after | 19.0 / 15.8 | 40.2 / 31.8 | 61.4 / 47.8 | - |

What changed, by payoff on the cuda tier (4K 2x):
1. Fused FasterNet blocks (CUDA): one warp does pconv -> expand -> project of 16 pixels in registers (an mma
   accumulator fragment is exactly the next product's A fragment), bit-identical to the three layers; stage 2's 13
   blocks 3.4 -> 0.8 ms. Coopmat and generic have fused kernels too (`mlfg_fn_coopmat.comp`, `mlfg_conv_fn.hlsl`; both
   checked, the coopmat one bit-identical to its layers) but they are slower than the layers on the RTX 4070 (coopmat
   +1.0 ms at 4K: one-subgroup groups and shared-memory round trips; generic D3D12 +7.7 ms at 1080p: register
   pressure), so they are off unless NFRUX_MLFG_FUSED=1.
2. Glue: an exact fast fp16 rounding for fp16-normal results (in the fp32 encoding; every kMlfg pass, 320 M random
   inputs checked against numpy), unrolled per-channel loops (no indexed temporaries), and the composite / stage-2
   input take their uniform quotients from the host instead of a long division per pixel (a first try that divided once
   a group and shared the result cost 0.6 ms in barriers): composite 2.3 -> 1.2 ms, refine 0.8 -> 0.6 ms; still
   bit-exact (`mlfg_passes.py` 68/68).
3. Convolution tiling (CUDA): 32 pixels a warp where that still gives 256 warps (each B fragment serves two products):
   the big 3x3 layers 0.10 -> 0.06 ms; 4 warps a block. Measured and not kept: one n tile a warp for low-resolution
   layers (more warps, but A loads per product rise: stage 1 2.8 -> 5.0 ms) and n tiles that pad nothing (6 for 96
   outputs: slower, the warps want 8 independent accumulators).
4. FP8 tensor cores (sm_89+): the fused FasterNet blocks as mma.m16n8k32 e4m3 -> fp32 (k order permuted so accumulator
   fragments stay A fragments; FasterNet weights packed again as e4m3 B fragments): 0.6 -> 0.46 ms for the eight
   48-channel blocks, -0.25 ms a generated frame at 4K. Accuracy drops by about 1.5 dB (48.3-49.2 dB against the
   fp64-summed net, 44.4-44.7 on the second sample; yardstick 50.2 / 45.4 minus 3: passes): Ada's e4m3 units sum more
   coarsely than its fp16 ones. NFRUX_MLFG_NO_FP8=1 keeps fp16. Not done on Vulkan (VK_EXT_shader_float8 / NV
   cooperative matrix 2: the KHR coopmat tier is not the fast one on NVIDIA anyway).
5. Generic tier: staging by 16-byte loads (8 channels a load) and, in the shipped build only, multiply-add fusion of
   the convolutions' sums (the strict build keeps them strict: still 0 differing bytes between D3D12 and Vulkan).
6. Weights: no more 25 MB generated header in git (see Architecture).

Where 4K 2x goes now (cuda, 8.3 ms; per-pass sum 7.2 ms): stage 1 layers 2.3 (latency-bound low-resolution layers:
block 1 / 2 at 30 / 60 pixels wide give 54-144 warps), composite 1.2, stage-2 network 0.9 + 3x3 input / head 0.55,
refine 0.6, splats 0.43, stage-2 input 0.32, forward motion 0.24, the rest 0.5. A generated frame costs 4.0 ms at 4K
(composite, stage 2, splats, stage-2 input).

How close to AMD: AMD quotes 2.2 ms at 4K (2x) on an RX 9070 XT. Its dense FP8 rate is about 3.3x the RTX 4070's
(and 1.3x the memory bandwidth), so the same work on this card would be about 5.5-7 ms if the tensor cores were the
limit; NFRUX's best is 8.3 ms, roughly 1.2-1.5x off that scaled figure. What limits it here: the glue passes are
AMD's exact arithmetic in integer code (fp16 rounding, point-sample texel choice, long division in refine / forward
motion / softmax: 2.8 ms of the 8.3 together with the composite), the stage-1 layers are too small to fill 46 SMs
(no split-K: an atomic one would not give the same bytes run to run), and each convolution reads its inputs from
global memory per tap (no shared-memory staging / ldmatrix in the PTX kernels yet). Next steps by expected payoff:
shared-memory staging for the CUDA 3x3 kernels, a deterministic two-pass split-K for blocks 1-2, CUDA on Vulkan
(VK_NVX_binary_import: Vulkan on NVIDIA would get the cuda tier's 8.3 instead of coopmat's 15.9 ms), and fusing the
softmax into the head's epilogue.

### Speed work, round 2 (2026-11-26)

Same RTX 4070, idle this time. `parity --bench` (16 frames a submission, median of 30, ms a rendered frame); "before" is
round 1's last commit (26878cb) rebuilt in a worktree and measured on the same idle GPU, so the columns compare directly
(round 1's own table above was taken under a training job and reads about 5-10% high).

| Tier | 1080p 2x | 1080p 4x | 4K 2x | 4K 4x |
|---|---|---|---|---|
| cuda D3D12, before | 3.97 | 6.28 | 8.41 | 16.5 |
| cuda D3D12, after | **2.80** | **4.78** | **7.09** | **13.96** |
| Vulkan on NVIDIA, before (coopmat) | 6.22 | 11.5 | 15.9 | 35.5 |
| Vulkan on NVIDIA, after (cuda) | **3.04** | **5.05** | **7.41** | **14.3** |
| coopmat (AMD RDNA3+ / Intel path; forced here), before | 6.22 | 11.5 | 15.9 | 35.5 |
| coopmat, after | **4.12** | **7.99** | **11.5** | **25.4** |
| generic D3D12 / Vulkan (no speed work: the int8 QAT model will cover GPUs without tensor cores) | 21.2 / 15.4 | 42.7 / 30.9 | 67.6 / 45.7 | - |

What changed, in the order done:
1. **CUDA on Vulkan** (b6728a0): the rhi's CUDA calls (module, function, launch, buffers with device addresses) on both
   APIs, Vulkan through VK_NVX_binary_import as the int8 network's `cuda_vulkan.cpp`; one full barrier before each launch
   and one after the last before the next pass. Crash marker `vulkan_mlfg_cuda_<device>`, coopmat as the fallback. The
   same kernels give identical bytes on both APIs (mlfg_net: cuda D3D12 == Vulkan bytes; ngx_harness 63 of 63 outputs).
2. **Shared-memory staging, CUDA 3x3** (`conv3_8s2`): the block's rows staged once a chunk, every tap's A fragment a
   shared load; same bits, but neutral on Ada (the per-tap global loads hit L1): built, off (`NFRUX_MLFG_STAGING=1`).
   In coopmat the staging itself got cheaper: one 16-byte load a pixel's 8 channels instead of eight 2-byte ones.
3. **Deterministic split-K** (CUDA, stage-1 3x3 layers that fill fewer than 256 warps): up to 4 splits of the channel
   chunks write fp32 partial sums, a second kernel adds them in split order and runs the epilogue: the same bytes every
   run and on both APIs. Stage 1 2.35 -> 2.05 ms at 4K (`NFRUX_MLFG_NO_SPLITK=1` turns it off).
4. **Glue passes**: `F16Round` without a branch (fp16 subnormals by `(x + 0.75) - 0.75` marked `precise`; without
   `precise` spirv-opt folds it away, which mlfg_passes caught): same bits everywhere, refine 0.65 -> 0.45 ms, composite
   1.19 -> 1.01 ms. What is left in the composite (measured by taking parts out): the 5-weight fp16 bilinear upsample
   ~0.5 ms, colour reads 0.1, splat coverage 0.08; a relaxed (non-strict) build of the composite was no faster, so it
   stays exact in both builds. The coopmat epilogue got the same fast fp16 rounding.
5. **Softmax fused into the head** (CUDA): the 3x3 32->5 layer's epilogue rounds the logits to fp16, gathers the five
   from the quad's lanes by shuffles and computes kMlfgSoftmax5 exactly as the shader (FExp's Cody-Waite + polynomial
   with separate IEEE multiplies / adds, sum in order, `div.rn`): bit-identical to the two passes; the pass is skipped.
6. **FP8 on Vulkan** (VK_EXT_shader_float8): the loader and the test hosts enable the extension and
   `shaderFloat8CooperativeMatrix` where the GPU has them (`kFeatureFloat8`), rhi `Caps.coopMatF8` (e4m3 16x16x32 -> fp32),
   `mlfg_conv_fp8.comp` for stage 2's fp8 1x1 layers (inputs converted to e4m3 bytes in shared memory, exact; e4m3
   weights; fp32 sums). Accurate (49.1-49.4 dB against the fp64-summed net) but slower on the RTX 4070 (48->96: 0.18
   against 0.14 ms): these layers move fp16 tensors and are bound by that traffic and the epilogue, not the matrix
   rate. Opt-in (`NFRUX_MLFG_FP8MAT=1`) for devices where it may win; the real gain would need stage 2's tensors stored
   as e4m3 bytes (exact, they are e4m3 values), which is not done.
7. **Coopmat tuning**: output tiles a group per layer: 2 for the 17-32-output 3x3 (stage-2 input layer 1.24 -> 0.44
   ms), 3 for the 48 / 96-output 1x1 (no padded tiles; stage 2 4.97 -> 4.40 ms). More pixels a group (32 / 64, each B
   matrix used 2-4 times) was slower (accumulator pressure); built in the shader as MLFG_MT, not used. The fused
   FasterNet coopmat kernel re-checked with the new staging: still slower (14.5 against 12.2 ms), stays off.
8. **CUDA tiling**: 4 n tiles (not 8 with 4 of zeros) for 17-32 outputs with a 32-pixel variant (stage-2 input layer
   0.30 -> 0.24 ms); a 64-pixel kernel for one-n-tile 3x3 (head 0.25 -> 0.21 ms, with the softmax epilogue too).
9. **fp16 accumulation** (shipped build only; `NFRUX_STRICT_MATH` builds compile none of it and keep their bytes):
   see below.

#### fp16 accumulation: which layers and why

GeForce runs `mma.sync` with f16 accumulate at twice the f32 rate. Three forms were measured on every CUDA layer at 4K:
fp32 (as before), fp16 over the whole reduction, and fp16 within each 8-channel chunk (all taps; the chunk sums added
into fp32). Whole-reduction fp16 everywhere fails the yardstick (46.6-47.1 dB against 46.8-47.2 needed); chunked fp16
everywhere passes narrowly but is slower overall (7.7 against 7.3 ms): the tensor cores are rarely the limit here,
and on the 96-192-channel 3x3 layers the f16 kernels ran slower (more instructions a chunk, conversions). So the
choice is per layer, kept only where it is faster:

| Layers | Accumulator | Why |
|---|---|---|
| 1, 2, 5, 6 (encoders 3x3 32->32), 10 (2x2 96->192), 23 (2x2 64->128), 48 (3x3 24->24) | fp16 within each chunk, fp32 across chunks | 25-30% faster each (0.04 -> 0.03 ms, 0.032 -> 0.022, 0.025 -> 0.018); error not measurable on its own (within +-0.3 dB of fp32) |
| 60 (stage-2 input, 3x3 24->32) | fp32 | 8% faster but -0.5 to -1.4 dB on the 4x outputs: its inputs carry the largest magnitudes |
| every other plain layer (96-192-channel 3x3, transposed, the head, split-K) | fp32 | f16 slower or no faster |
| fused FasterNet (fp16 and FP8 kernels) | fp32 | not tried for FP8 f16-accumulate: the fp16 result above (latency, not rate, bound) gives no reason to expect a gain |

Gain -0.08 ms at 4K 2x; accuracy with the list against the fp64-summed net: 47.8 / 48.3 / 47.8 / 48.8 dB (sample 1,
2x and 4x; fp32 47.9-49.2) and 45.0 / 44.6 / 45.0 / 44.1 dB (sample 2; fp32 44.5-45.2). `NFRUX_MLFG_NO_ACC16=1` turns it
off, `NFRUX_MLFG_ACC16_LAYERS=a,b,..` replaces the list (tests). Coopmat: chunked float16_t-accumulator variants exist
(3x3, 2x2, 3x3 with 1 / 2 tiles); on the RTX 4070 they are slower (stage 1 2.67 -> 2.78 ms) and cost ~1.9 dB, so no
layer uses them by default; `NFRUX_MLFG_MAT_ACC16_LAYERS=*|a,b,..` selects them for measurement on other vendors.

#### Where 4K 2x goes now (cuda D3D12, 7.1 ms; per-pass sum 6.3 ms)

Stage-1 layers 1.97, composite 1.01, stage-2 network 0.82 (FasterNet FP8 0.45 + 64-ch 0.15 + rest), splats 0.43,
refine 0.45, stage-2 input pass 0.32, its 3x3 layer 0.24, head + softmax 0.21, forward motion 0.24, dilation 0.13, the
rest ~0.5. Vulkan on NVIDIA is 0.3 ms behind D3D12 (the full barriers around launches).

How close to AMD: 2.2 ms quoted on an RX 9070 XT, whose dense FP8 rate is about 3.3x the RTX 4070's and memory
bandwidth 1.3x; scaled, the same work here would be about 5.5-7 ms if the tensor cores were the limit. NFRUX is at
7.1 (D3D12) / 7.4 (Vulkan), at or just above the upper end of that range. What limits it now is not the matrix rate
(fp16 accumulation and FP8 on Vulkan both showed that): the stage-1 layers are latency-bound (30-60-pixel-wide maps),
the glue is AMD's exact integer arithmetic (composite upsample, point-sample texel choice), and stage 2 moves fp16
tensors. Not done, by expected payoff: stage-2 tensors as e4m3 bytes (halves its traffic; exact), a separable
precomputed weight upsample for the composite, fusing the stage-2 input pass with its 3x3 layer, ldmatrix-based
staging for the 16-pixel kernels.