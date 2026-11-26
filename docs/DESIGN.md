# NFRUX design

Status: revision 10, agreed by all three reviewers (docs/REVIEWS.md). Nothing under "Acceptance" counts as done until its
evidence is in `docs/RESULTS.md`.

## 1. Goal and requirements

NFRUX is neural multi-frame generation (2x to 6x) that a game or OptiScaler uses in place of `nvngx_dlssg.dll`.

| # | Requirement | Evidence |
|---|---|---|
| R1 | Acts as a DLSS-G replacement DLL (NGX feature FrameGeneration) | A4, A10 |
| R2 | Can be an OptiScaler frame generation provider | A6 |
| R3 | Native D3D12 and native Vulkan, no API interop | A11 |
| R4 | Tested with Streamline_Sample on both APIs | A4, A5 |
| R5 | Detects and uses what the GPU has (shader model, cooperative matrix, CUDA/NVAPI); optical flow hardware, SM 6.9 cooperative vectors and DirectML (suggested): measured or checked, not used, numbers told to the project (2026-11-25; D3, D4) | A7, A8, A12 |
| R6 | Output visually matches between Vulkan and D3D12: the shipped build by A2b's limits, the strict build byte for byte (A2a) | A2, A5, A9 |
| R7 | Reuses existing NGX code where useful (UI handling) | A9, section 2 |
| R8 | Design documented and agreed by independent reviewers | `docs/REVIEWS.md` |
| R9 | The shipped build is one DLL, easy to install and remove; the package holds only what is needed; no source-project names in files | A15 |

Not goals: super resolution, presentation and pacing (Streamline / OptiScaler own them), anti-cheat games, other frame generators' code
(its repository has no license).

## 2. Structure

```
loader/     process-local NGX router, active when the DLL is named as a Windows DLL (D6)
frontend/   NGX exports, parameter reading, batch protocol (section 4)
core/       interpolator: pass list, constants, histories, UI, subrects. API neutral
core/ml     TOSA lowering and CPU reference; Vulkan, D3D12 and CUDA kernels
core/mlfg   the second interpolator: AMD FSR Frame Generation 4's network and its runtime (docs/FFX_MLFG.md)
rhi/        Device / Texture / Buffer / Pass / Network (rhi.h); one implementation on NVIDIA NRI (rhi_nri.cpp)
tools/      build_shaders.py, build_nri.ps1
tests/      conformance, parity, NGX harness, quality reference
experiments/ measured and not used: DirectML, optical flow hardware; the CUDA launch spike
```

| Part | Source | License |
|---|---|---|
| Passes, pass order, Vulkan network runtime, TOSA lowering | Arm NGSDK 1.1.1 (the nfru-mfg branch), ML Emulation Layer flow | MIT; Apache-2.0 (flow) |
| UI, subrect, no-warp, menu, distortion rules | earlier NGX work, rewritten | MIT |
| NGX ABI | earlier NGX work | MIT |
| Loader | ours | MIT |
| Tests | `tests/` (A1 from the SDK's runtime test; the rest ours); the sample's scripting patch is kept outside the repository (A4) | MIT |
| Weights | `nfru_mfg_v1_int8` (the early fine-tune; v14 was softer at 2x by LPIPS and by eye, 2026-11-25, RESULTS.md), derived from Arm's NFRU v2 | Arm AI Model Community License 1.0 (PRE-1154) |
| NVAPI, CUDA functions | declared by us for the few entry points used, loaded dynamically | checked in M0, in THIRD_PARTY.md |
| OptiScaler, dlssg-to-fsr3 | GPL-3.0: interface facts only | - |

Model license, read in full: clause 2 grants modification, derivatives and distribution. Clause 3 requires shipping the
license, its clause 4 as an enforceable term, original notices, and a statement of changes. Distribution is governed by the clause 3.1 and 3.2 duties (THIRD_PARTY.md, A13).

## 3. Pipeline

Capacity: `kMaxGenerated = 5`, one constant for core, frontend and tests (histories, arrays, descriptor counts). n is always
`DLSSG.MultiFrameCount` as given; t = (i + 1) / (n + 1) is never computed from a clamped count. n > 5: frames 1..5 at their
own t, the last slots (k > 5, shown just before the real frame) are the back buffer (logged). The reported maximum is 5 (6x); `NFRUX_MAX_FRAMES` (generated frames, 1-5) or nfrux.ini `MaxMultiplier` (2-6) lowers it, the environment winning. What Streamline
and the host allow on a given GPU is theirs; A4 records it.

Per rendered frame: luma pyramid, hints (camera reprojection of the previous depth, as the SDK; no
scattered game vectors: measured no gain), optical flow once, reduction to the motion lane (<= 540 rows). Per generated frame: warps, hole fill, 20-channel
tensor, network, inpainting, composite. A batch runs stage by stage.

The graph is v1 (v14's inputs): channels 16-19 are t, t|mv|/64, (1-t)|mv|/64, 0; 17-18 are clamped to 0..1 (saturate). All quantisation constants are
`nfru_mfg_v1_int8_metadata.json`'s, written in `interpolator.cpp` (checked end to end by A3; v1: input 0.003921557 / -128 (SNORM 0.4980377 / -1.00787), params 0.2640706 / 89, gate 0.003921557 / -128; v2's and v14's input scale 1/255 are not used: corrected 2026-11-26 after review).

Flow at more than 1080 display rows: the SDK's colour reduction pass (01) gets an entry point and the flow, hints, flow scale
and `_OfInputSizeRcp` use the reduced size (aspect kept, 1080 rows), as the SDK does. Sizes to 8K.

Reset (first frame, `DLSSG.Reset`, size change, after menu / not rendering / release): every output of the batch is a copy of
the **back buffer** (UI included, subrect rules applied); the history stores the interpolation source.

Distortion field: the whole batch (undistort once, n interpolations, n composites).

## 4. NGX contract

Protocol per rendered frame: Evaluate is called once per generated frame with `MultiFrameIndex` k = 1..n. k = 1 makes all n
into private images and copies out frame 1; k > 1 copies out frame k. Keyed by order: k = 1 always starts a batch (no frame id is relied on; Streamline gives none, BackbufferFrameID is used when given).
k > 1 with no valid batch, a k above the batch size, or a failed batch: the back buffer. A change of n applies at the next k = 1 (histories do not depend on n). Reset,
NotRenderingGameFrames and menu are latched at k = 1 for the batch.

**OutputReal is written on every Evaluate** (a copy of the back buffer), on every path including reset, menu, failure.
OutputInterpolated is always written too. A failure is returned only for a missing Backbuffer / OutputInterpolated, or when nothing can be recorded (D5:
32 recordings unfinished after 2 s; an image that cannot be wrapped).

| Parameter (DLSSG.*) | Use | Missing |
|---|---|---|
| Backbuffer, OutputInterpolated | required | FAIL_InvalidParameter |
| Depth, MVecs | required to interpolate | back buffer, Success, logged once |
| OutputReal, HUDLess, UI, UIAlpha, NoWarp, BidirectionalDistortionField | optional | rule without it |
| *Subrect{BaseX,BaseY,Width,Height} per input; InputBackbuffer* preferred over the deprecated Backbuffer* | region | whole image (or the back buffer's, same size) |
| OutputInterpolatedSubrect*, OutputRealSubrect* | where the region goes in the output | the region's own place |
| EvalFlags bit 0 (UpdateOnlyInsideExtents) | only the region is written: several viewports, a feature each, on one back buffer | the whole image is written (the back buffer around the region, as strips that do not overlap it) |
| OutputDisableInterpolation | a 4-byte word: 1 when this batch holds copies only (menu, not rendering, reset failures, not ready), else 0; written at k = 1. D3D12: in COPY_DEST. Vulkan: NGX buffer resource with TRANSFER_DST usage; a host that maps it waits for the submission | not written |
| BackbufferFrameID | an id that is not the last or the last + 1 restarts the history | not used |
| DynamicResolution, InternalWidth/Height | depth and vectors resampled (nearest) to one size, the given or half the colour's, so a changing region keeps the history | the region's size |
| InvertXAxis, InvertYAxis | negate MvecScale | 0 |
| CameraMotionIncluded | 0: the camera's motion (depth, ClipToPrevClip) is added to the vectors; without ClipToPrevClip it cannot be (logged) | 1 |
| MvecScaleX/Y | to motion-vector pixels, **not** multiplied by the size again | 1 |
| ClipToPrevClip (row-major, row vectors) | camera motion; VP is chained from it | camera basis reconstruction |
| CameraViewToClip, CameraPos/Up/Right/Fwd, Near, Far, FOV, OrthoProjection | depth and fallback camera | defaults 0.1 / 1000 / 1.0 |
| JitterOffsetX/Y, DepthInverted, ColorBuffersHDR, Reset, NotRenderingGameFrames, MenuDetectionEnabled | flags | 0 |
| MultiFrameCount, MultiFrameIndex | protocol | 1, 1 |
| ...LowPrecision.{IsLowPrecision,Scale,Bias} | distortion decoding | none |
| CmdQueue, CmdAlloc, IsRecording | not used | - |
| MvecJittered | the jitter's change taken out of the vectors (the SDK's jitter cancellation) | 0 |
| MvecDilated, MvecInvalidValue, ClipToLensClip | not used: the network takes vectors as the SDK does; RTX Remix passes 0 as the invalid value, which is a real vector | - |

Parameters are read through the caller's `NVSDK_NGX_Parameter` vtable.
PopulateParameters_Impl / capability values: `FrameGeneration.Available` = 1, `.NeedsUpdatedDriver` = 0,
`.MinDriverVersionMajor/Minor` = 0, `.FeatureInitResult` = Success, `DLSSG.MultiFrameCountMax` = MaxFrames() (kMaxGenerated, lowered by `NFRUX_MAX_FRAMES` or `MaxMultiplier`), `DLSSG.EstimateVRAMCallback` (an estimate from the display's size).
`NVSDK_NGX_*_EvaluateFeature_C` is exported beside `EvaluateFeature`.

States (DLSS-FG Programming Guide): D3D12 inputs arrive and leave in NON_PIXEL_SHADER_RESOURCE, outputs in UNORDERED_ACCESS. Vulkan outputs GENERAL; inputs
GENERAL, or SHADER_READ_ONLY_OPTIMAL when the NGX resource is not read-write (RTX Remix). Streamline (debug layer, 2026-11-25)
follows that for the back buffer and outputs only: the disable word is in COPY_DEST, and tagged inputs are in the state the game left them, its copies of tags valid only
now in COPY_DEST. Seen in Streamline_Sample with the production Streamline 2.14.16 plugins (over the air; sl.common identical to the SDK's); its guide asks that a tag's state be right when Streamline uses it. Therefore:
- the loader wraps slSetTag / slSetTagForFrame (imports and GetProcAddress) and, once NFRUX's D3D12 feature exists, hands each valid-until-present tag's declared state to the feature
  (`NFRUX_NoteState`, by resource). Nothing is changed in the call;
- the feature transitions such an input from that state and back where its list can name the state; on a COMPUTE list an input whose state also has pixel bits is read as it is;
- only a tag a COMPUTE list cannot use (RENDER_TARGET, DEPTH_WRITE, pixel-only; a colour, which is also copied from, with any pixel or attachment bit) is passed on as valid only now, so that Streamline
  copies it on the game's list. Streamline then takes the declared state as the one there: an assumption, right in the sample, wrong for a game whose present-time state differs from its state at the
  call. Cost: one copy a tag and frame, not measured. No command list given: used as it is;
- Streamline's copies are in COPY_DEST, known by their debug name ("clone.tagVolatile", present in the release sl.common.dll 2.14.16). Another Streamline that does not set the name: read in the wrong declared
  state, as before this rule. eValidUntilEvaluate tags and D3D11 hosts are left alone; Streamline 1's slSetTag arguments are told apart (its second argument is a small number).
On the OptiScaler route nothing hands the states over: the guide's state is assumed, which is known to be wrong for tags in other states (section 7).
Diagnosis switches in the DLL, off unless set: NFRUX_TEST_D3D12_MESSAGES (the debug layer's messages and barriers into the log), NFRUX_TEST_PASSTHROUGH, NFRUX_TEST_ABSENT, NFRUX_TEST_NAMES.
Lists may be COMPUTE or DIRECT; NRI is patched (build_nri.ps1) to use 0x40, not 0xC0, for shader resources on DIRECT lists. Barriers use compute-legal states only
(NON_PIXEL_SHADER_RESOURCE, UNORDERED_ACCESS, COPY_*), since lists may be COMPUTE type. No blits: scaling or converting copies
are a compute pass. Depth: plane 0, the R32F / R24X8 / R16 SRV of its typeless family; Vulkan depth aspect. After a D3D12
Evaluate the caller must rebind descriptor heaps, root signature and PSO (NGX's rule); after Vulkan, compute bindings.

Failures: OOM / pipeline failure / unsupported size at Create -> FAIL_PlatformError or UnsupportedParameter, nothing leaked.
During Evaluate -> back buffer, Success, interpolator disabled for that size, logged.

## 5. Decisions

**D1. One shader source.** GLSL -> SPIR-V -> (SPIRV-Cross) HLSL -> (DXC) DXIL. Rejected: hand HLSL (drift), HLSL first (rewrites about 8k working lines). Numeric policy, enforced by
`build_shaders.py`:
- source-level 32-bit floats: `float16_t`, `f16vec*`, `mediump`, `lowp` map away, `HF` literals replaced; the packed f16 match is
  removed, the shared-memory match is used everywhere;
- the build fails if a pass SPIR-V declares Float16, Int16, Int8, 8/16-bit storage or any group operation;
- contraction: CMake option `NFRUX_STRICT_MATH`. ON (the proof build): the SPIR-V is post-processed to put `NoContraction` on every float
  arithmetic result before both consumers and DXIL is built with `-Gis`; the two APIs must then give the same bytes. OFF (the shipped
  build): compilers may fuse, except in the reduction passes (01-03), which round to a texel, often from an exact half, and are strict in every build; judged by A2's limits.
- opaque operations (matrix products, dot, mix, fma, length, normalize) are written out; the build refuses a pass that has one, as neither
  switch fixes how a driver fuses inside them. Denormals: no pass relies on them; both APIs' defaults (D3D flush allowed) are covered by the A2 budget;
- remaining expected differences: division, exp, sqrt, texture filtering rounding. A discrete flip (block-match tie, 0.07 UI
  threshold, hole test) can exceed them locally; A2 judges these as a counted class.
**Reference.** Passes, constants, sizes and formats follow the SDK branch (`ngsdk-steamdeck`, `nfru-mfg`, 38cfb23)
wherever sources differ (2026-11-25). Remaining differences from the branch give the same
results: packed integer types, selects for multiplies by 0 / 1, batched loads in the same order, log2 / exp2 for log / exp, the timestep
prior as softmax weights instead of logarithms (within rounding), 02 / 03 / 01 as compute passes; and the UI additions. The optical flow passes
port Arm's ML Emulation Layer, which the branch runs through VK_ARM_data_graph_optical_flow.

**D1b. NRI.** (Adopted 2026-11-25.) rhi.h has one implementation, on NVIDIA's NRI (MIT), commit 993db1a, built by
`tools/build_nri.ps1` (which changes one thing: a shader resource's state on DIRECT lists, 0x40) with D3D11, Agility SDK, NVAPI, AGS and NVTX off (its wrapped-device teardown crashed with them; which one was not
established). It wraps the host's device, command buffer and images. The native D3D12 backend that existed was bit-identical and no faster in
the sample (564 against 572 fps at 4x) and was deleted; that removes an independent implementation, so correctness evidence is A3 and A14
(outside references), and the validation runs, and A3 is required before M5's exit. Ours, on native handles: state tracking, descriptor recycling,
completion tokens, network kernels.

What NRI needs of a wrapped Vulkan device: 1.2 and the Vulkan 1.0 features the GPU reports. `DeviceCreationVKDesc::unknownFeatures = true` (new
after v192, the reason for the pin) makes it treat every optional 1.1+ feature (bufferDeviceAddress, synchronization2, maintenance*, memoryPriority,
...) as absent; `minorVersion` is always given as 2 and no extensions are listed, whatever the GPU reports, as the host's own choice is not known. Below
1.2: creation refused, logged. NRI's "all supported 1.0 features enabled" is an assumption on both routes; the ones our call set uses are
shaderStorageImageExtendedFormats, which the loader adds. No queue family is given to NRI (it would ask for a queue the host may not have made; NFRUX never submits through it). On the OptiScaler route the
assumptions are (a) 1.2, (b) shaderStorageImageExtendedFormats, and on NVIDIA (c) the optional features of D3, all logged at creation. A minimal-host test is A10.
Static linking of d3d12 and vulkan-1 is replaced by delay-load: a game on one API never loads the other.
**D2. Exact integer network.** Every path gives the CPU reference's bytes. Cooperative matrix is int8, fp16 or bf16: the float forms hold every int8 value and
x - zero point exactly, and their float32 sums are the same integers while every partial sum is below 2^24. The bound (per output channel, max
|x - zp| * sum |w| + |bias|) is computed from the weights when a network is made: 7,021,312 of 16,777,216 for this one; at 2^24 or more the float forms are not
used. A prefix of a sum is at most the sum of absolute values, so partial sums are covered. Order when a float form is refused or fails its self-check: int8, dot4, integer. fp16 is first on NVIDIA (20% faster than int8 on the RTX 4070), int8 elsewhere. bf16 is reached only where it is known or assumed enabled
(NFRUX_SetVulkanFeatures in the tests; the OptiScaler route's assumed features on NVIDIA; the loader does not enable it, so `coopmat_bf16` there falls back). D3D12 rescale uses a 32x32 -> 64 multiply from 16-bit halves, so Int64ShaderOps is not needed. Float
requantisation is off. `conv2d.hlsl` implements a restricted layer class (rescale input zero point 0, single rounding,
dilation 1, square kernel and stride, resize on source 0 only, shift >= 1, source strides equal to the words read): graph creation
checks each layer and the rung fails if any is outside it. The shape list is written by hand in `build_shaders.py`; a shape missing from it fails creation.
The `[flatten]` two-source load is split so no out-of-range address is formed. Outputs go to buffers and are copied to the
params / gate textures on every path (paths that write images directly are an optimisation, checked by the same test).

**D3. Ladders.** Chosen at creation, logged in one line, forced by `NFRUX_ML` (tests: strict, no crash markers); a forced rung that is unavailable fails creation. For users, `nfrux.ini` beside the DLL (`core/log.cpp`; read once; `;` `#` and inline ` ;` comments; unknown keys logged; `auto` or a missing line is NFRUX's choice; flags take 0/1/true/false/on/off/yes/no). An environment variable wins over its key:

| Section, key | Environment | Values |
|---|---|---|
| General MaxMultiplier | `NFRUX_MAX_FRAMES` (generated frames, so one less) | 2-6 |
| General IgnoreCrashMarkers | `NFRUX_IGNORE_MARKERS` | flag |
| General Log | `NFRUX_LOG` | flag (nfrux.log and nfrux_loader.log beside the DLL) or a file path |
| D3D12 Network | `NFRUX_ML` | cuda, dot4, scalar |
| Vulkan Network | `NFRUX_ML` | cuda, coopmat, coopmat_int8 / fp16 / bf16, dot4, scalar |
| Vulkan AssumeFeatures | `NFRUX_VK_ASSUME_FEATURES` | flag |
| Vulkan Dot4Tiled, Dot4Subgroup | `FFX_ML_DOT4_TILED`, `FFX_ML_DOT4_SUBGROUP` | flag; subgroup size |
| Vulkan FullBarriers | `NFRUX_FULL_BARRIERS` | flag |
| General Model | `NFRUX_MODEL` | nfru (default), amd (FSR FG4, docs/FFX_MLFG.md) |
| General AmdWeights | `NFRUX_AMD_WEIGHTS` | a `tools/mlfg_weights.py --bin` file |
| D3D12 / Vulkan AmdNetwork | `NFRUX_MLFG` (tests: strict, no markers) | generic, cuda (D3D12) / generic, coopmat, cuda (Vulkan) |

`Network` from the ini falls back to the automatic choice (logged, for the rest of that device) when it cannot be made, and counts crash markers like the automatic choice; the self-check applies to every choice. A Vulkan `cuda` (ini or environment) does not consult `AssumeFeatures`: it asks for the launch function itself and assumes buffer addresses with it, as the automatic NVIDIA choice does (below).

| | 1st | 2nd | floor |
|---|---|---|---|
| Vulkan (NVIDIA tensor cores, Intel XMX, AMD WMMA) | NVIDIA: CUDA kernels (VK_NVX_binary_import, the same as D3D12's), then cooperative matrix fp16, int8; elsewhere int8, fp16 (exact, D2) | dot4 (`shaderIntegerDotProduct`) | 32-bit integer |
| D3D12 NVIDIA RTX (20 and later) | CUDA via NVAPI | SM 6.4 dot4 | SM 6.0 |
| D3D12 other (AMD, Intel, GTX) | SM 6.4 dot4 | - | SM 6.0 |
| FSR FG4 (Model=amd), D3D12 | NVIDIA RTX: CUDA fp16 tensor cores, FP8 for stage 2's FasterNet blocks on RTX 40+ (NVAPI, crash marker) | - | generic (strict HLSL) |
| FSR FG4 (Model=amd), Vulkan | NVIDIA RTX: the same CUDA kernels (VK_NVX_binary_import, crash marker; identical bytes to D3D12) | fp16 16x16x16 cooperative matrices (crash marker off NVIDIA; FP8 1x1 layers opt-in) | generic (strict HLSL) |

Vulkan baseline (passes and floor): 1.2 (NRI with unknownFeatures, D1b) and one feature, `shaderStorageImageExtendedFormats` (storage images in r8, rg16f,
rgb10_a2, r11f_g11f_b10f; Streamline_Sample does not enable it). BGRA8 outputs: the composite stores blue first into an RGBA8 image of
its own, copied raw (size-compatible) into the host's. Everything
else is optional and needs to be *enabled*. Source of truth:
- Standalone route: the loader intercepts `vkCreateDevice`, obtained through GetProcAddress of vulkan-1.dll
  and through vkGetInstanceProcAddr / vkGetDeviceProcAddr, whose returned pointer for that name it wraps. It appends (and records what it
  appended), when supported per `vkGetPhysicalDeviceFeatures2`: shaderStorageImageExtendedFormats,
  cooperativeMatrix, vulkanMemoryModel (+ DeviceScope), shaderInt8, storageBuffer8BitAccess, shaderFloat16, storageBuffer16BitAccess, shaderIntegerDotProduct,
  subgroupSizeControl, computeFullSubgroups
  and their extensions (integer dot product and subgroup size control, without the 1.3 aggregate: only when the device lists the extension). It also wraps vkDestroyDevice (what is on the device is destroyed first). Merge rule: known structures of the pNext chain are copied, unknown ones kept in place (their pNext is changed for the call and put back); a feature that belongs to an aggregate
  present in the chain (VkPhysicalDeviceVulkan11/12/13Features, VkPhysicalDeviceFeatures2) is set in that copy; its individual struct
  present is set in that copy; else the individual struct is chained. Core features go into pEnabledFeatures when the app uses that, else
  Features2. The app's own values are never cleared. If creation with additions fails, the app's original call is made. Only the
  app's own process and device are affected; additions are logged.
- OptiScaler route: the enabled set is unknown. The assumptions of D1b (Vulkan 1.2, shaderStorageImageExtendedFormats) are made and logged. On NVIDIA the GPU's optional features are used as well (project,
  2026-11-25: outside strict validity when the game has not enabled them; its driver takes them, RTX 4070 tested), behind the crash marker of CUDA's kind and the self-check; elsewhere the floor.
  `NFRUX_VK_ASSUME_FEATURES=0 / 1` decides on any GPU. Markers (both rungs) are cleared when 40 recordings have completed since the network first ran, or on an orderly end.
The loader hands the DLL the recorded set per VkDevice (export `NFRUX_SetVulkanFeatures`); no record means the OptiScaler behaviour.
A4's validation run uses the standalone loader, which makes the cooperative-matrix rung valid. Fallback when M0a cannot intercept
vkCreateDevice: Vulkan uses the floor, A4 claims only that, and the ladder text is amended.
M0d result (RESULTS): SM 6.9 cooperative vectors are in neither the inbox runtime (SM 6.8) nor the retail SDK; DirectML's integer convolution is
exact but slower than dot4. Neither is a rung.

CUDA: `core/ml/cuda_network.cpp` writes PTX (sm_75, `mma.sync` m8n8k16 s8, one kernel a layer). Nothing is precompiled: the user's own driver links it for
the GPU it is, on the worker thread, with a short-lived context matched by LUID (checked: the same PTX links for sm_75, 80, 86, 89, 90, 100 and 120). m8n8k16 is as
fast here as sm_80's m16n8k16, so there is one form. GTX 16 (and T400 to T1200, MX450) reports sm_75 without tensor cores and is told apart by name, whole words (a word that begins with RTX, or is exactly T4 or T4G; a table test in `ml_conformance_d3d12`); GTX 10 and 16 use dot4, the
same hardware instruction CUDA would. Cubins are kept in memory only. NVAPI `CreateCuModule`, `CreateCuFunction`, `LaunchCuKernelChain` by public interface id
launch it. UAV barriers between launches, inputs in UNORDERED_ACCESS. Layer class of this rung (else it fails creation): square kernel up to 7, equal strides, dilation 1, weight zero point 0, single-rounding 32-bit rescale with shifts 2-62, up to 64 output channels, two sources only without resize. Layout: a warp computes 8 output pixels by 8 channels per mma tile; a layer of 20,000 output pixels or more gives a warp 2 such columns by up to 8 channel tiles, a smaller one (the low-resolution end of the U-Net) 1 by 1, so it has 8x the warps to fill the GPU (5-8% faster overall on the RTX 4070, 46 SMs; the threshold is not scaled by SM count; exact either way, RESULTS).

On Vulkan the same kernels are launched by vkCmdCuLaunchKernelNVX (`core/ml/cuda_vulkan.cpp`): buffers by address, a full memory barrier where a layer reads what the last wrote. The loader enables the extension and
bufferDeviceAddress; Streamline enables the extension itself, so on the OptiScaler route its presence is asked (the function's address) and buffer addresses are assumed with it. Same marker and self-check. A batch's networks run a
layer at a time on both APIs (independent launches between barriers).

AMD and Intel matrix hardware is reached through Vulkan cooperative matrices (behind a crash marker of their own off NVIDIA, since untested there: refused, dot4; tested on the RTX 4070 by forcing it, `NFRUX_TEST_MAT_MARKER`): any M, N up to 16 and K a multiple of 4; subgroups of 32 on AMD, of 16 (the tile's N) on Intel when
the size can be required, else the driver's own. Untested on such GPUs (none here): the self-check and the next rung are the safety. On D3D12 there is no public shader interface to
them (no cooperative vectors in the retail runtime; AMD's and Intel's are device-creation or vendor-driver extensions), so dot4.
Crash safety, for what depends on vendor-specific launches (CUDA on both APIs) or on features the game may not have enabled (the NVIDIA default on the OptiScaler route): before the first use in a process an "attempts" file in %LOCALAPPDATA%\NFRUX (wide paths; logged with its name), named by
rung, device (D3D12: also the adapter LUID) and nvcuda.dll's file time, is counted up; it is removed when 40 recordings have completed since the network first ran at its real size (the 64-wide self-check does not count), or on an orderly end, and kept after a failed
self-check or (D3D12) a device removal; on Vulkan a device loss is not detected at teardown, so an orderly end after one clears it. At two, that way is not used (one line in the log; deleting the file or `NFRUX_IGNORE_MARKERS` tries again). No folder or not writable: not used, logged. The assumed features' marker is begun only when a network that is not CUDA's uses them. A game closed twice in its
first seconds has the same effect. Tested with a process that ends at the first real run. A device loss in the game's process cannot be recovered and is not claimed to be. A network made in place of one that could not be created at
the real size has not been self-checked itself (the rung was: 64 wide). Where the features are known enabled, the cooperative-matrix and dot4 rungs are plain API use and have only the self-check.
**D4. Optical flow.** The compute block match the network was trained with, everywhere. NVIDIA's optical flow hardware was measured (A12,
offline, VK_NV_optical_flow, same network, 70 held-out pairs): 0.65 dB worse at 3x, 0.11 at 5x, 0.34 at 6x, 0.05 better at 4x. Idle RTX 4070: it takes
0.44 to 0.85 ms at 720p and 0.68 to 1.77 ms at 1080p (its three levels, its own engine); the block match costs 0.12 to 0.2 ms of shader time. It fails this section's earlier rule (within 0.05 dB), so it is not delivered, not even opt-in: it would also need the command
list split on D3D12 and was never shown valid on Vulkan. Neither flow had hints in that comparison, and
the network was trained on the block match. Reopened only with a network trained on that flow.

**D5. Lifetime, threads, latency.** Completion token: at the end of every recording the list writes that recording's own value into that recording's own slot of host
memory (D3D12 `WriteBufferImmediate` MARKER_OUT: ID3D12GraphicsCommandList2 is needed, Windows 10 1709; Vulkan `vkCmdFillBuffer` to a
host-visible buffer + host barrier). 32 slots; a slot is taken again only when it holds its own value (exact match, so order and abandoned lists do not
matter). None free: waited for once, at most 2 s; then the call fails with nothing recorded, and later calls fail at once until one completes. A recording
never submitted parks its slot. Descriptors, constants, views and readbacks follow their slot (views are kept 16 recordings longer, as the host's command
buffer may still count as pending); retired interpolators and cut-outs are freed 256 recordings later, more than can be in flight.
Release: the feature stays warm (history restart on next use), keyed by device, except on Vulkan without the loader (nobody reports the device's end
there, and its address can come back). Shutdown: the caller must be idle
(NGX's rule); NFRUX waits up to 2 s for tokens, then leaks the rest with a log line rather than free in-flight memory. Device
removed: no waits.
Locks: each feature owns its own rhi Device (recording state is never shared) and one mutex around Evaluate; Evaluate holds shared ownership, so a
concurrent Release cannot destroy it; released generators are kept warm per host device; a process mutex for caps, the pipeline cache and the NVAPI/nvcuda
tables. Interpolators, the self-check's network and its reference bytes are made on worker threads; until ready, outputs are the back
buffer, and the first real batch counts as a reset. Two interpolators per feature are kept (format switch), older ones retired as above. Nothing is destroyed at process exit. D3D12 and Vulkan features may coexist.

**D6. Loading.**

0. One file (2026-11-25): `nvngx_dlssg.dll` holds everything (shaders, weights, NRI, the loader, the C++ runtime statically). Its role is its file name:
   `nvngx_dlssg.dll` or `dlssg_to_fsr3_amd_is_better.dll` the feature; `dbghelp.dll` or `version.dll` the feature and the loader (their exports are
   forwarded to Windows' own). Uninstalling is deleting it. The `package` target holds the DLL, INSTALL.txt, THIRD_PARTY.md, the optional nfrux.ini (all auto) and the license texts, nothing else. The DLL pins
   itself when it hooks (patched imports point into it).
1. OptiScaler, two routes:
   a. Releases (0.9.5 the newest): the dlssg-to-fsr3 slot. The DLL as `dlssg_to_fsr3_amd_is_better.dll` beside OptiScaler, `FGInput=nukems`, `FGOutput=nukems`.
      That slot offers 2x only. This is the route R2 is judged by.
   b. Builds with `FGNvngxReplacement`: `OptiScaler/custom/NFRUX/dlssg/nvngx_dlssg.dll`, `FGNvngxReplacement=NFRUX`; up to
      what it asks for. On D3D12 that build crashes the sample by itself (control: `FGInput=nofg`), so it is judged on Vulkan only.
   OptiScaler loads the DLL directly and supplies parameters and presentation. Vulkan features are unknown there (D3). Whether it calls Shutdown is
   not known: NFRUX does not depend on it.
2. Standalone. The NGX static library inside Streamline loads the core `_nvngx.dll` from the driver folder named in the registry,
   and no machine-wide setting may be changed.
   Mechanism: the file as `dbghelp.dll` (sl.interposer.dll imports it; `version.dll` for hosts that do not) in the game folder hooks
   `LoadLibraryExW` and `GetProcAddress` by import-table patching of the modules loaded at attach (enumerated) and later ones through
   LdrRegisterDllNotification (in-process only; Streamline loads its over-the-air plugins from ProgramData). When the core is loaded, the returned exports of interest are wrapped:
   Vulkan Init* (note device), Shutdown* (NFRUX first, then the core), GetCapabilityParameters / GetParameters (add the section 4 capability values),
   GetFeatureRequirements (feature 11 supported; both APIs), CreateFeature / EvaluateFeature(_C) / ReleaseFeature. Not wrapped: AllocateParameters, Vulkan extension
   requirements, LoadLibraryA/W (the sample runs did not need them). Feature 11 goes to NFRUX, everything else to the real core untouched. Handles NFRUX returns are its own
   allocations, recognised by address set, so ids cannot collide. Without a core (non-NVIDIA) the standalone route is not available.
   Production DLLs are used unmodified.
   Rejected: machine-wide settings, patching files, replacing the driver's core.

   M0a status: routing, capability override, vkCreateDevice hook and real frames at 2x-6x (D3D12) and 2x-4x (Vulkan) done in short runs; the 60 s
   and validation parts run with A4. M0 pass condition: in Streamline_Sample with Streamline 2.14.16 and driver 610.88, both APIs, `slIsFeatureSupported(DLSS_G)` is
   ok, Create and Evaluate for feature 11 arrive in a stub that copies the back buffer, and frames are presented at 2x for 60 s.
   On Vulkan also: vkCreateDevice seen, features recorded and appended, device created, 0 validation errors, and the cooperative-
   matrix pipeline buildable on that device.
   If import patching misses a call, MinHook inline hooks are the second attempt. If both fail: R1 standalone is reported as not
   met, R4 runs through OptiScaler, and it is reported; it is not silently substituted.

**D7. Self-check.** One 64 x 64 hashed tensor made by a compute pass, expected bytes from the CPU reference, both made on a worker started at the first Evaluate; same code as the real network.
Recorded when the worker is done, read back under the completion token; outputs stay the back buffer until it passes. "Not complete" is not "mismatch"; on mismatch the next lower rung
is made and checked; after the last one NFRUX shows the back buffer. Fault injection (`NFRUX_TEST_BAD_CHECK=N` fails the first N checks) is tested.

**D8. UI.** DLSS-G's rules: HUD-less + UI alpha / UI layer, difference test, no HUD-less, no-warp, subrects, menu detection, distortion. HUD-less in another
format than the back buffer is converted when both are SDR formats (else ignored, logged); sRGB and UNORM images are compared as stored. Thresholds 0.07 and
0.4 are applied in the buffer's own values, SDR or PQ. The UI tests decide per 2x2 block; DLSS-G has 2x2 and 4x4, and how it chooses is not known, so there is
one.
**D9. Formats.** RGBA8, BGRA8 (and their sRGB images), RGB10A2 (SDR or HDR10 PQ), as DLSS-G. Float back buffers (scRGB) are not supported (2026-11-25):
the back buffer is shown, logged. Values are always the stored ones: sRGB inputs are copied raw into a UNORM image of NFRUX's and read from there (a UNORM
view of them is not always legal); outputs are copied raw. fp16 is used only inside, for the staged path. A D3D12 device without BGRA8 typed store cannot
generate BGRA8 (said in the log). A2R10G10B10 is not supported.
**D10. Build.** CMake, MSVC x64, Vulkan SDK >= 1.4.363 tools and Python 3 (standard library only). Clean checkout:
`tools\build_nri.ps1` (network, once), then `cmake -B build -A x64` and `cmake --build build --config Release`. Run time needs no compiler.

## 6. Milestones

| M | Content | Exit |
|---|---|---|
| M0a | Loader probe (D6) and state probe (section 4) in the sample, both APIs | D6 pass condition; states recorded |
| M0b | CUDA spike: one 3x3 layer in a compute list, bytes against the reference | done: exact, rung stays |
| M0d | (requested) SM 6.9 cooperative vectors, when the host exposes them or through NVAPI, and DirectML / metacommand integer convolution (exact int32 sums, our rescale after): one 3x3 layer each, exactness and time against dot4 and CUDA | done: dropped |
| M0e | Optical flow hardware | closed: measured offline, not used (D4) |
| M0c | THIRD_PARTY.md, notices | written |
| M1 | NRI rhi, network, completion token, build system (done, A1 partly) | A1 |
| M2 | core on Vulkan, capture/replay | A3, A14 |
| M3 | D3D12 through NRI, dot4 and floor (done, A2 partly); Graphics Tools or an app-local D3D12SDKLayers for the debug layer, and the section 4 state check with it, before M5 exits | A1 D3D12, A2 |
| M4 | frontend, NGX harness | A10, A11 |
| M5 | sample runs | A4, A5 |
| M6 | CUDA rung | A1, A7, A12 |
| M7 | UI, menu, distortion | A9 |
| M8 | OptiScaler, cost | A6, A8 |

M1-M3 do not depend on M0a.

## 7. Risks

| Risk | Handling |
|---|---|
| Interception fails on this Streamline version | M0a first; D6 fallback, reported |
| CUDA launch refused, hangs | M0b; crash marker (D3) |
| Streamline limits multi-frame on RTX 40 | A4 records what it grants; 3x-6x then proven by A10 and A2, said so |
| Cross-API rounding larger than budget | intermediates dumped per pass; the first differing pass is found |
| First-transition corruption at 5x (Vulkan): fixed by waiting for earlier work; mechanism not established | regression case in A2; synchronization validation with a messenger that fails the run |
| Vulkan: a Reflex cap is not kept when generation and presenting are heavy (5x and up; RESULTS) | open; D3D12 keeps it. Suspected: Streamline's pacer presenting while vkLatencySleepNV runs (validation reports the race) |
| NRI changes | pinned commit; A1 and A2 on update |
| CUDA on Vulkan and the NVIDIA default under OptiScaler: short sample runs only (RESULTS) | A4 and A6 on Vulkan before a package is handed out |
| OptiScaler route (R2), D3D12: tagged inputs in another state than the guide's are read without a transition. On NVIDIA no fault was seen (2x, sample); others not tested | known; only a host that gives the states can fix it |

## 8. Acceptance

Evidence rule: a row's final evidence in RESULTS names the commit and the build (strict or shipped). After a change to shaders, kernels, rhi or front end
A2, A5 (replay), A7, A9 and A10 are run again; after one to passes or constants A3 and A14 too; after one to kernels or the network runtimes A1; A4 and A6 on
the release candidate. Older entries are history.

Inputs: `parity`'s synthetic scenes (pan and moving square, made in the test), recordings of the sample (`NFRUX_DUMP`, `NFRUX_DUMP_FRAMES`, `--replay`), and
the held-out Blender clips (`C:\dev\modelgym\data\nfru6x\test_new`, `test_bal`, 180 fps truth) as `quality_reference.py` exports them. Encodings RGBA8, BGRA8,
RGB10A2; normal and inverted depth; odd sizes.

| # | Test (`tests/`) | Pass |
|---|---|---|
| A1 | `ml_conformance_d3d12`, `ml_conformance_vulkan`: each rung vs CPU reference graph outputs; sizes 64x64, 744x464, 960x544 on D3D12; random seeds, all -128, all 127, alternating, and on Vulkan inputs with the signs of the first layer's weights; the 2^24 bound (D2); the tensor-core names | 0 differing bytes |
| A2a | `parity --strict`, strict build: synthetic scenes (1080p, 4K, 8K, ultra-wide, odd; RGBA8, BGRA8, RGB10A2; both depths) and 64-frame recordings, 2x / 4x / 6x; exit code | 0 differing bytes in every output and intermediate |
| A2b | `parity`, shipped build, the same | PSNR >= 60 dB, >= 99.9% of values within 1 code, <= 0.01% off by more than 3 (RGB10A2 by 10-bit code); exit code |
| A3 | `tests/quality_reference.py`: port with v1 int8 vs the torch v1 int8 pipeline (`eval_mfg` pipeline, `C:\dev\modelgym\output\nfru_mfg\qat1\last.pt`, same clips, 8-bit colour and half-float vectors for both) | PSNR against the true frames within 0.2 dB of torch per multiplier |
| A4 | Sample, D3D12 and Vulkan, scripting patch: 2x and the maximum granted, 1920x1080, 5 min each on release DLLs; 60 s with D3D12 debug layer + GPU validation / Vulkan validation + sync | no crash; 0 messages naming NFRUX objects or thrown inside its recordings; `numFramesActuallyPresented` = n + 1 for >= 95% of states logged; OutputReal written every evaluate; dumped generated frames lie between neighbours (mean luma distance to both <= distance between them + 2 codes, >= 99% of frames) |
| A5 | Sample with fixed timestep, scripted camera, DLSS off: dump 64 frames on both APIs (HUD-less colour); replay each dump on both APIs | replay meets A2a (strict) and A2b (shipped); live: generated PSNR between the APIs no more than 1 dB below their inputs' PSNR (the absolute 35 dB of earlier revisions is dropped: the sample's own renderers differ by 31.4 dB, so it measured them, not NFRUX); image pairs kept |
| A6 | Route a (release 0.9.5, ini lines recorded) with the sample, D3D12 and Vulkan; route b on Vulkan | a: DLL loaded, >= 1000 evaluates, 2x presented, per API; b: more than 2x presented |
| A7 | Detection log; each rung forced; negative: nvapi / nvcuda absent, feature off, bad self-check, stale marker | `ngx_harness` passes on each; falls to the expected rung |
| A8 | Idle GPU, `parity --bench` (16 frames a submission, recording time off, median of 30): each rung, API, 1080p / 1440p / 4K, 2x / 4x / 6x; VRAM | recorded; default rung at 1080p 2x <= 2.5 ms on RTX 4070 or the miss explained |
| A9 | `ngx_harness` scenes: each D8 rule with expected pixels, SDR (UNORM and sRGB images), RGB10A2 (PQ UI and quality), float back buffer shown, reset + HUD-less, distortion at 5x, subrects, output subrects, two viewports on one back buffer, the disable word, frame ids, inverted axes, camera motion excluded, dynamic resolution, both APIs | all |
| A10 | `ngx_harness`: loads the DLL like `Nvngx_DllProxy`; protocol cases of section 4 (missing, repeated (a new batch between identical frames: outputs equal the frame within A2), out of order, n changing, n > 5), release/recreate, resize, format and HDR switches, D3D12 and Vulkan together, two features on one device with a concurrent Release, a minimal Vulkan 1.2 host (only shaderStorageImageExtendedFormats) under validation, 2000 frames | outputs as specified; no leak (VRAM and handles flat) |
| A11 | Import table and API call check: the DLL opens no shared handles and loads no other graphics API's DLL at run time (both are delay-loaded) (nvapi, nvcuda exempt on their rungs) in each mode | pass |
| A12 | Optical flow hardware vs block match: true-frame PSNR on `test_bal`, `test_new` and cost, offline | recorded; delivered only within 0.05 dB (it is not) |
| A13 | Licensing: THIRD_PARTY.md, model license shipped, changes stated | recorded |
| A14 | Quality: LPIPS (alex) and PSNR vs true frames, 2x-6x, NFRUX vs Arm NFRU v2 (as released, and with NFRUX's timestep prior), held-out clips, `tests/quality_reference.py`; and a crop sheet looked at | recorded; mean LPIPS not above Arm's at any multiplier (PSNR recorded only: it rewards blending candidates, which looks soft; changed 2026-11-26 after review). v1: fails at 2x and 4x (RESULTS), open until a model that passes replaces it. A replacement is judged on test_new and test_bal, on LPIPS and PSNR both (neither worse than Arm's at any multiplier) and on the crop sheet, so the bar is not fitted to one model. Release gate: no package is handed out until A14 passes, or the project accepts the shipped weights knowing they fail it; A4 long runs are repeated on that candidate |
| A15 | `package` target: its files listed; the DLL's imports; every license text THIRD_PARTY.md names is in it; no file name or text of a source project (search); installed and removed by one file copy as dbghelp.dll and as version.dll | as R9 |

Why these numbers: a uniform 1-code error everywhere is 48 dB, 1 code on 6% of channels is 60 dB. 0.2 dB is the run-to-run noise of the training evals.

Hardware: one RTX 4070. AMD and Intel ladders are exercised by forcing (correctness only); their speed and cooperative-matrix tile
sizes are untested, and RESULTS says so.
