# NFRUX design

Status: revision 7, under review (revision 6 was agreed by all three reviewers; docs/REVIEWS.md). Nothing under "Acceptance" counts as done until its
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

Not goals: super resolution, presentation and pacing (Streamline / OptiScaler own them), anti-cheat games, other frame generators' code
(its repository has no license).

## 2. Structure

```
loader/     process-local NGX router, active when the DLL is named as a Windows DLL (D6)
frontend/   NGX exports, parameter reading, batch protocol (section 4)
core/       interpolator: pass list, constants, histories, UI, subrects. API neutral
core/ml     TOSA lowering and CPU reference; Vulkan, D3D12 and CUDA kernels
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
| Tests | base: `nfru/test/ffx_ml_runtime_test.cpp` (A1), `tests/fg_vk_test.cpp` (A9, A10), `tests/fg_d3d12_ordering_test.cpp`, `tests/slsample_fg_scripting.patch` (A4) | MIT |
| Weights | `nfru_mfg_v14_int8` (ngsdk fork 38cfb23), derived from Arm's NFRU v2 | Arm AI Model Community License 1.0 (PRE-1154) |
| NVAPI, CUDA functions | declared by us for the few entry points used, loaded dynamically | checked in M0, in THIRD_PARTY.md |
| OptiScaler, dlssg-to-fsr3 | GPL-3.0: interface facts only | - |

Model license, read in full: clause 2 grants modification, derivatives and distribution. Clause 3 requires shipping the
license, its clause 4 as an enforceable term, original notices, and a statement of changes. Distribution is governed by the clause 3.1 and 3.2 duties (THIRD_PARTY.md, A13).

## 3. Pipeline

Capacity: `kMaxGenerated = 5`, one constant for core, frontend and tests (histories, arrays, descriptor counts). n is always
`DLSSG.MultiFrameCount` as given; t = (i + 1) / (n + 1) is never computed from a clamped count. n > 5: frames 1..5 at their
own t, the last slots (k > 5, shown just before the real frame) are the back buffer (logged). The reported maximum is 5 (6x), `NFRUX_MAX_FRAMES` lowers it. What Streamline
and the host allow on a given GPU is theirs; A4 records it.

Per rendered frame: luma pyramid, hints (camera reprojection of the previous depth, as the SDK; no
scattered game vectors: measured no gain), optical flow once, reduction to the motion lane (<= 540 rows). Per generated frame: warps, hole fill, 20-channel
tensor, network, inpainting, composite. A batch runs stage by stage.

The graph is v14: channels 16-19 are t, t|mv|/64, (1-t)|mv|/64, 0. All quantisation constants are
`nfru_mfg_v14_int8_metadata.json`'s, written in `interpolator.cpp` (checked end to end by A3; v14: params 0.0621184 / 54, gate 0.00392156 / -128; v2's are not used).

Flow at more than 1080 display rows: the SDK's colour reduction pass (01) gets an entry point and the flow, hints, flow scale
and `_OfInputSizeRcp` use the reduced size (aspect kept, 1080 rows), as the SDK does. Sizes to 8K.

Reset (first frame, `DLSSG.Reset`, size change, after menu / not rendering / release): every output of the batch is a copy of
the **back buffer** (UI included, subrect rules applied); the history stores the interpolation source.

Distortion field: the whole batch (undistort once, n interpolations, n composites).

## 4. NGX contract

Protocol per rendered frame: Evaluate is called once per generated frame with `MultiFrameIndex` k = 1..n. k = 1 makes all n
into private images and copies out frame 1; k > 1 copies out frame k. Keyed by order: k = 1 always starts a batch (no FrameID is relied on; M0a records whether one exists).
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
| *Subrect{BaseX,BaseY,Width,Height} per resource | region | whole image (or the back buffer's, same size) |
| MvecScaleX/Y | to motion-vector pixels, **not** multiplied by the size again | 1 |
| ClipToPrevClip (row-major, row vectors) | camera motion; VP is chained from it | camera basis reconstruction |
| CameraViewToClip, CameraPos/Up/Right/Fwd, Near, Far, FOV, OrthoProjection | depth and fallback camera | defaults 0.1 / 1000 / 1.0 |
| JitterOffsetX/Y, DepthInverted, ColorBuffersHDR, Reset, NotRenderingGameFrames, MenuDetectionEnabled | flags | 0 |
| MultiFrameCount, MultiFrameIndex | protocol | 1, 1 |
| ...LowPrecision.{IsLowPrecision,Scale,Bias} | distortion decoding | none |
| CmdQueue, CmdAlloc, IsRecording | not used | - |
| MvecJittered | the jitter's change taken out of the vectors (the SDK's jitter cancellation) | 0 |
| MvecDilated, MvecInvalidValue, CameraMotionIncluded | not used | - |

Parameters are read through the caller's `NVSDK_NGX_Parameter` vtable.
PopulateParameters_Impl / capability values: `FrameGeneration.Available` = 1, `.NeedsUpdatedDriver` = 0,
`.MinDriverVersionMajor/Minor` = 0, `.FeatureInitResult` = Success, `DLSSG.MultiFrameCountMax` = kMaxGenerated.

States. Observed in M0a, to be shown with the debug layer / validation before M5 exits: D3D12 COMMON in and out; Vulkan GENERAL in and out. Barriers use compute-legal states only
(NON_PIXEL_SHADER_RESOURCE, UNORDERED_ACCESS, COPY_*), since lists may be COMPUTE type. No blits: scaling or converting copies
are a compute pass. Depth: plane 0, the R32F / R24X8 / R16 SRV of its typeless family; Vulkan depth aspect. After a D3D12
Evaluate the caller must rebind descriptor heaps, root signature and PSO (NGX's rule); after Vulkan, compute bindings.

Failures: OOM / pipeline failure / unsupported size at Create -> FAIL_PlatformError or UnsupportedParameter, nothing leaked.
During Evaluate -> back buffer, Success, interpolator disabled for that size, logged.

## 5. Decisions

**D1. One shader source.** GLSL -> SPIR-V -> (SPIRV-Cross) HLSL -> (DXC) DXIL. 78 variants. Rejected: hand HLSL (drift), HLSL first (rewrites about 8k working lines). Numeric policy, enforced by
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
`tools/build_nri.ps1` with D3D11, Agility SDK, NVAPI, AGS and NVTX off (its wrapped-device teardown crashed with them; which one was not
established). It wraps the host's device, command buffer and images. The native D3D12 backend that existed was bit-identical and no faster in
the sample (564 against 572 fps at 4x) and was deleted; that removes an independent implementation, so correctness evidence is A3 and A14
(outside references), and the validation runs, and A3 is required before M5's exit. Ours, on native handles: state tracking, descriptor recycling,
completion tokens, network kernels.

What NRI needs of a wrapped Vulkan device: 1.2 and the Vulkan 1.0 features the GPU reports. `DeviceCreationVKDesc::unknownFeatures = true` (new
after v192, the reason for the pin) makes it treat every optional 1.1+ feature (bufferDeviceAddress, synchronization2, maintenance*, memoryPriority,
...) as absent; `minorVersion` is always given as 2 and no extensions are listed, whatever the GPU reports, as the host's own choice is not known. Below
1.2: creation refused, logged. NRI's "all supported 1.0 features enabled" is an assumption on both routes; the ones our call set uses are
shaderStorageImageExtendedFormats, which the loader adds. No queue family is given to NRI (it would ask for a queue the host may not have made; NFRUX never submits through it). On the OptiScaler route the
assumptions are exactly (a) 1.2, (b) shaderStorageImageExtendedFormats, logged at creation. A minimal-host test is A10.
Static linking of d3d12 and vulkan-1 is replaced by delay-load: a game on one API never loads the other.
**D2. Exact integer network.** Every path gives the CPU reference's bytes. Cooperative matrix is int8 only (the runtime's fp16
preference is overridden). D3D12 rescale uses a 32x32 -> 64 multiply from 16-bit halves, so Int64ShaderOps is not needed. Float
requantisation is off. `conv2d.hlsl` implements a restricted layer class (rescale input zero point 0, single rounding,
dilation 1, square kernel and stride, resize on source 0 only, shift >= 1, source strides equal to the words read): graph creation
checks each layer and the rung fails if any is outside it. The shape list is written by hand in `build_shaders.py`; a shape missing from it fails creation.
The `[flatten]` two-source load is split so no out-of-range address is formed. Outputs go to buffers and are copied to the
params / gate textures on every path (paths that write images directly are an optimisation, checked by the same test).

**D3. Ladders.** Chosen at creation, logged in one line, forced by `NFRUX_ML`; a forced rung that is unavailable fails creation.

| | 1st | 2nd | floor |
|---|---|---|---|
| Vulkan (NVIDIA tensor cores, Intel XMX, AMD WMMA) | cooperative matrix: int8, else fp16, else bf16 (exact sums) | dot4 (`shaderIntegerDotProduct`) | 32-bit integer |
| D3D12 NVIDIA RTX (20 and later) | CUDA via NVAPI | SM 6.4 dot4 | SM 6.0 |
| D3D12 other (AMD, Intel, GTX) | SM 6.4 dot4 | - | SM 6.0 |

Vulkan baseline (passes and floor): 1.2 (NRI with unknownFeatures, D1b) and one feature, `shaderStorageImageExtendedFormats` (storage images in r8, rg16f,
rgb10_a2, r11f_g11f_b10f; Streamline_Sample does not enable it). BGRA8 outputs: the composite stores blue first into an RGBA8 image of
its own, copied raw (size-compatible) into the host's. Everything
else is optional and needs to be *enabled*. Source of truth:
- Standalone route: the loader intercepts `vkCreateDevice`, obtained through GetProcAddress (vulkan-1.dll and sl.interposer.dll
  exports) and through vkGetInstanceProcAddr / vkGetDeviceProcAddr, whose returned pointer for that name it wraps. It records the enabled
  extensions and features and appends, when supported per `vkGetPhysicalDeviceFeatures2`: shaderStorageImageExtendedFormats,
  cooperativeMatrix, vulkanMemoryModel (+ DeviceScope), shaderInt8, storageBuffer8BitAccess, shaderIntegerDotProduct
  and their extensions. It also wraps vkDestroyDevice (what is on the device is destroyed first). Merge rule: known structures of the pNext chain are copied, unknown ones kept in place (their pNext is changed for the call and put back); a feature that belongs to an aggregate
  present in the chain (VkPhysicalDeviceVulkan11/12/13Features, VkPhysicalDeviceFeatures2) is set in that copy; its individual struct
  present is set in that copy; else the individual struct is chained. Core features go into pEnabledFeatures when the app uses that, else
  Features2. The app's own values are never cleared. If creation with additions fails, the app's original call is made. Only the
  app's own process and device are affected; additions are logged.
- OptiScaler route: the enabled set is unknown. The floor is used, with the two assumptions of D1b (Vulkan 1.2, and that
  `shaderStorageImageExtendedFormats` works (reported by every desktop driver; outside strict validity when the app has not enabled it,
  and logged so). Formats rgba8/rgba16f/r32f/r32ui need no feature. `NFRUX_VK_ASSUME_FEATURES=1` (opt-in, same caveat) enables the
  optional rungs; the installer offers it for drivers listed as tested.
The loader hands the DLL the recorded set per VkDevice (export `NFRUX_SetVulkanFeatures`); no record means the OptiScaler behaviour.
A4's validation run uses the standalone loader, which makes the cooperative-matrix rung valid. Fallback when M0a cannot intercept
vkCreateDevice: Vulkan uses the floor, A4 claims only that, and the ladder text is amended.
M0d result (RESULTS): SM 6.9 cooperative vectors are in neither the inbox runtime (SM 6.8) nor the retail SDK; DirectML's integer convolution is
exact but slower than dot4. Neither is a rung.

CUDA: `core/ml/cuda_network.cpp` writes PTX (sm_75, `mma.sync` m8n8k16 s8, one kernel a layer). Nothing is precompiled: the user's own driver links it for
the GPU it is, on the worker thread, with a short-lived context matched by LUID (checked: the same PTX links for sm_75, 80, 86, 89, 90, 100 and 120). m8n8k16 is as
fast here as sm_80's m16n8k16, so there is one form. GTX 16 reports sm_75 without tensor cores and is told apart by name; GTX 10 and 16 use dot4, the
same hardware instruction CUDA would. Cubins are kept in memory only. NVAPI `CreateCuModule`, `CreateCuFunction`, `LaunchCuKernelChain` by public interface id
launch it. UAV barriers between launches, inputs in UNORDERED_ACCESS.

AMD and Intel matrix hardware is reached through Vulkan cooperative matrices: any M, N up to 16 and K a multiple of 4; subgroups of 32 on AMD, of N on Intel when
the size can be required. Untested on such GPUs (none here): the self-check and the next rung are the safety. On D3D12 there is no public shader interface to
them (no cooperative vectors in the retail runtime; AMD's and Intel's are device-creation or vendor-driver extensions), so dot4.
Crash safety for CUDA (the one rung that depends on vendor-specific launches): before the first use in a process an "attempts" file, named by device,
adapter and driver file time, is counted up; it is removed when ten recordings have completed, or on an orderly end, and kept after a failed
self-check or a device removal. At two, the rung is not used (one line in the log, `NFRUX_IGNORE_MARKERS`). No folder: not used. A device loss in the
game's process cannot be recovered and is not claimed to be. The cooperative-matrix and dot4 rungs are plain API use and have only the self-check.

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
   `nvngx_dlssg.dll` or `dlssg_to_fsr3_amd_is_better.dll` the feature; `dbghelp.dll`, `version.dll` or `winmm.dll` the feature and the loader (their exports are
   forwarded to Windows' own). Uninstalling is deleting it. The `package` target holds the DLL, INSTALL.txt, THIRD_PARTY.md and the license texts, nothing else.1. OptiScaler, two routes:
   a. Releases (0.9.5 the newest): the dlssg-to-fsr3 slot. The DLL as `dlssg_to_fsr3_amd_is_better.dll` beside OptiScaler, `FGInput=nukems`, `FGOutput=nukems`.
      That slot offers 2x only. This is the route R2 is judged by.
   b. Builds with `FGNvngxReplacement`: `OptiScaler/custom/NFRUX/dlssg/nvngx_dlssg.dll`, `FGNvngxReplacement=NFRUX`; up to
      what it asks for. On D3D12 that build crashes the sample by itself (control: `FGInput=nofg`), so it is judged on Vulkan only.
   OptiScaler loads the DLL directly and supplies parameters and presentation. Vulkan features are unknown there (D3). Whether it calls Shutdown is
   not known: NFRUX does not depend on it.
2. Standalone. The NGX static library inside Streamline loads the core `_nvngx.dll` from the driver folder named in the registry,
   and no machine-wide setting may be changed.
   Mechanism: the file as `dbghelp.dll` (sl.interposer.dll imports it; `version.dll` / `winmm.dll` for hosts that do not) in the game folder hooks
   `LoadLibraryExW` and `GetProcAddress` by import-table patching of the modules loaded at attach (enumerated) and later ones through
   LdrRegisterDllNotification (in-process only; Streamline loads its over-the-air plugins from ProgramData). When the core is loaded, the returned exports of interest are wrapped:
   Vulkan Init* (note device), Shutdown* (NFRUX first, then the core), GetCapabilityParameters / GetParameters (add the section 4 capability values),
   D3D12 GetFeatureRequirements (feature 11 supported), CreateFeature / EvaluateFeature / ReleaseFeature. Not wrapped: AllocateParameters, extension
   requirements, `_C` evaluates, LoadLibraryA/W (the sample runs did not need them). Feature 11 goes to NFRUX, everything else to the real core untouched. Handles NFRUX returns are its own
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
| The D3D12 debug layer cannot run here (Graphics Tools needs elevation; Agility's layers are refused by this runtime) | A4 stays partly met until it is installed or accepts the gap in writing; NRI's validation is no replacement (it does not check wrapped images' states) |

## 8. Acceptance

Evidence rule: a row's final evidence in RESULTS names the commit and the build (strict or shipped). After a change to shaders, kernels, rhi or front end
A2, A5 (replay), A7, A9 and A10 are run again; after one to passes or constants A3 and A14 too; after one to kernels or the network runtimes A1; A4 and A6 on
the release candidate. Older entries are history.

Inputs: `parity`'s synthetic scenes (pan and moving square, made in the test), recordings of the sample (`NFRUX_DUMP`, `NFRUX_DUMP_FRAMES`, `--replay`), and
the held-out Blender clips (`C:\dev\modelgym\data\nfru6x\test_new`, `test_bal`, 180 fps truth) as `quality_reference.py` exports them. Encodings RGBA8, BGRA8,
RGB10A2; normal and inverted depth; odd sizes.

| # | Test (`tests/`) | Pass |
|---|---|---|
| A1 | `ml_conformance_d3d12`, `ml_conformance_vulkan`: each rung vs CPU reference graph outputs; sizes 64x64, 744x464, 960x544 on D3D12; random seeds, and all -128, all 127, alternating on D3D12 | 0 differing bytes |
| A2a | `parity --strict`, strict build: synthetic scenes (1080p, 4K, 8K, ultra-wide, odd; RGBA8, BGRA8, RGB10A2; both depths) and 64-frame recordings, 2x / 4x / 6x; exit code | 0 differing bytes in every output and intermediate |
| A2b | `parity`, shipped build, the same | PSNR >= 60 dB, >= 99.9% of values within 1 code, <= 0.01% off by more than 3 (RGB10A2 by 10-bit code, RGBA16F by value with the scene's peak as 255 codes); exit code |
| A3 | `tests/quality_reference.py`: port with v14 int8 vs the torch v14 int8 pipeline (`eval_mfg` pipeline, `C:\dev\modelgym\output\nfru_mfg_v14\qat15\last.pt`, same clips, 8-bit colour and half-float vectors for both) | PSNR against the true frames within 0.2 dB of torch per multiplier |
| A4 | Sample, D3D12 and Vulkan, scripting patch: 2x and the maximum granted, 1920x1080, 5 min each on release DLLs; 60 s with D3D12 debug layer + GPU validation / Vulkan validation + sync | no crash; 0 messages naming NFRUX objects or thrown inside its recordings; `numFramesActuallyPresented` = n + 1 for >= 95% of states logged; OutputReal written every evaluate; dumped generated frames lie between neighbours (mean luma distance to both <= distance between them + 2 codes, >= 99% of frames) |
| A5 | Sample with fixed timestep, scripted camera, DLSS off: dump 64 frames on both APIs (HUD-less colour); replay each dump on both APIs | replay meets A2a (strict) and A2b (shipped); live: generated PSNR between the APIs no more than 1 dB below their inputs' PSNR (the absolute 35 dB of earlier revisions is dropped: the sample's own renderers differ by 31.4 dB, so it measured them, not NFRUX); image pairs kept |
| A6 | Route a (release 0.9.5, ini lines recorded) with the sample, D3D12 and Vulkan; route b on Vulkan | a: DLL loaded, >= 1000 evaluates, 2x presented, per API; b: more than 2x presented |
| A7 | Detection log; each rung forced; negative: nvapi / nvcuda absent, feature off, bad self-check, stale marker | `ngx_harness` passes on each; falls to the expected rung |
| A8 | Idle GPU, `parity --bench` (16 frames a submission, recording time off, median of 30): each rung, API, 1080p / 1440p / 4K, 2x / 4x / 6x; VRAM | recorded; default rung at 1080p 2x <= 2.5 ms on RTX 4070 or the miss explained |
| A9 | `ngx_harness` scenes: each D8 rule with expected pixels, SDR (UNORM and sRGB images), RGB10A2 (PQ UI and quality), float back buffer shown, reset + HUD-less, distortion at 5x, subrects, both APIs | all |
| A10 | `ngx_harness`: loads the DLL like `Nvngx_DllProxy`; protocol cases of section 4 (missing, repeated (a new batch between identical frames: outputs equal the frame within A2), out of order, n changing, n > 5), release/recreate, resize, format and HDR switches, D3D12 and Vulkan together, two features on one device with a concurrent Release, a minimal Vulkan 1.2 host (only shaderStorageImageExtendedFormats) under validation, 2000 frames | outputs as specified; no leak (VRAM and handles flat) |
| A11 | Import table and API call check: the DLL opens no shared handles and loads no other graphics API's DLL at run time (both are delay-loaded) (nvapi, nvcuda exempt on their rungs) in each mode | pass |
| A12 | Optical flow hardware vs block match: true-frame PSNR on `test_bal`, `test_new` and cost, offline | recorded; delivered only within 0.05 dB (it is not) |
| A13 | Licensing: THIRD_PARTY.md, model license shipped, changes stated | recorded |
| A14 | Quality: PSNR vs true frames, 2x-6x, NFRUX vs Arm NFRU v2 (as released, and with NFRUX's timestep prior), held-out clips | recorded; mean not below Arm's at any multiplier |

Why these numbers: a uniform 1-code error everywhere is 48 dB, 1 code on 6% of channels is 60 dB. 0.2 dB is the run-to-run noise of the training evals.

Hardware: one RTX 4070. AMD and Intel ladders are exercised by forcing (correctness only); their speed and cooperative-matrix tile
sizes are untested, and RESULTS says so.
