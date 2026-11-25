# Design reviews

| Revision | Reviewer | Verdict | Blocking |
|---|---|---|---|
| 1 | architecture | DISAGREE | 7 |
| 1 | code accuracy | DISAGREE | 9 |
| 1 | readiness and tests | DISAGREE | 14 |

Revision 2 answers: precision and contraction (D1), int8-only exactness and layer checks (D2), enabled Vulkan features (D3), NVOF rule
(D4), completion token / threads / latency (D5), loader mechanism and gate (D6), NGX contract (4), capacity 5 (3), reset and distortion
(3), 4K flow (3), license (2), milestones (6), measurable acceptance (8).

| Revision | Reviewer | Verdict | Blocking |
|---|---|---|---|
| 2 | architecture | AGREE | 0 |
| 2 | code accuracy | AGREE | 0 |
| 2 | readiness and tests | DISAGREE | 1 (Vulkan baseline feature, merge rule, interception) |

Revision 3: that blocker (D3, D6 gate), plus the reviewers' suggestions on tokens, markers, FrameID, A3, A11, D7.
| 3 | readiness and tests | AGREE | 0 |

Architecture and code accuracy agreed on revision 2; revision 3 changed only what the third reviewer required plus their own suggestions.

| 4 | architecture | DISAGREE | 2 |
| 4 | code accuracy | DISAGREE | 2 |
| 4 | readiness and tests | DISAGREE | 4 |
| 5 | architecture | AGREE | 0 |
| 5 | code accuracy | AGREE | 0 |
| 5 | readiness and tests | AGREE | 0 |

Revision 5 at commit 61d7e69 is the agreed design; the edits after it are the reviewers' own text corrections.

Revision 6 (after M0d, A12, A3 and two decisions): strict / relaxed math (D1); the nfru-mfg branch as the reference (D1); optical flow
hardware dropped with numbers (D4); cooperative vectors and DirectML dropped (D3); scatter removed; pass 01 in; D7 as built; MvecJittered; A3 as
run; sizes to 8K.

| 6 | architecture | DISAGREE, then AGREE | 2 (sRGB images; Shutdown and device end), fixed |
| 6 | code accuracy | DISAGREE, DISAGREE, then AGREE | 2 (Vulkan video memory not measured; A3 table owed), fixed |
| 6 | readiness and tests | DISAGREE, then AGREE | 4 (R5, OptiScaler routes, strict / shipped criteria, evidence rule), fixed |

Revision 6 is the agreed design (2026-11-25). A1 was narrowed to what the conformance programs run.

Revision 7: one DLL (role by file name, static runtime, no files on disk); scRGB / float back buffers out; CUDA from sm_75 with one mma form, per-GPU link by the driver; fp16 / bf16 cooperative matrices where int8 is missing, Intel subgroup rule; WARP and lavapipe tested; file names; low-frame-rate test.

| 7 | architecture | DISAGREE | CUDA name test; R9 / A15 and the package's license texts; winmm role |
| 7 | code accuracy | DISAGREE | fp16 / bf16 exactness not shown; bf16 not reachable through the loader |
| 7 | readiness and tests | DISAGREE | the same, and worst-case inputs on Vulkan |

Revision 8: those fixed (name test by whole words with a table test; 2^24 bound computed and enforced, extreme inputs; D3 says where bf16 is reached; R9, A15, all
license texts in the package; winmm dropped, version.dll run); DLSS-G guide gaps (host states, output subrects, UpdateOnlyInsideExtents, disable word, frame ids, axes,
camera motion, dynamic resolution, VRAM callback, `_C` evaluates); fp16 first on NVIDIA Vulkan; 64-bit atomics removed; the DLL pins itself.

| 8 | architecture | AGREE (253ebe3) | 0 |
| 8 | code accuracy | DISAGREE, DISAGREE, then AGREE (db5dc31) | 4 (jitter under dynamic resolution; 1.3 extensions; the int8 retry; overlapping copies), fixed |
| 8 | readiness and tests | AGREE (253ebe3) | 0 |

Revision 8 is the agreed design (2026-11-25). Architecture and readiness agreed before the fixes the third reviewer asked for and checked.

Revision 9 (the D3D12 debug layer's findings: input states, tags, disable word, readback lifetime, 0x40 on DIRECT lists; Vulkan validation; the comparison).

| 9 | architecture | DISAGREE (the tag hook was too wide), then AGREE | 1, fixed |
| 9 | code accuracy | DISAGREE (the same), then AGREE | 1, fixed |
| 9 | readiness and tests | DISAGREE, then AGREE | 5 (DLL provenance, commit and A4, Vulkan attribution, UI column, ceiling), fixed |

Revision 9 is the agreed design (2026-11-25). After it: the reviewers' small points on stale state entries and UI alpha.
