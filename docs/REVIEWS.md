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
