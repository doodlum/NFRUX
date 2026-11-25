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
