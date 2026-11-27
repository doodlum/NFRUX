# Optimising NFRUX for a GPU

NFRUX runs one network in several ways (variants), and picks the fastest that works on each GPU. This guide is for two kinds of
helper:
- **testers**, who run the benchmark kit on their GPU and send the report;
- **optimisers**, who make a variant faster.

What exists and how much we know about it is tracked in [VARIANTS.md](VARIANTS.md).

## The rules

1. **Same bytes.** Every variant gives exactly the same output as every other (DESIGN D2). A change that alters an output byte is a
   model change, not an optimisation. The conformance tests and `parity --strict` enforce this.
2. **Add, don't replace.** A new kernel or tuning becomes a new variant beside the old one. The first-start timing (DESIGN D3,
   `core/rungs.cpp`) measures both on each GPU and keeps the faster one, so a speed-up for one GPU can't become a slow-down for
   another. Remove a variant only when reports show it losing on every family that can run it.
3. **Gate by capability, not by name.** Anything that needs an instruction, extension or feature checks for it (compute capability,
   enabled Vulkan feature, shader model) and falls back otherwise. Vulkan features count only when they are enabled on the game's
   device (DESIGN, "Vulkan features").
4. **Measure, then keep.** Any speed-up that is measurably real and exact is kept, however small. Anything that doesn't measure faster
   is reverted. Compare by running the old and new builds alternately, at least twice each, on an idle GPU.
5. **Say where it was tuned.** A commit and VARIANTS.md name the hardware the numbers came from.
6. Name no other frame-generation product in the repository (code, comments, docs, commit messages).

## Running the kit (testers)

You need Windows, a recent GPU driver and about ten minutes with nothing else using the GPU.

1. Get `NFRUX_bench_kit_<version>.zip` from the release page, and unpack it into a new folder.
2. Close games, video players and anything else using the GPU.
3. Right-click `run_bench.ps1` > **Run with PowerShell**. Use `-Quick` for a 3-minute run at 1080p only.
4. Send `report\nfrux_bench_<GPU>_<date>.md`, for example as a GitHub issue titled "bench: <GPU>". If a check says
   **FAILED**, attach the `report\raw\<date>` folder too: that is a correctness bug and more urgent than speed.

The kit installs nothing, opens no window, and writes only inside its folder. The report holds the GPU's name and PCI ID, the
driver and Windows version, the correctness results, what NFRUX chose, and the timings.

## Reading a report and updating the tracker (maintainers)

1. Put the report in `docs/bench/` (keep the file name).
2. Check the correctness table first. Any failure is a bug: open an issue with the raw folder and stop.
3. Read "What NFRUX chose": the family (`rungs: ...`), the order, any quirk, and the timing result (`rung timing: ...`).
4. In the CSV, compare the variants of each API at each size and multiplier. `ms` is the frame cost; `first` is the evaluate that
   runs the network; `cpu` is NFRUX's own CPU time. `not available` means the path can't run there; `NO OUTPUT` is a bug.
5. Update that family's rows in VARIANTS.md: **untested**/**inherited** becomes **measured**, with the report linked. Look for:
   - the automatic choice losing to a forced variant by more than run-to-run noise (about 3%): the timing or the plan needs a
     look (`core/rungs.cpp`);
   - a variant much slower than its neighbours (relative to the RTX 4070 report): a **gap**. Add it under "Open gaps" with numbers.

## Building the kit (maintainers)

```powershell
cmake --build <build dir> --config Release --target bench_kit
```

`<build dir>\bench_kit` holds the DLL, `ngx_harness`, both `ml_conformance` tests, `parity`, `run_bench.ps1`, its README and the
licences. CI attaches it to each release as `NFRUX_bench_kit_<version>.zip`.

## Optimising a variant (optimisers)

### 1. Pick a target

Take a **gap** or a **measured** row from VARIANTS.md. Work on hardware that runs that variant: the timing tools force a variant,
but they can't emulate another GPU.

### 2. Profile

| Tool | What it tells you |
|---|---|
| `ngx_harness <dll> --time [WxH[@N]] d3d12\|vulkan` | ms a rendered frame per evaluate, plus NFRUX's CPU time (N generated frames: 1 = 2x ... 5 = 6x) |
| `NFRUX_TEST_TIMES=1` + `NFRUX_LOG=<file>` | GPU time per pass, median a recording (Vulkan; D3D12 only on a device with a timestamp queue) |
| `ml_conformance_d3d12 layers [w h variant]` | CUDA GPU time per network layer, 1 and 5 networks |
| `model_layers [w h]` | the network's layers, multiply-adds and activation bytes: the ceiling to compare with |
| `NFRUX_ML`, `NFRUX_TEST_CUDA_VARIANT`, `FFX_ML_DOT4_TILED`, `FFX_ML_DOT4_SUBGROUP` | force a variant (VARIANTS.md, "The variants") |

Set `LOCALAPPDATA` to an empty folder while testing: NFRUX keeps its crash markers and the timed choice there, and a cached choice
would hide the variant you're testing.

Compare against what the hardware can do: multiply-adds over the time against the GPU's int8 or fp16 peak, and bytes over the
time against its memory bandwidth. A pass close to bandwidth needs fewer bytes, not faster code.

### 3. Change it

| Kind of change | Where | Notes |
|---|---|---|
| A new CUDA kernel variant | `core/ml/cuda_network.cpp` MakeCudaPlan; raise `kCudaVariants` | The timing tries every variant (`rhi/rhi_nri.cpp` StartTrial); the cache text is `cuda/<n>` |
| Vulkan coopmat or dot4 tuning for a vendor | `core/ml/vulkan_network.cpp` TuningFor | One row a vendor; a different setting for one architecture needs a new variant or a device-ID check with a fallback |
| A family's default order, or a quirk | `core/rungs.cpp` `kPriors` / `kQuirks` | Add the GPU to `tests/rungs_test.cpp` |
| A pass (flow, warp, post-process ...) | `core/interpolator.cpp`, `core/shaders`, `core/gpu` | Shared by every GPU: check it doesn't slow another family (ask for reports) |
| Recording cost (CPU) | `rhi/rhi_nri.cpp` | Measure with the `calls ... ms CPU` column |

### 4. Gates (all must pass before a commit)

```powershell
ml_conformance_d3d12; ml_conformance_vulkan                       # EXACT for every path this GPU has
parity                                                            # worst PSNR >= 99 dB (shipped build)
parity --strict                                                   # 0 differing bytes (NFRUX_STRICT_MATH build), for shader changes
ngx_harness <dll> d3d12; ngx_harness <dll> vulkan                 # 78 checks, 0 failed; again with NFRUX_ML=dot4, scalar, coopmat
rungs_test                                                        # the plans for GPUs not on this machine
```

### 5. Record it

- **Commit message:** what changed, the hardware, and before / after numbers.
- **docs/RESULTS.md:** a dated entry with the before/after table and what didn't help.
- **docs/VARIANTS.md:** the row's status and evidence, and the gap closed or narrowed.
- **A fresh kit report** from the hardware you tuned on, in `docs/bench/`.
