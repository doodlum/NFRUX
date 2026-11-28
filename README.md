# NFRUX

NFRUX is frame generation for games that support NVIDIA DLSS Frame Generation (DLSS-G). It is one DLL that answers where DLSS-G's
`nvngx_dlssg.dll` would. The game, or OptiScaler, asks for DLSS-G and NFRUX makes the frames: 2x to 6x, on D3D12 and Vulkan.

Each generated frame comes from a small int8 neural network, fine-tuned from Arm's NFRU v2, that blends motion-compensated candidates of the two
rendered frames around it. The game's UI is kept out of the interpolation.

NFRUX is an independent project, not affiliated with or endorsed by NVIDIA or Arm.

## Status: 0.0.1 (first release)

| GPU | What runs | Tuned |
|---|---|---|
| NVIDIA RTX 40 | CUDA int8 tensor-core kernels (D3D12 and Vulkan) | yes, on an RTX 4070 |
| NVIDIA RTX 20, 30, 50 | the same CUDA kernels | no: RTX 40 settings, timed against the alternatives on first start |
| NVIDIA GTX, AMD, Intel | Vulkan cooperative matrices, integer dot products (dot4) or 32-bit integers; D3D12: dot4 or integers | no: generic settings, timed on first start |

On its first start on a GPU, NFRUX times every path that works there and keeps the fastest. Every path gives exactly the same image.
Only the RTX 4070 has actually been measured. If you have other hardware, the [benchmark kit](docs/OPTIMISING.md#running-the-kit-testers)
takes about ten minutes and helps tune NFRUX for your GPU. What is known per GPU family is in [docs/VARIANTS.md](docs/VARIANTS.md).

Cost on the RTX 4070, ms per rendered frame (headless harness, idle GPU; details in [docs/RESULTS.md](docs/RESULTS.md)):

| | 1080p 2x / 6x | 4K 2x / 6x |
|---|---|---|
| NFRUX, D3D12 | 0.77 / 2.35 | 1.17 / 3.81 |
| NFRUX, Vulkan | 0.92 / 2.59 | 1.47 / 4.29 |
| NVIDIA DLSS-G 310.9, D3D12 (2x only on RTX 40) | 1.76 / - | 3.63 / - |

## Install

Get `NFRUX_<version>.zip` from the [releases](../../releases). NFRUX is one file, `nvngx_dlssg.dll`, plus an optional `nfrux.ini`.
Use **one** route:

- **OptiScaler with `FGNvngxReplacement`** (2x-6x): put the DLL at `OptiScaler\custom\NFRUX\dlssg\nvngx_dlssg.dll`, and in `OptiScaler.ini` set
  `FGInput=nvngxfg` and `FGNvngxReplacement=NFRUX`.
- **OptiScaler 0.9.5** (2x): put it beside OptiScaler as `dlssg_to_fsr3_amd_is_better.dll`, and set `FGInput=nukems` and `FGOutput=nukems`.
- **A game with DLSS-G, no OptiScaler**: put it beside the game's `.exe` as `dbghelp.dll`, or as `version.dll` if the game doesn't load `dbghelp.dll`.
  This route also turns on the Vulkan features NFRUX's fastest Vulkan paths need when the game creates its device.

To uninstall, delete the file. Nothing is registered. A crash marker lives in `%LOCALAPPDATA%\NFRUX` during a game's first frames, and the timed choice is cached
there. Not for games with anti-cheat.

## Options

`nfrux.ini` beside the DLL (every line is optional):

| Key | Values |
|---|---|
| `[General] MaxMultiplier` | 2-6: the highest multiplier offered to the game |
| `[General] Log` | 1: write `nfrux.log` beside the DLL (attach it to bug reports) |
| `[D3D12] Network`, `[Vulkan] Network` | `auto`, or force a path: `cuda`, `coopmat` (Vulkan), `dot4`, `scalar` |
| `[Vulkan] FullBarriers` | 1: stricter GPU synchronisation, if you see flicker or corruption |

## Requirements

- Windows 10 or 11, x64.
- D3D12: shader model 6.0; 6.4 for dot4.
- Vulkan: 1.2, through a game that uses Streamline (all the routes above).
- CUDA paths: an NVIDIA RTX GPU and a current driver.
- Back buffers: 8-bit or 10-bit (RGB10A2) colour.

NFRUX uses only the Vulkan features that are enabled on the game's device (docs/DESIGN.md, "Vulkan features").

## Problems

Set `Log=1` in `nfrux.ini`, reproduce, and attach `nfrux.log` (and `nfrux_loader.log` if it is there) to an issue. `Network=scalar` takes the
plainest path, which helps tell a driver problem from an NFRUX one.

## Building

You need:
- Visual Studio 2022 (MSVC, x64);
- CMake 3.20 or newer;
- the Vulkan SDK (headers, loader library, glslang and DXC);
- Python 3.

```powershell
tools\build_nri.ps1                                   # NRI at its pinned commit, into third_party\NRI
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release                  # build\NFRUX\Release\nvngx_dlssg.dll and the tests
cmake --build build --config Release --target package # build\package: what a release ships
cmake --build build --config Release --target bench_kit
```

`-DNFRUX_STRICT_MATH=ON` makes the strict build: no fused or reordered shader arithmetic, so D3D12 and Vulkan give identical bytes. It's
for regression checks; releases use the default build. The tests (`ml_conformance_d3d12`, `ml_conformance_vulkan`, `parity`, `ngx_harness`, `rungs_test`)
are headless and need a GPU, except `rungs_test`. [docs/OPTIMISING.md](docs/OPTIMISING.md) lists the gates a change must pass.

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md): how NFRUX works and why.
- [docs/RESULTS.md](docs/RESULTS.md): every measurement, dated.
- [docs/VARIANTS.md](docs/VARIANTS.md): each way the network runs, which GPUs use it, and what's tuned where.
- [docs/OPTIMISING.md](docs/OPTIMISING.md): the benchmark kit and how to make NFRUX faster on a GPU.

## Licence

The code is MIT ([LICENSE](LICENSE)). The network weights are modified from Arm's NFRU v2 and are under the Arm AI Model Community License
1.0, whose clause 4 (acceptable use) binds anyone who receives them. Third-party parts and their licences are in [THIRD_PARTY.md](THIRD_PARTY.md).
