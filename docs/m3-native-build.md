# Native Apple M3 build (local only)

The shipped artifacts are **portable**: the target is `arm64-apple-macos26.0`
with no CPU flag, so the codegen baseline is clang's default for the triple
(`apple-m1`). That is deliberate — an M1 through an M6 can run the release, and
`tools/release.sh` and `tools/build_library.sh` never add a CPU flag.

`tools/build_m3.sh` is the opt-in for a machine that will **not** ship the
result: it builds every product with `-target-cpu apple-m3` (Swift) and
`-mcpu=apple-m3` (the C kernels), into a separate scratch path so the portable
build is untouched.

## Use

```bash
tools/build_m3.sh                 # clean build into .build/m3
M3_KEEP=1 tools/build_m3.sh       # reuse the scratch
M3_SCRATCH=/path/.build/m3-other tools/build_m3.sh
```

It requires an arm64 host whose `machdep.cpu.brand_string` starts with
`Apple M3` (M3, M3 Pro, M3 Max, M3 Ultra). The default scratch is
`.build/m3/`, which is gitignored.

## What it builds

| Artifact | Product |
|---|---|
| `TinyTitanCLI`, `TinyTitanServer`, `TinyTitanRepack`, `TinyTitanBench`, `ContinuityDemo`, `tinytitan-memory` | executables |
| `libTinyTitanLib.a` | static library |
| `libTinyTitanLibDynamic.dylib` | dynamic library |

The Metal shaders are not compiled in this step: `MetalContext` compiles the
`.metal` sources at load time on the device, so there is no build-time GPU
target to flag. The C kernels do get `-mcpu=apple-m3`, and the Swift products
get `-target-cpu apple-m3`.

## Verification

The script is not satisfied by intending the flags:

- It builds from a clean scratch with `-v` and fails unless the compiler log
  contains `-target-cpu apple-m3` and `-mcpu=apple-m3`.
- It checks every executable is present, `lipo -archs` reports exactly `arm64`,
  and `vtool -show-build` reports a macOS platform.
- It checks both library artifacts the same way.

## What it is not

- **Not a release artifact.** `tools/release.sh` does not call it, and its
  output must not be handed to another Mac or published.
- **Not a measured speed win.** Naming this Mac's CPU measured about **1 %**,
  inside the noise, because the CPU kernel is float-based rather than an integer
  dot product (see `AGENTS.md`). A 3-pair, 32-token decode of the 4B on this M3
  measured 22.44–22.50 tok/s for the portable build and 22.45–22.54 tok/s for
  the M3 build: no difference outside the noise. The M3 build is about using
  the codegen that matches the machine, not a performance claim.
- **Not a change to the default.** A normal `swift build -c release` remains the
  portable build, and it is what CI and the release verify.
