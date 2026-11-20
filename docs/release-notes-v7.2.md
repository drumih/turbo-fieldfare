# TinyTitan 7.2 — Ornith decode restored; dense Qwen 3.5 qualified and shipped

This release fixes the Ornith 1.5 35B-A3B 4-bit decode regression that shipped
in 7.1, and promotes the dense Qwen 3.5 family from "runs on 7.0 engines" to
**qualified and registry-shipped** on both CPU and GPU. It also lands the stable
log-softmax preparation and removes the dead deprecated declarations.
No model was added or converted in this cut.

### Ornith 4-bit decode restored

7.1 routed Ornith 4-bit through `Ornith35B4Bit`, but that variant's factory
loaded a fixed **32-slot** expert cache and read only
`TINYTITAN_EXPERT_CACHE_SLOTS`. The session's resolved slot count — the 10 GiB
profile, `--ram-budget`, `--expert-cache-slots` — was computed and never passed
to the model. The 6.x profile for this pair is 128–160 slots and measured
20.4–21.0 tok/s; the 7.1 variant measured **~7 tok/s**, and a RAM-budget sweep
was really six runs of the same 32-slot engine.

`EngineFactory` now has an optional `buildModelWithSlots(directory, arch,
context, slots)`, `ServerModelSession.load` passes its resolved load-plan slot
count, and the Ornith variant uses it; the environment override remains the
operator's last word. A routing test pins the resolved count (64) through the
path.

Measured on the shipped 4-bit install, 192 greedy tokens, three interleaved
rounds (corrected record in `docs/ornith-ram-budget-decode-2026-11-20.md`):

| RAM budget | Mean tok/s | Range |
| --- | ---: | ---: |
| 8G | **16.41** | 15.08–17.31 |
| 10G | **19.02** | 17.39–20.20 |
| 12G | **20.10** | 18.78–20.81 |
| 14G | **21.33** | 21.07–21.52 |
| 16G | **20.96** | 20.62–21.60 |
| max (no flag) | **21.35** | 20.81–21.64 |

The `ornith-4` golden is byte-identical and the full suite and twenty gates are
green. 8G is still usable, but it now costs ~23% against max; 10–12G is the
low-pressure compromise.

### Dense Qwen 3.5 is qualified and registry-shipped

All twelve dense variants (2B/4B/9B at 4 and 8 bits, CPU and GPU) now carry a
qualification record: the golden body SHA-256, a two-run 64-token decode mean,
hardware and date. The CPU variants had no baselines of their own, so
`tools/golden-baseline.sh` gained an explicit `--engine cpu` mode and six CPU
dense baselines were captured and re-checked byte-identically. The release gate
now checks both the GPU and the CPU dense baselines.

| Model | 4-bit GPU / CPU | 8-bit GPU / CPU |
| --- | ---: | ---: |
| 2B | 54.10 / 19.64 | 35.71 / 19.13 |
| 4B | 27.68 / 8.28 | 17.23 / 9.11 |
| 9B | 15.92 / 4.44 | 9.38 / 5.19 |

`shipped: true` is set on all twelve pairs; `catalog --check` and
RegistryInventoryTests are green. The registry's Ornith qualification still
quotes the old 32-slot 11.55 tok/s; re-qualifying at the new 160-slot default is
tracked as TT-108.

### Stable log-softmax adopted

`StableLogSoftmax` implements the arXiv:2610.31642 margin form: the argmax is
excluded from the log-sum-exp, the loss is `softplus(-margin)`, and the paper's
`[9,-9,-9]` case returns `3.046e-8` where the standard float form returns 0. The
CPU NLL scorer delegates to it. No logprobs API exists yet; when one is added it
must report these stable quantities.

### Also in this release

- The three unreferenced deprecated helpers (`releaseLayerStreamer`,
  `routedExpertBits(layer:expert:)`, `requireSharedScalarGateBuffer`) are gone
  from the 6.x runtime and the Ornith/DenseGPU copies (TT-050).
- The layer-major-prefill record no longer names a deleted helper.
- A tracker row (TT-107) records the Ornith slot regression; TT-108 tracks the
  Ornith re-qualification at the new default.

### Performance

The Ornith 4-bit table above is the release's main performance correction; the
dense table is the qualification measurement. Both were taken on the same 24 GB
M3, and no benchmark was re-run for the other families. The earlier 7.1
Ornith figure (~7 tok/s) was a configuration defect, not the engine's rate.

### Checks not run

The installed models on this machine are Ornith 1.5 35B-A3B 4-bit and the six
dense Qwen 3.5 installs; the release gate checks their thirteen golden targets
(seven GPU, six CPU). No model may be fetched to fill a row. The following
baselines are **not checked** — no install under `models/`: `ornith-8`,
`qwen36-4`, `qwen36-8`, `qwen38-4`, `qwen38-8`, `k2-4`, `k2-8`,
`agentworld-4`, `agentworld-8`, `katcoder-4`, `katcoder-8`, `ornith122-4` and
`ornith122-8`.

### Checksum

`tinytitan-7.2-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-7.2-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-7.2-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-7.2-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-7.2-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-7.2-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
