# TinyTitan 7.0 — one engine per model: the EngineVariant architecture, shipping Ornith 1.5 35B-A3B 4-bit

This release is the 7.0 redesign: one engine path per (model, quantization)
pair as an `EngineVariant` target under `sources/Engines/`, a generic
`TinyTitanCore` library that carries no model knowledge, and a single
registry (`registry/registry.json`) checked by `tinytitan catalog --check`.
The 7.0 release ships **one variant: Ornith 1.5 35B-A3B 4-bit**
(`Ornith35B4Bit`), built and parity-verified against the 6.x runtime. The
dense Qwen 3.5 ports and the generic Core foundation also landed, as the
base future variants build on — they are not the 7.0 shipped focus.

### One engine path per (model, quantization)

A variant is a SwiftPM target under `sources/Engines/` that conforms to the
`EngineVariant` protocol and owns its engine: geometry, quantization,
kernels and runner. The `variant-isolation` lint gate enforces the boundary
— a variant imports only `TinyTitanCore`, `TinyTitanFormat`, the C kernels
and system frameworks, never another variant and never the 6.x runtime —
and `core-purity` refuses model names, `ModelFamily`, `ModelProfile` or
`ModelCatalog` references inside Core. Both gates fail on an empty scan, so
a check that read nothing cannot report a pass. Backed by the two gate
suites (`test_lint_core_purity.py`, `test_lint_variant_isolation.py`, 10
tests, both directions) and the full lint chain, now twenty checks.

### TinyTitanCore: the generic foundation

The contract types (`EngineVariant`, `VariantID`, `QuantSpec`/`QuantRole`,
`GeometryContract`, `PerformanceProfile`, `SamplingProfile`,
`Qualification`) live in `sources/TinyTitanCore/Variant/`, test-first (13
contract tests: 6-bit refused by name, quant resolution perTensor > roles >
default, geometry names the first mismatch). The complete generic
install-reading stack moved with them — `ResidentIndex`,
`SSDAIModelDirectory`, `BoundedMetadataRead`, the manifest types and
`SSDAIManifestReader` (the runtime's reader delegates to it) — plus
`MetalContext` and the shader bundle with an injectable shader-module list,
so Core's default names no model. `ArchConfig.family` is now a raw string,
which is what let the geometry configs leave the runtime; the full serial
suite passed after the decoupling (2,371 tests, commit `bdbf4069`).

### One registry, checked: `tinytitan catalog --check`

`registry/registry.json` is the one inventory of (model, quantization)
pairs — 9 models, 19 pairs, converter-pinned sources — and
`tinytitan catalog --check` refuses a registry entry with no target, an
orphan variant target, or a shipped-but-unqualified variant (targets read
from `swift package describe`, not a hand-kept list). `--markdown` renders
the model list. Backed by `CatalogCheckTests` (6 tests) and
`RegistryInventoryTests`; the check exits 0 on a consistent registry and
exits 1 naming a shipped-unqualified stub.

### The shipped variant: Ornith 1.5 35B-A3B 4-bit

`Ornith35B4Bit` owns the qwen36-family GPU engine with SSD expert streaming
for this one pair. `OrnithParityTests` drives the variant and the original
6.x runtime engine on the same install and asserts identical greedy
tokens: passed (24 tokens, commit `bc112a90`). The install is about
19.5 GiB (20,954,280,960 bytes per the registry); it streams routed experts
from SSD, so the 35B MoE runs on a 24 GB Mac. Server-side decode measured
about **7.66 tok/s on a 24 GB M3** during the 7.0 variant work — a
measurement on that host, not a re-certified benchmark table (see below).
The install path is the usual one (`docs/adding-a-model.md`); the repacker
bug that wrote the family enum instead of its `rawValue` into
`manifest.json` was fixed on the way (`17529070`), and the Ornith install
completes and verifies with `--verify-install`.

### Also in this release

- The dense Qwen 3.5 family is ported to variants — `Qwen35Dense4B4Bit`,
  `Qwen35Dense4B8Bit`, `Qwen35Dense9B4Bit`, `Qwen35Dense9B8Bit` run the
  config-driven dense CPU engine, and `DenseParityTests` asserts greedy
  output identical to the 6.x runtime engine on their own installs (all
  four passed). `Qwen35Dense2B4Bit` carries the scaffold and the model-free
  matrix test (every role resolves, the 8-bit-as-nibbles trap pinned).
  These ports stay in the tree as code for future releases; 7.0's shipped
  focus is the Ornith 4-bit variant alone.
- The 6.x runtime, CLI, server and installer keep working as before; the
  variant architecture lands beside them, and the 6.x machinery retires
  only when every shipped pair is a variant (tracker TT-087).

### Performance

No benchmark table was re-run for this cut. The ~7.66 tok/s Ornith decode
figure above is the server-side measurement taken on the 24 GB M3 during
the 7.0 variant work; the wiki Benchmarks page's older rows stand labelled
by their engine. The parity claim is token identity (`OrnithParityTests`,
`DenseParityTests`), not a speed claim.

### Checks not run

Only `models/ornith-1.5_35B_A3B_4Bit` is installed on the machine this
release was prepared on, so `ornith-4` is the one golden target the release
dry run can check; its re-check runs in `tools/release.sh v7.0` before
publish and had not been run when these notes were written. All other
baselines are **not checked** — no install under `models/`, and none may be
fetched to fill a row: `ornith-8`, `ornith-bf16dense-4`, `ornith122-4`,
`ornith122-8`, `qwen36-4`, `qwen36-8`, `qwen38-4`, `qwen38-8`,
`agentworld-4`, `agentworld-8`, `katcoder-4`, `katcoder-8`, `k2-4`, `k2-8`,
`qwen35-2b-4`, `qwen35-2b-8`, `qwen35-4b-4`, `qwen35-4b-8`, `qwen35-9b-4`
and `qwen35-9b-8`.

### Checksum

`tinytitan-7.0-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-7.0-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-7.0-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-7.0-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-7.0-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-7.0-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes