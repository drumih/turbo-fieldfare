# TinyTitan 6.3 — the as-is GGUF path, the Qwen3.8 4/8/16 install, and opt-in speculative drafting

This release ships the as-is GGUF decoder as a feature, changes what the
Qwen3.8-Flash-Next 4-bit build is (routed experts 4-bit, every resident tensor
8-bit), adds opt-in wide-verify drafting and expert-cache warm start, and
makes the OpenCode provider file list every model the installer supports.

### The as-is GGUF path ships (experimental, CLI only)

A directory of Unsloth Qwen3.8-Flash-Next-UD-IQ3_XXS `.gguf` shards passed to
`--model` runs in place with no conversion. It agrees with llama.cpp on the
same file (same top-1 and top-5 tokens), checked stage by stage by
`GGUFLogitProbeTests` and the bit-identity suites. Measured on a 24 GB M3,
200 tokens greedy: **4.58 / 4.61 tok/s at `--ram-budget 10G`** and
**4.82 / 4.81 tok/s at `12G`** — the speed of the converted model, a
convenience path rather than a faster one. `--ram-budget` sizes its expert
cache; `TINYTITAN_GGUF_TRACE=1` prints the per-phase split.

### Qwen3.8-Flash-Next ships two install types with fixed recipes

The installer's 4-bit type is the **4/8/16 hybrid**: routed experts 4-bit,
every resident tensor (attention, GDN, hyper-connection, PLE, indexer, shared
expert, embedding, LM head, router) 8-bit, the n-gram table fp16 — about
164 GB, quality-forward. The 8-bit type is uniform 8-bit over the same fp16
table. The recipe is hardcoded per type (`--rest-8bit` at width 4), the MTP
draft head is no longer an install type, and the no-Python fallback no longer
suggests the repacker's RockTalk streaming (a 4/4/16 recipe the installer
does not produce). The converter gained `--rest-8bit` and a local
re-quantiser (`tools/requantize_qwen38_rest.py`) that rebuilds the rest width
from two existing installs with a decode self-check. Backed by the converter
suites (92 tests), a full `--verify-install` (57 files, 175.8 GB), and the
re-captured `qwen38-4` golden, byte-identical on re-check.

### Warm expert start and bf16-rest installs

`TINYTITAN_WARM_EXPERTS=<n>` preloads the expert slot cache from the routing
profile at startup (off by default); `TINYTITAN_EXPERT_PROFILE=0` keeps a run
unseeded. Installs whose embedding, LM head and dense weights are bf16 (the
converter's `--rest-bf16` recipe) load and run.

### Wide verify (experimental, opt-in)

`TINYTITAN_WIDE_VERIFY=1` drafts with a multi-row verify pass
(`TINYTITAN_WIDE_VERIFY_ROWS` sizes it). A dedicated W-row pass costs 3.53
scalar tokens at W=8; the GDN prefill path is pinned bit-identical to decode.
Identity is not closed yet — the layer-1 PLE injection, the shared expert and
the multi-row mixer still drift (bisected by `TinyTitanBench verifyprobe`) —
so drafts can be wrongly rejected and text can change. It is off by default
for exactly that reason.

### The OpenCode provider lists every installer model

`plugins/opencode-tinytitan/opencode.json` — the block `server_launcher.sh`
merges — now carries all nineteen installer models keyed by their `/v1/models`
ids, with the Qwen3.8 hybrid as the default. A model whose install is missing
shows in the picker and serves 404 until installed. Verified with
`opencode models`.

### Also in this release

- The Qwen 4.0 Flash family (`qwen4flash`) is ported as a code path; no
  checkpoint is published, nothing is measured, and there is no installer row.
- The 2026-10-06 audit records were deleted with their docs-gate machinery
  (owner-approved after retention); the tag-citation and derived-count checks
  remain.
- 38 unreferenced files were removed by a reference scan; `benchmark/golden/`,
  the CI-registered suites and their subjects were kept.
- `minimatch` is overridden to 11.0.7 (GHSA-g3hv-75wg-m249); the CodeQL action
  is at 4.39.3.

### Checks not run

Only `models/qwen3.8-flash-next_125B_A6B_4Bit` is installed on the machine
this release was built on, so `qwen38-4` is the one golden checked. All other
baselines were reported **not checked** — the targets are `ornith-4`,
`ornith-8`, `ornith-bf16dense-4`, `ornith122-4`, `ornith122-8`, `qwen36-4`,
`qwen36-8`, `qwen38-8`, `agentworld-4`, `agentworld-8`, `katcoder-4`,
`katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`, `qwen35-4b-4`, `qwen35-4b-8`,
`qwen35-9b-4`, `qwen35-9b-8`, `k2-4` and `k2-8`. No model was fetched to fill
a row. No performance table was re-run for this cut; the numbers above are the
measurements cited in the commit history and were not re-certified here.

### Checksum

`tinytitan-6.3-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-6.3-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-6.3-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-6.3-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-6.3-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-6.3-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
