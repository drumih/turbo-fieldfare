# TinyTitan 7.4 — Engine plugins (TPP/1.1), Inkling 2 Mini full support, and Engine 7 phase 2

This release introduces the TinyTitan Plugin Protocol (TPP/1.1) with standalone
plugins for every model on disk, graduates Inkling 2 Mini (both original
NVFP4/MXFP8 and Unsloth UD-Q4_K_XL) to full `.ssdai` plugins with goldens and
multi-slot concurrency, re-qualifies Ornith 1.5 4-bit at the 160-slot default,
and cuts external 6.x `import TinyTitan` sites from 78 to 23.

### TinyTitan Plugin Protocol v1.1 (`TPP/1.1`)

- Every installed model family runs through its own standalone compiled-in
  plugin (`Ornith35B4BitPlugin`, `DenseQwen35Plugin`, `Inkling2GGUFQ4Plugin`,
  `Inkling2NVFP4Plugin`), backed by `PluginRegistry`, `TemplatePlugin`, and an
  out-of-tree plugin example (`examples/out-of-tree-plugin`).
- `PluginParameterGate` checks every declared parameter before load: explicit
  unsupported flags (such as `--kv-bits 8` on Inkling or `--mtp-model` where
  unsupported) are refused by name with the plugin's reason, and unsupported
  defaults are substituted and announced.
- `TinyTitanCLI plugin-check` verifies plugins and installs against the static
  conformance contract.
- TPP/1.1 adds `ChatDialect` (`ChatMLDialect`, `K2Dialect`, `InklingDialect`)
  on `EnginePlugin.chatDialect(for:)` and `ResolvedPlugin.chatDialect`, moving
  host prompt encoding, prompt-cache bridging, and assistant stream decoding
  off `TokenizerDialect` switches.

### Inkling 2 Mini: full `.ssdai` plugins (original + Unsloth UD-Q4_K_XL)

- Both `inkling2mini` (original HF NVFP4/MXFP8 checkpoint) and
  `inkling-q4-k-xl-unsloth` (Unsloth UD-Q4_K_XL) stream from Hugging Face into
  verified `.ssdai` installs without staging raw checkpoints, and retire the
  v7.3 side routes.
- Both plugins support the full server API (`tools/api-feature-check.py` 10/10),
  multi-slot `--max-concurrent-sequences` batching, 32K default context under
  `--ram-budget` (preserving expert slots), and captured golden baselines
  (`inkling2-gguf-q4`, `inkling2-nvfp4`).
- Word-wide CPU expert repack and E4M3 lookup tables cut NVFP4 repack to
  1.10 ms/expert.
- Opt-in DSpark MTP speculative decode (`TINYTITAN_INKLING_MTP=on`, default
  OFF) runs the bundled 3-layer + 7-step `rnn_hidden` Markov head over
  `model_weights.bin` with exact 3-tap conv rollback and byte-for-byte greedy
  identity; kept OFF by default because 8-token verify under SSD expert
  streaming measured 3.404 vs 8.047 tok/s (0.423×, 14.7% acceptance).

### Also in this release

- **Ornith 1.5 35B-A3B 4-bit re-qualification:** `registry/registry.json`
  updated from the old 32-slot `11.55 tok/s` to **15.69 tok/s** median at the
  160-slot default (`cache=10.55G`, golden byte-identical).
- **Engine 7 phase 2 (TT-134 slices 1–6):** moved `Quantization`,
  `TensorOpsPolicy`, `SideEngine`, `RuntimeLimits`, and `ChatDialect` into
  `TinyTitanCore`; removed `TinyTitan` dependency from `TinyTitanVariants` and
  `TinyTitanCPUEngine`; cut external 6.x `import TinyTitan` sites from 78 to 23.
- **Dense CPU tool calls & dense 4-bit GPU KV fix:** `--cpu` renders tool
  templates and emits tool calls; dense Qwen 3.5 4-bit GPU reads 8-bit K/V
  weights accurately.
- **Housekeeping:** removed deprecated `ServerCore` helpers and fixed
  `golden-baseline.sh --server` model selection.

### Performance

Measured on Apple M3 MacBook Pro (24 GB, macOS 27.2.1, Xcode 27 / Swift 6.4):

| Model | Configuration | Decode tok/s |
| --- | --- | ---: |
| Ornith 1.5 35B-A3B 4-bit | Default 160 slots (`cache=10.55G`) | **15.69** (14.31 / 15.69 / 15.80) |
| Inkling 2 Mini UD-Q4_K_XL | `--ram-budget 14G` (93 slots, 32K ctx) | **10.70** |
| Inkling 2 Mini UD-Q4_K_XL | `--ram-budget 18G --max-context 4096` (128 slots, all-resident) | **17.93** |
| Inkling 2 Mini (NVFP4/MXFP8) | `--ram-budget 12G` (59 slots, MTP off vs on) | **8.05** vs **3.40** (0.423×) |

### Checks not run

The installed models on this machine are Ornith 1.5 35B-A3B 4-bit, Inkling 2
Mini (`inkling2-gguf-q4` and `inkling2-nvfp4`), and the six dense Qwen 3.5
installs (15 golden targets checked: 9 GPU, 6 CPU). No model may be fetched to
fill a row. The following baselines are **not checked** because nothing is
installed under `models/`: `ornith-8`, `qwen36-4`, `qwen36-8`, `qwen38-4`,
`qwen38-8`, `k2-4`, `k2-8`, `agentworld-4`, `agentworld-8`, `katcoder-4`,
`katcoder-8`, `ornith122-4` and `ornith122-8`.

### Checksum

`tinytitan-7.4-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-7.4-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-7.4-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-7.4-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-7.4-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-7.4-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
