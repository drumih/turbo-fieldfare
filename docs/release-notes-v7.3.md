# TinyTitan 7.3 — Inkling-2-Mini GGUFs run natively; dense Qwen 3.5 gets a 1M default

This release adds a native Swift/Metal decoder for Unsloth's Inkling-2-Mini
GGUFs (`UD-Q2_K_XL`, `UD-Q4_K_XL`). It matches llama.cpp's greedy tokens and
runs at least 1.25× its speed on decode and prefill on a 24 GB M3. The dense
Qwen 3.5 GPU path now defaults to a 1M context without changing short answers.
Inkling is a **preview**: it is not on the README's supported list and has no
golden target. No model was added, converted or installed for this cut.

### Inkling-2-Mini GGUFs run natively, faster than llama.cpp

- `TinyTitanCLI --model <file.gguf>` and `TinyTitanServer --model <file.gguf>`
  detect an Inkling2 GGUF and run it in place on the new decoder. No repack is
  needed.
- Kernels are picked per tensor, because neither file is mostly Q2_K or Q4_K.
  The supported formats are IQ1_S, IQ2_XXS/XS/S, IQ3_XXS, IQ4_XS, Q2_0, Q2_K
  through Q8_0, NVFP4 and MXFP4.
- Parity: greedy output matches llama.cpp **96/96** on both files, token by
  token and with the prompt in one chunk (`Inkling2GGUFParityTests`).
- Speed, paired with `llama-bench -ngl 99 -fa 1` in the same rounds, at commit
  `0981b7dd` on a 24 GB M3:

| | Q2_K_XL | Q4_K_XL |
| --- | ---: | ---: |
| Decode (64 tokens) | **1.28–1.31×** (21.6–21.8 tok/s) | **1.34–1.38×** (18.9–19.6 tok/s) |
| Prefill, 64 tokens | **1.26–1.36×** | **1.26–1.28×** |
| Prefill, 512 tokens | **1.33–1.38×** | **1.39–1.40×** |

- Limits:
  - The server route is text only. It returns no tool calls and applies no
    stop strings.
  - It serves one request at a time, and the context is capped at 16,384
    tokens.
- The details, the rejected variants and what is left are in
  `docs/handover-inkling2-gguf.md`.

### Inkling-2-Mini from the original checkpoint (experimental)

- `--model models/inkling-2-mini` runs the NVFP4/MXFP8 safetensors checkpoint
  text-only from the CLI and the loopback server. Experts stream from disk and
  the quantization is kept as shipped.
- Quality evidence is a six-prompt factual check and an exact chat-template
  continuation. Its speed is far below the GGUF route.

### Dense Qwen 3.5: a 1M default without changing short answers

- The dense GPU path now defaults to a 1M context cap with a fixed 1M YaRN
  extension profile. `--max-context` picks another rung, from 512K to 16M.
- Positions below the native 262K use the native RoPE frequencies, so the
  cap changes only capacity, not short answers. The dense goldens check this.
- The dense CPU engine has no YaRN and stays at 262K.

### Also in this release

- `tools/release.sh` gates on the compile/test CI only. CodeQL is reported but
  is advisory.
- The gitleaks allowlist records `Inkling2GGUFTensor`, a Swift type name the
  generic-api-key rule matched. The full-history scan is clean again.
- The release-precondition test follows the renamed CI section of
  `tools/release.sh`.

### Performance

The Inkling table above was measured at `0981b7dd` on a 24 GB M3. Nothing
since then touches the Inkling code. No other family's benchmark was re-run
for this release.

### Checks not run

The installed models on this machine are Ornith 1.5 35B-A3B 4-bit and the six
dense Qwen 3.5 installs. The release gate checks their thirteen golden targets
(seven GPU, six CPU). Inkling-2-Mini has no golden target; its parity suite is
model-gated and was run during development, not by the release gate. No model
may be fetched to fill a row. The following baselines are **not checked**
because nothing is installed under `models/`: `ornith-8`, `qwen36-4`,
`qwen36-8`, `qwen38-4`, `qwen38-8`, `k2-4`, `k2-8`, `agentworld-4`,
`agentworld-8`, `katcoder-4`, `katcoder-8`, `ornith122-4` and `ornith122-8`.

### Checksum

`tinytitan-7.3-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-7.3-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-7.3-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-7.3-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-7.3-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-7.3-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
