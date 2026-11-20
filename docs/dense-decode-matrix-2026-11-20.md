# Dense decode matrix — 2B / 4B / 9B on CPU and GPU (2026-11-20)

A same-session sample of the six dense Qwen 3.5 installs on both 7.0 engines.
It is **not** a publication-grade record: the host was busy throughout (load
average 5.79 at the end; Chrome and WindowServer were the top CPU users), so
the absolute figures are depressed. The GPU/CPU ratios are the stable part.
The published README table remains the authority for quoted numbers until a
quiet-host interleaved rerun replaces this file.

## Method

- Machine: M3 MacBook Pro, 24 GB, macOS 27.2.1, Swift 6.4.2, commit
  `1220cc1e`.
- One `TinyTitanServer --models-dir models` process per install, 7.0 routing:
  the base id selects `DenseGPU` for the dense family; the `<id>@cpu` request
  alias switches the same install to its per-width `Qwen35Dense*` variant.
- Prompt: "Write a short paragraph about the sea." (20 tokens), greedy,
  `max_tokens=96`, one discarded 8-token warm-up.
- Decode tok/s is computed from the streaming reply: `(completion_tokens - 1) /
  (last content chunk - first content chunk)`. Prefill and load are excluded.
- The raw JSON and the driver live in the gitignored `.build/bench/`
  (`matrix.json`, `matrix-loaded.json`, `measure.py`).

## Results

| Model | 4-bit GPU | 4-bit CPU | 8-bit GPU | 8-bit CPU |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 2B | **40.4** | **15.3** | **27.5** | **14.5** |
| Qwen 3.5 4B | **18.6** | **4.8** | **9.9** | **5.4** |
| Qwen 3.5 9B | **10.0** | **3.2** | **6.1** | **2.7** |

GPU/CPU ratio: 2B 2.6× / 1.9×, 4B 3.8× / 1.8×, 9B 3.1× / 2.2×
(4-bit / 8-bit in that order).

## Load sensitivity (same build, same method)

Two later repeats on the same machine, both with two measured runs per cell,
show what the host moves:

| Load | 2B4 | 2B8 | 4B4 | 4B8 | 9B4 | 9B8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12.8 (GPU / CPU) | 29.3 / 11.3 | 21.2 / 10.3 | 12.9 / 4.3 | 6.6 / 3.9 | 6.0 / 1.9 | 3.6 / 1.2 |
| 10.3, Xcode+Roblox (GPU / CPU) | 12.5 / 2.5 | 11.8 / 3.3 | 6.0 / 1.4 | 5.3 / 2.6 | 5.9 / 1.3 | 4.1 / 1.8 |

The ordering is stable in every sample; the absolute figures are not. The
table above is the quietest sample, not a guaranteed rate.

## Reading it

- All six installs are healthy at this commit. The 4B 8-bit entry exists only
  because its install was rebuilt the same day: it had been 99.98% zero blocks
  (TT-100), and before the repair it emitted repeated `!`.
- The absolute numbers sit below the published README rows (for example 4B
  4-bit 22.8 GPU / 7.5 CPU there), which is the expected effect of the load.
  Do not quote these as the product's speed.
- The CPU rows are the 7.0 dense variants, the GPU rows the shared `DenseGPU`
  engine; this matrix is the first same-session pairing of the two after the
  2B/4B8 repairs.

## Found while measuring

`tools/server_launcher.sh --engine cpu` starts the server with
`--model <id>@cpu`, but the startup catalog lookup only knew base ids and
refused it as "not a model the catalog can serve" while the same spelling
worked as an HTTP request field. Fixed at `1220cc1e`
(`ModelCatalog.entry(idOrPath:)` resolves `@cpu`/`@gpu`, with
`theModelFlagFindsAnEngineAlias`), and verified live for both directions. The
measurements above used the HTTP alias; the engines are the same.

## Reproduce

```bash
# GPU arm
.build/release/TinyTitanServer --models-dir models --model qwen3.5-4b_4-Bit \
  --reasoning off --port 8300 --max-concurrent-sequences 1
# then POST /v1/chat/completions with that id, stream=true, max_tokens=96

# CPU arm: the same server, the alias as the request model
#   "model": "qwen3.5-4b_4-Bit@cpu"
# or start it directly (after 1220cc1e):
.build/release/TinyTitanServer --models-dir models --model qwen3.5-4b_4-Bit@cpu \
  --reasoning off --port 8300 --max-concurrent-sequences 1
```

A publishable run needs a quiet host, an interleaved GPU/CPU/CPU/GPU order,
and at least two measured repeats per cell.
