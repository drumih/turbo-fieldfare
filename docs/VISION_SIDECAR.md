# Optional vision sidecar

Gemma 4 vision weights are installed separately from the language model. This
keeps the existing text-only `.gturbo` package unchanged and lets an existing
completed install gain image support without downloading the language-model
weights again.

Install the normal model first:

```bash
swift run -c release TurboFieldfareRepack \
  --output scratch/gemma4.gturbo
```

Then stream and install only the vision tensors:

```bash
swift run -c release TurboFieldfareRepack \
  --install-vision \
  --output scratch/gemma4.gturbo
```

Continue an interrupted vision download:

```bash
swift run -c release TurboFieldfareRepack \
  --install-vision \
  --resume \
  --output scratch/gemma4.gturbo
```

Discard only the saved vision download state, without touching the completed
language model:

```bash
swift run -c release TurboFieldfareRepack \
  --discard-vision-partial \
  --output scratch/gemma4.gturbo
```

Verify the completed sidecar independently:

```bash
swift run -c release TurboFieldfareRepack \
  --verify-vision \
  --input-gturbo scratch/gemma4.gturbo
```

Use `--overwrite` with `--install-vision` only when replacing an already
completed sidecar.

## Layout and source binding

The installer adds these files and does not rewrite the root model manifest,
receipt, resident weights, tokenizer, or packed experts:

```text
scratch/gemma4.gturbo/
└── vision/
    ├── weights.bin
    ├── manifest.json
    └── verified-install.json
```

`weights.bin` uses the same page-aligned resident index as
`model_weights.bin`. Quantized scale and bias companions are represented by
offsets on their `.weight` entry. The sidecar manifest records both the source
snapshot SHA-256 and the root `manifest.json` SHA-256, so vision weights from a
different checkpoint cannot be attached to the installed language model.

## Bounded download behavior

The sidecar installer reads only safetensors headers and the byte ranges that
contain `vision_tower.*` and `embed_vision.*`. It never downloads a complete
checkpoint or shard and never holds a complete tensor in Swift `Data`.
Payloads are copied from bounded temporary range files through the existing
512 KiB scratch tile. Progress is checkpointed after each durably written
range.

Files are assembled in `.vision.partial` and promoted to `vision/` by an
atomic directory rename. Cancellation preserves verified ranges for
`--resume`; `--discard-vision-partial` removes only the sidecar's temporary
directory and checkpoint.
