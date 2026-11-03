# Qwen 4.0 Flash (`qwen4-omni-120b-think`) — port record and runbook

Status: **code path prepared, checkpoint not published.** Nothing in this
document was measured on Qwen 4.0 weights yet; the geometry below is the
Qwen3.8-Flash-Next preview's verified contract, copied as the starting point.

## What the model is

| | |
| --- | --- |
| Model name | `qwen4-omni-120b-think` |
| Display name | Qwen 4.0 Flash (Qwen4-Omni-120B-Think) |
| TinyTitan family | `qwen4flash` (target), `qwen4flash_mtp` (draft head) |
| Expected `model_type` | `qwen4_omni_moe`, with `qwen4_omni`/`qwen4_flash` and their `_text` forms accepted |
| Text shape (starting contract) | 48 layers, 3 GDN : 1 QSA, hidden 2560, 512 experts top-10, GDN 16 K / 48 V heads, hyper-connections 4 x 320, indexer budget 2048 / compress 4, PLE on layer 1 |
| Modality | text-only in this runtime; vision/audio towers are excluded, not repacked |

`qwen4_exp` is deliberately **not** an alias: it belongs to Qwen3.8-Flash-Next
and stays on `qwen38flash`. The two families are separate code paths even where
their fields agree, so a 4.0 checkpoint can move a tensor without changing the
3.8 install.

## What is in the tree

- `ArchInfo.loadQwen4Flash` — the config reader (`text_config` fields, group
  size, 1-based `ple_layer_ids` conversion) and `crossCheckQwen4Flash`, which
  gates the production shape.
- `ArchInfo.qwen4FlashMTP` — the one-layer draft derived from a 4.0 target, plus
  `ArchInfo.draftHead(from:)` so local and remote import share one dispatch.
- `ModelFamily.qwen4flash` / `.qwen4flashMTP` and the `RepackModelFamily`
  mirrors; `ArchConfig.qwen4Flash` / `.qwen4FlashMTP` and their registry rows.
- `TensorSchema.qwen4flash` / `.qwen4flashMTP` and `Qwen4FlashTensors`, the
  standalone copy of the hyper-connection, indexer, GDN and PLE names.
- The runtime branches: validation, PLE constants/table file, sampling rows,
  reasoning levels, decode tuning, MTP predicates, session model id, server
  catalogue and display name, doctor scan.
- The repack branches: classification, passthrough requirements, MTP
  namespace stripping, local/remote draft import.
- Tests: `Qwen4FlashArchInfoTests`, `Qwen4FlashClassifyTests`,
  `Qwen4FlashArchTests`, `Qwen4FlashSchemaTests`, plus the manifest-resolution
  cases for a differently-shaped 4.0 install.

## What is deliberately missing

The installer and launcher still have no `qwen4flash` row, because a pinned
source needs a repository, revision and index SHA-256, and publishing those
before the checkpoint exists would be a guess. The runtime can already import a
local snapshot, which is the weights-day path.

## Runbook for the day the weights land

1. **Read the real config.** Confirm `model_type` is accepted; if it is a new
   spelling, add it to `ArchInfo.qwen4FlashModelTypes` deliberately. Confirm
   `model_output` / `text_config` geometry against `ArchConfig.qwen4Flash` and
   `crossCheckQwen4Flash`; reconcile both together, never one alone.
2. **Compare tensor names.** Run the classification tests against the
   checkpoint's own `model.safetensors.index.json` names (replace or add the
   fixture), then diff each name the classifier accepts against
   `Qwen4FlashTensors`. A rename is one edit in that namespace.
3. **Import a local snapshot.** `TinyTitanRepack --input-snapshot <dir>
   --output models/qwen4-omni-120b-think_4Bit --model-id
   qwen4-omni-120b-think-4bit` (add `--draft-head` from the same snapshot for
   the MTP sidecar). Verification is the ordinary `--verify-install`.
4. **Pin the remote source** once an MLX build exists: add a
   `SupportedModelSource` with the 4-bit repo/revision/index SHA-256 and the
   measured download/installed bytes, then an `install_models.sh` catalogue row,
   a `tinytitan_models.sh` key, and a launcher menu row.
5. **Re-measure, then add a `ModelProfile` row.** Until then a 4.0 install
   resolves its family defaults (12 GiB expert cache, depth 1, sampling
   1.0 / 0.95) and `keepExpertCacheWired` stays false.
6. **Golden baseline.** Add `qwen4-4` / `qwen4-8` to
   `tools/golden-baseline.sh` only when an install exists, then capture it.

## Verification used for this preparation

```
swift build
swift test --filter 'Qwen4|ArchConfigManifest|Qwen38|ModelCatalog|ModelIdentity'
tools/lint.sh all
```

The classification suite proves every one of the 3,164 pinned 3.8 names
classifies identically under `qwen4flash`; the arch suite proves the 4.0
`model_type` dispatches to the new family and `qwen4_exp` still does not.
