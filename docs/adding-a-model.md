# Adding a model to the family

This is the checklist for turning a Hugging Face checkpoint into a first-class
TinyTitan member at 4-bit and 8-bit. It exists because the work is spread over
eight places in the tree and the order matters: the research decides whether it
is a wiring job or a runtime job, and nothing is called *supported* until a real
model has been run.

`KAT-Coder-V2.5-Dev` is the worked example throughout — a Qwen3.6-35B-A3B
fine-tune, which is the cheap case. A checkpoint whose architecture is not
already implemented is a different project; see the tracker's "planned work".

## 0. Research the checkpoint before touching the tree

Read it from the source, not from the model card's prose:

```bash
# Identity, gating, parameter count, per-file sizes, the pinned sha.
curl -sL 'https://huggingface.co/api/models/<org>/<name>?blobs=true' | python3 -m json.tool | head -40

# Geometry: hidden size, layers, experts, rope block, tie_word_embeddings.
curl -sL 'https://huggingface.co/<org>/<name>/raw/<sha>/config.json'

# Sampling the authors actually asked for (not the base model's).
curl -sL 'https://huggingface.co/<org>/<name>/raw/<sha>/generation_config.json'

# Tensor names, and how many there are.
curl -sL 'https://huggingface.co/<org>/<name>/resolve/<sha>/model.safetensors.index.json' \
  | python3 -c 'import json,sys; wm=json.load(sys.stdin)["weight_map"]; print(len(wm))'
```

Record, because each one has burned a release or a conversion before:

- **`gated`** — if true, stop and ask the human for a token; do not paste one
  into a script.
- **The pinned sha**, not `main`: the install receipt records the source, and a
  moved `main` must not silently change what "4-bit" means.
- **`tie_word_embeddings`** — decides whether the head exists. A vision-language
  wrapper often unties it.
- **The rope block.** A multimodal wrapper can carry `mrope_interleaved` /
  `mrope_section` even in text-only use. Compare it against a model the runtime
  already serves rather than reasoning about it: if it matches, the text path
  needs no new kernel.
- **The EOS ids.** `generation_config.json` may list two. Check they resolve as
  stop tokens by *string* (`<|endoftext|>`, `<|im_end|>`) rather than by number,
  because the numbers move between releases.
- **Recommended sampling**, including `presence_penalty` and any extra template
  kwargs (`preserve_thinking`). Note which of them this runtime cannot do, and
  say so in the docs rather than quietly dropping it. Do not state a limitation
  the code no longer has: this file said "presence penalty `0.0` only" until
  v5.10 implemented the non-zero values, and a port written after that copied the
  sentence into its own deviations. Presence penalty is supported across OpenAI's
  `-2…2` on the wire and the same range on `--presence-penalty`, so what a card
  outside that range asks for is the deviation worth naming; what this runtime
  still cannot do is `frequency_penalty` and `min_p` (both refused unless zero;
  neither filter is implemented).

## 1. Decide: wiring job or runtime job

Compare the config's geometry and the index's tensor names against a family the
runtime already serves. For a converter built on `prepare_agentworld.py`, the
only namespaces it knows are `model.language_model.*`, `lm_head.weight` and
`mtp.*`; anything else is a new case. Verify rather than trust:

```bash
python3 - <<'PY'
import json
wm = json.load(open("/tmp/index.json"))["weight_map"]
known = ("model.language_model.", "model.visual.", "mtp.")
outside = [k for k in wm if k != "lm_head.weight" and not k.startswith(known)]
print(len(wm), "tensors;", len(outside), "outside the known namespaces")
print("visual:", sum(1 for k in wm if k.startswith("model.visual.")),
      "mtp:", sum(1 for k in wm if k.startswith("mtp.")),
      "head:", "lm_head.weight" in wm)
PY
```

A `vision_config` in `config.json` does **not** mean the checkpoint carries a
vision tower: KAT declares one and ships language weights only. Check the index.

### The checks that cost nothing and catch the expensive mistakes

All four run before a byte of weights is fetched. They exist because a
conversion is 69 GB of download and an hour of quantization, and each of these
has a failure mode that would otherwise appear at the end of it.

**1. Diff the chat template against a sibling the runtime already serves.** The
renderer uses the model's own `chat_template.jinja` (its absence is a hard
error), so the template *is* the prompt, the tool-call dialect and the
reasoning markers. KAT's differs from AgentWorld's in two places — AgentWorld
has an extra `audio` content branch, and a mid-conversation system message
raises on AgentWorld where KAT renders it inline — and nowhere else: the ChatML
markers, the `<tool_call>` block, the thinking block and the generation prompt
are byte-identical. That answers "will the parser accept its tool calls?" by
construction rather than by hoping.

**2. Resolve every key the repacker's arch reader requires.** Read the
`loadQwen35MoE` branch of
`sources/TinyTitanRepack/Core/Format/ArchInfo+Loaders.swift`,
extract its `try i("…")` keys, and check each against the checkpoint's
`text_config`. A missing one is a `configJsonInvalid` at the end of the
conversion; KAT has all seventeen, plus `layer_types` and both rope keys.

**3. Check the production cross-check.** The same file carries
`crossCheckProductionQwen35MoE`, which refuses a 2048/40 model whose geometry
differs in any field from the shipped one. If the checkpoint matches it
exactly, the repacker will treat it as *the* production geometry; if it does
not, that is a runtime question to answer before converting, not after.

**4. Confirm the tokenizer sidecar.** The converter's `TOKENIZER_FILES` marks
`tokenizer.json` and `tokenizer_config.json` required and the rest optional, and
skips a missing optional file rather than failing. But the *runtime* needs
`chat_template.jinja` — it errors with "installed tokenizer is missing
chat_template.jinja" — so a checkpoint that omits it needs a template supplied
even though the converter would not complain.

## 2. The eight wiring points

| # | Where | What |
| --- | --- | --- |
| 1 | `tools/prepare_agentworld.py` — `MODELS` | The repo and the **pinned sha**, with a comment recording what was verified about the checkpoint |
| 2 | `tools/install_models.sh` — `CATALOGUE` | Two rows (`<key>`, `<key>-8bit`) and the preset→served-id `case` |
| 3 | `tools/tinytitan_models.sh` | The key/stem/label `case` with the fallback fields it carries beside them — `ENGINES`, `THINKING` and `FAMILY` — because that list is what the launcher offers when the server cannot report a catalog; plus the unknown-model help text, `TINYTITAN_ALL_MODELS`, and the `tinytitan_install_key` stem→catalogue-key row (without it the launcher cannot fetch the new model for you, and `test_launcher_install` says so) |
| 4 | `tools/server_launcher.sh` | The model-key list in the header comment and in the unknown-model error |
| 5 | `ModelProfile.swift` | One row per width. Sample from the **checkpoint's** config; say in the comment when cache/prefetch values are inherited from identical geometry rather than measured |
| 6 | `ModelCatalog.swift` — `displayNames` | The served id → human name (`/v1/models` reads this) |
| 7 | `tests/` | `ModelProfileTests.shipped` (and the table count) |
| 8 | ANE prefill sidecar | `tools/ane_sidecars.sh <install>` — export and verify it. A GPU-path install without one has no ANE prefill at all: the switch is on by default, the runtime asks for the sidecar, and finds nothing. Three kinds are skipped. The exporter has no graph for the one-layer MTP draft, which the runtime verifies rather than prefills on the ANE. The ANE has been *measured not to pay* for `qwen38flash`: its GPU path already attends to only the indexer's ~2,051 selected keys while the ANE graph is dense over the context, so a sidecar there only slows the default path (0.72×, `benchmark/ane-prefill/README.md`); its block is still exportable — the runtime folds the QSA selection into the mask — but export one explicitly to re-measure, never to install. And Ornith 1.5 35B-A3B's sidecar fails the identical-greedy quality gate (the ANE arm diverges at the first character and prefills slower), so `tools/ane_sidecars.sh` refuses to export one until tracker TT-099 is resolved; an existing sidecar must be renamed `ane_prefill-rejected` because the runtime selects `ane_prefill` when it exists |

Validate 1 before any download:

```bash
python3 tools/prepare_agentworld.py --model <key> --plan
```

It fetches the config and the index only, and prints the shard count, the
tensor split per width, the bf16 keeps and the output size. `--plan` writing no
files is the point: it is the cheap place to discover that a last dimension is
not group-aligned.

The converters need **Python 3.10+ with numpy, ml_dtypes and safetensors**, and
they resolve that themselves: `tools/lib/python.sh` tries `python3.14` down to
`python`, testing version *and* imports for each, and takes the first that
passes. Use whichever name it resolved, or point `TINYTITAN_PYTHON` at a specific
interpreter (a virtualenv, say) — it skips the search but not the test, so a
typo there is refused where it is read rather than at the first converter call.
A bare `python3` is not safe to assume: on a
stock macOS it is 3.9 from `/usr/bin`, older than these scripts' syntax and
without the packages.

## 3. Convert and install

The hard rule (AGENTS.md, "Models"): every Hugging Face download is converted to
`.ssdai` as it arrives — each shard or byte range converted and deleted before
the next one is staged — and only the `.ssdai` install is served. A new source
format means a new streaming converter, never a raw folder under `models/`.

```bash
tools/install_models.sh <key>            # 4-bit
tools/install_models.sh <key>-8bit       # 8-bit
```

The installer streams up to twenty source shards at once (about 90 GB in
flight for a 35B-A3B, plus a few finished shards waiting for the converter) and
deletes each after use; it does **not** stage the full checkpoint.

**An interrupted conversion is not resumable the way it looks.** Each source
shard is deleted once converted, and the snapshot's index and `config.json` are
written only at the very end, so a run that dies at shard N re-downloads those N
shards on the next attempt, and `OutputWriter` never clears its output directory
— orphan `model-*.safetensors` from the dead run sit beside the new ones, which
is wasted space in exactly the disk budget you were trying to fit. Before
re-running, remove the partial output (`rm -rf .build/<key>-affine-*`), and run
the conversion as a supervised background job with a log rather than in a
session that might be interrupted.

**What *does* resume is the download inside a run.** Each shard is fetched as a
sequence of 64 MiB ranged requests, and a partial shard is continued at its own
offset, so a dropped connection costs one chunk rather than the whole file.
That matters on a link that truncates a long transfer: the host this was
written against kills a 5.3 GB response every few minutes
(`curl: (18) end of response with N bytes missing`), and a whole-file download
could never finish because each attempt was shorter than the interval between
truncations. Every chunk is length-checked against a size derived from the
shard's own header, which is also what makes resuming safe — a wrongly appended
file is the wrong length and is rejected rather than decoded into silently
wrong weights.

**The download is usually the whole critical path, so measure it before
starting.** Three things were worth doing on a slow link, in this order:
`--http1.1` (this host resets HTTP/2 streams continuously; 214 KB/s → 908 KB/s
on the same range), a pool of three concurrent downloads (the throttle is per
connection — a second connection added ~490 KB/s beside a ~205 KB/s one), and
the chunking above. What did **not** help: relaying through a fast remote host.
A VPS that pulled the checkpoint at 40 MB/s still delivered it to the working
Mac at 1.40 MB/s, because the Mac's own link was the ceiling.
Both widths come from one download when the converter is called with `--bits 4
8`, but the **snapshot and the install exist at the same time**, so budget:

| Build | Shards | Snapshot | Install | Peak |
| --- | ---: | ---: | ---: | ---: |
| 35B-A3B 4-bit | ~11 GB | ~20 GB | ~20 GB | ~50 GB |
| 35B-A3B 8-bit | ~11 GB | ~38 GB | ~35 GB | ~84 GB |

Delete the snapshot after the install (`rm -rf .build/<key>-affine-*`) and do
one width at a time on a full disk: 8-bit needs the space the 4-bit snapshot and
install are holding. Check `df -h .` against the table before starting, and say
so rather than starting a conversion that will die at 90%.

## 4. Verify before calling it supported

The project's bar, in this order:

1. **It loads and answers** through the CLI, and the continuations this project
   uses behave: `Once upon a` → " time", `The capital of France is` → " Paris",
   `The quick brown fox jumps over the lazy` → " dog".
2. **A golden baseline**, added to `tools/golden-baseline.sh` (target table) and
   to `release.sh`'s `check_golden` list, captured only for a deliberate
   numerics change — never re-captured to make a mismatch go away. Declare both
   **before** the install exists: `release.sh` verifies only the targets that
   have an install under `models/`, reports every one it could not check, and
   `--publish` requires the notes to name them — so the gate neither fails on a
   machine that has not installed the model nor lets that pass unrecorded, and
   it never downloads the model to close the gap. Until the baseline is
   captured, a direct `--check <target>` fails closed with "no verified install"
   or "no baseline", which is the guard working.
3. **The receipt**: `TinyTitanRepack --verify-install --input-ssdai <dir>` passes,
   and the manifest's `sourceSnapshotHash` matches the snapshot that produced
   it.
4. **The catalog is right**: `/v1/models` lists the id with the name from
   `displayNames` and the sampling from the profile row, and the launcher
   resolves the key, the stem and both widths.
5. **A first measured row** (TTFT, decode) from the model's own install, with
   the machine and commit stated.
6. **The whole API but vision** (the AGENTS.md rule): start the server on the
   install and run `tools/api-feature-check.py --port <port>`. All ten rows —
   chat, streaming, tool calls plain and streamed, the tool-result turn, stop
   strings, a strict JSON schema, thinking, Anthropic Messages and Responses
   with tools — must pass, on every engine the model is served on. A failing
   row is a gap to fix, not a limit to document.

## 5. Document it where a user will look

- `README.md` — the supported list, with the status stated honestly.
- The wiki: `Getting-Started` (the install table), `Features` (the model row),
  `Runtime-Controls` (thinking levels, and the temperature line if this
  checkpoint differs from its base's), `Project-Tracker` (what was checked,
  what landed, what is pending, and any deviation from the model's own
  recommendations).
- Deviations are stated, not hidden: presence penalty, extra template kwargs,
  an unverified tool-call dialect, an inherited profile value.

Never mark a model "supported" before §4 passes. "Install path landed,
verification pending" is the honest state, and it is worth writing down.

## 5b. A wider member of an existing family

Ornith-1.5-122B is the worked example of a *geometry* addition rather than an
architecture one. It is the same Qwen3.5-MoE text graph as the 35B-A3B
(`family: qwen36`), at hidden 3072, 48 layers, 256 experts top-8, F=1024,
32 Q / 2 KV heads and 16 GDN K / 64 V heads. Three things differ from the 35B
checklist above:

1. **The runtime must serve two shapes under one family.** `ArchConfig.resolved`
   returns the 35B preset when the manifest declares the 35B shape and otherwise
   builds the config from the manifest's own `arch` block (all conventions are
   `required`, never defaulted). The repacker's `crossCheckOrnith122` pins the
   3072/48 geometry the same way `crossCheckProductionQwen35MoE` pins 2048/40.
2. **The MoE threadgroup tile.** `kMoEXMaxD` in `moe.metal` and
   `Model.maximumThreadgroupTileWidth` are 3072, not 2816. A hidden size above
   the tile is a silent out-of-bounds write in the kernel, so the load guard is
   the thing that makes this a refusal rather than wrong numbers. The GDN tile is
   already 4096.
3. **Disk, not compute, decides the conversion shape.** The source is 250 GB,
   the 4-bit snapshot 69.5 GB and the 8-bit 129.8 GB; one `--bits 4 8` pass
   stages 199 GB and then wants a 125 GB install beside it. The `ornith122` rows
   are therefore `convert_qwen35moe_solo`: one width per run, converter fetchers
   capped at four, and `TinyTitanRepack --input-snapshot --consume-snapshot`
   unlinks each snapshot shard after its last planned range is committed. That is
   a ~142 GB peak for the 8-bit width instead of ~255 GB. It consumes the
   snapshot: a failed repack after a retirement needs the snapshot rebuilt, which
   is the trade the checklist's "converted widths are reproducible from the
   source" already assumes.

## 6. After the checkout moves

An install receipt is bound to its absolute path. Moving or renaming the
checkout invalidates **every** install's receipt, with
`trusted receipt invalid: model directory mismatch` — not corruption, and no
re-download. Re-issue each in place:

```bash
for d in models/*/; do
  [ -f "$d/verified-install.json" ] || continue
  swift run -c release TinyTitanRepack --verify-install --input-ssdai "$d"
done
```

Never hand-edit `verified-install.json`: the path binding is what detects a
moved or swapped directory.

## Checklist

- [ ] Gating, pinned sha, geometry, rope block, tie-embeddings and EOS ids read
      from the source
- [ ] The index's tensor names checked against the converter's namespaces
- [ ] `--plan` classifies every tensor and the output size fits the disk table
- [ ] All eight wiring points above
- [ ] Conversion and install, one width at a time, snapshots deleted after
      (a 122B install additionally uses `--consume-snapshot`)
- [ ] Continuations, golden target, receipt, catalog, launcher keys
- [ ] `tools/api-feature-check.py` 10/10 on every engine the model is served on
- [ ] README, wiki pages, tracker, roadmap — with deviations and status
- [ ] Committed, pushed, and the receipts re-issued if the checkout moved
