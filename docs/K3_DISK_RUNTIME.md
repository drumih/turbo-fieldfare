# K3 disk runtime

Kimi K3 selects 16 routed experts in each of 92 MoE layers. Exact inference
must evaluate all selected matrices; an index can find their offsets in O(1),
but cannot know or omit a selected expert before the router runs. Disk
optimization therefore targets bytes served from SSD, wasted speculative
reads, and overlap—not fewer exact matrix products.

## Metal wired-memory prerequisite

The int8 trunk's `model_weights.bin` is about 57.8 GiB. On the validated
128 GB M5 Max, macOS may report only a 32 GiB GPU wired-memory ceiling after a
reboot; in that state the first load-time MLA plane expansion fails even with
`--expert-cache-gib 0` and a short context. Check both values before a run:

```bash
sysctl iogpu.wired_limit_mb
memory_pressure -Q
```

The ceiling is a machine-wide safety control, not a per-process reservation.
An operator who accepts the tradeoff can raise it temporarily; on a 128 GB
machine, start at 80 GiB rather than assigning nearly all unified memory to
GPU wiring:

```bash
sudo sysctl -w iogpu.wired_limit_mb=81920
sysctl iogpu.wired_limit_mb
```

This resets on reboot. Do not automate a larger value: leave room for macOS,
the 24 GiB ordinary-RAM expert arena, and other applications. Validate first
with cache 0 and one output token, then increase the cache while monitoring
memory pressure.

## Exact direct-resident expert banks

`--expert-cache-gib N` divides one page-aligned RAM arena into complete
per-layer banks. Each has 16 canonical packed MXFP4 slots. The single backing
allocation avoids creating one large VM allocation per MoE layer. Only the layer being executed
receives a temporary `bytesNoCopy` Metal view; the other resident banks remain
ordinary RAM and do not consume the GPU working-set budget. A hit is checked
only after the real router selected `(layer, expert)` and stays in place. A
miss reads directly from SSD into the compute slot. There is no cache-to-bank
hit copy and no miss-population copy; verbose `ramCopy` must remain zero.

The canonical blob is 17,547,264 bytes and one complete layer bank is
280,756,224 bytes. A 24 GiB budget covers 91 of 92 layers (1,456 fixed slots);
the remaining layer uses a transient bank. Resident bytes survive sequential
CLI batch jobs and server requests; KDA, MLA, routing prediction, and logits do
not. The option remains opt-in because the banks compete with the int8 trunk,
MLA context, applications, and the OS. Compare verbose `ramCache`, `ramCopy`,
`read`, and `tok/s` counters in a long real-weight A/B.

At the maximum 262144-token context, the MLA latent KV cache and the int8
trunk leave less headroom than at smaller contexts: a real-checkpoint run at
that context needed `--expert-cache-gib 16` rather than 24 to keep
`memory_pressure -Q` acceptable (976 resident slots at 16 GiB, matching
`entries=976/976` in the verbose footer below). Start large-context runs at
16 GiB and only raise it after confirming pressure stays green.

### Cache-size sweep (8 / 16 / 24 GiB, real checkpoint, 262144 context)

A direct A/B/C on the same real checkpoint and prompt (`--expert-predict off`,
`--expert-io-splits 1`, `--expert-io-cache uncached`, `trusted-install`)
confirms 16 GiB is the correct default here — 24 GiB's larger resident cache
looks better on the raw hit-count but loses on wall time, because 256K
context plus the int8 trunk already leave little headroom and the extra
6 GiB of banked RAM pushes memory pressure hard enough to outweigh the I/O
it saves.

A short 2–3 new-token probe first isolates the effect: comparing the *second*
decode step's hit rate against the first (cold) one shows how much of the
resident arena is actually getting reused before generation is long enough
for the wall-clock cost to matter:

| Cache | Resident slots (`entries`) | 2nd-decode `ramCache` hits | SSD `read` | forward time |
|---|---|---|---|---|
| 8 GiB | 480/480 | 93/960 | 649,996 MB | 9.22 s |
| 16 GiB | 976/976 | 303/1952 | 646,482 MB | 9.40 s |
| 24 GiB | 1,456/1,456 | 491/2912 | 643,269 MB | 13.91 s |

24 GiB reuses the largest share of its (larger) resident set, and reads the
least from SSD — but is ~51% slower than 8 or 16 GiB, because at this
context length the extra RAM commitment itself has a real cost. 16 GiB reads
almost as little as 24 GiB while matching 8 GiB's speed, which is why it was
carried into the full-length run below rather than 24 GiB.

The decisive comparison is a realistic `--max-new 64` generation, run
identically at 8 GiB and 16 GiB (24 GiB was excluded per the probe above):

| Cache | Decode | Throughput | SSD `read` | Expert hits over the run |
|---|---|---|---|---|
| 8 GiB | 138.78 s | 0.461 tok/s | 2,007,070 MB | 8,790 / 92,736 |
| 16 GiB | 124.99 s | 0.512 tok/s | 1,776,956 MB | 22,541 / 92,736 |

Same output text both times. 16 GiB decodes ~13.8 s faster (~10%), reads
~230 GB less from SSD (~11.5%), and reuses experts roughly 2.6× more often
than 8 GiB — with none of the memory-pressure penalty 24 GiB showed in the
probe. **`--expert-cache-gib 16` is therefore the confirmed default for a
128 GB M5 Max at the 262144-token context ceiling**; re-run this sweep with a
longer `--max-new` (64 or 256) before trusting a different value on another
context length or machine.

### Measured example (real checkpoint)

```bash
.build/release/TurboFieldfareCLI \
  --model scratch/kimi-k3.gturbo \
  --messages-file messages.json \
  --no-thinking \
  --temperature 0 \
  --max-new 64 \
  --max-context 262144 \
  --prefill chunked \
  --prefill-chunk 32 \
  --expert-predict off \
  --expert-cache-gib 16 \
  --expert-io-workers auto \
  --expert-io-splits 1 \
  --expert-io-cache uncached \
  --model-verification trusted-install \
  --verbose
```

```text
[stop=maxTokens prefill=68tok new=64tok decode=124.99s tok/s=0.512 ttft=51.872s fwd=248530.4ms experts=22541/92736hit io=1776956MB chunks=3x17283.2ms]
[prefill=chunked(32) sampling=greedy demand=22541/92736hit ramCache=22541/61488hit entries=976/976 ramCopy=0MB prefetch=0 skippedBusy=0 skippedCold=0 read=1776956MB ioWorkers=4 ioSplits=1 ioCache=uncached ioPeak=4 ioTune=done ioBatches=8170 verification=trusted-install position=131]
```

0.512 tok/s decode and a 51.9 s TTFT (three 32-token prefill chunks at
~17.3 s each) on the real 2.78T checkpoint land inside the §3 feasibility
estimate of 0.3–1 tok/s, and `ramCopy=0MB` confirms no cache-to-compute copy
occurred at either a cache hit or a miss. This is one run, not a throughput
guarantee — treat it as a known-good starting configuration, not a ceiling.

## Selective prefetch

`--expert-predict selective` takes the ordered intersection of the last two
routes for the same layer and prefetches at most four experts. The remaining
experts are demand reads. `on` or `full` retains the former full top-16 replay
only for comparison. Start with prediction `off` when using the exact RAM
cache so the measurements do not mix two policies.

## Sequential batch mode

`--batch-file` accepts UTF-8 JSONL and keeps a single loaded engine and expert
resident banks. Each row contains exactly one of `prompt` or `messages`, plus
optional `id`, `max_new`, and `temperature`:

```json
{"id":"raw-1","prompt":"The capital of France is","max_new":32}
{"id":"chat-1","messages":[{"role":"user","content":"Write a Swift LRU cache."}],"max_new":256}
```

```bash
swift run -c release TurboFieldfareCLI \
  --model scratch/kimi-k3.gturbo \
  --batch-file jobs.jsonl \
  --max-context 262144 \
  --prefill chunked \
  --prefill-chunk 32 \
  --expert-predict off \
  --expert-cache-gib 16 \
  --expert-io-workers auto \
  --expert-io-splits 1 \
  --expert-io-cache uncached \
  --model-verification trusted-install
```

Output is JSONL with text, latency, throughput, exact expert bytes read, and
resident-bank hits. This mode amortizes startup and can reuse exact compressed
experts across jobs. It is not continuous multi-sequence batching: independent
KDA recurrent states and MLA caches are still evaluated one sequence at a
time. True dynamic batching requires batched state slabs and layer kernels.

## Multi-SSD stripes

The `.gturbo` v2 bundle remains unchanged. Repeat `--expert-shard-root` once
per prepared SSD root. Each root contains `expert-shard.json` and regular
files (symlinks are rejected). Every descriptor uses this shape:

```json
{
  "format": "turbofieldfare-k3-expert-shard-v1",
  "modelID": "moonshotai/Kimi-K3",
  "sourceSnapshotHash": "sha256:...",
  "expertStride": 17547264,
  "expertsPerLayer": 896,
  "shardIndex": 0,
  "shardCount": 2,
  "layers": [
    {
      "layer": 1,
      "file": "layer_01.bin",
      "experts": [0, 2, 4],
      "size": 7861174272,
      "sha256": "..."
    }
  ]
}
```

The abbreviated example omits the remaining expert ids and layers. A layer
file concatenates the listed logical expert blobs in list order. Across all
roots, every shard index, MoE layer, and logical expert must appear exactly
once. Geometry and model identity are validated immediately; files are opened
with `O_NOFOLLOW`, size-checked, and SHA-256 verified. The base bundle's
`trusted-install` receipt does not cover external roots, so shard payloads are
always fully hashed until a separately bound shard receipt is implemented.
Consequently the first traversal of a prepared stripe set has a large
verification cost and is not yet the recommended steady-state deployment.

Layer-level distribution is not enough: layers execute serially. This contract
stripes experts *within every layer*, allowing the existing bounded pread pool
to read one top-16 set across SSDs. An offline atomic/resumable writer for the
1.5 TB transformation is not yet included; do not hand-edit or move the live
bundle. RAID 0 remains the no-code alternative, but it changes the failure
domain and must be provisioned separately.

## Remaining exact opportunity

Each expert blob contains equal-size `w1`, `w2`, and `w3` regions. Phase 1 uses
`w1+w3`; phase 2 uses `w2`. A future split command-buffer path can read the
two phase-1 regions, commit phase 1, and overlap the final third of SSD I/O
with Metal work. This preserves total bytes and logits but changes bank
lifetime and completion ordering, so it needs a dedicated real-weight timing
gate before becoming a runtime option.
