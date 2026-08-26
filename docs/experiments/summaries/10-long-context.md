# Long-context ladder and retrieval

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Server guide](../../OPENAI_SERVER.md) |
[Runtime controls](../../RUNTIME_CONTROLS.md)

Extending the server's `--max-context` allowlist to the Gemma 4 26B-A4B native
256K raised two separate questions. Whether a prompt that long is *admitted* —
accepted, prefilled, and memory stable — and whether it is *used*, meaning the
model still attends across it. They need different harnesses and they are
reported separately here.

| Current result | Disposition |
| --- | --- |
| 256K admission, prefill, and memory stability | Measured on M5 Max; see caveats below |
| `--prefill off` above 64K | Rejected during argument parsing |
| Long-context retrieval at ladder rungs | **Not yet measured** |
| Multi-token decode throughput at long context | Not measured; ladder uses a one-token completion |

## Why the sliding window makes this two questions

`ArchConfig.gemma4_26B_A4B` sets `slidingWindow: 1024` and marks layers 5, 11,
17, 23, and 29 as full attention. Twenty-five of thirty layers therefore see a
1024-token window, and every long-range dependency rests on five layers.

Raising the context cap allocates KV and admits the prompt. It does not by
itself widen what those five layers can carry. A model whose long-range
attention had degraded would still return HTTP 200 with a well-formed
one-token completion, so the ladder below cannot detect that failure and must
not be read as evidence against it.

## Admission ladder

`Scripts/context_ladder.py` starts a fresh release server per rung, submits one
request with `max_completion_tokens=1`, and samples memory every ten seconds. A
rung passes on HTTP 200 with the intended prompt token count, no rejection,
timeout, cancellation, or OOM. Resource pressure is recorded rather than used
as a hidden threshold.

The prompt is `"word " * N`. That makes the token count reproducible, and it is
also the measurement's main limitation: repeated filler is not representative
of agent history and can produce unusually stable expert routing.

### Recorded run

Apple M5 Max MacBook Pro, 18 CPU cores, 32-core GPU, 36 GB unified memory,
macOS 26.5.2 (25F84), Apple Swift 6.3.3. Reserve was a uniform 8,192 tokens
across every rung.

| Context cap | Actual prompt | E2E server time | Prefill | Prefill tok/s | Peak footprint | Peak Metal | Minimum free memory | Swap | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64K | 57,344 | 236.158s | 236.000s | 242.983 | 3,441 MB | 1,537 MB | 76% | 0 MB | completed |
| 96K | 90,112 | 464.218s | 463.937s | 194.233 | 4,082 MB | 2,177 MB | 76% | 0 MB | completed |
| 128K | 122,880 | 701.019s | 700.724s | 175.361 | 4,714 MB | 2,817 MB | 75% | 0 MB | completed |
| 192K | 188,416 | 1,291.620s | 1,290.896s | 145.958 | 6,032 MB | 4,097 MB | 68% | 0 MB | completed |
| 256K | 253,952 | 1,920.565s | 1,919.868s | 132.276 | 7,320 MB | 5,377 MB | 62% | 0 MB | completed |

Prompt length grows 4.43x from 64K to 256K while end-to-end time grows 8.13x.
Time per prompt token rises from 4.118 ms to 7.563 ms and prefill throughput
falls from 242.983 to 132.276 tokens/s, so cost is superlinear in length even
though a least-squares fit over these five points is close to linear
(R² = 0.996). No rung touched swap.

### Provenance and a known staleness

These numbers were produced before this branch merged `main`, which brought in
"Speed up prefill on pre-Apple10 Macs" (#159). That change rewrote prefill
attention pipeline selection and the prefill command-buffer structure, which is
exactly the path the timing columns measure.

**The timing columns above therefore describe a superseded prefill
implementation and must be re-measured before they are quoted as current.** The
memory columns are structural — they follow from KV allocation, not from
scheduling — and are expected to hold; see the check below. The rungs were also
run across more than one commit, with the 64K baseline recorded last, so they
are not a matched set even among themselves.

Re-running the full ladder on a single binary is tracked as blocking question 1
in the pull request.

### Memory model cross-check

FP16 KV allocation follows directly from `KVCacheManager`: with the ring
enabled, each of the 25 sliding-window layers holds
`slidingWindow + prefillChunkTokens` = 1,152 tokens at
`numKVHeads * headDim * 2` bytes, while each of the 5 full-attention layers
holds `maxContext` at `numFullKVHeads * fullHeadDim * 2` bytes, doubled for K
and V:

```
KV(ctx) = 25 * 1152 * 4096 * 2  +  5 * ctx * 2048 * 2
```

| Context | Predicted KV | Observed peak Metal | Difference |
|---:|---:|---:|---:|
| 64K | 1,505 MB | 1,537 MB | +32 MB |
| 96K | 2,145 MB | 2,177 MB | +32 MB |
| 128K | 2,785 MB | 2,817 MB | +32 MB |
| 192K | 4,065 MB | 4,097 MB | +32 MB |
| 256K | 5,345 MB | 5,377 MB | +32 MB |

A constant 32 MB offset across a 4.43x span means the model accounts for the
scaling term and the residual is fixed overhead. This is also why
`--prefill off` is rejected above 64K: without the ring every layer allocates
at `maxContext`, and the same formula gives about 55 GiB at 256K.

## Retrieval

`Scripts/context_retrieval.py` answers the second question. It plants a
distinctive fact at a fractional depth in filler, asks for it back with a real
multi-token completion, and sweeps depth because sliding-window degradation is
position dependent — a needle near the end stays inside every layer's local
window, while one in the middle must survive on the full-attention layers.

### Recorded run

Same machine and build as above.

| Context cap | Actual prompt | Depths | Recall |
|---:|---:|---|---|
| 4K | 1,543 | 0.1 / 0.5 / 0.9 | 3/3 |
| 16K | 14,043 | 0.1 / 0.5 / 0.9 | 3/3 |
| 64K and above | — | — | **not measured** |

14,043 tokens already exceeds the 1024-token sliding window by more than 13x,
so the full-attention path demonstrably carries information well past the local
window. That is what makes the harness trustworthy at larger sizes; it is not
evidence about those larger sizes.

**No retrieval measurement exists for 57K through 254K — the exact range the
256K cap adds.** Until it does, the supported claim is that long prompts are
admitted and memory stable, not that they are usable. Tracked as blocking
question 2 in the pull request.

## Reproducing

```bash
swift build -c release --product TurboFieldfareServer

# Admission ladder: all rungs, or one at a time merged into the aggregate.
python3 Scripts/context_ladder.py --model scratch/gemma4.gturbo
python3 Scripts/context_ladder.py --model scratch/gemma4.gturbo --levels 256

# Retrieval at a given cap.
python3 Scripts/context_retrieval.py --model scratch/gemma4.gturbo \
  --max-context 262144 --filler-words 250000 --depths 0.1,0.5,0.9
```

Both harnesses write JSON under `benchmark-results/`, which is ignored by git;
copy any result worth keeping into this note. Every rung is only comparable to
rungs produced by the same binary, so record the commit and re-measure the
whole ladder after any change to the prefill or attention path.
