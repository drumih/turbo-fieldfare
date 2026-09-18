# Long-context decode: 8.14 to 18.02 tokens/s

On an M5 Pro with 24 GB of memory, grouped full attention decoded at
18.02 tokens/s after a 110,000-token prompt. The old exact split-KV path
reached 8.14 tokens/s. Both generated the same 256 token IDs.

That is a 2.21x decode speedup in one matched pair, measured on September 18,
2026. Both runs used 32 expert-cache slots, above the shipping default of 16.
The result supports the grouped kernel selected in this 0.9.0 draft. Retrieval
and memory checks for the larger context settings remain open.

## What changed

Gemma 4 has five full-attention layers with 16 query heads sharing two KV
heads. The grouped kernel assigns one SIMD group to each query head and puts
the eight queries sharing a KV head in one threadgroup. It removes the
per-key threadgroup barriers while retaining the existing split-KV combine.
The sliding-window layers and FP16 KV format are unchanged.

Floating-point order mattered. An earlier grouped implementation failed the
short quality gate. The repaired kernel follows the old kernel's two-level
score reduction and explicit fused multiply-add order in the online-softmax
recurrence. Eighteen synthetic identity cases passed through 64K, including
cancellation, late maxima and a two-key recurrence regression. The short
9,216-row quality comparison then reported identical summary metrics on M2
and M5. Those checks preceded the 110K run.

## The 110K comparison

| Measurement | Old exact attention | Current grouped attention |
| --- | ---: | ---: |
| Prompt tokens | 110,000 | 110,000 |
| Generated tokens | 256 | 256 |
| Decode time | 31.458 s | 14.209 s |
| Decode throughput | 8.138 tok/s | 18.017 tok/s |
| Prefill time | 1,315.812 s | 1,202.890 s |
| Process footprint after decode | 6,234.49 MiB | 6,234.24 MiB |
| New swapout pages | 0 | 0 |

The host was an M5 Pro (Mac17,8), 24 GiB, running macOS 26.5.1 (25F80).
One release binary ran the old path first and the grouped path second, in
separate sequential processes. No competing model process was present at
launch, and sampled host checks reported no thermal warning.

Both used the pinned Gemma 4 26B-A4B IT 4-bit pack, greedy decoding, 131,072
context capacity, 32 LFU expert-cache slots, 256-token prefill chunks, FP16
KV and the fused output head. The prompt repeated a prose fixture and was
truncated to exactly 110,000 tokens before model loading. Prompt-token hashes
matched. Profiling and capture were disabled.

Only attention selection differed between the resolved runtime settings.
The measurement used the private benchmark executable, with temporary CLI
wiring for the existing cache-slot and prefill-chunk options. That executable
and its raw reports are not part of the public package.

The prefill implementation was the same in both runs. Its timing difference
is not evidence of a prefill improvement from this change. OS cache state was
uncontrolled, and the pair was not repeated. The memory values above are
samples after decode, not peak measurements.

## What this establishes

The grouped kernel was faster on this workload, and no difference appeared
in the 256 greedy output tokens. Matching IDs does not prove identical logits
or rule out quality differences on other prompts. This pair does not measure
the gain at the default 16 cache slots or compare TurboFieldfare with MLX.

A prior 128K-budget run completed 130,816 prompt tokens and 256 generated
tokens with grouped attention. Its exact-attention control was interrupted,
so that run supplies no paired speed or token-parity result.

## Context settings in the 0.9.0 draft

The model's position ceiling is 262,144. The app offers 128K and 256K where
the host-memory admission calculation permits them; the server also accepts
96K and 192K. App and CLI defaults remain 8K, the server default remains 16K,
and the expert-cache default remains 16 slots. Admission estimates FP16 KV
plus a 2 GiB runtime allowance at 16 expert-cache slots and reserves 3 GiB
for the host. Larger caches add their page-aligned allocation across all 30
layers: about 0.75 GiB for 24 slots or 1.50 GiB for 32. An 8 GB host therefore
refuses 128K with either larger cache. The app adjusts context choices when
Slots changes and clamps an incompatible selection with a visible notice.

Before release, the outstanding checks are retrieval at 10%, 50% and 90%
depth for 64K, 128K and 256K; repeated 128K/256K execution; same-checkpoint MLX
parity at 32K and 64K; public/private runtime parity; and the 128K memory test
on an 8 GB Mac. The 110K speed result does not establish useful 256K answer
quality. Keep client context budgets at their existing settings until those
checks pass.

The server context work follows the community investigation in
[PR #156](https://github.com/drumih/turbo-fieldfare/pull/156) and
[issue #157](https://github.com/drumih/turbo-fieldfare/issues/157). Its M5 Max
measurements are separate from the M5 Pro comparison above.

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Attention and KV cache](05-attention-and-kv-cache.md)
