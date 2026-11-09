# Prompt: quality-neutral decode-speed plan for Qwen3.8 on the 24 GB M3

Use this as the complete brief for an implementation agent. It is deliberately
limited to changes that do **not** degrade model quality and do **not** require
new hardware.

---

## Role and objective

You are working in the TinyTitan checkout on an M3 Mac with 24 GB unified
memory. Raise **single-stream decode tok/s** for the installed Qwen3.8-Flash-Next
4-bit model. Work only on execution, scheduling, layout, cache and same-math
kernel changes. Do not quantize further, prune, merge, distill, approximate
attention, or change the sampler. Do not assume a faster SSD or more RAM.

The target is a measured, output-identical improvement. Every number below is a
projection from the current measurements; it is not an entitlement. Measure each
work package independently, then the combination.

## Measured baseline

Use this exact trace as the reference:

- Model: Qwen3.8-Flash-Next 125B-A6B, 4-bit install, 262,144 context.
- Prompt: `difference swift vs c++ in detail`, `--max-new 256`,
  `temperature 0`, `presence_penalty 0`, `seed 41`.
- Baseline: **4.62 tok/s = 216 ms/token**.
- Device I/O: **403 MB/token** at **3.95 GB/s** = **102 ms**.
- GPU busy: **102 ms/token**, occupancy ~44 %.
- Host/launch gaps: **~31 ms/token**.
- Current overlap: **~19 ms/token** (102 + 102 + 31 − 216 = 19).
- Exposed I/O: **~83 ms/token**.
- Output identity for the baseline run: record and compare SHA-256 of the
  generated text.

Do not use the older 5.7/6.9 tok/s figures: they predate the QSA indexer's
full-context scratch and are not reachable at the shipped context on this
machine. Do not use the 3.83 tok/s busy-machine MTP figure as a baseline either.

## Quality bar

- **Exact changes** (permutations, cache, prefetch, scheduling, layout, lossless
  dedup) must produce **byte-identical output** to the baseline on the fixed
  prompt.
- **Same-math kernel changes** may differ in the last FP bits. They must keep
  top-1 agreement and KLD within the noise of the current build on a held-out
  prompt set; otherwise revert.
- No quantization, pruning, merging, distillation, sparsity, early exit, KV
  compression, approximate indexer, or changed sampling. Those are out of scope.

## Estimation model

Use these formulas so every estimate is reproducible:

```
T = G + I + H - O
r = 1000 / T
```

where `G` is GPU busy, `I` is device I/O, `H` is host/launch time, and `O` is
the overlap between I/O and compute. Baseline: `G=102`, `I=102`, `H=31`,
`O=19`, `T=216`, `r=4.62`.

For cache changes, demand bytes scale with the miss rate:

```
I_new = I_old * (1 - hit_new) / (1 - hit_old)
```

with `hit_old = 0.824`. For overlap changes, `O_new` is the newly hidden I/O;
do not count the same milliseconds twice.

A combined estimate is not the sum of percentages. Recompute `T` with the new
`G`, `I`, `H`, `O` and compare `1000/T`.

## Ideas, estimates and justification

These are ordered by expected value per engineering day.

### 1. Hide more expert I/O behind compute (prefetch/pipeline)

**Change:** improve the predictive ring, issue the union fetch earlier, overlap
it with shared-expert and attention work that does not depend on it, and carry
the fetch across the next layer's independent GPU stages. Fall back to a demand
read on a miss; a wrong prefetch is discarded.

**Calculation:** exposed I/O is `I − O = 83 ms`. Hiding 50 % of it moves
`O` from 19 to 60 ms: `T = 102 + 102 + 31 − 60 = 175 ms` → **5.71 tok/s,
+23.6 %**. Hiding 75 % gives `O=81`, `T=154` → **6.49 tok/s, +40.5 %**. Perfect
overlap with the current host gaps gives `T = 102 + 31 = 133` → **7.52 tok/s,
+62.8 %**.

**Estimate: +20 to +40 % realistic, +63 % at the overlap ceiling.**

### 2. Raise the expert-cache hit rate without adding memory

**Change:** pin the hot expert set, use the persisted usage profile, and order
cache slots by co-activation so the resident set covers more of each token's
top-k. Keep the 64-slot budget; 96 slots already measured slower at full
context. This is an exact permutation/pinning change.

**Calculation:** `403 MB` at `hit=0.824` implies `403/(1−0.824) = 2,290 MB` of
unconstrained demand. At `hit=0.854` (+3 points) the demand is
`2,290 × 0.146 = 334 MB`; at 3.95 GB/s that is `84.6 ms`, saving `17.4 ms`.
With the same 19 ms overlap, `T = 102 + 84.6 + 31 − 19 = 198.6` → **5.03 tok/s,
+8.9 %**. At +5 points: `I = 73 ms`, `T = 187` → **5.35 tok/s, +15.8 %**.

**Estimate: +9 % for +3 points, +16 % for +5 points.**

### 3. Remove host/launch gaps

**Change:** reuse command buffers, argument buffers, scratch and encoders;
replace per-layer allocations with persistent objects; reduce commits and waits
where the dependency graph allows. Do not change the arithmetic.

**Calculation:** `H=31 ms`. Cutting it to 15 ms saves `16 ms`: `T=200` →
**5.00 tok/s, +8.2 %**. Cutting it to 10 ms gives `T=195` → **5.13 tok/s,
+11.0 %**.

**Estimate: +5 to +11 %.**

### 4. Same-math kernel efficiency

**Change:** improve GEMV/GEMM tiling and occupancy, fuse norm/elementwise
launches, reduce redundant dispatches. The measured GPU is 102 ms at ~44 %
occupancy; the GEMV roles run at ~67 GB/s against a ~100 GB/s device.

**Calculation:** if `G` falls from 102 to 90 ms, `T=204` → **4.90 tok/s,
+6.1 %**; at 80 ms, `T=194` → **5.15 tok/s, +11.5 %**. This is a bandwidth
argument, not a promise: the same kernels already sit near the memory roofline.

**Estimate: +3 to +11 %, measure per kernel.**

### 5. Expert/layout permutation and packing

**Change:** permute experts and router rows per layer; cluster co-activated
experts; place gate/up/down and scales/biases contiguously; align experts to
16 KB pages; avoid any read that crosses a page or pulls padding.

**Calculation:** the device already delivers 3.95 GB/s, which is 97–100 % of
its measured wall. A 3 % effective-bandwidth gain moves `I` from 102 to 99 ms:
`T=213` → **4.69 tok/s, +1.5 %**. There is no headroom for a large win here;
this is a tail-latency/read-amplification fix, not a throughput fix.

**Estimate: 0 to +3 %.**

### 6. Exact speculative decoding (optional, currently negative)

**Change:** keep target verification exact; no sampling approximations.

**Calculation:** the current MTP pass is 480 ms for 1.869 tokens at 86.9 %
acceptance; break-even is 404 ms. Even a perfect 1.3× two-row fusion plus a
1.27× union fetch and lower overhead projects 381–435 ms, i.e. **4.30–4.90
tok/s against scalar 4.62**. This is not a win on this machine.

**Estimate: −7 to +6 %; do not prioritize.**

### 7. Batching independent requests (throughput, not single-stream)

**Change:** run two independent sessions through one weight stream, sharing
resident weights and overlapping one session's expert I/O with the other's
compute.

**Calculation:** per-stream latency is unchanged; aggregate throughput can rise
because fixed weight traffic is shared. Model it as `1 + f` where `f` is the
fraction of weight reads shared; do not report a single-stream gain. Measure
aggregate tok/s over two concurrent 256-token requests.

**Estimate: +30 to +60 % aggregate for two streams, 0 % single-stream.**

### 8. Lossless files and dedup

**Change:** hardlink identical tables, deduplicate identical tensors, use
lossless frames only if it does not slow decode.

**Calculation:** decode bytes are unchanged, so `I` and `T` are unchanged.
Measured lossless expert compression was −42 % decode on this engine; keep it
out unless a fast path exists. **Estimate: 0 % decode; do not count it.**

### 9. Environmental and cache-policy checks

**Change:** run plugged in, close competing GPU work, keep the process out of
swap, and A/B `F_NOCACHE`/readahead and reader-thread counts. Reuse the prefix
cache where the KV is identical.

**Calculation:** these do not change the model or the arithmetic. Removing a 3 %
contention penalty is +3 %; page-cache or reader tuning can move `I` by a few
percent, never beyond the 3.95 GB/s wall. Prefix caching changes TTFT, not
decode tok/s. **Estimate: 0 to +3 %, and 0 % for prefix caching on decode.**

### 10. Prompt/template correctness

**Change:** merge leading system/developer messages, map `reasoning_effort:
high` to `xhigh`, implement `preserve_thinking`, validate tool-call arguments.
This is a quality/token-use fix, not a decode-speed lever.

**Estimate: 0 % decode tok/s; may reduce generated tokens and improve accuracy.**

## Combined scenarios (recompute, do not sum)

| scenario | G | I | H | O | T | tok/s | vs 4.62 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 102 | 102 | 31 | 19 | 216 | 4.62 | — |
| conservative | 100 | 89 (+2 hit points) | 25 | 35 | 179 | 5.59 | +21 % |
| stretch | 95 | 85 (+3 hit points) | 20 | 50 | 150 | 6.67 | +44 % |
| overlap ceiling | 102 | 102 | 31 | 102 | 133 | 7.52 | +63 % |

The conservative row assumes small gains in three independent places; the
stretch row assumes the overlap work succeeds substantially and the cache/kernel
work lands. Neither is a sum of the individual percentages.

## Work order and budget

1. **Baseline and instrumentation** (0.5 day): 5 runs, median, output SHA,
   runner/kernel stats, phase trace.
2. **Cache hit work** (1–2 days): measure hit rate and hot-set coverage; pin and
   reorder; A/B; exact output.
3. **Host/launch cleanup** (1–2 days): reuse buffers/encoders; A/B.
4. **Prefetch/overlap** (3–5 days): the largest expected win; add fallback and
   correctness tests.
5. **Kernel work** (3–5 days): only after the above; per-role GPU timings.
6. **Layout** (1 day): low priority; measure read amplification first.
7. **Combined run** (1 day): all accepted changes together, fresh baseline.

Do not start kernel work before the cache/overlap work is measured: the kernel
changes are the least certain and the most expensive.

## Measurement protocol

- Use the release server, the same fixed prompt, `temperature 0`,
  `presence_penalty 0`, `max_new_tokens 256`, `seed 41`.
- 5 runs per arm, interleaved A/B/A/B, report median and spread. Reject a change
  whose improvement is inside the run-to-run spread.
- Record `TINYTITAN_RUNNER_STATS=1` and `TINYTITAN_KERNEL_STATS=1`.
- Record the output SHA-256 for every arm. Exact changes must match the baseline.
- For same-math kernels, also run the held-out KLD/top-1 set and require no
  regression beyond the FP noise band.
- Report `G`, `I`, `H`, `O`, `T`, tok/s and the estimate's assumed values in one
  table, so the prediction can be checked against the measurement.

## Acceptance

- **Ship** a work package only if median tok/s improves by at least 3 % beyond
  the measured spread, with output identical (or KLD/top-1 within tolerance).
- **Revert** anything that regresses or cannot be separated from noise.
- **Combined target:** +15–25 % conservative (5.3–5.8 tok/s), +35–45 % stretch
  (6.2–6.7 tok/s). The theoretical overlap ceiling with the current GPU is
  about 7.5 tok/s (133 ms) and 9.8 tok/s if host gaps also vanish; do not claim
  the ceiling without a measurement.
- **Out of scope:** quantization, pruning, merging, distillation, low-rank,
  sparsity, early exit, approximate attention, KV compression, new hardware.

## Deliverables

A commit series, each with the measured `G/I/H/O/T` table and output hash, plus
a final `docs/` note stating which work packages were accepted, which were
reverted and why, and the combined measured tok/s against the 4.62 baseline.
