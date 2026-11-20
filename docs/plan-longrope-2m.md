# A 2M context window (LongRoPE): what shipped, what did not

**Status (2026-10-18; corrected 2026-11-20): the 2M rung is implemented and
tested; the LongRoPE non-uniform rescale search is not.** The engine now accepts
`--rope-scaling yarn --max-context 2097152` on any host. 256K stays the native
default and `1 048 576` stays the YaRN default, because the engine serves
machines other than the one it is built on and choosing the window is the
operator's call. What is *not* implemented is the paper's per-dimension `λ`
search, which is the part that buys quality at 8× rather than just a position
range. On a 24 GB host a 2M window does not fit either (arithmetic below), so the
rung exists for larger machines and refuses through the normal memory guard
rather than silently clamping.

Requested 2026-10-18: "implement a 2M context window based on
<https://arxiv.org/abs/2402.13753>".

## What LongRoPE does

The paper (Ding et al., 2024) extends LLaMA2/Mistral to **2 048k** tokens with
three pieces:

1. **Non-uniform positional interpolation.** Per-dimension rescale factors
   `λ_i` plus a position threshold `n̂`: the rotation angle for dimension `i` at
   position `n` is `n / β^i` for `n < n̂` and `n / (β^i · λ_i)` for `n ≥ n̂`,
   with `λ` found by a search that minimises loss on long documents. This alone
   gives an **8× extension without fine-tuning** — and 262 144 × 8 =
   **2 097 152**, exactly the request.
2. **Progressive extension.** Fine-tune at 256k, then interpolate again to 2M.
3. **Re-adjust at 8k** to recover short-context quality.

## What it costs in memory

Qwen3.8-Flash-Next's geometry, from the installed manifest: 48 layers, of which
**12 are full attention** (the other 36 are linear/GDN with constant state),
`numFullKVHeads = 2`, `fullHeadDim = 256`.

| KV precision | bytes/token | at 1M | **at 2M** |
| --- | ---: | ---: | ---: |
| 8-bit | 12 KB | 12.6 GB | **24.6 GB** |
| 4-bit | 6 KB | 6.3 GB | **12.3 GB** |

The model's own resident estimate is **15.6 GB**, and the expert cache and OS sit
on top of that. So 2M needs **28–40 GB before any expert cache** — it does not
fit in 24 GB at either precision. The engine's YaRN rungs now include
`2 097 152` and now `4 194 304`, `8 388 608` and `16 777 216`
(`RuntimeConfiguration.supportedYaRNContextTokens`; `1 048 576` remains the
default), and even the 1M rung is marginal here at 4-bit KV.

### Dense Qwen 3.5: 1M is the default, with a threshold

The dense GPU path now defaults to a **1M cap** and a fixed **1M extension
profile**. The thresholded kernel keeps every position below the native 262,144
on the native frequencies and attention magnitude, so short answers are
byte-identical to the old native default; only positions past 262,144 use the
extension. The cap is capacity: `--max-context 4M/8M/16M` selects a larger KV
budget, not a different rope function. The larger rungs are for hosts that can
carry the KV; quality above 2M is unmeasured. The dense **CPU** engine has no
YaRN path and stays at 262K.

### Which YaRN this is

The 2M rung is generic factor-8 YaRN (`factor = 8`, `betaFast = 32`,
`betaSlow = 1`), not the paper's per-dimension `λ` table and not a per-model
tuned set. A 6.4 build that used different tuned parameters is not represented
by those values; porting them would need the actual table.

## The engine gap

RoPE is currently a scalar-theta kernel
(`sources/TinyTitan/Metal/Primitives/rope.metal`):

```
exponent = -2 * pair / frequency_divisor
angle    = position * pow(theta, exponent)
```

LongRoPE needs a **per-pair frequency table** and a **position threshold**, so the
angle above `n̂` uses `1/λ_i` extra scaling. That is a kernel-interface change
affecting every model's attention (prefill and decode), which is why it is not
being made speculatively for a window that cannot be served here.

## Scoped implementation, in dependency order

| # | Change | Verifiable on this Mac? |
| --- | --- | --- |
| 1 | Per-dimension RoPE table + `n̂` in the kernels, with a CPU reference test (arbitrary `λ`, both branches of `n̂`) | **yes** — synthetic tensors |
| 2 | A `longrope` rope mode that reads `λ` from the model directory and folds it into the table | yes, unit-level |
| 3 | The search harness: minimise long-document loss over `λ`, per the paper | **no** — needs the 2M KV itself (≥ 40 GB host) |
| 4 | `2 097 152` added to the YaRN rungs, behind a KV memory guard that refuses below ~40 GB | yes, the refusal |

Steps 1–2 are mechanism; 3–4 are what make it a product. **A 2M rung without a
searched `λ` would be a promise the tree cannot validate**: naive 8× scaling of a
model trained at 256K produces fluent text with broken long-range attention, and
no gate here could tell the difference.

## Decision

- On this host the honest ceiling is **1M with 4-bit KV**, and the decode rate at
  that context is far below the 262K figure in the README.
- A 2M window is a **≥ 40 GB target**. Start at step 1 on that machine, or on this
  one only when a synthetic-reference test is the deliverable, not a 2M run.
