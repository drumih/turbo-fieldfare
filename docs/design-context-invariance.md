# Context-size invariance: a cap that cannot change the answer

**Status: proposal (2026-11-20). Not implemented.** This is the design for making
`--max-context` a capacity decision rather than a numerics decision.

## The problem

The context cap currently selects the YaRN target, and YaRN rescales RoPE
frequencies at **every** position. Choosing 1M instead of native 262K therefore
changes a short prompt: measured on the 2B golden prompt, the same greedy 96-token
run produces different text at 1M. That is a quality change, not a capacity
change.

## The requirement

For one text and one model, any context cap that is at least the text length must
produce the **same output**. Where a native reference exists (text ≤ 262,144),
that output must be the native one.

## The design

Three separate concepts:

1. **Capacity (`maxContext`)** — the maximum sequence length and the KV arena
   size. It may refuse a run that does not fit; it never enters the RoPE math.
2. **Rope profile** — a fixed per-model function independent of `maxContext`:
   - positions `p < n̂` use the model's native frequencies;
   - positions `p ≥ n̂` use one fixed extended frequency table and threshold,
     with `n̂ = 262,144` (native).
   The profile is chosen once per model, not once per cap.
3. **KV allocation** — sized to the actual sequence/limit, not eagerly to the
   cap, so a large cap does not by itself cost memory.

With that, a 5K prompt is native at every cap; a 500K prompt uses the same fixed
extended profile whether the cap is 1M or 16M. The cap only decides whether the
run is allowed and how much memory is reserved.

## What is mathematically impossible

For text longer than 262,144 there is no native reference: the native model
cannot process it. Any extension changes attention, so “identical to a native
long run” is not a target. The achievable guarantee is:

- `text ≤ 262,144`: bit-identical to the current native output, at every cap.
- `text > 262,144`: identical across every cap (one fixed profile), with quality
  against a true long-context reference to be measured — not assumed.

## Kernel shape

The current YaRN kernels hold one inverse-frequency table and apply it to all
positions. The thresholded version needs two tables (native and extended) and a
per-position selection in prefill and decode. Attention-factor scaling must be
conditional too: 1 below the threshold, the extension factor above it. The
same check must cover the dense CPU path, which has no YaRN today; until then CPU
stays native and refuses beyond it.

## Acceptance

- Golden/greedy output for every dense install is byte-identical at native and at
  every cap ≥ text length, including 262K vs 1M vs 16M on a short prompt.
- One long text produces the same output at caps 1M, 2M, 4M, 8M and 16M.
- KV memory tracks the actual sequence, not the cap.
- Long-context quality (needle/retrieval and the project's checked prompts) is
  measured; the extension profile is rejected if it degrades below the accepted
  floor.

## Why not just “use native always”

A native-only path cannot serve a prompt longer than 262K. The point of the
profile split is that short text keeps native quality **and** a large cap can be
offered without changing it.
