# Handover: Ornith ANE prefill — the tail split (2026-11-19, evening)

**Paste this into the next session:**

> Continue the TinyTitan work in this checkout. Read `AGENTS.md`, then
> `docs/handover-tinytitan-7.md`, then this file, then the wiki
> `Project-Tracker` row **TT-099**, then
> `.superpowers/sdd/2026-11-18-ornith-ane-prefill/progress.md` (gitignored; the
> measured runbook and probe log). The Ornith ANE prefill now produces a
> **coherent, faster** answer but is **not token-identical** to the GPU arm:
> common prefix 35 characters, live logits 4.34%. The remaining decision is
> whether that is acceptable or the ANE stays GPU-only.
>
> **The 10G benchmark is complete:** `benchmark/ane-prefill/v7-ornith-ane-ram10.json`
> — GPU 140.10 s, ANE 118.00 s, 1.19×, 2 runs/arm, `used_ane` true. Nothing is
> running.
>
> **Do next, in order:** (1) review the completed 10G table above with the machine state; (2) decide with the owner whether
> "coherent + 35-char agreement" is the bar or identical text is mandatory;
> (3) if accepted, update the README/wiki/Tracker and close TT-099; if not,
> rename `models/ornith-1.5_35B_A3B_4Bit/ane_prefill` to
> `ane_prefill-rejected` and record the negative result; (4) run
> `swift test --no-parallel`, `tools/lint.sh all`, the golden `ornith-4` and
> push `main` plus the wiki `master` (two pushes).

## What was wrong

The ANE executed the same sidecar package at **4.245%** error against the fp32
reference on a real captured normed, while the identical package on the GPU
scored **0.075%**. Stage-by-stage on the ANE: attention core **0.64%**, output
gate **1.74%**, `wo` projection **3.96%**, gate+`wo` **4.25%**. The tail loses
the result to fp16 accumulation/cancellation (`gated` mean 0.008 against 0.1–0.7
partial products cancelling to a 0.0101 output). Splitting `wo` into 4/16 fp32
partials and manual fp16/fp32 softmax are both canonicalized by CoreML and
change nothing; `compute_precision=FLOAT32` leaves 4.248%.

## What changed (the “tail split”)

- `tools/export_ane_prefill.py`: the graph returns `attn` (attention core,
  `[t, qDim]`) and `gate` (pre-sigmoid, `[t, qDim]`) instead of `out`; metadata
  records `"tail": "split"`. Version stays 1.
- Runtime (`sources/TinyTitan/…` and the byte-identical
  `sources/Engines/Ornith35B4Bit/…` copies): a split sidecar predicts
  `attn`/`gate`, the bridge blits them to `attentionOutput`/`attnGate`, and
  applies `sigmoid` + the `o` projection on the GPU (`encodeSigmoidGateMul` +
  `encodeAffineProjection`). A sidecar without `tail` keeps the historical
  fused `out` path, so old sidecars still load.
- `tools/verify_ane_sidecar.py` applies the fp32 gate+`wo` tail itself for a
  split sidecar.
- `benchmark/ane_prefill_ab_matrix.py` gained `--ram-budget` (passed through to
  the CLI).
- `tests/TinyTitan/Runtime/Prefill/ANEPrefillAttentionTests.swift`: pins that
  `tail: "split"` decodes and an absent field means fused.

## Measured (M3 24 GB, chunk 4,096, 4,333-token prompt)

| Gate | Result |
|---|---|
| 1 verify (split sidecar) | synthetic **0.147%**, real normed **0.619%** — pass |
| 2 layer diff | `on` logits **4.34%** (was 5.55%); L3_after 0.602%, L39 1.49% |
| 3 identical greedy | **false**, common prefix **35** (was 0); the ANE answer is coherent: “# Swift vs C++: A Fair Comparison\n\nBefore judging, let me name the axes…” |
| 3 single-run prefill | ANE **78.14 s** vs GPU **101.34 s** (1.30×) at the default budget |
| 4 paired @10G | GPU **140.10 s**, ANE **118.00 s**, **1.19×** (22.10 s saved), 2 runs/arm, `used_ane` true; host load 6.0 at record time |

`benchmark/ane-prefill/ane-correctness-ane-correctness-ornith15-4bit-split.json`
`ane-layer-diff-ornith15-4bit-split.json` and `v7-ornith-ane-ram10.json` are committed. The older
`*-baseline`, `*-fp32softmax`, `*-fp32core` records are the failing history and
must not be overwritten.

## Exact commands

```bash
# rebuild after any Swift change (the benchmark scripts use the release CLI)
swift build -c release

# gate 4 (paired, 10G) — the deliverable
python3 benchmark/ane_prefill_ab_matrix.py \
  --models ornith-1.5_35B_A3B_4Bit --repeats 2 --ram-budget 10G \
  --label v7-ornith-ane-ram10 --record

# re-run the quality gates if needed
~/.venvs/coreml-py311/bin/python tools/verify_ane_sidecar.py \
  --model models/ornith-1.5_35B_A3B_4Bit \
  --activation-dump .superpowers/sdd/2026-11-18-ornith-ane-prefill/real-L3_in.f16
python3 benchmark/ane_prefill_correctness.py --models ornith-1.5_35B_A3B_4Bit \
  --max-new 32 --label ane-correctness-ornith15-4bit-split --record

# gates and publication
swift test --no-parallel
PATH="$HOME/.local/bin:$PATH" tools/lint.sh all
tools/golden-baseline.sh --check ornith-4
git push origin main
cd .qwen/wiki && git push origin master
```

## Traps

- **Do not delete `models/ornith-1.5_35B_A3B_4Bit/ane_prefill-rejected/`** — it
  is the fused sidecar that proves the tail split. `ane_prefill/` is the split
  one; the runtime selects it automatically.
- The split is **only** in the graph/contract; a re-export with an older
  checkout writes a fused sidecar that the new runtime still handles.
- A sidecar's `tail` field is the contract: absent = fused `out`, `"split"` =
  `attn`/`gate`. The verifier and the runtime both branch on it.
- The benchmark host is a GUI Mac (Chrome/WindowServer load); report the load
  with the numbers. Do not benchmark while a model process is running.
- Rebuild release after any Swift edit; the scripts have no way to notice a
  stale CLI.
- The 10G run is slow (expert streaming); allow ~30 min for two arms ×
  warm-up + 2 repeats.
- If the decision is “identical only”, the residual 4.34% comes from the ANE
  attention core (0.62% block on the real input) and is likely inherent; moving
  the core to the GPU removes the ANE benefit.
