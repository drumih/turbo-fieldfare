# Project Maya closeout (T31)

Status: closed on this host after T1-T34 were each implemented, measured,
rejected with evidence, or recorded as blocked by a resource this checkout does
not have. The closed-out status record is `docs/maya-task-status.md`; this page is the
scorecard and the verification footer.

## Environment

- Apple M3, 24 GB unified memory, macOS 27.0.1 (26A434), Xcode 27.1 /
  Swift 6.4.
- Installs used: Ornith 1.5 35B-A3B 4-bit (19 GB) and 8-bit (34 GB),
  Qwen3.8-Flash-Next 125B-A6B 4-bit (96 GB). Nothing was downloaded,
  converted, repacked or re-installed for this work.
- Base commit when the round opened: `23f8757`.

## What was added

| Task | Outcome |
| --- | --- |
| T22 edited-turn resume | `PromptCheckpointPlanner`, the prefill split hook and the longest-prefix restore. A turn-cache miss tries `lastTurnStart`, then `systemEnd`, then full prefill. Qwen3.8 carries QSA/PLE state in `InferenceAuxiliaryState` (1,407-token edit 44.38 s -> 2.76 s, 16.1x). |
| T23 checkpoint tests | Planner/store/live-runner matrix, including prefix, domain, limits, corruption, dedup and concurrency. |
| T24 checkpoint benchmark | 24.5K-token edited-turn read: 1119.3 s -> 3.07 s (**365x**), byte-identical text; 16K grid 465.2 s (2.41x). Literal 100K is host-blocked (see below). |
| T25 page-cache policy | Machine-size rule (`RAM >= 2 * installBytes`), explicit override, fail-closed inputs. No change on 24 GB. |
| T26 policy tests | Boundary, precedence, invalid override, unreadable inputs, install-size overflow. |
| T28 estimator | Adjacent-route distinct-expert predictor; MAE 1.881 (W=2) against 6.063 for the topK baseline on a recorded 48-layer trace. |

## What was rejected or blocked

| Task | Outcome |
| --- | --- |
| T11-T14 host-KV at scale | Blocked (model): the only QSA install is Qwen3.8 4-bit, which does not fit this 24 GB host at the target windows; Ornith has no indexer. T14 depends on T13. |
| T19 multi-Mac validation | Blocked (hardware): one Mac here. |
| T24 literal 100K | Host-blocked: the 100K request (about 61K tokens at the measured character ratio) was still prefilling at the 3600 s timeout. A 24.5K read is the feasible scale; the placement conclusion is unchanged and the scaling is documented. |
| T27 large-memory benchmark | Blocked (hardware): no 64/128 GB Mac. |
| T29 speculation gate | Rejected: measured verify pass 2.238x a scalar token, 1.857 emitted/pass at 85.7% acceptance, so even perfect acceptance is 0.894x. A +5% gate cannot fire; no controller code is kept. |
| T30 MTP controller benchmark | **Rejected after measurement.** The Ornith MTP sidecar was installed from the pinned HF source (`tools/install_models.sh ornith15-mtp`, 4.18 GB shard -> 43 MB install). At 64/128 tokens MTP measured 11.16/10.74 vs 11.76/12.21 tok/s aggregate (−5.1%/−12.0%; paired medians −14.2%/−13.4%), text identical; acceptance 70.3%/64.9%. No >5% win. |
| T34 fused decode | Rejected earlier on wall clock; its opt-in path remains for a 64 GB retest, not wired. |

## Performance kept, performance not kept

- Kept: the message-boundary checkpoint tier (365x on the edited turn), the
  machine-size page-cache *policy* (a no-op on this host, the measured 8-bit win
  on a host that fits), and the estimator as an offline benchmark tool.
- Not kept: a speculation controller (its only reachable state is off), a
  per-16K/32K grid as a default (2.41x at 16K and strictly less precise than
  `lastTurnStart`), and any capture on a family whose restore cannot succeed
  (QSA/PLE).
- The grid remains an explicit benchmark opt-in (`TINYTITAN_CHECKPOINT_GRID`,
  `TINYTITAN_CHECKPOINT_GRID_ONLY`); the default is unchanged.

## Additional model tuning (Qwen3.8-Flash-Next 4-bit)

The same round profiled the 125B model on this host. The only change kept is
its prefill chunk: **4,096 -> 2,048**, measured at 5-16% faster decode at the
same prefill on ~200- and ~851-token chat prompts, with the generated text
byte-identical to the golden baseline. The 8-bit row and the family fallback
keep 4,096 because a 10k-token prompt prefills ~11% faster with it; an operator
running long prompts can pass `--prefill-chunk 4096`. Expert cache slots (96),
wired cache, read advice, the page cache, GPU key select, HC fusion and the
LFU use cap were each A/B'd and left at their current values; see
`docs/qwen38-4bit-tuning-2026-10-22.md` for the arms and the losses.

`docs/qwen38-4bit-verification-2026-10-22.md` maps every task-list
optimization onto this model: width-2 batching has no aggregate win (and at a
lower slot count produced different text until QSA/PLE models were clamped to
one sequence), the fused loop is rejected on Qwen3.8 correctness, and the
page-cache policy correctly stays bounded for the 96 GB install. Interior
checkpoints now work on Qwen3.8 as well: a 1,407-token edited last turn fell
from 44.38 s to 2.76 s (16.1x, identical text) with the QSA/PLE auxiliary
payload, and the turn-end cache carries the same payload across a process
restart (`tier=ssd`, 6.22 s prefill vs 14.33 s cache-off, identical text);
`docs/qwen38-checkpoint-auxiliary.md`.

## Verification

| Gate | Result |
| --- | --- |
| `swift test --no-parallel` | **PASS**, 1,947 tests across the seven test bundles (927 + 421 + 164 + 189 + 85 + 63 + 98), 0 failures |
| `tools/lint.sh all` | **PASS**, exit 0: all eighteen gates, including swiftlint 0.65.1, swift-format and ruff 0.16.7 |
| `swift build -c release` | **PASS**, warnings-as-errors, 0 warnings |
| `tools/golden-baseline.sh --check 4` | **PASS**, `output identical to baseline` (Ornith 1.5 4-bit) |
| `tools/golden-baseline.sh --check qwen38-4` | **PASS**, `output identical to baseline` (Qwen3.8-Flash-Next 4-bit) |
| T24 edited-turn | **PASS**, text hash `b344d80e24a36799…` identical across none/message/16k |

The T24 command and per-arm rows are in
`docs/interior-checkpoint-benchmark-2026-10-22.md`; the T28 command and error
table are in `docs/speculation-cost-model-verdict.md`.

## Residual

The literal 100K placement suite (T24) and the large-memory page-cache arm
(T27) are the two measurements this host cannot produce. Both are recorded as
blocked rather than passed. A 64/128 GB Mac is the next useful instrument for
T24's 100K arm, T27, and a fused-decode retest (T34); nothing in the shipped
default was changed on the strength of an unmeasured arm.
