# Ornith 1.5 35B-A3B 4-bit — decode vs the RAM budget (2026-11-20, corrected)

**This file was corrected after its first version.** The first sweep ran through
the 7.0 `Ornith35B4Bit` variant, whose factory hard-coded 32 expert-cache slots
and ignored the session's resolved slot count; `--ram-budget` and
`--expert-cache-slots` therefore had no effect, and the six cells measured the
same 32-slot configuration at ~7.2 tok/s. The variant now receives the resolved
slot count (`EngineFactory.buildModelWithSlots`, used by
`ServerModelSession.load`), with the env override still winning; the numbers
below are the re-run with that fix. The old table is withdrawn.

## Method

- Machine: Apple M3 MacBook Pro, 24 GB, macOS 27.2.1.
- Ornith 1.5 35B-A3B 4-bit install; release CLI.
- Prompt: three paragraphs of a technical description plus "continue for at
  least 700 more words" via `--messages-file`, so `--max-new 192` binds rather
  than the model stopping.
- Generation: 192 tokens, `--temperature 0`; a fresh process per run, so load
  and first-token costs are outside the footer's decode window.
- Budgets: `--ram-budget 8G`, `10G`, `12G`, `14G`, `16G`, and **max** = no
  `--ram-budget` at all (the install's profile, 10 GiB / 160 slots for this
  pair). Three interleaved rounds across all six budgets.
- Decode tok/s is the CLI footer's own value (`new=`/`decode=`), not a
  wall-clock estimate around the process.
- Every one of the 18 runs generated 192 tokens and stopped on `maxTokens`.

## Results

| RAM budget | Mean tok/s | Range (3 runs) | Runs |
| --- | ---: | ---: | --- |
| 8G | **16.41** | 15.08–17.31 | 15.080, 16.832, 17.310 |
| 10G | **19.02** | 17.39–20.20 | 17.386, 19.488, 20.199 |
| 12G | **20.10** | 18.78–20.81 | 18.782, 20.707, 20.813 |
| 14G | **21.33** | 21.07–21.52 | 21.406, 21.069, 21.517 |
| 16G | **20.96** | 20.62–21.60 | 20.621, 20.657, 21.601 |
| max (no flag) | **21.35** | 20.81–21.64 | 20.807, 21.614, 21.641 |

## Why 6.x reported 10–20 tok/s

The 6.x profile for this pair (`ModelProfile.swift`) is a **10 GiB expert-cache
budget**, resolved to the 128/160-slot rungs and measured at 20.4–21.0 tok/s in
its own 2026-09-05 sample; the README's 21.7 tok/s is the warm-up-pass figure.
The 7.0 variant's first factory defaulted to **32 slots**, deliberately kept
small for its parity runs, and did not read the session's resolved budget. That
is the entire gap: 32 slots ~7 tok/s, the 6.x profile ~21 tok/s.

The fix is not a new kernel. It passes the already-computed
`EngineEngineLoadPlan.expertCacheSlots` into the variant's model load; the
env override remains the operator's last word. The corrected sweep lands where
the 6.x profile does: 8G 16.4, 10G 19.0, 12G 20.1, 14G–max 21.0–21.4 tok/s.

The registry's Ornith qualification (`decodeTokS: 11.55`) was measured on the
old 32-slot configuration; a re-qualification at the new default is tracked
separately. This file is a performance measurement only: no generation output
changed (the `ornith-4` golden is byte-identical).

The raw rows are in `.build/ornith-ram-sweep-fixed.json` (gitignored); the table
above carries every value.
