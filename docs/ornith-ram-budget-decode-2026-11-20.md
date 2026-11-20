# Ornith 1.5 35B-A3B 4-bit — decode tok/s against the RAM budget (2026-11-20)

A same-session sweep of the shipped Ornith 4-bit install at six resident-memory
targets. It answers whether a larger expert cache buys decode throughput on
this model, and it is a measurement only: no code or runtime default changed.

## Method

- Machine: Apple M3 MacBook Pro, 24 GB, macOS 27.2.1; commit `5dffa771`.
- Binary: `.build/release/TinyTitanCLI`, release build.
- Model: `models/ornith-1.5_35B_A3B_4Bit` (the shipped 4-bit install; no 8-bit
  install is present on this host).
- Prompt: three paragraphs of a technical description plus "continue for at
  least 700 more words" via `--messages-file`, so the 192-token cap binds rather
  than the model stopping early.
- Generation: `--max-new 192 --temperature 0`; each run is a fresh process, so
  load and first-token costs are outside the footer's decode window.
- Budgets: `--ram-budget 8G`, `10G`, `12G`, `14G`, `16G`, and **max** = no
  `--ram-budget` at all (the install's own profile). The flag is a
  whole-process target; the routed-expert cache gets what remains after the
  resident weights and the runtime reserve.
- Three rounds, interleaved across the six budgets (8G,10G,…,max, repeat), so
  thermal or background drift lands on every budget rather than on whichever
  ran last.
- Decode tok/s is the CLI's own footer value (`decode=` seconds and `new=`
  tokens), not a wall-clock estimate around the process.

All 18 runs generated **192 tokens** and stopped on `maxTokens`; no run stopped
early, so the windows are comparable.

## Results

| RAM budget | Mean tok/s | Range (3 runs) | Runs |
| --- | ---: | ---: | --- |
| 8G | **7.25** | 6.91–7.55 | 6.912, 7.284, 7.552 |
| 10G | **7.20** | 6.76–7.52 | 7.337, 6.760, 7.517 |
| 12G | **7.30** | 6.81–7.63 | 7.477, 6.806, 7.627 |
| 14G | **7.59** | 7.38–7.89 | 7.377, 7.507, 7.888 |
| 16G | **7.53** | 7.37–7.67 | 7.669, 7.369, 7.549 |
| max (no flag) | **7.58** | 7.44–7.71 | 7.444, 7.705, 7.600 |

## Reading

- 8G–12G sit together at 7.20–7.30 tok/s. 14G–16G and max sit at 7.53–7.59:
  about **+0.3 tok/s (~4–5%)** in the means.
- That difference is at the edge of the run-to-run spread (the 8G runs include
  7.55, the 14G runs include 7.38), so this is not a clean monotonic scaling
  claim. The only defensible statement is that **14G and above cluster slightly
  above 8–12G on this host**, and **max is not faster than 14–16G**.
- The floor at 8G is still usable: the model runs and decodes at ~7.2 tok/s.
  The 4-bit install's resident footprint plus runtime leaves enough room for
  the cache at that target.
- Do not quote these as the model's ceiling. This is a single 24 GB M3, three
  runs per cell, one prompt; the absolute rate is lower than the Ornith
  production figures measured on a quieter host/other prompt (the wiki's
  Ornith 7.0 table stands separately).

The raw rows are in `.build/ornith-ram-sweep.json` (gitignored); the table above
carries every value.
