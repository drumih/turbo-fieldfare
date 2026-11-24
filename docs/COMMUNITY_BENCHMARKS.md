# Contribute a benchmark result

Run the three [frozen chat cases](benchmark-prompts/real-generation-v1) and
share the complete outputs. The standard profile uses prefetch **off** and
16 cache slots. The commands below require a build with `--early-expert-read`.

## Prepare the Mac

[Install the model](../README.md#command-line-interface), connect to power,
disable Low Power Mode and quit demanding apps. Use runtime defaults without
`TURBO_FIELDFARE_*` or `GTURBO_*` overrides or profiling.

```bash
swift build -c release --product TurboFieldfareCLI
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

Continue only when the process check prints nothing. Run one process at a time.

## Record the machine

Run these blocks in the same shell. Each session gets a new results directory.

```bash
mkdir -p benchmark-results
benchmark_run=$(mktemp -d benchmark-results/run.XXXXXX)
mkdir -p "$benchmark_run"/{system,warmup,measured}
{
  git status --short
  git rev-parse HEAD
  sw_vers
  swift --version
  system_profiler SPHardwareDataType |
    awk -F': ' '/Model Name|Model Identifier|Chip|Total Number of Cores|Memory/ { print $1 ": " $2 }'
  shasum -a 256 scratch/gemma4.gturbo/manifest.json
  shasum -a 256 docs/benchmark-prompts/real-generation-v1/*.json
} 2>&1 | tee "$benchmark_run/system/system.txt"
```

Review files for personal information before sharing.

## Run the cases

This runs all three discarded warmups, then all three measured cases in fresh
processes. Keep these prompts, settings and seeds unchanged.

```bash
(
  set -e
  : "${benchmark_run:?Run the machine-recording block first}"
  for phase in warmup measured; do
    for case_seed in short-explanation:20260721 medium-review:20260722 long-synthesis:20260723; do
      case_id="${case_seed%%:*}"
      seed="${case_seed##*:}"
      .build/release/TurboFieldfareCLI \
        --model scratch/gemma4.gturbo \
        --messages-file "docs/benchmark-prompts/real-generation-v1/${case_id}.json" \
        --max-new 1024 --max-context 4096 \
        --early-expert-read off --expert-cache-slots 16 \
        --temperature 0.2 --top-k 64 --top-p 0.95 --seed "$seed" \
        > "${benchmark_run}/${phase}/${case_id}.stdout" \
        2> "${benchmark_run}/${phase}/${case_id}.stderr"
    done
  done

  grep -h '^\[stop=' "$benchmark_run"/measured/*.stderr
)
```

Require three measured runs with `stop=endOfTurn`. Read their outputs: loops,
repeated blocks or incomplete answers invalidate a speed result.

## Report the result

Open a [benchmark issue](https://github.com/drumih/turbo-fieldfare/issues/new?template=benchmark.yml)
with:

- the system file and all three measured stdout/stderr pairs;
- the exact command, effective prefetch setting (`false` for off, `true` for
  router), and any overrides;
- storage, power mode, other active workloads and protocol deviations.

Keep full stderr, including token counts, timing and `[early-read ...]` when
present. Missing historical settings remain `unknown`.

## Optional speculative prefetch comparison

Use the same release binary, model, host, settings and per-case seeds; change
only `--early-expert-read off` to `router`. Save each run separately. Do not
purge caches, rebuild, change settings or enable diagnostics between modes.

| Stage | Case order | Mode order per case |
| --- | --- | --- |
| Discarded warmups | Short, medium, long | Off, router |
| Measured pair 1 | Short, medium, long | Off, router |
| Measured pair 2 | Long, medium, short | Router, off |

These are 18 fresh processes. Report each case and pair separately:
`100 * (router_tok_s / off_tok_s - 1)`. Require matching complete output bytes,
token counts and `endOfTurn`; investigate mismatches before interpreting speed.
Report raw rates, run order and request latency as well as percentages. Two
pairs are preliminary; retain mixed and negative results.

If collecting memory, use `/usr/bin/time -l` for both modes. Distinguish RSS
from footprint and swap-counter deltas from existing swap occupancy; use the
page size reported by `vm_stat`. Missing measurements limit memory conclusions.

## Community results

Submitted results are not a controlled hardware comparison. Prefetch is
`false` for verified pre-feature revisions and `unknown` when unverified.
Maintainer M2/M5 results are in [Benchmarks](BENCHMARKS.md).

| Source | Mac | Memory | Storage | Prefetch | Prompt / generated tokens | Decode | Measurement |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| [bangddong](https://github.com/bangddong) · [issue #80](https://github.com/drumih/turbo-fieldfare/issues/80) | A18 Pro MacBook Neo | 8 GB | Internal SSD | false | 61 / 516 | 4.134 tok/s | One submitted run |
| [bangddong](https://github.com/bangddong) · [issue #80](https://github.com/drumih/turbo-fieldfare/issues/80) | A18 Pro MacBook Neo | 8 GB | Internal SSD | false | 430 / 780 | 3.672 tok/s | One submitted run |
| [bangddong](https://github.com/bangddong) · [issue #80](https://github.com/drumih/turbo-fieldfare/issues/80) | A18 Pro MacBook Neo | 8 GB | Internal SSD | false | 3,015 / 617 | 3.012 tok/s | One submitted run |
| [Maxim Voronin](https://github.com/maxslamdunk) · [issue #116](https://github.com/drumih/turbo-fieldfare/issues/116) | M1 MacBook Air | 8 GB | Internal 256 GB SSD | false | 61 / 511 | 3.340 tok/s | One submitted run |
| [Maxim Voronin](https://github.com/maxslamdunk) · [issue #116](https://github.com/drumih/turbo-fieldfare/issues/116) | M1 MacBook Air | 8 GB | Internal 256 GB SSD | false | 430 / 669 | 3.053 tok/s | One submitted run |
| [Maxim Voronin](https://github.com/maxslamdunk) · [issue #116](https://github.com/drumih/turbo-fieldfare/issues/116) | M1 MacBook Air | 8 GB | Internal 256 GB SSD | false | 3,015 / 604 | 2.642 tok/s | One submitted run |
| [German](https://github.com/iGerman00) · [issue #76](https://github.com/drumih/turbo-fieldfare/issues/76) | M1 MacBook Air | 16 GB | Internal SSD | false | 61 / 511 | 7.802 tok/s | One submitted run |
| [German](https://github.com/iGerman00) · [issue #76](https://github.com/drumih/turbo-fieldfare/issues/76) | M1 MacBook Air | 16 GB | Internal SSD | false | 430 / 669 | 7.173 tok/s | One submitted run |
| [German](https://github.com/iGerman00) · [issue #76](https://github.com/drumih/turbo-fieldfare/issues/76) | M1 MacBook Air | 16 GB | Internal SSD | false | 3,015 / 604 | 6.093 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #94](https://github.com/drumih/turbo-fieldfare/issues/94) | M1 Mac mini | 16 GB | Internal 512 GB SSD | false | 61 / 511 | 6.754 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #94](https://github.com/drumih/turbo-fieldfare/issues/94) | M1 Mac mini | 16 GB | Internal 512 GB SSD | false | 430 / 669 | 6.265 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #94](https://github.com/drumih/turbo-fieldfare/issues/94) | M1 Mac mini | 16 GB | Internal 512 GB SSD | false | 3,015 / 604 | 5.570 tok/s | One submitted run |
| [Felix Li](https://github.com/dt1dr) · [issue #37](https://github.com/drumih/turbo-fieldfare/issues/37) | M1 Pro MacBook Pro | 16 GB | Internal SSD | false | 61 / 511 | 9.477 tok/s | One submitted run |
| [Felix Li](https://github.com/dt1dr) · [issue #37](https://github.com/drumih/turbo-fieldfare/issues/37) | M1 Pro MacBook Pro | 16 GB | Internal SSD | false | 430 / 669 | 8.576 tok/s | One submitted run |
| [Felix Li](https://github.com/dt1dr) · [issue #37](https://github.com/drumih/turbo-fieldfare/issues/37) | M1 Pro MacBook Pro | 16 GB | Internal SSD | false | 3,015 / 604 | 7.805 tok/s | One submitted run |
| [Vaibhav Malik](https://github.com/VaibhavMalik4187) · [issue #64](https://github.com/drumih/turbo-fieldfare/issues/64) | M1 Max MacBook Pro | 64 GB | Internal SSD | false | 61 / 511 | 22.888 tok/s | One submitted run |
| [Vaibhav Malik](https://github.com/VaibhavMalik4187) · [issue #64](https://github.com/drumih/turbo-fieldfare/issues/64) | M1 Max MacBook Pro | 64 GB | Internal SSD | false | 430 / 669 | 22.043 tok/s | One submitted run |
| [Vaibhav Malik](https://github.com/VaibhavMalik4187) · [issue #64](https://github.com/drumih/turbo-fieldfare/issues/64) | M1 Max MacBook Pro | 64 GB | Internal SSD | false | 3,015 / 604 | 20.493 tok/s | One submitted run |
| [stefinuxha](https://github.com/stefinuxha) · [issue #131](https://github.com/drumih/turbo-fieldfare/issues/131) | M2 Mac mini | 16 GB | Not reported | false | 61 / 511 | 9.674 tok/s | One submitted run |
| [stefinuxha](https://github.com/stefinuxha) · [issue #131](https://github.com/drumih/turbo-fieldfare/issues/131) | M2 Mac mini | 16 GB | Not reported | false | 430 / 669 | 8.889 tok/s | One submitted run |
| [stefinuxha](https://github.com/stefinuxha) · [issue #131](https://github.com/drumih/turbo-fieldfare/issues/131) | M2 Mac mini | 16 GB | Not reported | false | 3,015 / 604 | 6.578 tok/s | One submitted run |
| [Matt 'Smartis' W](https://github.com/0xSmartis) · [issue #69](https://github.com/drumih/turbo-fieldfare/issues/69) | M2 Pro Mac mini | 16 GB | Internal SSD | false | 61 / 511 | 15.713 tok/s | One submitted run |
| [Matt 'Smartis' W](https://github.com/0xSmartis) · [issue #69](https://github.com/drumih/turbo-fieldfare/issues/69) | M2 Pro Mac mini | 16 GB | Internal SSD | false | 430 / 669 | 17.826 tok/s | One submitted run |
| [Matt 'Smartis' W](https://github.com/0xSmartis) · [issue #69](https://github.com/drumih/turbo-fieldfare/issues/69) | M2 Pro Mac mini | 16 GB | Internal SSD | false | 3,015 / 604 | 15.755 tok/s | One submitted run |
| [Eli Smith](https://github.com/eliliam) · [issue #139](https://github.com/drumih/turbo-fieldfare/issues/139) | M2 Pro MacBook Pro | 16 GB | Internal SSD | false | 61 / 511 | 11.749 tok/s | One submitted run |
| [Eli Smith](https://github.com/eliliam) · [issue #139](https://github.com/drumih/turbo-fieldfare/issues/139) | M2 Pro MacBook Pro | 16 GB | Internal SSD | false | 430 / 669 | 10.428 tok/s | One submitted run |
| [Eli Smith](https://github.com/eliliam) · [issue #139](https://github.com/drumih/turbo-fieldfare/issues/139) | M2 Pro MacBook Pro | 16 GB | Internal SSD | false | 3,015 / 604 | 9.384 tok/s | One submitted run |
| [elroyonline](https://github.com/elroyonline) · [issue #59](https://github.com/drumih/turbo-fieldfare/issues/59) | M2 Max Mac Studio | 32 GB | External PCIe NVMe | false | 61 / 511 | 25.310 tok/s | One submitted run |
| [elroyonline](https://github.com/elroyonline) · [issue #59](https://github.com/drumih/turbo-fieldfare/issues/59) | M2 Max Mac Studio | 32 GB | External PCIe NVMe | false | 430 / 669 | 25.172 tok/s | One submitted run |
| [elroyonline](https://github.com/elroyonline) · [issue #59](https://github.com/drumih/turbo-fieldfare/issues/59) | M2 Max Mac Studio | 32 GB | External PCIe NVMe | false | 3,015 / 604 | 23.730 tok/s | One submitted run |
| [Benjamin Schilling](https://github.com/benjamin-schilling) · [issue #27](https://github.com/drumih/turbo-fieldfare/issues/27) | M3 Pro MacBook Pro | 18 GB | Internal SSD | false | 61 / 516 | 19.09 tok/s | One submitted run |
| [Benjamin Schilling](https://github.com/benjamin-schilling) · [issue #27](https://github.com/drumih/turbo-fieldfare/issues/27) | M3 Pro MacBook Pro | 18 GB | Internal SSD | false | 430 / 780 | 16.50 tok/s | One submitted run |
| [Benjamin Schilling](https://github.com/benjamin-schilling) · [issue #27](https://github.com/drumih/turbo-fieldfare/issues/27) | M3 Pro MacBook Pro | 18 GB | Internal SSD | false | 3,015 / 617 | 13.88 tok/s | One submitted run |
| [castillovas-arch](https://github.com/castillovas-arch) · [issue #172](https://github.com/drumih/turbo-fieldfare/issues/172) | M4 Mac mini | 16 GB | Internal SSD | false | 61 / 489 | 11.649 tok/s | One measured run after one discarded warmup |
| [castillovas-arch](https://github.com/castillovas-arch) · [issue #172](https://github.com/drumih/turbo-fieldfare/issues/172) | M4 Mac mini | 16 GB | Internal SSD | false | 430 / 760 | 10.119 tok/s | One measured run after one discarded warmup |
| [castillovas-arch](https://github.com/castillovas-arch) · [issue #172](https://github.com/drumih/turbo-fieldfare/issues/172) | M4 Mac mini | 16 GB | Internal SSD | false | 3,015 / 594 | 8.058 tok/s | One measured run after one discarded warmup |
| [gandalfk7](https://github.com/gandalfk7) · [issue #95](https://github.com/drumih/turbo-fieldfare/issues/95) | M4 Mac mini | 16 GB | Internal 256 GB SSD | false | 61 / 516 | 10.847 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #95](https://github.com/drumih/turbo-fieldfare/issues/95) | M4 Mac mini | 16 GB | Internal 256 GB SSD | false | 430 / 780 | 9.878 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #95](https://github.com/drumih/turbo-fieldfare/issues/95) | M4 Mac mini | 16 GB | Internal 256 GB SSD | false | 3,015 / 617 | 8.658 tok/s | One submitted run |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | Internal 256 GB SSD | false | 61 / 516 | 11.38 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | Internal 256 GB SSD | false | 430 / 780 | 8.43 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | Internal 256 GB SSD | false | 3,015 / 617 | 8.58 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | External Samsung 990 Pro | false | 61 / 516 | 11.53 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | External Samsung 990 Pro | false | 430 / 780 | 10.79 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | External Samsung 990 Pro | false | 3,015 / 617 | 11.21 tok/s | Median of five runs |
| [dragonwang1019](https://github.com/dragonwang1019) · [issue #101](https://github.com/drumih/turbo-fieldfare/issues/101) | M4 Pro Mac mini | 24 GB | External NVMe SSD | false | 61 / 516 | 22.256 tok/s | One submitted run |
| [dragonwang1019](https://github.com/dragonwang1019) · [issue #101](https://github.com/drumih/turbo-fieldfare/issues/101) | M4 Pro Mac mini | 24 GB | External NVMe SSD | false | 430 / 780 | 28.964 tok/s | One submitted run |
| [dragonwang1019](https://github.com/dragonwang1019) · [issue #101](https://github.com/drumih/turbo-fieldfare/issues/101) | M4 Pro Mac mini | 24 GB | External NVMe SSD | false | 3,015 / 617 | 24.209 tok/s | One submitted run |
| [r1tz](https://github.com/r1tz) · [issue #175](https://github.com/drumih/turbo-fieldfare/issues/175) | M4 Pro Mac mini | 24 GB | Not reported | false | 61 / 489 | 33.343 tok/s | One submitted run |
| [r1tz](https://github.com/r1tz) · [issue #175](https://github.com/drumih/turbo-fieldfare/issues/175) | M4 Pro Mac mini | 24 GB | Not reported | false | 430 / 760 | 31.236 tok/s | One submitted run |
| [r1tz](https://github.com/r1tz) · [issue #175](https://github.com/drumih/turbo-fieldfare/issues/175) | M4 Pro Mac mini | 24 GB | Not reported | false | 3,015 / 594 | 25.186 tok/s | One submitted run |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | false | 61 / 516 | 34.060 tok/s | Automatic mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | false | 430 / 780 | 30.954 tok/s | Automatic mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | false | 3,015 / 617 | 25.292 tok/s | Automatic mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | false | 61 / 516 | 33.379 tok/s | High Power mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | false | 430 / 780 | 30.614 tok/s | High Power mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | false | 3,015 / 617 | 25.230 tok/s | High Power mode |
| [nxexcelsior](https://github.com/nxexcelsior) · [issue #111](https://github.com/drumih/turbo-fieldfare/issues/111) | M5 MacBook Air | 24 GB | Internal SSD | false | 61 / 497 | 11.982 tok/s | One submitted run |
| [nxexcelsior](https://github.com/nxexcelsior) · [issue #111](https://github.com/drumih/turbo-fieldfare/issues/111) | M5 MacBook Air | 24 GB | Internal SSD | false | 430 / 721 | 12.429 tok/s | One submitted run |
| [nxexcelsior](https://github.com/nxexcelsior) · [issue #111](https://github.com/drumih/turbo-fieldfare/issues/111) | M5 MacBook Air | 24 GB | Internal SSD | false | 3,015 / 610 | 9.783 tok/s | One submitted run |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 61 / 497 | 36.2 tok/s | Median of 3 runs, default 16 expert-cache slots |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 430 / 721 | 34.0 tok/s | Median of 3 runs, default 16 expert-cache slots |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 3,015 / 610 | 29.5 tok/s | Median of 3 runs, default 16 expert-cache slots |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 61 / 497 | 38.9 tok/s | Median of 3 runs, 24 expert-cache slots |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 430 / 721 | 36.5 tok/s | Median of 3 runs, 24 expert-cache slots |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 3,015 / 610 | 31.5 tok/s | Median of 3 runs, 24 expert-cache slots |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 61 / 497 | 40.8 tok/s | Median of 3 runs, 32 expert-cache slots |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 430 / 721 | 38.5 tok/s | Median of 3 runs, 32 expert-cache slots |
| [chongdlh-pixel](https://github.com/chongdlh-pixel) · [issue #170](https://github.com/drumih/turbo-fieldfare/issues/170) | M5 Pro MacBook Pro | 24 GB | Not reported | unknown | 3,015 / 610 | 33.3 tok/s | Median of 3 runs, 32 expert-cache slots |
| [Matt Skipton](https://github.com/mattskipton) · [issue #97](https://github.com/drumih/turbo-fieldfare/issues/97) | M5 Pro MacBook Pro | 64 GB | Not reported | false | 61 / 497 | 37.426 tok/s | One submitted run |
| [Matt Skipton](https://github.com/mattskipton) · [issue #97](https://github.com/drumih/turbo-fieldfare/issues/97) | M5 Pro MacBook Pro | 64 GB | Not reported | false | 430 / 721 | 34.194 tok/s | One submitted run |
| [Matt Skipton](https://github.com/mattskipton) · [issue #97](https://github.com/drumih/turbo-fieldfare/issues/97) | M5 Pro MacBook Pro | 64 GB | Not reported | false | 3,015 / 610 | 31.485 tok/s | One submitted run |
| [Ulises](https://github.com/ulises-c) · [issue #149](https://github.com/drumih/turbo-fieldfare/issues/149) | M5 Max MacBook Pro | 36 GB | Internal SSD | false | 61 / 497 | 43.440 tok/s | One submitted run |
| [Ulises](https://github.com/ulises-c) · [issue #149](https://github.com/drumih/turbo-fieldfare/issues/149) | M5 Max MacBook Pro | 36 GB | Internal SSD | false | 430 / 721 | 40.965 tok/s | One submitted run |
| [Ulises](https://github.com/ulises-c) · [issue #149](https://github.com/drumih/turbo-fieldfare/issues/149) | M5 Max MacBook Pro | 36 GB | Internal SSD | false | 3,015 / 610 | 37.370 tok/s | One submitted run |

Reported pre-feature revisions supporting the `false` values:

| Submission issues | Reported revision |
| --- | --- |
| #23, #27 | `f8abc442` |
| #37 | `1859181a` |
| #59, #64 | `add22ff1` |
| #67, #69 | `8648274d` |
| #76 | `fcd8f78a` |
| #80 | `f43822ec` |
| #94, #95, #97 | `3249be40` |
| #101 | `b6eaf02b` |
| #111, #116 | `417f3893` |
| #131 | `903c7bf` |
| #139 | `9df7b677` |
| #149 | `fd3c0b8b` |
| #172 | `6c044011` |
| #175 | `4c6db1e` |

These revisions precede `b3e82c32`, which introduced speculative prefetch.
The source links in the results table identify the submissions; the boolean
reflects their reported revision, not an independent rerun of those machines.

[Issue #170's median follow-up](https://github.com/drumih/turbo-fieldfare/issues/170#issuecomment-5529940211)
uses three different seeds per case and slot count. Its nine median rows are
therefore aggregates across different generated workloads. The displayed
generated-token counts come from the initial submission; the follow-up does
not give counts for every additional seed. Its custom-build source revision
and prefetch setting remain unverified. Keep these deviations when comparing
its cache-slot configurations with fixed-seed results.

These submissions use the public community prompts and generate until the end
of the model turn. Compare rows only when the case, prompt and generated-token
counts, settings, and stop reason match.

[Issue #172](https://github.com/drumih/turbo-fieldfare/issues/172) reports macOS
26.6.2 (25G83), Swift 6.3.3, and commit `6c044011c24dd55595af823cfef10f26b525cfca`.
Each measured case ran in a fresh process with the sampling settings above;
all three ended with `stop=endOfTurn`. The reporter reviewed the outputs and
reported no loops, repeated blocks or protocol deviations.

[Issue #175](https://github.com/drumih/turbo-fieldfare/issues/175) reports a
12-core M4 Pro Mac mini (Mac16,11), macOS 26.6.2 (25G83), Swift 6.4, and commit
`4c6db1e698ea861609109d4bc517410ff302af46`. The attached prompt hashes match
the frozen community prompts. All three measured runs ended with
`stop=endOfTurn`; the attached outputs finish without obvious loops or
repeated blocks. Storage, energy mode and other active workloads were not
reported. These are individual submitted runs, not repeated-run averages.
