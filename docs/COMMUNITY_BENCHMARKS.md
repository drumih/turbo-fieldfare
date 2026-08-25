# Contribute a benchmark result

TurboFieldfare's community benchmark uses three chat-framed generation cases:
a short explanation, a medium design review, and a long document synthesis.
They exercise different prompt lengths and require coherent text that reaches
the end of the model turn. A repeating calibration prompt is not a valid speed
result because repeated expert choices can make decode artificially fast.

Submitted measurements are listed under
[Community results](#community-results).

The frozen prompts are in
[`benchmark-prompts/real-generation-v1/`](benchmark-prompts/real-generation-v1).
Runs use the app sampling defaults with fixed seeds: temperature `0.2`, Top-K
`64`, Top-P `0.95`, a 4,096-token context, and up to 1,024 generated tokens.

## Prepare the Mac

Install the model with the [README instructions](../README.md#command-line-interface),
connect a laptop to power, turn off Low Power Mode, quit other demanding apps,
and build the release CLI:

```bash
swift build -c release --product TurboFieldfareCLI
```

Confirm that no other model process is running:

```bash
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

Continue only when that command prints nothing.

## Record the machine

```bash
mkdir -p benchmark-results/system benchmark-results/warmup benchmark-results/measured
{
  git status --short
  git rev-parse HEAD
  sw_vers
  swift --version
  system_profiler SPHardwareDataType |
    awk -F': ' '/Model Name|Model Identifier|Chip|Total Number of Cores|Memory/ { print $1 ": " $2 }'
  shasum -a 256 scratch/gemma4.gturbo/manifest.json
  shasum -a 256 docs/benchmark-prompts/real-generation-v1/*.json
} 2>&1 | tee benchmark-results/system/system.txt
```

Review the system file before sharing it. The filtered hardware command omits
the serial number and hardware UUID.

If you calculate memory from `vm_stat`, use the page size printed on its first
line. Do not assume 4 KB; the A18 Pro report in issue #80 used 16 KB pages.

## Run the cases

Run one discarded warmup for each case:

```bash
for case_seed in \
  short-explanation:20260721 \
  medium-review:20260722 \
  long-synthesis:20260723; do
  case_id="${case_seed%%:*}"
  seed="${case_seed##*:}"
  .build/release/TurboFieldfareCLI \
    --model scratch/gemma4.gturbo \
    --messages-file "docs/benchmark-prompts/real-generation-v1/${case_id}.json" \
    --max-new 1024 \
    --max-context 4096 \
    --temperature 0.2 \
    --top-k 64 \
    --top-p 0.95 \
    --seed "$seed" \
    > "benchmark-results/warmup/${case_id}.stdout" \
    2> "benchmark-results/warmup/${case_id}.stderr"
done
```

Then run the three measured cases in fresh processes:

```bash
for case_seed in \
  short-explanation:20260721 \
  medium-review:20260722 \
  long-synthesis:20260723; do
  case_id="${case_seed%%:*}"
  seed="${case_seed##*:}"
  .build/release/TurboFieldfareCLI \
    --model scratch/gemma4.gturbo \
    --messages-file "docs/benchmark-prompts/real-generation-v1/${case_id}.json" \
    --max-new 1024 \
    --max-context 4096 \
    --temperature 0.2 \
    --top-k 64 \
    --top-p 0.95 \
    --seed "$seed" \
    > "benchmark-results/measured/${case_id}.stdout" \
    2> "benchmark-results/measured/${case_id}.stderr"
done

grep -h '^\[stop=' benchmark-results/measured/*.stderr |
  tee benchmark-results/summary.txt
```

Every measured footer must say `stop=endOfTurn`. Read the three output files as
well: reject the result if an answer loops, repeats a block, or ends incomplete.
Report the prompt and generated token counts from each footer because different
output tokens create a different routing workload.

## Report the result

Open an issue titled `Benchmark: <chip>, <memory>, <macOS version>` and include:

- `benchmark-results/system/system.txt`;
- `benchmark-results/summary.txt`;
- the three measured stdout and stderr files;
- energy mode, whether the Mac was connected to power, and any other active
  workload; and
- any output-quality problem or protocol change.

Compare rows only when the case, prompt tokens, generated tokens, settings, and
stop reason match. The [reference M5 range](BENCHMARKS.md#m5-measured-decode)
uses controlled non-repeating continuations for stable token-for-token
measurement, while this public protocol checks autonomous product generation.

## Community results

These measurements were submitted by GitHub users. They are collected here
for reference, not as a controlled hardware comparison.
Prompt length, generated tokens, storage, cache state, and the number of runs
all affect decode speed.

| Source | Mac | Memory | Storage | Prompt / generated tokens | Decode | Measurement |
| --- | --- | ---: | --- | ---: | ---: | --- |
| [bangddong](https://github.com/bangddong) · [issue #80](https://github.com/drumih/turbo-fieldfare/issues/80) | A18 Pro MacBook Neo | 8 GB | Internal SSD | 61 / 516 | 4.134 tok/s | One submitted run |
| [bangddong](https://github.com/bangddong) · [issue #80](https://github.com/drumih/turbo-fieldfare/issues/80) | A18 Pro MacBook Neo | 8 GB | Internal SSD | 430 / 780 | 3.672 tok/s | One submitted run |
| [bangddong](https://github.com/bangddong) · [issue #80](https://github.com/drumih/turbo-fieldfare/issues/80) | A18 Pro MacBook Neo | 8 GB | Internal SSD | 3,015 / 617 | 3.012 tok/s | One submitted run |
| [Maxim Voronin](https://github.com/maxslamdunk) · [issue #116](https://github.com/drumih/turbo-fieldfare/issues/116) | M1 MacBook Air | 8 GB | Internal 256 GB SSD | 61 / 511 | 3.340 tok/s | One submitted run |
| [Maxim Voronin](https://github.com/maxslamdunk) · [issue #116](https://github.com/drumih/turbo-fieldfare/issues/116) | M1 MacBook Air | 8 GB | Internal 256 GB SSD | 430 / 669 | 3.053 tok/s | One submitted run |
| [Maxim Voronin](https://github.com/maxslamdunk) · [issue #116](https://github.com/drumih/turbo-fieldfare/issues/116) | M1 MacBook Air | 8 GB | Internal 256 GB SSD | 3,015 / 604 | 2.642 tok/s | One submitted run |
| [German](https://github.com/iGerman00) · [issue #76](https://github.com/drumih/turbo-fieldfare/issues/76) | M1 MacBook Air | 16 GB | Internal SSD | 61 / 511 | 7.802 tok/s | One submitted run |
| [German](https://github.com/iGerman00) · [issue #76](https://github.com/drumih/turbo-fieldfare/issues/76) | M1 MacBook Air | 16 GB | Internal SSD | 430 / 669 | 7.173 tok/s | One submitted run |
| [German](https://github.com/iGerman00) · [issue #76](https://github.com/drumih/turbo-fieldfare/issues/76) | M1 MacBook Air | 16 GB | Internal SSD | 3,015 / 604 | 6.093 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #94](https://github.com/drumih/turbo-fieldfare/issues/94) | M1 Mac mini | 16 GB | Internal 512 GB SSD | 61 / 511 | 6.754 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #94](https://github.com/drumih/turbo-fieldfare/issues/94) | M1 Mac mini | 16 GB | Internal 512 GB SSD | 430 / 669 | 6.265 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #94](https://github.com/drumih/turbo-fieldfare/issues/94) | M1 Mac mini | 16 GB | Internal 512 GB SSD | 3,015 / 604 | 5.570 tok/s | One submitted run |
| [Felix Li](https://github.com/dt1dr) · [issue #37](https://github.com/drumih/turbo-fieldfare/issues/37) | M1 Pro MacBook Pro | 16 GB | Internal SSD | 61 / 511 | 9.477 tok/s | One submitted run |
| [Felix Li](https://github.com/dt1dr) · [issue #37](https://github.com/drumih/turbo-fieldfare/issues/37) | M1 Pro MacBook Pro | 16 GB | Internal SSD | 430 / 669 | 8.576 tok/s | One submitted run |
| [Felix Li](https://github.com/dt1dr) · [issue #37](https://github.com/drumih/turbo-fieldfare/issues/37) | M1 Pro MacBook Pro | 16 GB | Internal SSD | 3,015 / 604 | 7.805 tok/s | One submitted run |
| [Vaibhav Malik](https://github.com/VaibhavMalik4187) · [issue #64](https://github.com/drumih/turbo-fieldfare/issues/64) | M1 Max MacBook Pro | 64 GB | Internal SSD | 61 / 511 | 22.888 tok/s | One submitted run |
| [Vaibhav Malik](https://github.com/VaibhavMalik4187) · [issue #64](https://github.com/drumih/turbo-fieldfare/issues/64) | M1 Max MacBook Pro | 64 GB | Internal SSD | 430 / 669 | 22.043 tok/s | One submitted run |
| [Vaibhav Malik](https://github.com/VaibhavMalik4187) · [issue #64](https://github.com/drumih/turbo-fieldfare/issues/64) | M1 Max MacBook Pro | 64 GB | Internal SSD | 3,015 / 604 | 20.493 tok/s | One submitted run |
| [stefinuxha](https://github.com/stefinuxha) · [issue #131](https://github.com/drumih/turbo-fieldfare/issues/131) | M2 Mac mini | 16 GB | Not reported | 61 / 511 | 9.674 tok/s | One submitted run |
| [stefinuxha](https://github.com/stefinuxha) · [issue #131](https://github.com/drumih/turbo-fieldfare/issues/131) | M2 Mac mini | 16 GB | Not reported | 430 / 669 | 8.889 tok/s | One submitted run |
| [stefinuxha](https://github.com/stefinuxha) · [issue #131](https://github.com/drumih/turbo-fieldfare/issues/131) | M2 Mac mini | 16 GB | Not reported | 3,015 / 604 | 6.578 tok/s | One submitted run |
| [Matt 'Smartis' W](https://github.com/0xSmartis) · [issue #69](https://github.com/drumih/turbo-fieldfare/issues/69) | M2 Pro Mac mini | 16 GB | Internal SSD | 61 / 511 | 15.713 tok/s | One submitted run |
| [Matt 'Smartis' W](https://github.com/0xSmartis) · [issue #69](https://github.com/drumih/turbo-fieldfare/issues/69) | M2 Pro Mac mini | 16 GB | Internal SSD | 430 / 669 | 17.826 tok/s | One submitted run |
| [Matt 'Smartis' W](https://github.com/0xSmartis) · [issue #69](https://github.com/drumih/turbo-fieldfare/issues/69) | M2 Pro Mac mini | 16 GB | Internal SSD | 3,015 / 604 | 15.755 tok/s | One submitted run |
| [Eli Smith](https://github.com/eliliam) · [issue #139](https://github.com/drumih/turbo-fieldfare/issues/139) | M2 Pro MacBook Pro | 16 GB | Internal SSD | 61 / 511 | 11.749 tok/s | One submitted run |
| [Eli Smith](https://github.com/eliliam) · [issue #139](https://github.com/drumih/turbo-fieldfare/issues/139) | M2 Pro MacBook Pro | 16 GB | Internal SSD | 430 / 669 | 10.428 tok/s | One submitted run |
| [Eli Smith](https://github.com/eliliam) · [issue #139](https://github.com/drumih/turbo-fieldfare/issues/139) | M2 Pro MacBook Pro | 16 GB | Internal SSD | 3,015 / 604 | 9.384 tok/s | One submitted run |
| [elroyonline](https://github.com/elroyonline) · [issue #59](https://github.com/drumih/turbo-fieldfare/issues/59) | M2 Max Mac Studio | 32 GB | External PCIe NVMe | 61 / 511 | 25.310 tok/s | One submitted run |
| [elroyonline](https://github.com/elroyonline) · [issue #59](https://github.com/drumih/turbo-fieldfare/issues/59) | M2 Max Mac Studio | 32 GB | External PCIe NVMe | 430 / 669 | 25.172 tok/s | One submitted run |
| [elroyonline](https://github.com/elroyonline) · [issue #59](https://github.com/drumih/turbo-fieldfare/issues/59) | M2 Max Mac Studio | 32 GB | External PCIe NVMe | 3,015 / 604 | 23.730 tok/s | One submitted run |
| [Benjamin Schilling](https://github.com/benjamin-schilling) · [issue #27](https://github.com/drumih/turbo-fieldfare/issues/27) | M3 Pro MacBook Pro | 18 GB | Internal SSD | 61 / 516 | 19.09 tok/s | One submitted run |
| [Benjamin Schilling](https://github.com/benjamin-schilling) · [issue #27](https://github.com/drumih/turbo-fieldfare/issues/27) | M3 Pro MacBook Pro | 18 GB | Internal SSD | 430 / 780 | 16.50 tok/s | One submitted run |
| [Benjamin Schilling](https://github.com/benjamin-schilling) · [issue #27](https://github.com/drumih/turbo-fieldfare/issues/27) | M3 Pro MacBook Pro | 18 GB | Internal SSD | 3,015 / 617 | 13.88 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #95](https://github.com/drumih/turbo-fieldfare/issues/95) | M4 Mac mini | 16 GB | Internal 256 GB SSD | 61 / 516 | 10.847 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #95](https://github.com/drumih/turbo-fieldfare/issues/95) | M4 Mac mini | 16 GB | Internal 256 GB SSD | 430 / 780 | 9.878 tok/s | One submitted run |
| [gandalfk7](https://github.com/gandalfk7) · [issue #95](https://github.com/drumih/turbo-fieldfare/issues/95) | M4 Mac mini | 16 GB | Internal 256 GB SSD | 3,015 / 617 | 8.658 tok/s | One submitted run |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | Internal 256 GB SSD | 61 / 516 | 11.38 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | Internal 256 GB SSD | 430 / 780 | 8.43 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | Internal 256 GB SSD | 3,015 / 617 | 8.58 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | External Samsung 990 Pro | 61 / 516 | 11.53 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | External Samsung 990 Pro | 430 / 780 | 10.79 tok/s | Median of five runs |
| [22f](https://github.com/22f) · [issue #23](https://github.com/drumih/turbo-fieldfare/issues/23) | M4 Mac mini | 16 GB | External Samsung 990 Pro | 3,015 / 617 | 11.21 tok/s | Median of five runs |
| [dragonwang1019](https://github.com/dragonwang1019) · [issue #101](https://github.com/drumih/turbo-fieldfare/issues/101) | M4 Pro Mac mini | 24 GB | External NVMe SSD | 61 / 516 | 22.256 tok/s | One submitted run |
| [dragonwang1019](https://github.com/dragonwang1019) · [issue #101](https://github.com/drumih/turbo-fieldfare/issues/101) | M4 Pro Mac mini | 24 GB | External NVMe SSD | 430 / 780 | 28.964 tok/s | One submitted run |
| [dragonwang1019](https://github.com/dragonwang1019) · [issue #101](https://github.com/drumih/turbo-fieldfare/issues/101) | M4 Pro Mac mini | 24 GB | External NVMe SSD | 3,015 / 617 | 24.209 tok/s | One submitted run |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | 61 / 516 | 34.060 tok/s | Automatic mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | 430 / 780 | 30.954 tok/s | Automatic mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | 3,015 / 617 | 25.292 tok/s | Automatic mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | 61 / 516 | 33.379 tok/s | High Power mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | 430 / 780 | 30.614 tok/s | High Power mode |
| [Marcelo Barros](https://github.com/marcelopbarros) · [issue #67](https://github.com/drumih/turbo-fieldfare/issues/67) | M4 Pro MacBook Pro | 24 GB | Internal SSD | 3,015 / 617 | 25.230 tok/s | High Power mode |
| [nxexcelsior](https://github.com/nxexcelsior) · [issue #111](https://github.com/drumih/turbo-fieldfare/issues/111) | M5 MacBook Air | 24 GB | Internal SSD | 61 / 497 | 11.982 tok/s | One submitted run |
| [nxexcelsior](https://github.com/nxexcelsior) · [issue #111](https://github.com/drumih/turbo-fieldfare/issues/111) | M5 MacBook Air | 24 GB | Internal SSD | 430 / 721 | 12.429 tok/s | One submitted run |
| [nxexcelsior](https://github.com/nxexcelsior) · [issue #111](https://github.com/drumih/turbo-fieldfare/issues/111) | M5 MacBook Air | 24 GB | Internal SSD | 3,015 / 610 | 9.783 tok/s | One submitted run |
| [Matt Skipton](https://github.com/mattskipton) · [issue #97](https://github.com/drumih/turbo-fieldfare/issues/97) | M5 Pro MacBook Pro | 64 GB | Not reported | 61 / 497 | 37.426 tok/s | One submitted run |
| [Matt Skipton](https://github.com/mattskipton) · [issue #97](https://github.com/drumih/turbo-fieldfare/issues/97) | M5 Pro MacBook Pro | 64 GB | Not reported | 430 / 721 | 34.194 tok/s | One submitted run |
| [Matt Skipton](https://github.com/mattskipton) · [issue #97](https://github.com/drumih/turbo-fieldfare/issues/97) | M5 Pro MacBook Pro | 64 GB | Not reported | 3,015 / 610 | 31.485 tok/s | One submitted run |
| [Ulises](https://github.com/ulises-c) · [issue #149](https://github.com/drumih/turbo-fieldfare/issues/149) | M5 Max MacBook Pro | 36 GB | Internal SSD | 61 / 497 | 43.440 tok/s | One submitted run |
| [Ulises](https://github.com/ulises-c) · [issue #149](https://github.com/drumih/turbo-fieldfare/issues/149) | M5 Max MacBook Pro | 36 GB | Internal SSD | 430 / 721 | 40.965 tok/s | One submitted run |
| [Ulises](https://github.com/ulises-c) · [issue #149](https://github.com/drumih/turbo-fieldfare/issues/149) | M5 Max MacBook Pro | 36 GB | Internal SSD | 3,015 / 610 | 37.370 tok/s | One submitted run |

These submissions use the public community prompts and generate until the end
of the model turn. Compare rows only when the prompt and generated-token counts
match.
