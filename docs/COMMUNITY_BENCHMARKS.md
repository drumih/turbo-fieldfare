# Contribute a benchmark result

TurboFieldfare's community benchmark uses three chat-framed generation cases:
a short explanation, a medium design review, and a long document synthesis.
They exercise different prompt lengths and require coherent text that reaches
the end of the model turn. A repeating calibration prompt is not a valid speed
result because repeated expert choices can make decode artificially fast.

The frozen prompts are in
[`benchmark-prompts/real-generation-v1/`](benchmark-prompts/real-generation-v1/).
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
pgrep -fl 'TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
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
