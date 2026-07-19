# Contribute a benchmark result

Results from other Apple Silicon Macs help show where TurboFieldfare runs well
and where it needs more work. This guide uses the
[canonical 1K prompt](benchmark-prompts/prefill-1024.txt) from the project's
prefill and decode benchmarks. It contains 1,017 model tokens including BOS.
Each run generates 256 greedy tokens so reports from different machines remain
comparable.

This is an explicit reproducibility workload, not the interactive product
default. The app and CLI normally allow up to 1,024 new tokens and sample with
temperature `0.1`, Top-K `64`, and Top-P `0.95`; the commands below override
those values with a fixed 256-token greedy decode.

The public package uses `TurboFieldfareCLI` for this measurement. Its timing
footer runs the production defaults and reports decode-only tokens per second.
It excludes model installation, model loading, and prompt prefill.

## Before you run

Install the model with the [README instructions](../README.md#command-line-interface),
then prepare the Mac:

1. Connect a laptop to power and turn off Low Power Mode. Use the normal
   Automatic energy mode unless you clearly report another mode.
2. Build the release executable before measuring:

   ```bash
   swift build -c release --product TurboFieldfareCLI
   ```

3. Quit every app you can, leaving only Terminal. In particular, close the
   TurboFieldfare Mac app, Xcode, browsers, video calls, games, screen recorders,
   local model runners, downloads, backups, and file indexing that you started.
4. Let background CPU activity and the Mac's temperature settle.
5. Check for another model process:

   ```bash
   pgrep -fl 'TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
   ```

   Continue only when this prints nothing. Quit a listed app normally; do not
   start two model runs at once.

Do not change the prompt, token count, temperature, or public runtime defaults
for the standard result.

## Record the machine

From the repository root, collect a privacy-safe system summary:

```bash
mkdir -p benchmark-results
{
  git status --short
  git rev-parse HEAD
  sw_vers
  swift --version
  system_profiler SPHardwareDataType |
    awk -F': ' '/Model Name|Model Identifier|Chip|Total Number of Cores|Memory/ { print $1 ": " $2 }'
  shasum -a 256 scratch/gemma4.gturbo/manifest.json
  shasum -a 256 docs/benchmark-prompts/prefill-1024.txt
} 2>&1 | tee benchmark-results/system.txt
```

An empty first line from `git status --short` means the checkout is clean. If it
prints changes, either use a clean checkout or describe those changes in the
report. The filtered hardware command omits the serial number and hardware UUID,
but review `benchmark-results/system.txt` before sharing it.

## Run the fixed workload

First run one warmup. It also confirms that the model loads and the prompt
completes:

```bash
.build/release/TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "$(cat docs/benchmark-prompts/prefill-1024.txt)" \
  --max-new 256 \
  --max-context 4096 \
  --temperature 0 \
  > benchmark-results/warmup.txt 2>&1

tail -n 2 benchmark-results/warmup.txt
```

Then run three measured rows, sequentially:

```bash
for run in 1 2 3; do
  output="benchmark-results/run-${run}.txt"
  .build/release/TurboFieldfareCLI \
    --model scratch/gemma4.gturbo \
    --prompt "$(cat docs/benchmark-prompts/prefill-1024.txt)" \
    --max-new 256 \
    --max-context 4096 \
    --temperature 0 \
    > "$output" 2>&1 || break
  tail -n 2 "$output"
done
```

Each successful file ends with a timing footer. Confirm that it contains this
workload shape:

```text
prefill=1017tok new=256tok
```

Collect the three timing lines for easy copy and paste:

```bash
grep -h '^\[stop=' benchmark-results/run-*.txt |
  tee benchmark-results/summary.txt
```

This protocol uses one warmup followed by three fresh CLI processes. The file
cache is warm but uncontrolled. Do not purge caches or average these rows with
runs made under different conditions.

## Report the result

Open an issue in this repository with the title
`Benchmark: <chip>, <memory>, <macOS version>`. Paste this template:

````markdown
## TurboFieldfare benchmark

- Energy mode: Automatic / High Power / other
- Connected to power: yes / no
- Other apps: quit / list anything left open
- Thermal state before run: settled / uncertain
- Protocol changes: none / describe every change

### System

```text
Paste benchmark-results/system.txt here.
```

### Results

```text
Paste benchmark-results/summary.txt here.
```

### Notes

Add unexpected output, errors, throttling, or other observations.
````

Attach the three `run-*.txt` files when possible. They contain the generated
completion and timing footer, which helps us spot incomplete or non-equivalent
runs.

If you want to test a different app setting or code change, submit the standard
result first. Then change one thing, repeat the same warmup and three-run policy,
and label the follow-up as experimental. Never combine rows from different
settings into one standard result.

See [Benchmarks](BENCHMARKS.md) for the current reference measurements and how
to interpret comparisons.
