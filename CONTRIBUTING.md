# Contributing

TurboFieldfare welcomes focused fixes, documentation improvements, and
benchmark reports from Apple Silicon Macs.

## Before opening a change

- Keep the package compatible with macOS 26, Swift 6.2, and Metal 4.
- Preserve the bounded-memory model path. Never load a complete checkpoint,
  shard, or large model tensor into Swift heap memory.
- Keep public runtime controls limited to those documented in
  [Runtime controls](docs/RUNTIME_CONTROLS.md).
- Add or update a focused test for behavior changes.

Run the release build, serial tests, and Markdown link check:

```bash
swift build -c release
Scripts/test.sh
ruby Scripts/check_markdown_links.rb
```

These checks do not download or load the model. For a real-model change, also
report the prompt, generated token count, output, timing footer, Mac model,
memory, macOS version, Swift version, and any protocol change.

## Benchmark reports

Follow the [community benchmark protocol](docs/COMMUNITY_BENCHMARKS.md). Review
all captured files before sharing them, and remove personal paths or unrelated
process details.

## Pull requests

Keep each pull request narrow. Explain the behavior change, tests run, and any
remaining limitation. By contributing, you agree that your work is licensed
under the repository's [Apache License 2.0](LICENSE).
