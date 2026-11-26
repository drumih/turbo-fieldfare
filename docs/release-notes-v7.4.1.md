# TinyTitan 7.4.1 — Fix prompt-cache disk eviction wedge on missing entries (#35)

This patch release fixes a prompt-cache disk eviction bug reported in
[Pummelchen/TinyTitan#35](https://github.com/Pummelchen/TinyTitan/issues/35).

### Bug fixes

- **Prompt-cache disk eviction no longer wedges on `ENOENT` (#35):**
  `ServerPromptStateStore.releaseDiskRecord` and `PromptCheckpointStore`'s
  eviction, reclaim, and deletion paths previously treated every
  `FileManager.removeItem` error as a volume refusal, leaving an already-absent
  directory (`NSFileNoSuchFileError` / `ENOENT`, from external cleanup or a
  concurrent `Task.detached` unlink) pinned at the head of `diskLRU`. Every
  subsequent turn then logged `TinyTitan prompt_cache disk_write_failed` and
  stopped evicting older entries, allowing `disk_bytes` to grow past the
  `--prompt-cache-disk-mib` cap. Both stores now recognize `ENOENT` as already
  removed and release the ledger entry while keeping refusal accounting intact
  for real permission or I/O errors.
- Added regression tests (`ServerPromptStateStoreTests.missingDiskEntryDoesNotWedgeEviction`
  and `PromptCheckpointStoreTests.missingDiskCheckpointDoesNotWedgeEviction`).

### Checks not run

The installed models on this machine are Ornith 1.5 35B-A3B 4-bit, Inkling 2
Mini (`inkling2-gguf-q4` and `inkling2-nvfp4`), and the six dense Qwen 3.5
installs (15 golden targets checked: 9 GPU, 6 CPU). No model may be fetched to
fill a row. The following baselines are **not checked** because nothing is
installed under `models/`: `ornith-8`, `qwen36-4`, `qwen36-8`, `qwen38-4`,
`qwen38-8`, `k2-4`, `k2-8`, `agentworld-4`, `agentworld-8`, `katcoder-4`,
`katcoder-8`, `ornith122-4` and `ornith122-8`.

### Checksum

`tinytitan-7.4.1-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-7.4.1-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-7.4.1-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-7.4.1-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-7.4.1-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-7.4.1-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
