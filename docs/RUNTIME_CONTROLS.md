# Runtime controls

The Mac app exposes a small set of runtime controls in its fixed right settings
pane. The defaults are the supported production configuration. Change one
control at a time, reload the model, and restore the defaults before normal
use.

## Generation controls

The Mac app and CLI share these interactive defaults:

| Control | Mac values | CLI flag | Default | Effect |
| --- | --- | --- | --- | --- |
| Maximum response | 1...4,096 tokens | `--max-new` | 1,024 | Caps newly generated tokens. It does not include prompt tokens. |
| Maximum context | 256...8,192 tokens | `--max-context` | 4,096 | Sets the prompt plus generation KV-cache budget. |
| Temperature | 0...2 in 0.05 steps | `--temperature` | 1.0 | `0` selects deterministic greedy decoding; positive values sample. |
| Top-K | Off or 1...256 | `--top-k` | 64 | Keeps at most K candidates. The CLI value `0` turns it off. |
| Top-P | Off or 0.01...1 | `--top-p` | 0.95 | Keeps candidates within the selected probability mass before Top-K. It is effective only while Top-K is enabled. |
| Repetition penalty | Off or 1...1.8 | `--repetition-penalty` | Off (`1.0`) | Values above `1.0` penalize tokens already present in the sequence. |

With positive temperature, a CLI Top-P below `1` requires Top-K between `1`
and `256`. To disable both truncation controls, pass `--top-k 0 --top-p 1`.
Generation controls apply to the next request and do not require a model
reload. They are interactive product settings, not the fixed community
benchmark protocol.

## Runtime settings

| Control | Values | Production default | Effect |
| --- | --- | --- | --- |
| Expert-cache slots | 8, 16, 24, 32 | 16 | More slots can retain more routed experts and reduce later reads, but values above 16 use more RAM. |
| Expert-cache policy | LFU, LRU | LFU | Selects which resident expert is evicted when a new expert needs a slot. |
| Prompt prefill | On, off | On | On processes known prompt tokens through the chunked prefill path. Off disables that path. |
| Prefill chunk | 32, 64, 128 tokens | 128 | Sets the number of prompt tokens processed per prefill chunk. It has no effect when prefill is off. |
| KV cache | FP16, TurboQuant K4/V4 | FP16 | K4/V4 reduces KV storage, but it is experimental and failed the full quality gate. |
| RDADVISE | Off, Default, Bounded, Adaptive | Off | Applies experimental read advice. Its effect depends on the workload; it may help a short decode and slow a long one. |
| Model verification | Full SHA-256, Trust verified install | Full SHA-256 | Full SHA-256 hashes the model files. Trust verified install checks the bounded install receipt and file sizes instead of hashing every model byte again. |

Changing a runtime control marks the loaded configuration as stale. Reload the
model before generating with the new setting.

## Run an experiment

1. Start from 16 LFU slots, prefill on with a 128-token chunk, FP16 KV,
   RDADVISE off, and Full SHA-256 verification.
2. Keep the prompt and generation controls fixed.
3. Record a baseline after a warmup.
4. Change one runtime control and reload the model.
5. Compare prompt prefill, request TTFT, decode rate, peak memory, and I/O per
   token over repeated runs.
6. Restore the production defaults when the experiment ends.

Use the [community benchmark protocol](COMMUNITY_BENCHMARKS.md) for a standard
production result. A run with changed runtime controls is experimental and must
name the changed setting.

## Read the results

- **Decode rate** measures generated tokens per second after prompt prefill.
- **Request TTFT** includes prompt prefill and the wait for the first generated
  token.
- **Peak memory** in Last run is the highest decode-service memory observed
  during the request. The HUD shows the service's current memory instead of the
  much smaller foreground UI process.
- **I/O / token** reports routed-expert read time per generated token.
- **Advanced** shows decode duration and per-token cb1, cb2, and output-head
  time. When RDADVISE runs, it also shows time, calls, data, and skipped advice.

During chunked prefill, the phase label reports the latest exact progress, for
example `Prefill (128/514)`. The app reports prefill mismatches, unsupported
paths, and RDADVISE failures only when they occur. Treat K4/V4 and RDADVISE as measured
experiments, not replacements for the production defaults. A result is a data
point, not a performance ceiling.
