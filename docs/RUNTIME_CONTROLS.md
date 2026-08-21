# Runtime controls

The Mac app, CLI, and local server expose generation and runtime controls. The
app keeps them in its fixed right settings pane. FP16 is the fixed KV format.
Generation settings apply to the next request; app load-time settings require a
reload.

## Generation controls

The Mac app and CLI expose these generation controls:

| Control | Mac values | CLI flag | Default | Effect |
| --- | --- | --- | --- | --- |
| Maximum response | Automatic | `--max-new` | App: remaining context; CLI: 1,024 tokens | The app can use the context space left after formatting the prompt. The CLI uses its explicit or default `--max-new` limit. |
| Maximum context | 4K, 8K, 16K, 32K, 64K | `--max-context` | CLI and app: 8K; server: 16K | Sets prompt plus response capacity, and 8K is what leaves room for an image and its prompt. The app shows the FP16 KV-memory delta. The server defaults higher still because agent clients routinely send prompts near 8K on their own. |
| Temperature | 0...2 in 0.05 steps | `--temperature` | 0.2 | `0` is greedy; positive values sample. |
| Top-K | Off or 1...256 | `--top-k` | 64 | Keeps at most K candidates. CLI `0` turns it off. |
| Top-P | Off or 0.01...1 | `--top-p` | 0.95 | Applies nucleus truncation before Top-K and is effective only while Top-K is enabled. |

With positive temperature, a CLI Top-P below `1` requires Top-K between `1`
and `256`. To disable both truncation controls, pass `--top-k 0 --top-p 1`.
Generation controls apply to the next request and do not require a model
reload. They are interactive product settings, not the fixed community
benchmark protocol.

## Runtime settings

The CLI and the [local server](OPENAI_SERVER.md) accept these flags with the
same names and values. Their defaults agree except for `--max-context`, which
the server defaults to 16,384 rather than 8,192. The server resolves them before it loads the
model, so an unsupported combination fails immediately with the usage text.

| Control | Mac values | CLI and server flag | Production default | Effect |
| --- | --- | --- | --- | --- |
| Expert-cache slots | 8, 16, 24, 32 | `--expert-cache-slots` | 16 | More slots can retain more routed experts and reduce later reads, but values above 16 use more RAM. Chunked prefill requires at least 16 slots. |
| Expert-cache policy | LFU | `--expert-cache-policy lfu\|lru` | LFU | Chooses which expert is evicted when the cache is full. |
| Prompt prefill | On, off | `--prefill on\|off` | On | On processes known prompt tokens through the chunked prefill path. Off disables that path. |
| Prefill chunk size | 128 | `--prefill-chunk-tokens 32\|64\|128\|256\|auto` | 128 | Sets the number of prompt tokens processed by each chunked-prefill step, and has no effect while prefill is off. Chunks loop outside layers, so each one re-reads that layer's routed experts: fewer, larger chunks read less. `auto` picks the smallest size that covers the prompt. On a 7,019-token prompt, 256 measured 16% faster than 128 with byte-identical output. |
| RDADVISE | Off, Default, Bounded, Adaptive | `--rdadvise off\|default\|bounded\|adaptive` | Off | Applies experimental read advice. Its effect depends on the workload; it may help a short decode and slow a long one. |

In the app, changing context length, expert-cache slots, or RDADVISE requires a
reload. Some sampling changes also require a reload because greedy and sampled
generation use different output-head paths. Prompt-prefill settings apply to
each request and do not require a reload. Each CLI invocation loads a new model
process, so its selected runtime settings apply immediately. The server fixes
its runtime settings at startup, so changing one means restarting the process.

## Image controls

Image input needs the companion pack installed beside the text model. Without
it, these controls have nothing to select.

| Control | Values | Default | Effect |
| --- | --- | --- | --- |
| Vision pack | directory | `<model>.vision.gturbo` beside the model | Where the companion pack is read from. If you pass a path that does not exist, the load fails. It does not fall back to serving text with the images missing. |
| Vision residency | `on-demand`, `keep-ready` | `on-demand` | `on-demand` releases the routed-expert streamers and their slot scratch before encoding, then recreates them for language prefill. `keep-ready` maps the tower during the load and leaves it mapped. Changing this requires a reload. |

The CLI takes `--vision-pack` and `--vision-residency`, plus `--image <path>`,
which is repeatable, and `--chat-prompt`. The server takes the first two.

A request whose images cannot fit the selected context is refused before any
pixel is decoded. The rejection reports the token cost, not the number of
images, because an image's cost scales with its dimensions.

## Run an experiment

1. Start from the 8K default context, 16 expert-cache slots, prefill on, and
   RDADVISE off.
2. Keep the prompt and generation controls fixed.
3. Record a baseline after a warmup.
4. Change one runtime control and reload the app model, or start a new CLI run.
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

During chunked prefill, the phase label reports exact progress, for example
`Prefill (128/514)`. Errors and unsupported configurations appear only when
they occur. RDADVISE remains experimental and is off by default. A measured
result is a data point, not a performance ceiling.
