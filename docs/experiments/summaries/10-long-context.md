# Speeding up attention for long prompts

The previous full-attention kernel made GPU threads wait for each other
at every cached token. With a long prompt, those synchronization steps
added up, slowing down generation.

The new grouped kernel gives each attention head its own small group of
GPU threads. Heads that share cached data run together, but each can
calculate its scores without waiting for the others. Removing those
repeated waits makes attention faster while still using the full context.

## How I tested it

On an M5 Pro with 24 GB of RAM, I ran both kernels with the same
110,000-token prompt made from repeated prose, then generated 256 tokens
using greedy decoding. Only the attention kernel changed between runs.
Both used 32 expert-cache slots, above the default of 16.

Generation speed went from **8.14 to 18.02 tokens/s**, a 2.21x improvement.
The 256-token answer took 14.21 seconds instead of 31.46, and every output
token ID matched. These times exclude reading the prompt, which still
took about 20 minutes.

## Measurement limits

This was one comparison, with the previous kernel running first and OS
cache state uncontrolled. It used the private benchmark tool and has not
been repeated with the public package or the default cache size.
Matching output on this prompt does not establish quality on other prompts
or at 256K. The [memory guide](../../RUNTIME_CONTROLS.md#context-length-and-memory)
covers context settings and the remaining memory checks.

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Attention and KV cache](05-attention-and-kv-cache.md)
