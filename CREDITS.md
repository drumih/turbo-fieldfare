# Credits

TinyTitan is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), whose
bounded-memory runtime, installer, CLI and local server this project builds on.
The Qwen 3.6 integration was created by
[NeelM0906](https://github.com/NeelM0906) in
[upstream PR #29](https://github.com/drumih/turbo-fieldfare/pull/29). Concise
mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll).

**Project Maya:** the 6.0 checkpoint, calibration and machine-size cache work was
carried under that name. The closed-out status record is
[docs/maya-task-status.md](docs/maya-task-status.md); the closeout is
[docs/maya-closeout-2026-10-22.md](docs/maya-closeout-2026-10-22.md).

**Kapitsa:** the cross-session shared system-prefix cache and the M4+ tensor-ops
prefill direction were informed by [nace-ai/kapitsa](https://github.com/nace-ai/kapitsa).
No code is copied from it.

**Related research:** [TinyTitan Datacenter](https://github.com/Pummelchen/TinyTitan_Datacenter)
runs large MoE models across a cluster of Mac minis and Studios, keeping them on
SSD/NVMe for near-linear scale of decode throughput.
