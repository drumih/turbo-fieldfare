<p align="center">
  <img width="1254" height="1254" alt="TinyTitan" src="https://github.com/user-attachments/assets/2a990de5-af24-48cd-b673-e875755741cc" />
</p>

# TinyTitan

[![Stars](.github/badges/stars.svg)](https://github.com/Pummelchen/TinyTitan/stargazers)
[![Views (14d)](.github/badges/views.svg)](https://github.com/Pummelchen/TinyTitan)
[![Last Commit](.github/badges/last-commit.svg)](https://github.com/Pummelchen/TinyTitan/commits/main)

# TinyTitan — Project Update & End of Development

For several months, **TinyTitan** has consistently outperformed
established LLM inference engines such as `llama.cpp` by **20–25%**
on Apple Silicon.

Due to the significant resources required to maintain this performance
advantage, TinyTitan will narrow its focus to three LLMs for its
remaining development:

- **Inkling 2 Mini** — Original and Unsloth weights
- **Ornith 1.5 35B A3B**
- **Qwen 4 Flash**

## Future Development

Future development resources will be redirected towards:

- **TinyTitan Datacenter**
- **XAIOS** — Our flagship project

## End of Development

> [!IMPORTANT]
> **TinyTitan v7.5** will mark the final release and the official
> end of the TinyTitan project.

## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB, after a warm-up pass.

| Model | Quantization | GPU | CPU | Terminal-Bench 2.x | Terminal-Bench 4.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.5 tok/s** | NA | 86% (2.1) | 25% |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.1 tok/s** | NA | 86% (2.1) | 25% |
| Ornith 1.5 122B | 4-bit | **2.9 tok/s** | NA | 81% (2.1) | NA |
| Ornith 1.5 122B | 8-bit | **1.2 tok/s** | NA | 81% (2.1) | NA |
| Inkling 2 Mini 26B-A5B | UD-Q4_K_XL (Unsloth) | **17.9 tok/s** | NA | 72% (2.1) | 5.6% |
| Inkling 2 Mini 26B-A5B | Original (NVFP4/MXFP8) | **8.1 tok/s** | NA | 72% (2.1) | 5.6% |
| Laguna-S-2.1 118B-A8B *(decode probe)* | 4-bit | **1.5 tok/s** | NA | 70% (2.1) | NA |
| Laguna-S-2.1 118B-A8B *(decode probe)* | 8-bit | **0.7 tok/s** | NA | 70% (2.1) | NA |
| Ornith 1.5 35B-A3B | 4-bit | **21.7 tok/s** | NA | 68% (2.1) | NA |
| Ornith 1.5 35B-A3B | 8-bit | **11.9 tok/s** | NA | 68% (2.1) | NA |
| K2-Horizon-MoVA-36B-A4B *(probe backend)* | 4-bit | **12.4 tok/s** | NA | 59% (2.1) | NA |
| K2-Horizon-MoVA-36B-A4B *(probe backend)* | 8-bit | **2.0 tok/s** | NA | 59% (2.1) | NA |
| Qwen 3.6 35B-A3B | 4-bit | **21.4 tok/s** | NA | 53% (2.1) | NA |
| Qwen 3.6 35B-A3B | 8-bit | **12.4 tok/s** | NA | 53% (2.1) | NA |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B *(CLI only)* | 4-bit | **6.7 tok/s** | **3.6 tok/s** | 51% (2.1) | NA |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B *(CLI only)* | 8-bit | **0.6 tok/s** | **0.9 tok/s** | 51% (2.1) | NA |
| Qwen-AgentWorld 35B-A3B | 4-bit | **21.7 tok/s** | NA | 40% (2.0) | NA |
| Qwen-AgentWorld 35B-A3B | 8-bit | **12.3 tok/s** | NA | 40% (2.0) | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 4-bit | **17.9 tok/s** | NA | 33% (2.1) | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 8-bit | **6.9 tok/s** | NA | 33% (2.1) | NA |
| Qwen 3.5 4B (dense) | 4-bit | **22.8 tok/s** | **7.5 tok/s** | 26% (2.1) | NA |
| Qwen 3.5 4B (dense) | 8-bit | **14.1 tok/s** | **7.9 tok/s** | 26% (2.1) | NA |
| Qwen 3.5 9B (dense) | 4-bit | **12.9 tok/s** | **4.4 tok/s** | 9% (2.0) | NA |
| Qwen 3.5 9B (dense) | 8-bit | **7.8 tok/s** | **4.4 tok/s** | 9% (2.0) | NA |
| Qwen 3.5 2B (dense) | 4-bit | **44.4 tok/s** | **17.4 tok/s** | 3% (2.1) | NA |
| Qwen 3.5 2B (dense) | 8-bit | **29.6 tok/s** | **18.1 tok/s** | 3% (2.1) | NA |


## Documentation & Credits

Everything is in the [Wiki](https://github.com/Pummelchen/TinyTitan/wiki).
TinyTitan is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare); the full
acknowledgements are in [CREDITS.md](CREDITS.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright (c) 2026 André Borchert.
