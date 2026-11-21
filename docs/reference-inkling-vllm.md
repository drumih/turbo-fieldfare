# Inkling 2 Mini reference: vLLM on NVIDIA

**Status: installer prepared; not run on this host.** The development Mac has no
NVIDIA GPU, so `tools/reference_inkling_vllm.sh` refuses here. It is the
reference for the TinyTitan port and must run on a supported NVIDIA host.

## Why this reference

Thinking Machines' model card lists vLLM among the supported local runtimes, and
the vLLM recipe pins the image and the model-specific flags. vLLM reads the
checkpoint's **original NVFP4 (layers 0–19) and MXFP8 (layers 20–23)** weights;
it is therefore the correct quality oracle. A GGUF conversion (llama.cpp,
Ollama, Unsloth) changes the quantized weights and cannot be the 100% quality
reference.

## Host

The vLLM recipe verifies: **GB200, B300, GB300, H100, H200 and DGX Spark**.
Its default configuration is 4× GB200, but one GPU at TP1 serves the model; on
H100 and DGX Spark the recipe caps context at 262,144 tokens. 1M context is a
GB200-class target.

Requirements: NVIDIA driver + Container Toolkit, Docker, and the ~20 GB
checkpoint. The reference image is:

```
vllm/vllm-openai:inkling-2-mini
```

Use that image until Inkling 2 Mini support reaches a vLLM release (recipe
pins vLLM 0.35.0+nightly).

## Run

```bash
# on the GPU host, with the checkpoint at models/inkling-2-mini
MODEL_DIR=models/inkling-2-mini PORT=8000 MAX_LEN=262144 \
  tools/reference_inkling_vllm.sh
```

The script:
- refuses without `nvidia-smi`, Docker, the image, or the checkpoint metadata;
- mounts the checkpoint read-only;
- serves on loopback only;
- keeps the model's own `inkling` reasoning parser.

For the TinyTitan Phase 1 text comparison, use the raw `/v1/completions`
endpoint with `temperature: 0` and a fixed prompt set, so the chat template and
reasoning parser are not part of the comparison. Record the exact image digest
and vLLM version with the outputs.

## What it does not do

- It does not run on the Mac.
- It does not install the model; the checkpoint download is a separate operator
  action.
- It does not enable DSpark speculative decoding for the base-model parity
  capture; that is a serving optimization, not part of the target-only text
  reference.
