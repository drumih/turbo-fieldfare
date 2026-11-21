#!/usr/bin/env bash
# Run the vLLM reference for Inkling 2 Mini on an NVIDIA Blackwell/Hopper host.
#
# The model card lists vLLM as a supported host and the vLLM recipe pins the
# image below. This is the reference the TinyTitan port must match; it loads the
# checkpoint's original NVFP4/MXFP8 weights, unlike a GGUF conversion.
#
# This script refuses on a host without an NVIDIA GPU. The Mac development host
# has no such device, so it is a remote-host installer, not a local run.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/inkling-2-mini}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-262144}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:inkling-2-mini}"
NAME="${CONTAINER_NAME:-tinytitan-inkling-reference}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "no nvidia-smi: the vLLM reference needs an NVIDIA Blackwell/Hopper GPU" >&2
  echo "model card targets: GB200, B200, B300, GB300, H100, H200, DGX Spark" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "no docker: install Docker + NVIDIA Container Toolkit on the GPU host" >&2
  exit 2
fi
if [ ! -f "$MODEL_DIR/config.json" ] || [ ! -f "$MODEL_DIR/hf_quant_config.json" ]; then
  echo "missing checkpoint metadata under $MODEL_DIR" >&2
  exit 2
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image $IMAGE is not local; pull it with: docker pull $IMAGE" >&2
  exit 2
fi

echo "starting vLLM reference: $IMAGE on port $PORT, max-model-len $MAX_LEN"
exec docker run --rm --name "$NAME" --gpus all --ipc=host \
  -p "127.0.0.1:$PORT:8000" \
  -v "$(cd "$(dirname "$MODEL_DIR")" && pwd)/$(basename "$MODEL_DIR"):/model:ro" \
  "$IMAGE" \
  --model /model \
  --served-model-name thinkingmachines/Inkling-2-Mini \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --reasoning-parser inkling
