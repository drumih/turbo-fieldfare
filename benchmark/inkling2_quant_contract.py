#!/usr/bin/env python3
"""Read and validate the Inkling 2 Mini checkpoint contract.

This is the Phase 0 inventory for the Inkling port: it does not load the 20 GB
weights. It reads the two small metadata files (locally, or from the Hugging
Face raw URLs) and answers which precision every routed-expert block uses, so
the port can never silently requantize a layer or miss one.

    python3 benchmark/inkling2_quant_contract.py --fetch
    python3 benchmark/inkling2_quant_contract.py \
        --config /path/config.json --quant-config /path/hf_quant_config.json

Exit 0 only when all 24 layers have exactly one NVFP4 or MXFP8 group, the
expected 20/4 split holds, and no tensor is claimed by two groups.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

MODEL = "thinkingmachines/Inkling-2-Mini"
BASE = f"https://huggingface.co/{MODEL}/raw/main"
QUANT_FORMATS = {"nvfp4", "mxfp8"}
EXPECTED_SPLIT = 20
LAYERS = 24


class ContractError(ValueError):
    pass


def read_source(value: str) -> dict:
    if value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=60) as response:
            return json.load(response)
    return json.loads(pathlib.Path(value).read_text(encoding="utf-8"))


def _format_of(group: dict) -> str:
    compute = str(group.get("compute_dtype", ""))
    if compute in QUANT_FORMATS:
        return compute
    if compute in {"bfloat16", "float32"}:
        return f"cast:{compute}"
    raise ContractError(f"unknown compute_dtype {compute!r}")


def contract(config: dict, quant: dict) -> dict:
    text = config.get("text_config") or {}
    layers = text.get("num_hidden_layers")
    if layers != LAYERS:
        raise ContractError(f"expected {LAYERS} text layers, got {layers!r}")
    if (config.get("quantization_config") or {}).get("quant_method") != "modelopt":
        raise ContractError("config.json does not declare ModelOpt quantization")
    groups = (quant.get("model") or {}).get("parameter_groups")
    if not isinstance(groups, list) or not groups:
        raise ContractError("hf_quant_config.json has no model.parameter_groups")

    tensor_format: dict[str, str] = {}
    group_counts: dict[str, int] = {}
    for group in groups:
        fmt = _format_of(group)
        tensors = group.get("tensors")
        if not isinstance(tensors, list) or not tensors:
            raise ContractError(f"group {fmt} names no tensors")
        group_counts[fmt] = group_counts.get(fmt, 0) + len(tensors)
        for name in tensors:
            if not isinstance(name, str):
                raise ContractError(f"non-string tensor name in {fmt}")
            if name in tensor_format:
                raise ContractError(f"{name} is claimed by both {tensor_format[name]} and {fmt}")
            tensor_format[name] = fmt

    per_layer: dict[int, dict[str, str]] = {}
    for layer in range(LAYERS):
        entry: dict[str, str] = {}
        for suffix in ("w13_weight", "w2_weight"):
            name = f"model.llm.layers.{layer}.mlp.experts.{suffix}"
            fmt = tensor_format.get(name)
            if fmt is None:
                raise ContractError(f"layer {layer} is missing {name}")
            expected = "nvfp4" if layer < EXPECTED_SPLIT else "mxfp8"
            if fmt != expected:
                raise ContractError(f"layer {layer} {suffix} is {fmt}, expected {expected}")
            entry[suffix] = fmt
        per_layer[layer] = entry

    other = {fmt: count for fmt, count in group_counts.items() if fmt not in QUANT_FORMATS}
    return {
        "model": MODEL,
        "layers": layers,
        "routed_experts": text.get("n_routed_experts"),
        "experts_per_token": text.get("num_experts_per_tok"),
        "hidden_size": text.get("hidden_size"),
        "context": text.get("model_max_length"),
        "loop": text.get("loop_config"),
        "quant_method": quant.get("quant_method"),
        "group_counts": group_counts,
        "per_layer": per_layer,
        "non_expert": other,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=f"{BASE}/config.json")
    parser.add_argument("--quant-config", default=f"{BASE}/hf_quant_config.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--fetch", action="store_true", help="read the two metadata files from Hugging Face"
    )
    args = parser.parse_args(argv)
    config_url = f"{BASE}/config.json" if args.fetch else args.config
    quant_url = f"{BASE}/hf_quant_config.json" if args.fetch else args.quant_config
    try:
        result = contract(read_source(config_url), read_source(quant_url))
    except (ContractError, OSError, ValueError) as error:
        print(f"inkling2 contract: {error}", file=sys.stderr)
        return 1
    if args.json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(
            f"Inkling 2 Mini: {result['layers']} layers, "
            f"{result['routed_experts']} experts top-{result['experts_per_token']}, "
            f"{result['group_counts']}"
        )
        print("  layers 0-19 routed experts: NVFP4; layers 20-23: MXFP8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
