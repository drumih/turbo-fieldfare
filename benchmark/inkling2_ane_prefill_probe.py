"""Probe: does the Apple Neural Engine speed up Inkling2's dense prefill work?

Inkling-2-Mini's routed experts (NVFP4, sparse top-8 routing) cannot run on
the ANE, which executes dense fp16 Core ML graphs. Its dense per-layer
projections can: Q, K, V, gate and the relative projection share one input
(8,576 rows x 2,304), and the attention output is 2,304 x 3,072. This probe
takes those real weights from one layer of the GGUF (dequantised with
llama.cpp's own gguf-py, so the values are exactly what the GGUF stores), builds
them as fp16 Core ML programs at fixed prefill chunk sizes, and runs the same
compiled model on the ANE (CPU_AND_NE), the GPU (CPU_AND_GPU) and the CPU
(CPU_ONLY). It reports, per chunk size and engine: median latency, effective
TFLOP/s, and the relative error against an fp32 NumPy reference — plus whether
the compute plan really placed the matmuls on the ANE.

It answers the go / no-go question for an ANE prefill path before any runtime
code is written. It does not load the model into TinyTitan and needs no GPU
model process. Requires coremltools >= 8 (MLComputePlan) and numpy; run with
the Core ML venv, e.g.::

    ~/.venvs/coreml-py311/bin/python benchmark/inkling2_ane_prefill_probe.py \\
        --gguf <Inkling-2-Mini-UD-Q4_K_XL.gguf> --layer 1 \\
        --output benchmark/ane-prefill/inkling2-ane-probe.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys
import tempfile
import time

import numpy as np

PROJECTIONS = ["attn_q", "attn_k", "attn_v", "attn_gate", "attn_rel"]


def load_gguf_py(llama_cpp: pathlib.Path):
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf  # noqa: PLC0415 - path is only known at run time

    return gguf


def dequantised(gguf, reader, name: str) -> np.ndarray:
    tensor = next(t for t in reader.tensors if t.name == name)
    values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    rows = int(tensor.shape[1])
    cols = int(tensor.shape[0])
    return np.asarray(values, dtype=np.float32).reshape(rows, cols), tensor.tensor_type.name


def build_model(ct, mb, weight: np.ndarray, tokens: int, path: pathlib.Path):
    """y[t, r] = sum_c x[t, c] * W[r, c], fp16 weights and compute."""
    columns = weight.shape[1]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(tokens, columns), dtype=ct.converters.mil.mil.types.fp16)]
    )
    def program(x):
        return mb.linear(x=x, weight=weight.astype(np.float16), name="y")

    model = ct.convert(
        program,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.save(str(path))
    return path


def ane_fraction(ct, package: pathlib.Path) -> float | None:
    try:
        from coremltools.models.compute_plan import MLComputePlan  # noqa: PLC0415
    except ImportError:
        return None
    compiled = ct.models.utils.compile_model(str(package))
    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    program = plan.model_structure.program
    total = 0
    on_ane = 0
    for function in program.functions.values():
        for op in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None or op.operator_name in ("const",):
                continue
            total += 1
            name = type(usage.preferred_compute_device).__name__
            on_ane += "NeuralEngine" in name
    return on_ane / total if total else None


def measure(
    ct, package: pathlib.Path, units, x: np.ndarray, repeats: int
) -> tuple[float, np.ndarray]:
    model = ct.models.MLModel(str(package), compute_units=units)
    feed = {"x": x.astype(np.float16)}
    out = model.predict(feed)["y"]  # warm-up and compile
    for _ in range(2):
        model.predict(feed)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        out = model.predict(feed)["y"]
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), np.asarray(out, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--llama-cpp", default=str(pathlib.Path.home() / ".unsloth/llama.cpp"))
    parser.add_argument("--tokens", default="64,128,256,512")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import coremltools as ct  # noqa: PLC0415 - optional dependency
    from coremltools.converters.mil import Builder as mb  # noqa: PLC0415

    gguf = load_gguf_py(pathlib.Path(args.llama_cpp))
    reader = gguf.GGUFReader(args.gguf)
    prefix = f"blk.{args.layer}."
    parts = [dequantised(gguf, reader, prefix + name + ".weight") for name in PROJECTIONS]
    fused = np.concatenate([w for w, _ in parts], axis=0)
    output, output_type = dequantised(gguf, reader, prefix + "attn_output.weight")
    stages = {
        "qkv_gate_rel": (fused, sorted({t for _, t in parts})),
        "attn_output": (output, [output_type]),
    }
    rng = np.random.default_rng(0)
    units = {
        "ane": ct.ComputeUnit.CPU_AND_NE,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        for stage, (weight, types) in stages.items():
            for tokens in [int(t) for t in args.tokens.split(",")]:
                package = build_model(
                    ct, mb, weight, tokens, pathlib.Path(scratch) / f"{stage}-{tokens}.mlpackage"
                )
                # RMS-normalised activations, the shape of a projection input.
                x = rng.standard_normal((tokens, weight.shape[1])).astype(np.float32)
                reference = x.astype(np.float16).astype(np.float32) @ weight.T
                row = {
                    "stage": stage,
                    "tokens": tokens,
                    "rows": int(weight.shape[0]),
                    "columns": int(weight.shape[1]),
                    "gguf_types": types,
                    "ane_op_fraction": ane_fraction(ct, package),
                }
                flops = 2.0 * tokens * weight.shape[0] * weight.shape[1]
                for engine, unit in units.items():
                    seconds, got = measure(ct, package, unit, x, args.repeats)
                    error = float(np.linalg.norm(got - reference) / np.linalg.norm(reference))
                    row[engine] = {
                        "ms": seconds * 1e3,
                        "tflops": flops / seconds / 1e12,
                        "rel_error": error,
                    }
                results.append(row)
                print(
                    f"{stage:13s} T={tokens:4d}  "
                    + "  ".join(
                        f"{e}: {row[e]['ms']:7.2f} ms {row[e]['tflops']:5.2f} TF/s err {row[e]['rel_error']:.2e}"
                        for e in units
                    )
                    + f"  ane-ops {row['ane_op_fraction']}"
                )
    payload = {
        "probe": "inkling2-ane-prefill",
        "gguf": pathlib.Path(args.gguf).name,
        "layer": args.layer,
        "machine": platform.platform(),
        "coremltools": ct.__version__,
        "repeats": args.repeats,
        "results": results,
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
