#!/usr/bin/env python3
"""Where does a wide-verify row leave scalar decode? Model-free reader of the
dumps `TinyTitanBench verifyprobe` writes (docs/plan-wide-verify.md, task 3).

For each pass row i at position P+i, compares the pass's residual after layer L
(`pos{P+i}/L{L}_rows_after.f16`) with scalar decode's residual entering layer
L+1 (`L{L+1}_entry.f16`, or `stack_out.f16` after the last layer), and the
pass's logits (`verify_logits.f16`) with scalar decode's (`logits.f16`). It
prints, per row, the first layer whose residual differs at all, the largest
absolute difference per layer, and whether the logits' argmax agrees.

    python3 benchmark/verify_identity_bisect.py .build/verify-probe <P> <width>
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from array import array


def read_f16(path: pathlib.Path) -> array:
    values = array("e")
    values.frombytes(path.read_bytes())
    return values


def max_abs_diff(a: array, b: array) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch {len(a)} != {len(b)}")
    return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)


def argmax(values: array) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


STAGES = ("attn_in", "attn_out", "hidden_post_attn", "mlp_in", "mlp_out")


def stage_diffs(directory: pathlib.Path) -> list[tuple[str, float]]:
    """Inside-layer stages the pass dumps (`L{L}_{stage}_rows`) against scalar
    decode's (`L{L}_{stage}`), in layer and pipeline order; pairs whose files
    are missing or differ in length are skipped, not guessed."""
    found = []
    for path in sorted(directory.glob("L*_*_rows.f16")):
        name = path.name[: -len("_rows.f16")]
        layer, stage = name.split("_", 1)
        scalar = directory / f"{name}.f16"
        if stage not in STAGES or not scalar.exists():
            continue
        a, b = read_f16(path), read_f16(scalar)
        if len(a) == len(b):
            found.append((int(layer[1:]), STAGES.index(stage), name, max_abs_diff(a, b)))
    return [(name, diff) for _, _, name, diff in sorted(found)]


def compare_row(directory: pathlib.Path) -> dict:
    """Per-layer max |pass - scalar| for one position's dumps."""
    layers = sorted(int(p.name[1 : p.name.index("_")]) for p in directory.glob("L*_rows_after.f16"))
    if not layers:
        raise FileNotFoundError(f"{directory}: no L*_rows_after.f16 dumps")
    diffs = []
    for layer in layers:
        scalar = directory / f"L{layer + 1}_entry.f16"
        if not scalar.exists():
            scalar = directory / "stack_out.f16"
        if not scalar.exists():
            raise FileNotFoundError(f"{directory}: no scalar residual after layer {layer}")
        diffs.append(
            (
                layer,
                max_abs_diff(read_f16(directory / f"L{layer}_rows_after.f16"), read_f16(scalar)),
            )
        )
    first = next((layer for layer, diff in diffs if diff > 0), None)
    result = {"diffs": diffs, "first_divergent_layer": first, "stages": stage_diffs(directory)}
    verify, scalar_logits = directory / "verify_logits.f16", directory / "logits.f16"
    if verify.exists() and scalar_logits.exists():
        v, s = read_f16(verify), read_f16(scalar_logits)
        result["logits_max_abs_diff"] = max_abs_diff(v, s)
        result["argmax_same"] = argmax(v) == argmax(s)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dumps", type=pathlib.Path)
    parser.add_argument("position", type=int)
    parser.add_argument("width", type=int)
    args = parser.parse_args(argv)
    for row in range(args.width):
        directory = args.dumps / f"pos{args.position + row}"
        try:
            result = compare_row(directory)
        except (FileNotFoundError, ValueError) as error:
            print(f"row {row}: refused: {error}", file=sys.stderr)
            return 2
        worst = max(result["diffs"], key=lambda item: item[1])
        logits = (
            f"logits max|d| {result['logits_max_abs_diff']:.4g} argmax {'same' if result['argmax_same'] else 'DIFFERENT'}"
            if "argmax_same" in result
            else "no logits dumped"
        )
        print(
            f"row {row}: first divergent layer {result['first_divergent_layer']}, "
            f"worst L{worst[0]} max|d| {worst[1]:.4g}; {logits}"
        )
        print("   " + " ".join(f"L{layer}:{diff:.3g}" for layer, diff in result["diffs"]))
        if result["stages"]:
            print("   stages: " + " ".join(f"{name}:{diff:.3g}" for name, diff in result["stages"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
