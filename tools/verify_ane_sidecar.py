#!/usr/bin/env python3
"""Independently verify an exported ANE sidecar's graph against NumPy.

One question: does this sidecar's Core ML program compute the attention block
the exporter meant to build? The exported layer is fed synthetic input and its
output compared against a NumPy implementation written from the graph's own
specification — packed q+gate, per-head q/k RMSNorm, NeoX rope on
`partialRotaryFactor * headDim`, GQA expansion, additive mask.

What this DOES verify is the **geometry**: the hidden width, the query and KV
head counts, the rotary dimension, the output gate and the mask. That is the
failure that matters, because a sidecar built for the wrong geometry does not
crash — it computes a *different* attention and produces fluent nonsense that
nothing downstream flags.

For a sparse-indexed family (`selectionFolded` in the sidecar metadata, Qwen
3.8's QSA indexer) it additionally verifies the **fold**: the graph is fed a
QSA-shaped additive mask, checked against the reference under that same mask,
and the run also reports what a causal-only mask would have got wrong — so the
claim "the runtime must fold, and folding is enough" is measured rather than
assumed.

What it does NOT verify is the **weight dequantization**: both sides read the
tensors through the exporter's own loader, so a width error would cancel out.
That is covered instead by `load_tensor` refusing a manifest whose declared
width disagrees with the tensor's byte count.

    ~/.venvs/coreml-py311/bin/python tools/verify_ane_sidecar.py \
        --model models/qwen3.5_4B_4Bit

Exits non-zero when the mean relative error exceeds `--max-relative-error`
(default 2%; the design documents ~1% fp16 deviation from fp32), or when the
folded check finds that the mask barely matters (a vacuous check).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "export_ane_prefill", pathlib.Path(__file__).resolve().parent / "export_ane_prefill.py"
)
ex = importlib.util.module_from_spec(_spec)
sys.modules["export_ane_prefill"] = ex  # dataclasses resolves the module here
_spec.loader.exec_module(ex)


def reference(
    normed: np.ndarray, weights: dict, geom, mask: np.ndarray, t: int, block_rows: int = 512
) -> np.ndarray:
    """The graph's arithmetic, in float32, from the graph's own definition.

    The attention is accumulated in query blocks: at a 4,096-token chunk the
    full `[1, 24, 4096, 4096]` fp32 score tensor is 1.6 GB, and the softmax
    would hold two more copies of it. Blocking changes no arithmetic and keeps
    the check runnable beside everything else on a 24 GB machine.
    """

    def rms_head(x, weight):
        x = x.astype(np.float32)
        ms = (x * x).mean(-1, keepdims=True)
        return x / np.sqrt(ms + np.float32(ex.EPS)) * weight.astype(np.float32)

    cos, sin = ex.rope_tables(0, geom)
    cos = cos.astype(np.float32)
    sin = sin.astype(np.float32)

    def rope(x):  # x: [t, heads, hd]
        r = geom.rotary // 2
        r1, r2, rest = x[..., :r], x[..., r : 2 * r], x[..., 2 * r :]
        return np.concatenate(
            [
                r1 * cos[:, None, :] - r2 * sin[:, None, :],
                r2 * cos[:, None, :] + r1 * sin[:, None, :],
                rest,
            ],
            axis=-1,
        )

    hd, qh, kvh = geom.head_dim, geom.q_heads, geom.kv_heads
    xn = normed.astype(np.float32)
    packed = (xn @ weights["wq"].astype(np.float32).T).reshape(t, qh, 2 * hd)
    q, gate = packed[..., :hd], packed[..., hd:]
    k = (xn @ weights["wk"].astype(np.float32).T).reshape(t, kvh, hd)
    v = (xn @ weights["wv"].astype(np.float32).T).reshape(t, kvh, hd)
    q = rope(rms_head(q, weights["q_norm"]))
    k = rope(rms_head(k, weights["k_norm"]))

    rep = qh // kvh
    k4 = np.repeat(k, rep, axis=1).transpose(1, 0, 2)[None]
    v4 = np.repeat(v, rep, axis=1).transpose(1, 0, 2)[None]
    q4 = q.transpose(1, 0, 2)[None]

    bias = mask.astype(np.float32)
    kt = k4.transpose(0, 1, 3, 2)
    scale = np.float32(geom.scale)
    attn = np.empty((1, qh, t, hd), dtype=np.float32)
    for start in range(0, t, block_rows):
        end = min(t, start + block_rows)
        scores = q4[:, :, start:end] @ kt
        scores *= scale
        scores += bias[..., start:end, :]
        scores -= scores.max(-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(-1, keepdims=True)
        attn[:, :, start:end] = scores @ v4
    gated = attn.transpose(0, 2, 1, 3) * (1.0 / (1.0 + np.exp(-gate)))
    return gated.reshape(t, geom.q_dim) @ weights["wo"].astype(np.float32).T


def causal_mask(t: int) -> np.ndarray:
    mask = np.full((1, 1, t, t), ex.NEG, dtype=np.float32)
    for row in range(t):
        mask[0, 0, row, : row + 1] = 0.0
    return mask


def selection_mask(t: int, budget: int, compress_ratio: int, seed: int):
    """A QSA-shaped selection for one full chunk, as the runtime builds it.

    The rule is `QSAIndexer.selectKeysPrefill`'s, in NumPy: a query keeps the
    ragged tail of its own block, then complete blocks in descending score
    order — ties to the lower block index — until the cell budget runs out.
    Scores are synthetic but deterministic. What is under test is that the
    graph honours an *arbitrary* mask; whether the indexer ranks blocks well is
    the indexer's business, not the sidecar's.
    """
    selection_width = budget + compress_ratio - 1
    rng = np.random.default_rng(seed)
    mask = np.full((1, 1, t, t), ex.NEG, dtype=np.float32)
    kept = np.zeros(t, dtype=np.int64)
    scores = rng.standard_normal(t // compress_ratio + 1)
    for row in range(t):
        visible = row + 1
        row_mask = mask[0, 0, row]
        if visible <= selection_width:
            row_mask[:visible] = 0.0
            kept[row] = visible
            continue
        complete = (visible // compress_ratio) * compress_ratio
        row_mask[complete:visible] = 0.0
        remaining = selection_width - (visible - complete)
        if remaining > 0:
            blocks = complete // compress_ratio
            for block in sorted(range(blocks), key=lambda b: (-scores[b], b)):
                if remaining <= 0:
                    break
                take = min(compress_ratio, remaining)
                base = block * compress_ratio
                row_mask[base : base + take] = 0.0
                remaining -= take
        kept[row] = int((row_mask[:visible] == 0.0).sum())
    return mask, kept


def load_activation_dump(path: pathlib.Path, t: int, hidden: int) -> np.ndarray:
    """Load a REAL captured normed — a `TINYTITAN_ACT_DUMP` `L*_in.f16`.

    The synthetic N(0,1) input below lands the score dot-products near 1, where
    the fp16 ulp is ~0.001, and structurally cannot reproduce the live input's
    correlated rows: on ornith-1.5_35B_A3B_4Bit the same graph-vs-reference
    comparison measured 0.555% synthetic but 4.245% on a real captured normed
    (2026-11-19), which is how an fp16 score materialization shipped blessed.
    Refuses a dump whose size does not match the chunk geometry.
    """
    raw = np.fromfile(path, dtype=np.float16)
    if raw.size != t * hidden:
        raise SystemExit(
            f"{path}: {raw.size} elements, expected {t}*{hidden} = {t * hidden} "
            "for this sidecar's chunk; wrong dump or wrong --chunk"
        )
    return raw.reshape(t, hidden)


def compare_outputs(
    got: np.ndarray, expected: np.ndarray, max_relative_error: float
) -> tuple[bool, dict]:
    """The accept/refuse verdict on a graph-vs-reference comparison.

    The old gate was `relative > max_relative_error` — but `nan > x` is
    False in Python, so a 0/0 = nan error (both sides all zero, or NaN/Inf in
    either output) passed. The zeroed 2B install shipped exactly that way.
    Non-finite comparisons and a vacuous all-zero side are refusals with a
    named reason, never a silent pass.
    """
    mean_expected = float(np.abs(expected).mean())
    stats: dict = {"mean_expected": mean_expected}
    if mean_expected == 0.0 or not math.isfinite(mean_expected):
        stats["reason"] = "the reference is all zero or non-finite — a vacuous comparison"
        return False, stats
    relative = float(np.abs(got - expected).mean() / mean_expected * 100.0)
    peak = float(np.abs(got - expected).max() / mean_expected * 100.0)
    stats["relative"] = relative
    stats["peak"] = peak
    if not (math.isfinite(relative) and math.isfinite(peak)):
        stats["reason"] = "the comparison is not finite (NaN/Inf in one side's output)"
        return False, stats
    if not np.isfinite(np.asarray(got, dtype=np.float64)).all():
        stats["reason"] = "the graph output contains NaN/Inf"
        return False, stats
    if float(np.abs(got).max()) == 0.0:
        stats["reason"] = "the graph output is all zero — the graph computes nothing"
        return False, stats
    if relative > max_relative_error:
        stats["reason"] = f"above --max-relative-error {max_relative_error} %"
        return False, stats
    return True, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to an installed .ssdai directory")
    parser.add_argument(
        "--layer", type=int, default=None, help="full-attention layer to check (default: the first)"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=ex.CHUNK,
        help=f"which sidecar width to check (default "
        f"{ex.CHUNK}). A model can carry several — "
        f"ane_prefill-1024 beside ane_prefill — and "
        f"checking the wrong one would report a pass for a "
        f"graph nobody asked about",
    )
    parser.add_argument("--max-relative-error", type=float, default=2.0)
    parser.add_argument(
        "--min-selection-effect",
        type=float,
        default=1.0,
        help="the fold check fails when a synthetic QSA "
        "selection changes the reference by less than this "
        "percent of its mean magnitude, which would make "
        "the check vacuous",
    )
    parser.add_argument(
        "--selection",
        action="store_true",
        help="run the folded-mask check even where the sidecar "
        "does not record selectionFolded (it runs "
        "automatically where it does)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--activation-dump",
        type=pathlib.Path,
        default=None,
        help="run one extra comparison on a REAL captured normed "
        "(a TINYTITAN_ACT_DUMP `L*_in.f16`, [chunk*hidden] fp16). The "
        "synthetic input cannot see amplitude- and structure-dependent fp16 "
        "rounding: on ornith-1.5_35B_A3B_4Bit it measured 0.555%% while the "
        "real captured normed measured 4.245%% (2026-11-19).",
    )
    args = parser.parse_args()

    import coremltools as ct

    model = pathlib.Path(args.model)
    directory = model / ex.sidecar_directory(args.chunk)
    if not (directory / "ane_prefill.json").exists():
        raise SystemExit(
            f"{model} has no sidecar for chunk {args.chunk} ({directory.name}); export one first"
        )
    manifest = json.loads((model / "manifest.json").read_text())
    entries = ex.read_index(model / "model_weights.bin")
    # The chunk is part of the geometry the graph was built for, so it comes
    # from the request, not from the manifest: `--chunk` selects both the
    # directory and the shapes the reference must use.
    geom = dataclasses.replace(ex.geometry_for(manifest, entries), chunk=args.chunk)
    sidecar = json.loads((directory / "ane_prefill.json").read_text())
    if sidecar.get("chunkTokens") != args.chunk:
        raise SystemExit(
            f"{directory.name} holds a "
            f"{sidecar.get('chunkTokens')}-token sidecar, not "
            f"{args.chunk}; checking it would validate the wrong "
            f"graph"
        )
    layer = args.layer if args.layer is not None else sidecar["layers"][0]
    if layer not in sidecar["layers"]:
        raise SystemExit(f"layer {layer} is not in the sidecar {sidecar['layers']}")

    print(
        f"{model.name}: {geom.family}, hidden {geom.hidden}, "
        f"{geom.q_heads}q/{geom.kv_heads}kv x{geom.head_dim}, "
        f"rope {geom.rotary}, layer {layer}"
    )

    t = geom.chunk
    rng = np.random.default_rng(args.seed)
    normed = rng.standard_normal((t, geom.hidden)).astype(np.float16)
    mask = causal_mask(t)
    with open(model / "model_weights.bin", "rb") as handle:
        weights = ex.load_layer_weights(handle, entries, layer, geom, manifest)
    expected = reference(normed, weights, geom, mask, t)

    package = directory / f"layer_{layer}.mlpackage"
    model_ml = ct.models.MLModel(
        str(package), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name="h0"
    )

    def graph_output(raw):
        """`out` for a fused sidecar; attn+gate -> fp32 gate+wo for a split one."""
        if "attn" in raw:
            attn = np.asarray(raw["attn"], dtype=np.float32).reshape(t, geom.q_dim)
            gate = np.asarray(raw["gate"], dtype=np.float32).reshape(t, geom.q_dim)
            gated = attn * (1.0 / (1.0 + np.exp(-gate)))
            return gated @ weights["wo"].astype(np.float32).T
        return np.asarray(raw["out"], dtype=np.float32).reshape(expected.shape)

    got = graph_output(model_ml.predict({"normed": normed, "mask": mask.astype(np.float16)}))

    passed, stats = compare_outputs(got, expected, args.max_relative_error)
    print(
        f"  mean |expected| {stats['mean_expected']:.4f} | mean |diff| {np.abs(got - expected).mean():.5f}"
    )
    if "relative" in stats:
        print(f"  relative error {stats['relative']:.3f} % (peak {stats['peak']:.3f} %)")
    if not passed:
        print(f"FAIL: {stats['reason']}", file=sys.stderr)
        return 1

    # The optional real-input check: the synthetic case above cannot see
    # amplitude- and structure-dependent fp16 rounding (see
    # load_activation_dump). Run it when a captured dump exists.
    if args.activation_dump is not None:
        real = load_activation_dump(args.activation_dump, t, geom.hidden)
        expected_real = reference(real, weights, geom, mask, t)
        got_real = graph_output(model_ml.predict({"normed": real, "mask": mask.astype(np.float16)}))
        passed_real, stats_real = compare_outputs(got_real, expected_real, args.max_relative_error)
        print(f"  real captured normed ({args.activation_dump.name}):")
        print(f"    mean |expected| {stats_real['mean_expected']:.4f}")
        if "relative" in stats_real:
            print(
                f"    relative error {stats_real['relative']:.3f} %"
                f" (peak {stats_real['peak']:.3f} %)"
            )
        if not passed_real:
            print(f"FAIL: {stats_real['reason']} (real captured input)", file=sys.stderr)
            return 1

    # A sparse-indexed family's graph is only correct because the runtime folds
    # the indexer's selection into the mask. Measure that: the same graph under
    # a QSA-shaped mask against the reference under the same mask, and what the
    # causal-only mask — the mistake this wiring exists to prevent — would cost.
    if sidecar.get("selectionFolded") or args.selection:
        arch = manifest["arch"]
        budget = int(arch.get("indexerBudget") or 0)
        ratio = int(arch.get("indexerCompressRatio") or 0)
        if budget <= 0 or ratio <= 0:
            raise SystemExit(
                "--selection/folded check needs an indexer geometry "
                "(indexerBudget, indexerCompressRatio) and this manifest has "
                "none"
            )
        folded, kept = selection_mask(t, budget, ratio, args.seed)
        expected_folded = reference(normed, weights, geom, folded, t)
        got_folded = model_ml.predict({"normed": normed, "mask": folded.astype(np.float16)})["out"]
        got_folded = np.asarray(got_folded, dtype=np.float32).reshape(expected_folded.shape)
        base = float(np.abs(expected_folded).mean())
        folded_error = float(np.abs(got_folded - expected_folded).mean() / base * 100.0)
        # How much the selection changes the attention at all, and what an ANE
        # run that forgot to fold would have produced against the causal
        # reference: the two should agree, and both must be visible.
        effect = float(np.abs(expected_folded - expected).mean() / stats["mean_expected"] * 100.0)
        causal_error = float(np.abs(got_folded - expected).mean() / stats["mean_expected"] * 100.0)
        print(
            f"  folded mask: {kept.mean():.0f} of {t} keys kept on average "
            f"({kept.min()}–{kept.max()})"
        )
        print(
            f"  folded relative error {folded_error:.3f} % (against reference under the same mask)"
        )
        print(
            f"  selection changes the reference by {effect:.3f} %, and a "
            f"causal-only mask is off by {causal_error:.3f} %"
        )
        if folded_error > args.max_relative_error:
            print(
                f"FAIL: folded error above --max-relative-error {args.max_relative_error} %",
                file=sys.stderr,
            )
            return 1
        if effect < args.min_selection_effect:
            print(
                f"FAIL: the synthetic selection moves the reference by only "
                f"{effect:.3f} % (< --min-selection-effect "
                f"{args.min_selection_effect} %); this check would pass "
                f"whatever the graph did with the mask",
                file=sys.stderr,
            )
            return 1
        print("  ok — the graph honours the folded QSA selection")

    print("  ok — the graph computes this model's attention block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
