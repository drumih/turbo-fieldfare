#!/usr/bin/env python3
"""Export the ANE prefill attention sidecar for a `.ssdai` model.

Produces `<model>/ane_prefill/layer_<L>.mlpackage` for every full-attention
layer: a multifunction Core ML program whose functions `h0, h4096, ...` share
one set of fp16 weights (dequantized from the model's int4 affine tensors)
and differ only in how much KV history they attend to. Chunk width is fixed
at 4096 — the production prefill chunk — with the causal mask and NeoX RoPE
tables built in-graph, so the Swift runtime feeds only the normed hidden
chunk and the token-major fp16 K/V history.

Why these choices (all measured, see docs/v4.4-decode-width-plan.md Track A):
- decomposed attention, never the fused SDPA op — the fused op produces
  NaN/inf on this M3's ANE from sequence length 2048;
- fp16 weights — they amortize over 4,096-token chunks, so quantized palettes
  buy nothing at prefill; the sidecar is ~52 MB per layer;
- fixed enumerated shapes — chunk boundaries in this runtime are always
  multiples of 4096, so history is too, and fixed shapes keep the ANE
  scheduler on the fast path;
- additive -30000 mask instead of -inf — exp() underflows identically and
  fp16 infinity arithmetic stays out of the graph. The mask is an *input*, so
  a family whose attention selects keys (Qwen 3.8's QSA indexer) is served by
  the runtime folding that selection into the same additive mask; the graph
  itself is identical, and `selectionFolded` in the metadata records that the
  sidecar was exported knowing this.

  ~/.venvs/coreml-py311/bin/python tools/export_ane_prefill.py \
      --model models/ornith-1.5_35B_A3B_4Bit --max-history 12288

Exits non-zero, writes nothing, and leaves any existing sidecar untouched when
the Neural Engine refuses to compile a variant. Core ML reports that failure on
the native stderr and otherwise exits 0, so without this an export could
"succeed" into a sidecar that the runtime then runs on the CPU at ~38x the GPU
prefill cost (issue #7). A successful export records `aneCompileVerified` in
`ane_prefill.json`, and the runtime refuses a sidecar that lacks it.

Compiling is not the last word either: every function the metadata records is
also *loaded* under its `h<history>` name before the export commits, because a
specialization the ANE refuses can still save and then fail at the runtime's
`MLModel(contentsOf:)` — which would advertise coverage the sidecar cannot
serve. Qwen 3.8's `h12288` is that case on this machine.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import pathlib
import shutil
import struct
import sys
import tempfile
import warnings

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

try:
    from coremltools.models.compute_plan import MLComputePlan
except Exception:  # noqa: BLE001 — optional
    MLComputePlan = None

# The chunk the sidecar is built around by default. The runtime routes a chunk
# to the ANE only when its configured prefill chunk is exactly the sidecar's, so
# this is a contract with `ANEPrefillAttention.eligibleChunk`, not a tunable.
CHUNK = 4096
# The chunk sizes the runtime will accept as a prefill chunk, copied from
# `RuntimeConfiguration.allowedPrefillChunkTokens`: a sidecar whose chunk is not
# in this set could never match a configuration and would be dead weight.
PREFILL_CHUNK_CHOICES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
EPS = 1e-6
NEG = -30000.0
EXPORT_VERSION = 1

# Families whose full-attention block this graph reproduces. `qwen38flash` is
# included because the only thing that separated it was the *mask*: its
# full-attention layers pick keys with a QSA sparse indexer, and dense attention
# matches that selection only through 2,051 visible keys, so a sidecar fed the
# causal mask would attend to keys the model drops. The graph takes an arbitrary
# additive mask, so the runtime folds the same selection in
# (`ANEPrefillAttention.selectionMask`) and the graph is unchanged; the metadata
# records `selectionFolded` so the runtime refuses a sidecar that does not carry
# the contract. `qwen38flash_mtp` is the one-layer MTP draft, which the runtime
# never routes to the ANE.
SUPPORTED_FAMILIES = ("qwen36", "qwen3_5_dense", "qwen38flash", "qwen4flash")


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The attention geometry a sidecar is built and validated against.

    The supported families share one attention block — packed q+gate, q/k
    RMSNorm, GQA, NeoX rope on `partialRotaryFactor * headDim`, additive causal
    mask — and differ only in these numbers and the tensor-name prefix. The
    exporter used to hard-code the 35B-A3B row (D=2048, 16/2 heads), which is
    why no other family could ever have a sidecar.
    """

    family: str
    prefix: str
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rotary: int
    theta: float
    scale: float
    chunk: int
    layers: tuple[int, ...]

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    def as_metadata(self) -> dict:
        return {
            "family": self.family,
            "hiddenSize": self.hidden,
            "numHeads": self.q_heads,
            "numKVHeads": self.kv_heads,
            "headDim": self.head_dim,
            "rotaryDim": self.rotary,
            "ropeTheta": self.theta,
            "attentionScale": self.scale,
            "chunkTokens": self.chunk,
            "fullAttentionLayers": list(self.layers),
        }


def _attention_prefix(entries: dict[str, dict], layer: int) -> str:
    """The tensor-name prefix this model uses, read from its own index.

    `language_model.model.layers.N.self_attn.*` for the qwen36 and dense
    families, `model.language_model.layers.N.self_attn.*` for the 3.8 one — so
    it is discovered rather than assumed.
    """
    suffix = f".layers.{layer}.self_attn.q_proj.weight"
    for name in entries:
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".layers."
    raise SystemExit(
        f"no tensor named *{suffix} in model_weights.bin; this is not a "
        f"supported .ssdai attention layout"
    )


def geometry_for(manifest: dict, entries: dict[str, dict]) -> Geometry:
    """Derive the geometry from the model itself.

    Everything the graph needs is in the manifest's `arch`, and every guard
    below refuses a model whose attention block differs from the one this graph
    computes — a refusal is cheap, and a graph that silently computes a
    *different* attention produces fluent nonsense that nothing downstream
    flags.
    """
    arch = manifest["arch"]
    family = arch["family"]
    if family not in SUPPORTED_FAMILIES:
        detail = ""
        if family.endswith("_mtp"):
            detail = (
                " — the MTP draft is verified rather than prefilled on "
                "the ANE, so it carries no sidecar"
            )
        raise SystemExit(
            f"ANE prefill export supports {', '.join(SUPPORTED_FAMILIES)}; "
            f"this model is {family}{detail}"
        )
    # Each guard refuses a model whose attention block differs from the one this
    # graph computes. A refusal is cheap; a graph that silently computes a
    # *different* attention produces fluent nonsense nothing downstream flags.
    if arch.get("attentionKEqV"):
        raise SystemExit(
            "attentionKEqV models are not supported: the graph computes K and V separately"
        )
    if arch.get("ropeNeoxSubdim") is False:
        raise SystemExit("non-NeoX rope is not supported: the graph applies rope in NeoX order")
    if arch.get("slidingWindow"):
        raise SystemExit(
            f"slidingWindow={arch['slidingWindow']} is not "
            f"supported: the graph has no sliding-window mask"
        )
    mask = arch["fullAttentionLayerMask"]
    layers = tuple(i for i, v in enumerate(mask) if int(v) == 1)
    if not layers:
        raise SystemExit("fullAttentionLayerMask selects no full-attention layer")
    head_dim = int(arch.get("fullHeadDim") or arch["headDim"])
    rotary = int(round(float(arch["partialRotaryFactor"]) * head_dim))
    return Geometry(
        family=family,
        prefix=_attention_prefix(entries, layers[0]),
        hidden=int(arch["hiddenSize"]),
        q_heads=int(arch["numHeads"]),
        kv_heads=int(arch.get("numFullKVHeads") or arch["numKVHeads"]),
        head_dim=head_dim,
        rotary=rotary,
        theta=float(arch.get("fullRopeTheta") or arch["ropeTheta"]),
        scale=float(arch["attentionScale"]),
        chunk=CHUNK,
        layers=layers,
    )


# Core ML does not raise when the Neural Engine refuses to compile a model: it
# logs the failure on the native stderr and then runs the program on the CPU.
# An export that exits 0 with these in its log writes a sidecar that is 38x
# slower at prefill than the GPU path (issue #7), so the exporter treats them
# as a hard failure.
ANE_COMPILE_ERROR_MARKERS = (
    "ANECCompile() FAILED",
    "MILCompilerForANE error",
    "failed to compile ANE model",
    "E5RT encountered an STL exception",
)


class ANEExportError(RuntimeError):
    """The Neural Engine refused to compile a variant the exporter built."""


def verify_variant_reaches_the_ane(variant: pathlib.Path, history: int, layer: int) -> int | None:
    """Assert the Neural Engine is assigned this variant's operations.

    The compile-marker scan and `verify_variants_load` both ask whether Core ML
    *complained*. This asks what `aneCompileVerified` actually claims: did the
    Neural Engine get the graph?

    It matters because issue #7's failure mode is silent. Measured on the 3.8
    sidecar's h12288 as a standalone variant: it loads, `predict()` is never
    possible, and **0 of its 173 operations** are assigned to the Neural Engine —
    so Core ML runs it on the CPU at roughly 38x the GPU prefill cost while a
    marker-only check sees nothing. A healthy variant of the same graph reports
    74 of 173.

    It is checked per *variant*, before the multifunction merge, because that is
    where the variant is the package's default function: `MLComputePlan` reports
    device assignments only for the default function, so every non-default
    function of a merged package reports none (measured: the same 2B graph is
    74/173 standalone and 0/173 as a non-default function of the merge).

    Returns the count of operations assigned to the Neural Engine, or nil when
    the running coremltools has no compute-plan API — a warning, not a failure,
    because otherwise a missing API would block an export that the marker scan
    and the load check still cover.
    """
    if MLComputePlan is None:
        global _warnedNoComputePlan
        if not _warnedNoComputePlan:
            _warnedNoComputePlan = True
            sys.stderr.write(
                "warning: this coremltools has no MLComputePlan, so whether the "
                "Neural Engine is assigned each variant cannot be checked; the "
                "sidecar is covered by the compile markers and the load check "
                "only. coremltools >= 8 provides the full check.\n"
            )
        return None
    compiled = ct.models.utils.compile_model(str(variant))
    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    program = plan.model_structure.program
    function = program.functions.get("main") or next(iter(program.functions.values()))
    operations = list(function.block.operations)
    if not operations:
        raise ANEExportError(f"layer {layer}: h{history} has no operations to assign")
    on_ane = 0
    for operation in operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        device = getattr(usage, "preferred_compute_device", None)
        if device is not None and "NeuralEngine" in type(device).__name__:
            on_ane += 1
    if on_ane == 0:
        raise ANEExportError(
            f"layer {layer}: h{history} converted, but the Neural Engine is "
            f"assigned none of its {len(operations)} operations, so Core ML "
            f"would run it on the CPU at roughly 38x the GPU prefill cost "
            f"(issue #7). Re-export with a smaller --max-history"
        )
    return on_ane


_warnedNoComputePlan = False


def verify_variants_load(package: pathlib.Path, histories, layer: int) -> None:
    """Load every function the metadata is about to advertise.

    The converter accepting a graph is not the same as Core ML being able to
    *load* it. The ANE can refuse a specialization that conversion was happy
    with: the merged multifunction still saves, and then the runtime's
    `MLModel(contentsOf:configuration:)` fails outright — Core ML reports
    "`.functionName` property must be nil unless the model type is ML Program",
    because the refused variant did not come back as an mlprogram at all.

    Recording such a history would hand the runtime a sidecar that claims
    coverage it cannot deliver, and the request that reached that history would
    die at load time instead of falling back. It is the same shape of failure as
    issue #7 (`aneCompileVerified`), one stage later, so it fails the export.

    Measured on this machine: Qwen 3.8's `h12288` does not load (24 heads of
    4096x16384 fp16 scores is a 3.2 GB arena) while `h0`/`h4096`/`h8192` do, and
    the 35B's `h12288` loads. The check costs one load per function per layer —
    minutes on a full export, against a sidecar that is loaded once per install.

    The failure is caught through Core ML's *warning* as well as its exceptions:
    this Python API does not raise for the refused variant, it warns that
    "You will not be able to run predict() on this Core ML model" and returns a
    model that cannot run. Only the Objective-C call the runtime uses raises, so
    an exception-only check here would pass and the sidecar would still lie.
    """
    for history in histories:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                ct.models.MLModel(
                    str(package),
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    function_name=f"h{history}",
                )
            except Exception as exc:  # noqa: BLE001 — reported
                raise ANEExportError(
                    f"layer {layer}: the converter accepted h{history} but Core "
                    f"ML cannot load it ({exc}). Recording it would advertise "
                    f"coverage the sidecar cannot serve; re-export with a "
                    f"smaller --max-history"
                ) from exc
        unusable = [
            str(w.message) for w in caught if "will not be able to run predict" in str(w.message)
        ]
        if unusable:
            raise ANEExportError(
                f"layer {layer}: h{history} loads but Core ML says it cannot "
                f"run ({unusable[0][:160]}). Recording it would advertise "
                f"coverage the sidecar cannot serve; re-export with a smaller "
                f"--max-history"
            )


@contextlib.contextmanager
def _capture_native_stderr():
    """Capture the C++ stderr coremltools writes ANE compiler errors to."""
    saved = os.dup(2)
    spill = tempfile.TemporaryFile()
    os.dup2(spill.fileno(), 2)
    try:
        yield spill
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


def run_checked(what, call):
    """Run `call`, fail loudly if the ANE compiler rejected the result.

    The captured stderr is echoed back, so a clean run looks exactly as it did
    before; only a run that produced compiler errors changes behaviour.
    """
    with _capture_native_stderr() as spill:
        result = call()
    spill.seek(0)
    log = spill.read().decode("utf-8", "replace")
    spill.close()
    if log:
        sys.stderr.write(log)
    failed = [marker for marker in ANE_COMPILE_ERROR_MARKERS if marker in log]
    if failed:
        raise ANEExportError(
            f"{what}: the Neural Engine refused to compile this variant "
            f"({', '.join(failed)}). The sidecar was not written."
        )
    return result


def read_index(path: pathlib.Path) -> dict[str, dict]:
    with open(path, "rb") as handle:
        index_size, _resident, entry_count = struct.unpack("<QQQ", handle.read(24))
        handle.seek(0)
        region = handle.read(index_size)
    entries: dict[str, dict] = {}
    for i in range(entry_count):
        off = 24 + i * 72
        name_off, name_len = struct.unpack_from("<IH", region, off)
        name = region[name_off : name_off + name_len].decode()
        file_off, size = struct.unpack_from("<QQ", region, off + 8)
        shape = struct.unpack_from("<4I", region, off + 24)
        scale_off, scale_size, bias_off, bias_size = struct.unpack_from("<QQQQ", region, off + 40)
        entries[name] = dict(
            dtype=region[off + 6],
            offset=file_off,
            size=size,
            shape=shape,
            scale=(scale_off, scale_size),
            bias=(bias_off, bias_size),
        )
    return entries


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def tensor_weight_bits(manifest: dict, full_name: str, fallback: int) -> int:
    """The width a tensor is stored at: its per-tensor override, else its slot.

    The manifest's `quant` object holds the five slot defaults *and* one entry
    per tensor that deviates, keyed by stem — the same table the runtime
    resolves a role's width from (`Model.roleWeightBits`). The dense Qwen 3.5
    installs are the reason it exists: they declare a 4-bit attention slot but
    store `k_proj`/`v_proj` at 8 bits, and reading those as nibbles yields
    confident nonsense.
    """
    stem = full_name[: -len(".weight")] if full_name.endswith(".weight") else full_name
    slot = (manifest.get("quant") or {}).get(stem)
    if isinstance(slot, dict) and "weightBits" in slot:
        return int(slot["weightBits"])
    return fallback


def load_tensor(handle, entry, weight_bits: int = 4, name: str = "tensor") -> np.ndarray:
    rows, cols = entry["shape"][0], entry["shape"][1]
    if entry["dtype"] == 1:  # bf16
        handle.seek(entry["offset"])
        flat = bf16_to_f32(handle.read(entry["size"]))
        return flat.reshape([d for d in entry["shape"] if d] or [flat.size])
    # The declared width and the stored byte count must agree. They disagree
    # only if a manifest lies about its own payload, which is worth stopping
    # for: the dequantization below is silent about it.
    elements = rows * cols
    expected = elements // 2 if weight_bits == 4 else elements
    if elements and entry["size"] != expected:
        raise SystemExit(
            f"{name}: manifest says {weight_bits}-bit but the tensor is "
            f"{entry['size']} bytes for {elements} elements (expected "
            f"{expected}); refusing to guess its width"
        )
    handle.seek(entry["offset"])
    packed = np.frombuffer(handle.read(entry["size"]), dtype=np.uint8)
    if weight_bits == 8:
        q = packed.reshape(rows, cols).astype(np.float32)
    elif weight_bits == 4:
        packed = packed.reshape(rows, cols // 2)
        q = np.empty((rows, cols), dtype=np.float32)
        q[:, 0::2] = (packed & 0x0F).astype(np.float32)
        q[:, 1::2] = (packed >> 4).astype(np.float32)
    else:
        raise SystemExit(f"{name}: unsupported weightBits {weight_bits}")
    handle.seek(entry["scale"][0])
    scales = bf16_to_f32(handle.read(entry["scale"][1])).reshape(rows, cols // 64)
    handle.seek(entry["bias"][0])
    biases = bf16_to_f32(handle.read(entry["bias"][1])).reshape(rows, cols // 64)
    return q * np.repeat(scales, 64, axis=1) + np.repeat(biases, 64, axis=1)


def load_layer_weights(
    handle, entries, layer: int, geom: Geometry, manifest: dict
) -> dict[str, np.ndarray]:
    prefix = f"{geom.prefix}{layer}.self_attn."
    fallback = int(manifest["quant"]["attention"]["weightBits"])

    def get(name):
        full = prefix + name
        return load_tensor(
            handle,
            entries[full],
            weight_bits=tensor_weight_bits(manifest, full, fallback),
            name=full,
        )

    def norm(*names):
        """A per-head norm, under whichever name this checkpoint uses.

        The families agree on the projection names but not on the norms: the
        qwen36 and dense builds store `q_norm.weight`, the 3.8 build stores
        `q_norm` as a plain bf16 vector. Resolving by what the index actually
        holds keeps a rename a refusal instead of a KeyError halfway through a
        long export.
        """
        for name in names:
            if prefix + name in entries:
                return get(name)
        raise SystemExit(f"{prefix}{names[0]}: the model has none of {', '.join(names)}")

    return {
        "wq": get("q_proj.weight").astype(np.float16),
        "wk": get("k_proj.weight").astype(np.float16),
        "wv": get("v_proj.weight").astype(np.float16),
        "wo": get("o_proj.weight").astype(np.float16),
        "q_norm": norm("q_norm.weight", "q_norm").astype(np.float16),
        "k_norm": norm("k_norm.weight", "k_norm").astype(np.float16),
    }


def rope_tables(start: int, geom: Geometry) -> tuple[np.ndarray, np.ndarray]:
    rotary = geom.rotary
    half = rotary // 2
    inv = geom.theta ** (-np.arange(half, dtype=np.float64) * 2 / rotary)
    pos = np.arange(start, start + geom.chunk, dtype=np.float64)[:, None] * inv[None, :]
    return (np.cos(pos).astype(np.float16), np.sin(pos).astype(np.float16))


def build_variant(history: int, weights: dict[str, np.ndarray], geom: Geometry):
    """One (chunk, history) function for this model's geometry. Inputs are
    token-major so the runtime can wrap its staging buffers zero-copy:
      normed  [chunk, hidden]        post-input-norm hidden
      k_hist  [H, kvDim]             rotated+normed K rows already in the cache
      v_hist  [H, kvDim]
    Outputs, token-major for the same reason:
      out     [chunk, hidden]        attention branch output (pre-residual)
      k_new   [chunk, kvDim]         rotated+normed K of this chunk (cache layout)
      v_new   [chunk, kvDim]
    """
    # The graph body is written in terms of these names; binding them to the
    # model's geometry here is the whole difference between a qwen36 sidecar and
    # a dense one.
    D = geom.hidden
    N_Q_HEADS = geom.q_heads
    N_KV_HEADS = geom.kv_heads
    HEAD_DIM = geom.head_dim
    Q_DIM = geom.q_dim
    KV_DIM = geom.kv_dim
    ROTARY = geom.rotary
    SCALE = geom.scale
    t = geom.chunk
    total = history + t
    cos_np, sin_np = rope_tables(history, geom)
    fp16 = ct.converters.mil.mil.types.fp16
    specs = [mb.TensorSpec(shape=(t, D), dtype=fp16)]
    if history > 0:
        specs += [
            mb.TensorSpec(shape=(history, KV_DIM), dtype=fp16),
            mb.TensorSpec(shape=(history, KV_DIM), dtype=fp16),
        ]
    # The causal mask is an input, not a baked or generated constant: MIL
    # const-folds any constant-shaped fill/band_part chain, and a folded
    # [4096, 8192] fp16 mask is 64 MB per function — it tripled the package.
    # The runtime allocates each variant's mask once and wraps it zero-copy.
    specs += [mb.TensorSpec(shape=(1, 1, t, total), dtype=fp16)]

    def body(normed, k_hist, v_hist, mask):
        def rms_head(x, weight_name):
            sq = mb.mul(x=x, y=x)
            mean = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
            denom = mb.rsqrt(x=mb.add(x=mean, y=np.float16(EPS)))
            return mb.mul(x=mb.mul(x=x, y=denom), y=weights[weight_name].reshape(1, 1, HEAD_DIM))

        def rope(x, heads):
            r1 = mb.slice_by_index(
                x=x,
                begin=[0, 0, 0],
                end=[heads, t, ROTARY // 2],
                begin_mask=[True, True, False],
                end_mask=[True, True, False],
            )
            r2 = mb.slice_by_index(
                x=x,
                begin=[0, 0, ROTARY // 2],
                end=[heads, t, ROTARY],
                begin_mask=[True, True, False],
                end_mask=[True, True, False],
            )
            rest = mb.slice_by_index(
                x=x,
                begin=[0, 0, ROTARY],
                end=[heads, t, HEAD_DIM],
                begin_mask=[True, True, False],
                end_mask=[True, True, True],
            )
            cos_b = cos_np.reshape(1, t, ROTARY // 2)
            sin_b = sin_np.reshape(1, t, ROTARY // 2)
            o1 = mb.sub(x=mb.mul(x=r1, y=cos_b), y=mb.mul(x=r2, y=sin_b))
            o2 = mb.add(x=mb.mul(x=r2, y=cos_b), y=mb.mul(x=r1, y=sin_b))
            return mb.concat(values=[o1, o2, rest], axis=-1)

        packed = mb.matmul(x=normed, y=weights["wq"].T)
        k = mb.matmul(x=normed, y=weights["wk"].T)
        v = mb.matmul(x=normed, y=weights["wv"].T)

        packed_h = mb.reshape(x=packed, shape=[t, N_Q_HEADS, 2 * HEAD_DIM])
        q = mb.slice_by_index(
            x=packed_h,
            begin=[0, 0, 0],
            end=[t, N_Q_HEADS, HEAD_DIM],
            begin_mask=[True, True, False],
            end_mask=[True, True, False],
        )
        gate = mb.slice_by_index(
            x=packed_h,
            begin=[0, 0, HEAD_DIM],
            end=[t, N_Q_HEADS, 2 * HEAD_DIM],
            begin_mask=[True, True, False],
            end_mask=[True, True, True],
        )

        q = mb.transpose(x=q, perm=[1, 0, 2])
        k_h = mb.transpose(x=mb.reshape(x=k, shape=[t, N_KV_HEADS, HEAD_DIM]), perm=[1, 0, 2])
        v_h = mb.transpose(x=mb.reshape(x=v, shape=[t, N_KV_HEADS, HEAD_DIM]), perm=[1, 0, 2])

        q = rope(rms_head(q, "q_norm"), N_Q_HEADS)
        k_h = rope(rms_head(k_h, "k_norm"), N_KV_HEADS)

        # Cache-layout outputs: token-major [t, 512].
        k_new = mb.reshape(x=mb.transpose(x=k_h, perm=[1, 0, 2]), shape=[t, KV_DIM])
        v_new = mb.reshape(x=mb.transpose(x=v_h, perm=[1, 0, 2]), shape=[t, KV_DIM])

        k_cur = mb.reshape(x=k_h, shape=[1, N_KV_HEADS, t, HEAD_DIM])
        v_cur = mb.reshape(x=v_h, shape=[1, N_KV_HEADS, t, HEAD_DIM])
        if history > 0:
            k_hh = mb.reshape(x=k_hist, shape=[history, N_KV_HEADS, HEAD_DIM])
            k_hh = mb.reshape(
                x=mb.transpose(x=k_hh, perm=[1, 0, 2]), shape=[1, N_KV_HEADS, history, HEAD_DIM]
            )
            v_hh = mb.reshape(x=v_hist, shape=[history, N_KV_HEADS, HEAD_DIM])
            v_hh = mb.reshape(
                x=mb.transpose(x=v_hh, perm=[1, 0, 2]), shape=[1, N_KV_HEADS, history, HEAD_DIM]
            )
            k_all = mb.concat(values=[k_hh, k_cur], axis=2)
            v_all = mb.concat(values=[v_hh, v_cur], axis=2)
        else:
            k_all, v_all = k_cur, v_cur

        rep = N_Q_HEADS // N_KV_HEADS

        def gqa_expand(x):
            x5 = mb.reshape(x=x, shape=[1, N_KV_HEADS, 1, total, HEAD_DIM])
            x5 = mb.concat(values=[x5] * rep, axis=2)
            return mb.reshape(x=x5, shape=[1, N_Q_HEADS, total, HEAD_DIM])

        k_g = gqa_expand(k_all)
        v_g = gqa_expand(v_all)

        q4 = mb.reshape(x=q, shape=[1, N_Q_HEADS, t, HEAD_DIM])
        scores = mb.matmul(x=q4, y=k_g, transpose_y=True)
        scores = mb.mul(x=scores, y=np.float16(SCALE))
        scores = mb.add(x=scores, y=mask)
        probs = mb.softmax(x=scores, axis=-1)
        attn = mb.matmul(x=probs, y=v_g)

        gated = mb.mul(
            x=mb.transpose(x=attn, perm=[0, 2, 1, 3]),
            y=mb.sigmoid(x=mb.reshape(x=gate, shape=[1, t, N_Q_HEADS, HEAD_DIM])),
        )
        merged = mb.reshape(x=gated, shape=[t, Q_DIM])
        out = mb.matmul(x=merged, y=weights["wo"].T)
        return out, k_new, v_new

    if history > 0:

        @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
        def prog(normed, k_hist, v_hist, mask):
            return body(normed, k_hist, v_hist, mask)
    else:

        @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
        def prog(normed, mask):
            return body(normed, None, None, mask)

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    # Stable I/O names for the Swift runtime.
    spec = model.get_spec()
    rename = {}
    for out_obj, want in zip(spec.description.output, ("out", "k_new", "v_new"), strict=False):
        rename[out_obj.name] = want
    for old, new in rename.items():
        ct.utils.rename_feature(spec, old, new)
    return ct.models.MLModel(spec, weights_dir=model.weights_dir)


def sidecar_directory(chunk: int) -> str:
    """Where a sidecar for this chunk lives inside a model directory.

    4,096 keeps the historical name, so existing installs and the runtime's
    default lookup are unchanged. Any other width gets its own directory, which
    is what lets one model carry more than one — the width that wins depends on
    the prompt, and the runtime picks the directory matching its configured
    prefill chunk.
    """
    return "ane_prefill" if chunk == CHUNK else f"ane_prefill-{chunk}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to the installed .ssdai directory")
    parser.add_argument(
        "--max-history",
        type=int,
        default=12288,
        help="largest KV history variant (a multiple of "
        "--chunk); prompts beyond max-history+chunk "
        "tokens fall back to the GPU path",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=CHUNK,
        choices=PREFILL_CHUNK_CHOICES,
        help=f"chunk tokens the graph is built for (default "
        f"{CHUNK}; {', '.join(map(str, PREFILL_CHUNK_CHOICES))}). "
        f"The runtime routes a chunk to the sidecar only "
        f"when its configured prefill chunk equals this, "
        f"and a chunk below {CHUNK} is what makes the band "
        f"under {CHUNK} tokens reachable at all",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="comma list of layer indices (default: all full-attention layers)",
    )
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model)
    weights_bin = model_dir / "model_weights.bin"
    if not weights_bin.exists():
        raise SystemExit(f"not a .ssdai directory: {model_dir}")
    if args.max_history % args.chunk != 0:
        raise SystemExit(f"--max-history must be a multiple of --chunk ({args.chunk})")
    histories = list(range(0, args.max_history + 1, args.chunk))

    manifest = json.load(open(model_dir / "manifest.json", encoding="utf-8"))
    entries = read_index(weights_bin)
    # The chunk is the operator's choice here and the runtime's gate later:
    # `eligibleChunk` only routes a chunk to the sidecar when the configured
    # prefill chunk equals this. A 4,096 chunk wins on long prompts (fewer
    # boundaries, fewer per-layer model reloads); a smaller one is what makes
    # the band below 4,096 tokens reachable at all.
    geom = dataclasses.replace(geometry_for(manifest, entries), chunk=args.chunk)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(geom.layers)
    unknown = [L for L in layers if L not in geom.layers]
    if unknown:
        raise SystemExit(
            f"--layers names {unknown}, which are not full-attention layers of "
            f"this model ({list(geom.layers)})"
        )
    print(
        f"{geom.family}: hidden {geom.hidden}, "
        f"{geom.q_heads}q/{geom.kv_heads}kv x {geom.head_dim}, "
        f"rope {geom.rotary}, theta {geom.theta:g}, scale {geom.scale:g}, "
        f"{len(geom.layers)} full-attention layers",
        flush=True,
    )
    # Attention weights are 4-bit in the 4-bit build and 8-bit in the 8-bit
    # build; both dequantize to the same fp16 graph, so only the unpack
    # differs. The sidecar itself is fp16 either way.
    weight_bits = manifest["quant"]["attention"]["weightBits"]
    # A sparse indexer is the model's own geometry, like the head count: it
    # decides whether the runtime must fold a selection into the mask, and the
    # runtime cross-checks this flag against its own configuration.
    selection_folded = int(manifest["arch"].get("indexerBudget") or 0) > 0
    if selection_folded:
        print(
            "sparse indexer: the runtime will fold this model's key "
            "selection into the mask (selectionFolded=true)",
            flush=True,
        )

    out_dir = model_dir / sidecar_directory(args.chunk)
    # Build beside the live sidecar and swap only after every layer has
    # compiled, so a failed export leaves the previous sidecar untouched and
    # never leaves a half-written one for the runtime to load.
    staging = model_dir / f".ane_prefill.export-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    handle = open(weights_bin, "rb")
    try:
        for layer in layers:
            weights = load_layer_weights(handle, entries, layer, geom, manifest)
            stage = staging / f".stage_layer_{layer}"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir()
            desc = ct.utils.MultiFunctionDescriptor()
            for history in histories:
                variant = run_checked(
                    f"layer {layer} h{history} convert",
                    lambda history=history, weights=weights: build_variant(history, weights, geom),
                )
                variant_path = stage / f"h{history}.mlpackage"
                run_checked(
                    f"layer {layer} h{history} save",
                    lambda variant=variant, variant_path=variant_path: variant.save(
                        str(variant_path)
                    ),
                )
                # Checked here, not after the merge: the variant is the package's
                # default function only while it stands alone, and that is what
                # the compute plan reports device assignments for.
                on_ane = run_checked(
                    f"layer {layer} h{history} reaches the ANE",
                    lambda variant_path=variant_path, history=history, layer=layer: (
                        verify_variant_reaches_the_ane(variant_path, history, layer)
                    ),
                )
                if on_ane is not None:
                    print(
                        f"layer {layer}: h{history} — {on_ane} operations on the Neural Engine",
                        flush=True,
                    )
                desc.add_function(
                    str(variant_path), src_function_name="main", target_function_name=f"h{history}"
                )
                print(f"layer {layer}: built h{history}", flush=True)
            desc.default_function_name = "h0"
            final = staging / f"layer_{layer}.mlpackage"
            run_checked(
                f"layer {layer} multifunction",
                lambda desc=desc, final=final: ct.utils.save_multifunction(desc, str(final)),
            )
            # Every recorded history must actually load; see the docstring.
            run_checked(
                f"layer {layer} variants load",
                lambda final=final, histories=histories, layer=layer: verify_variants_load(
                    final, histories, layer
                ),
            )
            shutil.rmtree(stage)
            print(f"layer {layer}: wrote {final}", flush=True)

        meta = {
            "version": EXPORT_VERSION,
            "family": geom.family,
            "sourceWeightBits": weight_bits,
            "chunkTokens": geom.chunk,
            "histories": histories,
            "layers": layers,
            # The geometry the graph was built for. The runtime refuses a
            # sidecar that does not match the model it is loaded for: a
            # mismatch would compute a *different* attention, plausibly.
            "geometry": geom.as_metadata(),
            # Binds the sidecar to the exact weights it was built from. The
            # runtime refuses a mismatch: a sidecar from different weights would
            # compute plausible-looking but wrong attention.
            "weightsSha256": manifest["files"]["model_weights.bin"]["sha256"],
            # The runtime requires this flag. A sidecar that predates it (or
            # was written by a failing export) is refused, so the ~38x
            # CPU-fallback prefill of issue #7 cannot happen.
            "aneCompileVerified": True,
            # True for a family whose full-attention layers select keys with a
            # sparse indexer: the runtime refuses such a model's sidecar unless
            # it records that the selection has to be folded into the mask, so a
            # sidecar built by an exporter that did not know cannot be run with
            # a causal-only mask.
            "selectionFolded": selection_folded,
        }
        with open(staging / "ane_prefill.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        previous = model_dir / f".ane_prefill.previous-{os.getpid()}"
        if previous.exists():
            shutil.rmtree(previous)
        if out_dir.exists():
            os.replace(out_dir, previous)
        os.replace(staging, out_dir)
        if previous.exists():
            shutil.rmtree(previous)
        print(f"wrote {out_dir / 'ane_prefill.json'}")
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        handle.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ANEExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "error: the ANE sidecar was NOT updated; re-run with a smaller "
            "--max-history (8192 exports cleanly) or on a machine whose ANE "
            "accepts the model.",
            file=sys.stderr,
        )
        sys.exit(2)
