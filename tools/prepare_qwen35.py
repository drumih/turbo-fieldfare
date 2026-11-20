#!/usr/bin/env python3.13
"""Convert Qwen3.5's dense text models (2B, 4B) from bf16 into affine snapshots.

These are the small models the CPU engine serves, and the same architecture
family the 35B installs already use: gated-DeltaNet with gated full attention
every fourth layer, a fused output gate in `q_proj`, and rotary confined to
the first 64 of 256 head elements. Three things differ from
`prepare_agentworld.py`, and they are the whole of this file's reason to
exist:

  * **Dense feed-forward.** `mlp.{gate,up,down}_proj` in every layer, where
    the 35B has a router, 256 routed experts and a shared expert. Nothing
    here fuses or splits an expert tensor.
  * **Tied output, except the 9B.** The 2B and 4B ship no `lm_head`; their
    card says the LM output is tied to the token embedding, and the snapshot
    records that rather than duplicating 508M (2B) or 636M (4B) parameters,
    so the reader must honour the tie. The 9B is the vision-language build
    and does ship a root-level `lm_head.weight`, which is carried through as
    `language_model.lm_head.weight` and marked untied. Either way the head
    itself is kept at 8 bits in both build widths.
  * **Streamed.** It is run beside a resident model that holds most of the
    machine's memory, so no tensor is ever held whole: every matrix is read
    and quantized in row chunks, and each output shard's header is planned
    from the source headers first so every chunk is written straight to its
    final offset. Peak RSS is a few hundred megabytes whatever the size;
    the 4B embedding alone is 1.3 GB at bf16 and would be 2.5 GB as the
    float32 the quantizer works in.

The three sizes differ in shape, not kind, and every dimension the runtime
needs is read from the config this writes:

    size  layers  hidden  attention q/kv  delta-rule k/v heads  source shards
    2b      24     2048       8 / 2            16 / 16                1
    4b      32     2560      16 / 4            16 / 32                2
    9b      32     4096      16 / 4            16 / 32                4

The 4B's 32 value heads over 16 key heads is the one shape the 2B never
exercised; the engine's head-sharing tests pin it, and the 9B repeats it at
a wider hidden size rather than adding a fourth shape. The 9B is the
vision-language build: its checkpoint nests the text model under
`text_config` and ships a `model.visual.*` tower that this converter drops,
so it produces the same text-only snapshot as the other two.

Verified against each checkpoint before converting it, not assumed:
`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` and the
final `norm` are stored zero-centred and the +1 is folded in here, exactly as
the 35B converters do. Means over every tensor of each kind, in that order:
2B +0.226, +0.140, +0.498, +0.495, +2.536; 4B +0.183, +0.112, +0.503,
+0.491, +2.195 -- each with minimums near -1, which a stored scale would not
have. `linear_attn.norm` is the gated norm, stored around one (2B mean
+0.915, minimum +0.246; 4B mean +0.961, minimum +0.117) and is left alone.
Getting that backwards was the only bug in the AgentWorld port. (The 2B
figures first recorded here, +0.096 and so on, were single tensors: layer
0's input norm, layer 3's q_norm.)

    python3.13 tools/prepare_qwen35.py --size 4b --plan
    python3.13 tools/prepare_qwen35.py --size 4b --bits 4 8 \\
        --output models/qwen3.5_4B_{bits}Bit --work .build/qwen35-4b-shards
    python3.13 tools/prepare_qwen35.py --size 2b --bits 8 \\
        --output .build/qwen35-2b-affine-8bit --work .build/qwen35-2b-shards

`tools/install_models.sh qwen35-2b|qwen35-4b|qwen35-9b` runs exactly this
and then imports the snapshot, so the result is a verified `.ssdai` install
rather than a bare snapshot the catalog cannot list.

Disk while running: the shard being converted and the one downloading behind
it (4B: 5.3 + 4.0 GB; 9B: 10.6 + 8.0 GB), plus the outputs (4B: 2.7 GB at
4-bit, 4.5 GB at 8; 9B: 6.1 GB at 4-bit, 9.5 GB at 8).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import NamedTuple

try:
    import ml_dtypes
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment, not logic
    sys.exit(
        f"missing dependency: {exc}\n"
        f"  install them for the interpreter running this file: {sys.executable}\n"
        "    -m pip install safetensors numpy ml_dtypes\n"
        "  (or run this file with an interpreter that has them; TINYTITAN_PYTHON\n"
        "   selects the interpreter that tools/install_models.sh runs these\n"
        "   files with, and this file does not read it)"
    )


# NamedTuple, not a dataclass: the precision tools load this file with
# `spec_from_file_location` and never register it in `sys.modules`, and a
# dataclass under postponed annotations looks its module up there and fails.
class Size(NamedTuple):
    repo: str
    # Pinned: the install receipt records the source, and a moved `main` must
    # not silently change what "Qwen3.5-4B 8-bit" means.
    commit: str
    layers: int
    hidden: int
    display_name: str
    # The installed GPU models are named `qwen3.6-35b-a3b_8-Bit`; the CPU
    # ones follow, so a catalog lists both kinds in one style.
    model_id_stem: str
    # 2B and 4B ship no `lm_head` and tie the output to `embed_tokens`; the
    # 9B is the vision-language build and carries a separate root-level
    # `lm_head.weight`. The snapshot records which it is, and the reader
    # honours it, so the head is never duplicated for the tied sizes.
    tied_output: bool


SIZES = {
    "2b": Size(
        "Qwen/Qwen3.5-2B",
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
        24,
        2048,
        "Qwen 3.5 2B",
        "qwen3.5-2b",
        True,
    ),
    "4b": Size(
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        32,
        2560,
        "Qwen 3.5 4B",
        "qwen3.5-4b",
        True,
    ),
    # 9B is the vision-language build: its config nests the text model under
    # `text_config` and carries a `model.visual.*` tower, which `skipped()`
    # drops, so it converts with the same code path as the two text-only
    # sizes. Its text geometry is 4B's attention shape at 9B's width --
    # 16 query heads over 4 key heads, 32 value heads -- so it exercises the
    # same head-sharing the 4B pinned, not a new one.
    "9b": Size(
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        32,
        4096,
        "Qwen 3.5 9B",
        "qwen3.5-9b",
        False,
    ),
}

# Module-level, because the transport and config writers read them and the
# precision tools import this file; `select_size` rebinds all three.
SIZE = SIZES["2b"]
REPO = SIZE.repo
COMMIT = SIZE.commit
BASE = f"https://huggingface.co/{REPO}/resolve/{COMMIT}"


def select_size(key: str) -> None:
    global SIZE, REPO, COMMIT, BASE
    SIZE = SIZES[key]
    REPO, COMMIT = SIZE.repo, SIZE.commit
    BASE = f"https://huggingface.co/{REPO}/resolve/{COMMIT}"


GROUP_SIZE = 64
BITS_4, BITS_8 = 4, 8
OUTPUT_SHARD_BYTES = 4 << 30
# Elements per row chunk: 32 MB as float32. The quantizer's temporaries are a
# handful of these, so this sets the converter's footprint, not the model.
CHUNK_ITEMS = 1 << 23

TOKENIZER_FILES = (
    ("tokenizer.json", True),
    ("tokenizer_config.json", True),
    ("special_tokens_map.json", False),
    ("chat_template.jinja", False),
    ("chat_template.json", False),
    ("generation_config.json", False),
    ("vocab.json", False),
    ("merges.txt", False),
)


# --- quantization ------------------------------------------------------------


def quantize_affine(value: np.ndarray, bits: int) -> tuple[np.ndarray, ...]:
    """Identical to prepare_agentworld.quantize_affine, deliberately.

    Duplicated rather than imported so neither converter can drift silently;
    a change to one must be made in the other.

    Groups run along the last axis, inside a row, so quantizing a matrix a
    block of rows at a time gives the same bytes as quantizing it whole.
    """
    value = value.astype(np.float32)
    if value.shape[-1] % GROUP_SIZE:
        raise ValueError(f"last dimension {value.shape[-1]} is not group-aligned")
    shape = (*value.shape[:-1], value.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    grouped = value.reshape(shape)
    bias = grouped.min(axis=-1)
    high = grouped.max(axis=-1)
    levels = (1 << bits) - 1
    scale = np.where(high == bias, np.float32(1), (high - bias) / levels)
    scale = scale.astype(ml_dtypes.bfloat16)
    bias = bias.astype(ml_dtypes.bfloat16)
    quantized = (
        np.rint(
            (grouped - bias.astype(np.float32)[..., None]) / scale.astype(np.float32)[..., None]
        )
        .clip(0, levels)
        .astype(np.uint32)
        .reshape(value.shape)
    )
    lanes = 32 // bits
    words = quantized.reshape(*quantized.shape[:-1], quantized.shape[-1] // lanes, lanes)
    packed = np.zeros(words.shape[:-1], dtype=np.uint32)
    for lane in range(lanes):
        packed |= words[..., lane] << np.uint32(bits * lane)
    return packed, scale, bias


# --- naming ------------------------------------------------------------------


HEAD_BITS = BITS_8  # --head-bits: the tied embedding slot
WITH_MTP = False  # --mtp: also convert the one-layer draft head

# Tensors carried a width above the build's, because measurement says they
# are worth it (tools/precision_plan_qwen35.py, against the 2B bf16 source):
#
#   kind        err@4    err@8    MB@4   error removed per MB by 4 -> 8
#   v_proj     0.1231   0.0067       3   0.037
#   k_proj     0.1173   0.0067       3   0.035
#   o_proj     0.0967   0.0062      13   0.007
#   q_proj     0.0903   0.0057      25   0.003
#   mlp.*      0.094-0.099          151   0.0006
#
# K and V are the worst at 4-bit and the smallest, so promoting them is an
# order of magnitude the best trade on the board -- six megabytes removes
# 94% of their error. They are also the only projections whose output is
# cached and reused for every later token: with KV heads shared by four
# query heads (2 of 8 in the 2B, 4 of 16 in the 4B), an error there is in
# the context for the rest of the session, where an error in a feed-forward
# is spent on one token. The 4B keeps the rule unmeasured: same kinds, same
# 64-wide groups, and K/V are still under 1% of its bytes (16 MB).
#
# Nothing is promoted above 8-bit. At 8-bit every kind measures 0.006 error
# and 0.99998 direction agreement; bf16 would double the bytes to remove
# what is already three orders below the 4-bit case.
PROMOTE_TO_8BIT = (
    ".self_attn.k_proj",
    ".self_attn.v_proj",
)
PROMOTE = True  # --no-promote: build a uniform-width snapshot


def skipped(name: str) -> bool:
    """The vision tower is 297 of the 2B checkpoint's 632 tensors and this
    snapshot is the text model. The MTP draft head rides along too; it is
    only converted when asked for, because nothing reads it yet."""
    if name.startswith("model.visual."):
        return True
    if name.startswith("mtp."):
        return not WITH_MTP
    return False


def rename(name: str) -> str:
    """Checkpoint name -> the MLX spelling this family is repacked from."""
    if name.startswith("mtp."):
        return "mtp." + name[len("mtp.") :]
    # The untied 9B keeps its output head at the archive root, where the
    # reader's `lmHead` slot expects it; only the 2B and 4B tie it to the
    # embedding. Same mapping prepare_agentworld.py uses.
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "language_model.model." + name[len(prefix) :]
    raise ValueError(f"unexpected tensor outside the language model: {name}")


# Zero-centred RMSNorm: transformers stores gamma - 1 and applies
# (1 + weight); the runtime applies the stored weight as is, so the +1 is
# folded here. Measured on both checkpoints (see the module docstring). The
# gated linear-attention norm is initialised at one and applied plainly, so
# it is not in this list. Reversing these two is the bug that broke the
# AgentWorld port.
UNIT_OFFSET_NORM_SUFFIXES = (
    ".input_layernorm",
    ".post_attention_layernorm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    "language_model.model.norm",
)


def is_mtp_norm(name: str) -> bool:
    """Every norm in the draft head is zero-centred, including
    `pre_fc_norm_embedding` and `pre_fc_norm_hidden`, whose names do not end
    in `norm`."""
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return name.startswith("mtp.") and "norm" in stem.rsplit(".", 1)[-1]


def fold_unit_offset(out_name: str, value: np.ndarray) -> np.ndarray:
    stem = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
    if is_mtp_norm(out_name) or stem.endswith(UNIT_OFFSET_NORM_SUFFIXES):
        return (value.astype(np.float32) + 1.0).astype(value.dtype)
    return value


# Kept at bf16 in both widths. The 35B also keeps its router and shared-expert
# scalar gate; a dense model has neither. What remains is the pair of
# gated-DeltaNet projections with one row per value head (16 in the 2B, 32 in
# the 4B), where a 64-wide group has almost nothing to amortise over.
# Together under 1 MB.
KEEP_BF16 = (
    ".linear_attn.in_proj_a",
    ".linear_attn.in_proj_b",
)


def kept_bf16(name: str) -> bool:
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return stem.endswith(KEEP_BF16)


def quant_bits(name: str, width: int) -> int | None:
    """Bits for a renamed tensor, or None to copy it through at bf16."""
    if not name.endswith(".weight"):
        return None  # A_log, dt_bias
    if name.endswith("conv1d.weight") or name.endswith("norm.weight"):
        return None
    if is_mtp_norm(name):
        return None
    if kept_bf16(name):
        return None
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return HEAD_BITS  # 8-bit: the head, tied or not
    if PROMOTE and width < BITS_8:
        stem = name[: -len(".weight")]
        if stem.endswith(PROMOTE_TO_8BIT):
            return BITS_8
    return width


# --- the plan: every output tensor's place, from the source headers ---------


DTYPES = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32, "U32": np.uint32}


class Planned(NamedTuple):
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def rows(self) -> int:
        return self.shape[0] if self.shape else 1

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * np.dtype(DTYPES[self.dtype]).itemsize


def planned(name: str, entry: dict, width: int) -> list[Planned]:
    """What one source tensor becomes at one width, without reading it.

    The shapes follow from the name, the source shape and the width alone,
    which is what lets every header be written before any data."""
    out = rename(name)
    bits = quant_bits(out, width)
    if bits is None:
        return [Planned(out, entry["dtype"], tuple(entry["shape"]))]
    rows, columns = entry["shape"]
    if columns % GROUP_SIZE:
        raise ValueError(f"{name}: {columns} columns is not group-aligned")
    stem = out[: -len(".weight")]
    groups = (rows, columns // GROUP_SIZE)
    return [
        Planned(stem + ".weight", "U32", (rows, columns * bits // 32)),
        Planned(stem + ".scales", "BF16", groups),
        Planned(stem + ".biases", "BF16", groups),
    ]


def parse_header(raw: bytes) -> dict:
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header


def in_file_order(header: dict) -> list[str]:
    """Converted in the order they sit on disk, so the reads are sequential."""
    return sorted(header, key=lambda name: header[name]["data_offsets"][0])


def file_size(header_length: int, header: dict) -> int:
    return 8 + header_length + max((e["data_offsets"][1] for e in header.values()), default=0)


# --- source ------------------------------------------------------------------


class SourceShard:
    """One checkpoint shard, read with plain reads into buffers this owns.

    Not a mapping, which is what `safe_open` gives: every page a mapping
    touches is charged to this process until it is unmapped, so walking a
    5 GB shard through one reports -- and for a while holds -- 5 GB, beside a
    resident model that has none to spare. `F_NOCACHE` keeps the reads out
    of the page cache as well, where bytes read exactly once would evict the
    resident model's pages.
    """

    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, "rb", buffering=0)
        fcntl.fcntl(self.file.fileno(), fcntl.F_NOCACHE, 1)
        (length,) = struct.unpack("<Q", self._bytes(0, 8).tobytes())
        self.header = parse_header(self._bytes(8, length).tobytes())
        self.payload = 8 + length

    def close(self) -> None:
        self.file.close()

    def _bytes(self, offset: int, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.uint8)
        view = memoryview(out)
        self.file.seek(offset)
        done = 0
        while done < count:
            got = self.file.readinto(view[done:])
            if not got:
                raise EOFError(f"{self.path.name}: short read at {offset + done}")
            done += got
        return out

    def rows(self, name: str, first: int, stop: int) -> np.ndarray:
        entry = self.header[name]
        dtype = np.dtype(DTYPES[entry["dtype"]])
        inner = tuple(entry["shape"][1:])
        row_bytes = math.prod(inner) * dtype.itemsize
        raw = self._bytes(
            self.payload + entry["data_offsets"][0] + first * row_bytes, (stop - first) * row_bytes
        )
        return raw.view(dtype).reshape((stop - first, *inner) if entry["shape"] else ())


# --- output ------------------------------------------------------------------


class SnapshotWriter:
    """Writes a snapshot whose every tensor was placed before conversion.

    Each output shard's header is written first and the file sized to its
    final length, then every row chunk is `pwrite`n to its own offset.
    Nothing accumulates in memory -- the previous writer held up to 4 GB of
    converted tensors before flushing a shard -- and nothing is rewritten.

    A source tensor's outputs never straddle a shard boundary: the reader
    looks a matrix's scales and biases up in the shard that holds its
    weight, and a byte-count split between `.weight` and `.scales` would
    produce a snapshot that fails to load.
    """

    def __init__(self, out: Path, groups: list[list[Planned]]):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise SystemExit(f"{out} is not empty; refusing to mix two snapshots")
        shards: list[list[Planned]] = [[]]
        filled = 0
        for group in groups:
            if filled >= OUTPUT_SHARD_BYTES:
                shards.append([])
                filled = 0
            shards[-1].extend(group)
            filled += sum(p.nbytes for p in group)
        self.index: dict[str, str] = {}
        self.slots: dict[str, tuple[int, int, Planned]] = {}
        self.written: dict[str, int] = {}
        self.files: list[int] = []
        self.total = 0
        for number, members in enumerate(shards, start=1):
            name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
            header, cursor = {}, 0
            for p in members:
                header[p.name] = {
                    "dtype": p.dtype,
                    "shape": list(p.shape),
                    "data_offsets": [cursor, cursor + p.nbytes],
                }
                cursor += p.nbytes
            raw = json.dumps(header, separators=(",", ":")).encode()
            raw += b" " * (-len(raw) % 8)  # payload 8-aligned, as safetensors pads
            fd = os.open(out / name, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self._pwrite(fd, struct.pack("<Q", len(raw)) + raw, 0)
            os.ftruncate(fd, 8 + len(raw) + cursor)
            self.files.append(fd)
            for p in members:
                start = 8 + len(raw) + header[p.name]["data_offsets"][0]
                self.slots[p.name] = (fd, start, p)
                self.written[p.name] = 0
                self.index[p.name] = name
            self.total += cursor

    @staticmethod
    def _pwrite(fd: int, data, offset: int) -> None:
        view = memoryview(data).cast("B")
        done = 0
        while done < len(view):
            done += os.pwrite(fd, view[done:], offset + done)

    def write(self, name: str, first_row: int, value: np.ndarray) -> None:
        fd, start, p = self.slots[name]
        if np.dtype(value.dtype) != np.dtype(DTYPES[p.dtype]):
            raise TypeError(f"{name}: planned {p.dtype}, got {value.dtype}")
        row_bytes = p.nbytes // p.rows
        data = np.ascontiguousarray(value).reshape(-1).view(np.uint8)
        offset = first_row * row_bytes
        if offset + data.nbytes > p.nbytes:
            raise ValueError(f"{name}: rows from {first_row} overrun the tensor")
        self._pwrite(fd, data, start + offset)
        self.written[name] += data.nbytes

    def finish(self) -> None:
        short = [n for n, (_, _, p) in self.slots.items() if self.written[n] != p.nbytes]
        if short:
            raise RuntimeError(f"{len(short)} tensors incompletely written, e.g. {short[0]}")
        for fd in self.files:
            os.fsync(fd)
            os.close(fd)
        (self.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": self.total}, "weight_map": self.index}, indent=1)
        )


def convert_shard(src: SourceShard, names: list[str], writers: dict[int, SnapshotWriter]) -> None:
    """One source shard into every requested width; each row block is read
    once and quantized for all of them."""
    for name in names:
        out_name = rename(name)
        shape = src.header[name]["shape"]
        rows = shape[0] if shape else 1
        step = max(1, CHUNK_ITEMS // max(1, math.prod(shape[1:])))
        for first in range(0, rows, step):
            stop = min(rows, first + step)
            piece = src.rows(name, first, stop)
            for width, writer in writers.items():
                bits = quant_bits(out_name, width)
                if bits is None:
                    writer.write(out_name, first, fold_unit_offset(out_name, piece))
                    continue
                stem = out_name[: -len(".weight")]
                packed, scales, biases = quantize_affine(piece, bits)
                writer.write(stem + ".weight", first, packed)
                writer.write(stem + ".scales", first, scales)
                writer.write(stem + ".biases", first, biases)


# --- config ------------------------------------------------------------------


def write_config(config: dict, out: Path, tensor_names, width: int) -> dict:
    """config.json with the `quantization` block the reader wants: a base
    width plus every tensor whose width differs, keyed by stem.

    The text config is lifted to the top level, because the checkpoint nests
    it under a vision-language wrapper this snapshot does not carry. The tie
    is recorded rather than resolved: the reader uses the embedding as the
    output projection, and a reader that cannot must be told, not silently
    handed the embedding's parameters twice. The 9B is the untied case -- it
    ships its own `lm_head.weight`, carried through as
    `language_model.lm_head.weight` -- so the flag is written from the size
    rather than hard-coded.

    `model_id` and `display_name` make the directory a catalog entry on its
    own: a server finds a CPU model as a directory under `models/` whose
    config names a family the CPU engine serves and carries this block.
    """
    text = {
        "model_id": f"{SIZE.model_id_stem}_{width}-Bit",
        "display_name": SIZE.display_name,
        **config.get("text_config", config),
    }
    text["model_type"] = "qwen3_5_dense"
    text["architectures"] = ["Qwen3_5DenseForCausalLM"]
    text["tie_word_embeddings"] = SIZE.tied_output
    text["source_repo"] = REPO
    text["source_commit"] = COMMIT
    # Written at the top level, where the runtime reads them. The checkpoint
    # nests both under `rope_parameters`; a reader that looks only at the
    # top level would otherwise inherit whatever default its library has,
    # and a snapshot whose rotation depends on which version reads it is
    # wrong in a way position 0 cannot show -- the rotation is the identity
    # there whatever the constants are, so they are checked by sequence
    # parity instead.
    #
    # The fallbacks match this project's Qwen 3.6 install manifest
    # (`arch.ropeTheta` 10000000, `arch.partialRotaryFactor` 0.25), the same
    # architecture family.
    rope = text.get("rope_parameters") or {}
    lifted = []
    for key, fallback in (("rope_theta", 10_000_000.0), ("partial_rotary_factor", 0.25)):
        if key in rope:
            lifted.append(key)
        text.setdefault(key, float(rope.get(key, fallback)))
    text["rope_constants_source"] = (
        f"text_config.rope_parameters of {REPO}@{COMMIT[:7]}"
        if len(lifted) == 2
        else f"architecture default; absent from {REPO}@{COMMIT[:7]}/config.json"
    )
    overrides = {}
    for name in tensor_names:
        if not name.endswith(".weight"):
            continue
        bits = quant_bits(name, width)
        if bits is None or bits == width:
            continue
        overrides[name[: -len(".weight")]] = {"bits": bits, "group_size": GROUP_SIZE}
    text["quantization"] = {
        "bits": width,
        "group_size": GROUP_SIZE,
        "mode": "affine",
        **overrides,
    }
    (out / "config.json").write_text(json.dumps(text, indent=1))
    return text


# --- transport ---------------------------------------------------------------


# Small fetches retry like the shard download does; a single TLS hiccup on
# the index fetch ended one 70 GB build before it started.
RETRY = ["--retry", "5", "--retry-delay", "5", "--retry-all-errors"]


def fetch(remote: str, byte_range: str | None = None) -> bytes:
    ranged = ["-r", byte_range] if byte_range else []
    return subprocess.run(
        ["curl", "-sfL", "--max-time", "120", *RETRY, *ranged, f"{BASE}/{remote}"],
        capture_output=True,
        check=True,
    ).stdout


def fetch_json(remote: str) -> dict:
    return json.loads(fetch(remote))


def remote_header(shard: str) -> tuple[dict, int]:
    """A shard's header by two range requests: shapes and dtypes for the
    plan, and the byte length a finished download must have."""
    (length,) = struct.unpack("<Q", fetch(shard, "0-7"))
    raw = fetch(shard, f"8-{7 + length}")
    if len(raw) != length:
        raise RuntimeError(f"{shard}: header fetch returned {len(raw)} of {length} bytes")
    header = parse_header(raw)
    return header, file_size(length, header)


_download: subprocess.Popen | None = None


SAMPLE_BYTES = 8192


def content_probe(path: Path, shard: str, expected_size: int | None = None) -> None:
    """Refuse a size-correct shard whose bytes are empty or garbage.

    curl's exit status and the byte count were the only checks a download used
    to make, and a filesystem hole or a zeroed write passes both. The
    2026-11-16 2B installs shipped zeroed payloads through exactly that gap
    (TT-098), so the file is sampled before it is trusted: the safetensors
    header must name a plausible JSON length and at least 1% of the payload
    samples must be nonzero.
    """
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(
            f"download content check: {shard} is {size} bytes, expected {expected_size}"
        )
    if size < 8:
        raise RuntimeError(f"download content check: {shard} is {size} bytes")
    with open(path, "rb") as handle:
        header = handle.read(8)
        header_bytes = int.from_bytes(header, "little")
        if not 0 < header_bytes < min(size - 8, 100_000_000):
            raise RuntimeError(
                f"download content check: {shard} has no safetensors header "
                f"({header_bytes} bytes named)"
            )
        step = max(1, (size - 8) // SAMPLE_BYTES)
        total = nonzero = 0
        handle.seek(8)
        for offset in range(8, size, step):
            handle.seek(offset)
            byte = handle.read(1)
            total += 1
            if byte not in (b"\x00", b""):
                nonzero += 1
    if total == 0 or nonzero * 100 < total:
        raise RuntimeError(
            f"download content check: {shard} is {nonzero}/{total} nonzero — "
            "a zeroed or hole-y download was refused"
        )


def download(shard: str, work: Path, size: int) -> Path:
    """Fetched under a `.part` name and renamed when whole, so an interrupted
    run never leaves a truncated shard that the next one would trust."""
    global _download
    target = work / shard
    if target.exists() and target.stat().st_size == size:
        content_probe(target, shard, size)
        print(f"  have {shard}", flush=True)
        return target
    partial = work / (shard + ".part")
    print(f"  fetching {shard}", flush=True)
    started = time.time()
    _download = subprocess.Popen(["curl", "-sfL", *RETRY, "-o", str(partial), f"{BASE}/{shard}"])
    if _download.wait() != 0 or partial.stat().st_size != size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {shard}")
    try:
        content_probe(partial, shard, size)
    except RuntimeError:
        partial.unlink(missing_ok=True)
        raise
    _download = None
    partial.rename(target)
    print(f"    {size / 1e9:.2f} GB in {time.time() - started:.0f}s", flush=True)
    return target


def stop_download() -> None:
    if _download and _download.poll() is None:
        _download.terminate()


def fetch_tokenizer(out: Path) -> None:
    for name, required in TOKENIZER_FILES:
        done = subprocess.run(
            ["curl", "-sfL", "--max-time", "300", *RETRY, "-o", str(out / name), f"{BASE}/{name}"],
            check=False,
        )
        if done.returncode != 0:
            (out / name).unlink(missing_ok=True)
            if required:
                raise RuntimeError(f"could not fetch {name}")


# --- driver ------------------------------------------------------------------


def plan(groups: list[list[Planned]], width: int) -> None:
    """Classify every tensor and size the output from the headers alone: no
    download, no memory."""
    kinds: dict[str, int] = {}
    for group in groups:
        head = group[0]
        kind = (
            "bf16"
            if len(group) == 1
            else (f"{32 * head.shape[1] // (group[1].shape[1] * GROUP_SIZE)}-bit")
        )
        kinds[kind] = kinds.get(kind, 0) + 1
    total = sum(p.nbytes for group in groups for p in group)
    print(f"\n{width}-bit plan for {REPO} @ {COMMIT[:8]}")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:9s} {count:4d} tensors")
    print(f"  {'converted':9s} {len(groups):4d} tensors, {total / 1e9:.2f} GB")


def output_path(output: Path, width: int, widths: list[int]) -> Path:
    if "{bits}" in str(output):
        return Path(str(output).replace("{bits}", str(width)))
    return output if len(widths) == 1 else Path(f"{output}-{width}bit")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--size",
        choices=sorted(SIZES),
        required=True,
        help="which checkpoint: "
        + ", ".join(f"{k} = {v.repo}@{v.commit[:7]}" for k, v in SIZES.items()),
    )
    ap.add_argument(
        "--head-bits",
        type=int,
        choices=(4, 8),
        default=8,
        help="tied embedding width (default 8; it is 248320 rows and "
        "serves as both the embedding and the output head)",
    )
    ap.add_argument(
        "--no-promote",
        action="store_true",
        help="uniform build width; by default a 4-bit snapshot keeps "
        "k_proj and v_proj at 8-bit, which removes 94%% of their "
        "error for under 1%% of the bytes (tools/precision_plan_qwen35.py)",
    )
    ap.add_argument(
        "--mtp", action="store_true", help="also convert the one-layer mtp.* draft head"
    )
    ap.add_argument(
        "--plan",
        action="store_true",
        help="classify and size from the shard headers, download nothing",
    )
    ap.add_argument(
        "--bits",
        type=int,
        choices=(4, 8),
        nargs="+",
        default=[8],
        help="one width, or both to write two snapshots from one download",
    )
    ap.add_argument(
        "--output",
        type=Path,
        help="snapshot directory; with two widths, a path containing "
        "{bits} (models/qwen3.5_4B_{bits}Bit) or a prefix that "
        "gets -4bit/-8bit",
    )
    ap.add_argument("--work", type=Path, help="scratch for in-flight shards")
    args = ap.parse_args(argv)
    global HEAD_BITS, WITH_MTP, PROMOTE
    select_size(args.size)
    HEAD_BITS = args.head_bits
    WITH_MTP = args.mtp
    PROMOTE = not args.no_promote

    config = fetch_json("config.json")
    if config.get("model_type") != "qwen3_5":
        raise SystemExit(f"unexpected model_type {config.get('model_type')!r}")
    text = config.get("text_config", {})
    if (text.get("num_hidden_layers"), text.get("hidden_size")) != (SIZE.layers, SIZE.hidden):
        raise SystemExit(
            f"unexpected geometry for {args.size}: "
            f"{text.get('num_hidden_layers')} layers of {text.get('hidden_size')}"
        )
    index = fetch_json("model.safetensors.index.json")
    shards = sorted({s for n, s in index["weight_map"].items() if not skipped(n)})
    headers = {shard: remote_header(shard) for shard in shards}
    order = {
        shard: [n for n in in_file_order(headers[shard][0]) if not skipped(n)] for shard in shards
    }
    listed = {n for n in index["weight_map"] if not skipped(n)}
    found = {n for names in order.values() for n in names}
    if listed != found:
        raise SystemExit(f"index and shard headers disagree on {len(listed ^ found)} tensors")
    widths = sorted(set(args.bits))
    groups = {
        w: [planned(n, headers[s][0][n], w) for s in shards for n in order[s]] for w in widths
    }
    if args.plan:
        for width in widths:
            plan(groups[width], width)
        return 0
    if not args.output or not args.work:
        ap.error("--output and --work are required unless --plan")
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    outputs = {w: output_path(args.output, w, widths) for w in widths}
    writers = {w: SnapshotWriter(outputs[w], groups[w]) for w in widths}

    queue: Queue = Queue(maxsize=1)

    def fetcher() -> None:
        for shard in shards:
            try:
                queue.put((shard, download(shard, work, headers[shard][1])))
            except Exception as exc:  # noqa: BLE001
                queue.put(exc)
                return
        queue.put(None)

    def interrupted(*_: object) -> None:
        stop_download()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    thread = threading.Thread(target=fetcher, daemon=True)
    thread.start()
    started = time.time()
    try:
        while True:
            item = queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            shard, path = item
            print(f"  converting {shard}", flush=True)
            src = SourceShard(path)
            if src.header != headers[shard][0]:
                raise SystemExit(f"{shard} on disk is not the shard that was planned")
            convert_shard(src, order[shard], writers)
            src.close()
            path.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    for width, writer in writers.items():
        writer.finish()
        out = outputs[width]
        written = write_config(config, out, writer.index.keys(), width)
        fetch_tokenizer(out)
        print(
            f"  {out}: {written['model_id']}, {writer.total / 1e9:.2f} GB, "
            f"{len(writer.index)} tensors"
        )
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
