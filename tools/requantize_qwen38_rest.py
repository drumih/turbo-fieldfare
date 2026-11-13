#!/usr/bin/env python3
"""Build a Qwen3.8 install with the non-expert weights at a new width, locally.

    python3 tools/requantize_qwen38_rest.py \\
        --base   models/qwen3.8-flash-next_125B_A6B_4Bit \\
        --source models/qwen3.8-flash-next_125B_A6B_4Bit_BF16Rest \\
        --output models/qwen3.8-flash-next_125B_A6B_4E-8R-16N --bits 8

No download. The two existing installs already hold everything:

- `--base` (the 4-bit install) supplies the routed experts, the n-gram table,
  the tokenizer and every unquantised resident tensor. Its large files are
  hardlinked, so the new install costs only its own `model_weights.bin`.
- `--source` (BF16Rest) holds every quantised resident tensor of the base at
  the checkpoint's own bf16, which is the converter's input. Each one is
  quantised again with `prepare_qwen38.quantize_affine` -- the same function,
  so the bytes are what a fresh `--rest-8bit` conversion would write.

Before writing anything, every tensor is self-checked: re-quantising its bf16
rows at the base's own width must reproduce the base's bytes exactly. That
proves the bf16 data is the converter's input with no transform in between; a
mismatch stops the run with nothing written.

Only the resident file and the manifest are new. Run
`TinyTitanRepack --verify-install --input-ssdai <output>` afterwards: it hashes
every file and writes the receipt the runtime requires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_qwen38 as prepare  # noqa: E402

RESIDENT = "model_weights.bin"
HEADER_BYTES = 24
ENTRY_BYTES = 72
INDEX_ALIGNMENT = 16_384
DTYPE_U32, DTYPE_BF16 = 0, 1
GROUP = prepare.GROUP_SIZE
SLOTS = ("embedding", "attention", "router", "sharedExpert", "routedExpert")
# Files at least this large are hardlinked rather than copied.
LINK_MIN_BYTES = 64 * 1024 * 1024
# Rows quantised per step: bounds the float32 working set of the 248k-row head.
ROW_CHUNK = 8192
# Rows compared per tensor in the self-check: the first and the last block.
CHECK_ROWS = 512


@dataclass(frozen=True)
class Entry:
    name: str
    dtype: int
    offset: int
    size: int
    shape: tuple[int, int, int, int]
    scale_offset: int
    scale_size: int
    bias_offset: int
    bias_size: int


@dataclass
class Resident:
    path: Path
    index_size: int
    resident_size: int
    entries: list[Entry]

    def by_name(self) -> dict[str, Entry]:
        return {e.name: e for e in self.entries}


Planned = tuple[Entry, tuple[int, ...], Callable[[], list[bytes]]]


def read_resident(path: Path) -> Resident:
    with path.open("rb") as f:
        index_size, resident_size, count = struct.unpack("<QQQ", f.read(HEADER_BYTES))
        f.seek(0)
        index = f.read(index_size)
    entries = []
    for i in range(count):
        o = HEADER_BYTES + i * ENTRY_BYTES
        name_off, name_len, dtype, reserved = struct.unpack_from("<IHBB", index, o)
        if reserved:
            raise SystemExit(f"{path}: entry {i} has a non-zero reserved byte")
        offset, size = struct.unpack_from("<QQ", index, o + 8)
        shape = struct.unpack_from("<4I", index, o + 24)
        so, ss, bo, bs = struct.unpack_from("<QQQQ", index, o + 40)
        name = index[name_off : name_off + name_len].decode()
        entries.append(Entry(name, dtype, offset, size, shape, so, ss, bo, bs))
    actual = path.stat().st_size
    if index_size + resident_size != actual:
        raise SystemExit(
            f"{path}: index {index_size} + payload {resident_size} != file size {actual}; "
            "this tool's reading of the resident layout does not match the install"
        )
    return Resident(path, index_size, resident_size, entries)


def payload_alignment(resident: Resident) -> int:
    """The largest power of two (<= 16 KiB) every payload offset honours."""
    offsets = [resident.index_size]
    for e in resident.entries:
        offsets += [e.offset] + [o for o in (e.scale_offset, e.bias_offset) if o]
    align = INDEX_ALIGNMENT
    while align > 1 and any(o % align for o in offsets):
        align //= 2
    return align


def read_bytes(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(size)
    if len(data) != size:
        raise SystemExit(f"{path}: short read at {offset}")
    return data


def bf16_rows(source: Resident, entry: Entry) -> np.ndarray:
    rows, cols = entry.shape[0], entry.shape[1]
    if entry.dtype != DTYPE_BF16 or entry.shape[2:] != (0, 0) or entry.size != rows * cols * 2:
        raise SystemExit(f"{entry.name}: expected a 2-D bf16 tensor in --source, got {entry}")
    return np.memmap(
        source.path, dtype=ml_dtypes.bfloat16, mode="r", offset=entry.offset, shape=(rows, cols)
    )


def width_of(entry: Entry, cols: int) -> int:
    """The base entry's bit width, from its payload size and the bf16 shape.

    Installs record a quantised tensor's *logical* shape ([rows, cols], as the
    bf16 tensor has it); the width is what the packed payload's size implies.
    The packed convention ([rows, cols * bits / 32]) is accepted as well.
    """
    rows = entry.shape[0]
    bits, rem = divmod(entry.size * 8, rows * cols)
    if rem or bits not in (4, 8):
        raise SystemExit(f"{entry.name}: {entry.size} bytes is not a 4- or 8-bit {rows}x{cols}")
    if entry.shape[1] not in (cols, cols * bits // 32):
        raise SystemExit(f"{entry.name}: shape {entry.shape} matches neither convention")
    groups = rows * (cols // GROUP) * 2
    if entry.scale_size != groups or entry.bias_size != groups:
        raise SystemExit(f"{entry.name}: scale/bias sizes do not match {rows}x{cols}")
    return bits


def shape_at(entry: Entry, cols: int, old: int, bits: int) -> tuple[int, int, int, int]:
    """The entry's shape at the new width, in whichever convention it uses."""
    if entry.shape[1] == cols:
        return entry.shape
    return (entry.shape[0], cols * bits // 32, *entry.shape[2:])


def quantize_rows(value: np.ndarray, bits: int) -> tuple[bytes, bytes, bytes]:
    packed, scales, biases = [], [], []
    for start in range(0, value.shape[0], ROW_CHUNK):
        p, s, b = prepare.quantize_affine(np.asarray(value[start : start + ROW_CHUNK]), bits)
        packed.append(p.tobytes())
        scales.append(s.tobytes())
        biases.append(b.tobytes())
    return b"".join(packed), b"".join(scales), b"".join(biases)


def self_check(base: Resident, entry: Entry, value: np.ndarray, bits: int) -> None:
    rows, cols = value.shape
    word_bytes = cols * bits // 8
    group_bytes = cols // GROUP * 2
    blocks = {0, max(0, rows - CHECK_ROWS)}
    for start in sorted(blocks):
        stop = min(rows, start + CHECK_ROWS)
        p, s, b = prepare.quantize_affine(np.asarray(value[start:stop]), bits)
        n = stop - start
        expect = (
            read_bytes(base.path, entry.offset + start * word_bytes, n * word_bytes),
            read_bytes(base.path, entry.scale_offset + start * group_bytes, n * group_bytes),
            read_bytes(base.path, entry.bias_offset + start * group_bytes, n * group_bytes),
        )
        if (p.tobytes(), s.tobytes(), b.tobytes()) != expect:
            raise SystemExit(
                f"self-check failed: {entry.name} rows {start}-{stop} re-quantised at "
                f"{bits} bits from --source do not reproduce --base. The bf16 tensor is "
                "not the converter's input as-is; nothing was written."
            )


class ResidentWriter:
    """Writes the index first, then streams each tensor's blobs from `produce`.

    Sizes are known up front (they follow from the shapes), so the layout is
    fixed before any tensor is quantised and only one tensor is in memory.
    """

    def __init__(self, path: Path, items: list[Planned], align: int):
        self.path, self.items, self.align = path, items, align

    def write(self) -> tuple[int, str]:
        names = b""
        name_offsets = []
        table_end = HEADER_BYTES + ENTRY_BYTES * len(self.items)
        for entry, _, _ in self.items:
            name_offsets.append(table_end + len(names))
            names += entry.name.encode()
        index_size = -(-(table_end + len(names)) // INDEX_ALIGNMENT) * INDEX_ALIGNMENT
        cursor = index_size
        placed = []
        for _, sizes, _ in self.items:
            offsets = []
            for size in sizes:
                cursor = -(-cursor // self.align) * self.align
                offsets.append(cursor)
                cursor += size
            placed.append(offsets)
        total = cursor
        index = bytearray(index_size)
        struct.pack_into("<QQQ", index, 0, index_size, total - index_size, len(self.items))
        for i, ((entry, blob_sizes, _), offsets) in enumerate(zip(self.items, placed, strict=True)):
            sizes = list(blob_sizes) + [0, 0]
            offs = offsets + [0, 0]
            struct.pack_into(
                "<IHBB",
                index,
                HEADER_BYTES + i * ENTRY_BYTES,
                name_offsets[i],
                len(entry.name.encode()),
                entry.dtype,
                0,
            )
            struct.pack_into(
                "<QQ4IQQQQ",
                index,
                HEADER_BYTES + i * ENTRY_BYTES + 8,
                offs[0],
                sizes[0],
                *entry.shape,
                offs[1],
                sizes[1],
                offs[2],
                sizes[2],
            )
        index[table_end : table_end + len(names)] = names
        digest = hashlib.sha256()
        tmp = self.path.with_suffix(".bin.partial")
        with tmp.open("wb") as f:
            f.write(index)
            digest.update(index)
            position = index_size
            for (entry, sizes, produce), offsets in zip(self.items, placed, strict=True):
                blobs = produce()
                if [len(b) for b in blobs] != list(sizes):
                    raise SystemExit(f"{entry.name}: produced sizes differ from the plan")
                for blob, offset in zip(blobs, offsets, strict=True):
                    pad = b"\0" * (offset - position)
                    f.write(pad)
                    digest.update(pad)
                    f.write(blob)
                    digest.update(blob)
                    position = offset + len(blob)
            f.flush()
            os.fsync(f.fileno())
        tmp.rename(self.path)
        return total, digest.hexdigest()


def plan_entries(base: Resident, source: Resident, bits: int) -> list[Planned]:
    """Self-check every quantised tensor and fix the output layout.

    Nothing is written here: a self-check failure stops the run before the
    output directory exists.
    """
    src = source.by_name()
    out = []
    quantised = changed = 0
    for entry in base.entries:
        if entry.dtype != DTYPE_U32:
            if entry.scale_size or entry.bias_size:
                raise SystemExit(f"{entry.name}: unquantised entry carries scales")
            out.append(
                (
                    entry,
                    (entry.size,),
                    lambda e=entry: [read_bytes(base.path, e.offset, e.size)],
                )
            )
            continue
        if entry.name not in src:
            raise SystemExit(f"{entry.name}: missing from --source")
        value = bf16_rows(source, src[entry.name])
        rows, cols = value.shape
        if rows != entry.shape[0]:
            raise SystemExit(f"{entry.name}: {rows} rows in --source, {entry.shape[0]} in --base")
        old = width_of(entry, cols)
        self_check(base, entry, value, old)
        new = replace(entry, shape=shape_at(entry, cols, old, bits))
        groups = rows * (cols // GROUP) * 2
        out.append(
            (
                new,
                (rows * cols * bits // 8, groups, groups),
                lambda v=value: list(quantize_rows(v, bits)),
            )
        )
        quantised += 1
        changed += old != bits
    print(f"{quantised} quantised tensors self-checked against --base, {changed} change width")
    return out


def rewrite_manifest(manifest: dict, bits: int, size: int, sha: str, model_id: str) -> dict:
    quant = manifest["quant"]
    routed = quant["routedExpert"]["weightBits"]
    for slot in SLOTS:
        if slot != "routedExpert":
            quant[slot]["weightBits"] = bits
    for key, value in quant.items():
        if key in SLOTS or key == "promoted" or not isinstance(value, dict):
            continue
        if ".switch_mlp." not in key:
            value["weightBits"] = bits
    quant.pop("promoted", None)
    quant["routedExpert"]["weightBits"] = routed
    manifest["files"][RESIDENT] = {"size": size, "sha256": sha}
    manifest["modelID"] = model_id
    return manifest


def populate(base: Path, output: Path) -> None:
    """Hardlink the base's large files and copy its small ones into `output`.

    A file already present (for example experts or the n-gram table linked in
    by hand beforehand) is accepted only when it is the base's own file.
    """
    for item in sorted(base.rglob("*")):
        rel = item.relative_to(base)
        if rel.parts[0] in (RESIDENT, "manifest.json", "verified-install.json"):
            continue
        target = output / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif target.exists():
            if not (os.path.samefile(item, target) or same_bytes(item, target)):
                raise SystemExit(f"{target} exists and is not {item}")
        elif item.stat().st_size >= LINK_MIN_BYTES:
            os.link(item, target)
        else:
            shutil.copy2(item, target)


def same_bytes(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size or a.stat().st_size >= LINK_MIN_BYTES:
        return False
    return a.read_bytes() == b.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bits", type=int, choices=(4, 8), default=8)
    ap.add_argument("--model-id-suffix", default=None)
    args = ap.parse_args()

    ours = [args.output / n for n in (RESIDENT, "manifest.json", "verified-install.json")]
    if any(p.exists() for p in ours):
        raise SystemExit(f"{args.output} already holds an install; remove it first")
    created = not args.output.exists()
    base = read_resident(args.base / RESIDENT)
    source = read_resident(args.source / RESIDENT)
    align = payload_alignment(base)
    print(f"base {len(base.entries)} resident tensors, payload alignment {align} B")

    entries = plan_entries(base, source, args.bits)

    args.output.mkdir(parents=True, exist_ok=True)
    try:
        populate(args.base, args.output)
        size, sha = ResidentWriter(args.output / RESIDENT, entries, align).write()
        manifest = json.loads((args.base / "manifest.json").read_text())
        model_id = manifest["modelID"] + (args.model_id_suffix or f"-{args.bits}bitrest")
        manifest = rewrite_manifest(manifest, args.bits, size, sha, model_id)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    except BaseException:
        if created:
            shutil.rmtree(args.output, ignore_errors=True)
        else:
            for path in ours + [args.output / (RESIDENT[:-4] + ".bin.partial")]:
                path.unlink(missing_ok=True)
        raise
    print(f"wrote {args.output}: {RESIDENT} {size / 1e9:.2f} GB, modelID {model_id}")
    print(f"next: TinyTitanRepack --verify-install --input-ssdai {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
