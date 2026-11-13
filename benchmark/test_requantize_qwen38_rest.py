"""tools/requantize_qwen38_rest.py on a toy pair of installs.

The toy is two resident files the tool's own writer lays out: a 4-bit base and
a bf16 source holding the same values. No model, no install of a real family.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

try:
    import ml_dtypes
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import prepare_qwen38 as prepare
    import requantize_qwen38_rest as rq

    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - CI installs the deps
    rq = None
    IMPORT_ERROR = exc

QUANTISED = (
    ("model.language_model.layers.0.self_attn.q_proj.weight", 128, 256, 4),
    ("model.language_model.embed_tokens.weight", 64, 128, 8),
)
NORM = "model.language_model.layers.0.input_layernorm.weight"


def toy_values(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(ml_dtypes.bfloat16)


def write_install(root: Path, *, bf16: bool, corrupt: bool = False) -> None:
    root.mkdir()
    items = []
    norm = np.ones(256, dtype=ml_dtypes.bfloat16).tobytes()
    items.append(
        (
            rq.Entry(NORM, rq.DTYPE_BF16, 0, len(norm), (256, 0, 0, 0), 0, 0, 0, 0),
            (len(norm),),
            lambda b=norm: [b],
        )
    )
    for seed, (name, rows, cols, bits) in enumerate(QUANTISED):
        value = toy_values(rows, cols, seed)
        if corrupt:
            value = (value.astype(np.float32) * 1.01).astype(ml_dtypes.bfloat16)
        if bf16:
            blob = value.tobytes()
            entry = rq.Entry(name, rq.DTYPE_BF16, 0, len(blob), (rows, cols, 0, 0), 0, 0, 0, 0)
            items.append((entry, (len(blob),), lambda b=blob: [b]))
        else:
            p, s, b = (x.tobytes() for x in prepare.quantize_affine(value, bits))
            entry = rq.Entry(name, rq.DTYPE_U32, 0, 0, (rows, cols * bits // 32, 0, 0), 0, 0, 0, 0)
            items.append((entry, (len(p), len(s), len(b)), lambda t=(p, s, b): list(t)))
    rq.ResidentWriter(root / rq.RESIDENT, items, 64).write()
    quant = {slot: {"weightBits": 4} for slot in rq.SLOTS}
    quant["embedding"]["weightBits"] = 8
    quant["router"]["weightBits"] = 8
    quant["model.language_model.embed_tokens"] = {"weightBits": 8}
    manifest = {"modelID": "toy-4bit", "quant": quant, "files": {rq.RESIDENT: {}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "big.bin").write_bytes(b"\0" * rq.LINK_MIN_BYTES)
    (root / "tokenizer.json").write_text("{}")


@unittest.skipIf(rq is None, f"converter deps unavailable: {IMPORT_ERROR}")
class RequantizeRestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp))
        write_install(self.tmp / "base", bf16=False)

    def run_tool(self, source: Path) -> Path:
        out = self.tmp / "out"
        argv = [
            "x",
            "--base",
            str(self.tmp / "base"),
            "--source",
            str(source),
            "--output",
            str(out),
        ]
        with unittest.mock.patch.object(sys, "argv", argv):
            self.assertEqual(rq.main(), 0)
        return out

    def test_rest_is_requantised_at_8_bits_and_the_rest_is_kept(self) -> None:
        write_install(self.tmp / "src", bf16=True)
        out = self.run_tool(self.tmp / "src")
        resident = rq.read_resident(out / rq.RESIDENT)
        entries = resident.by_name()
        for seed, (name, rows, cols, _) in enumerate(QUANTISED):
            entry = entries[name]
            self.assertEqual(entry.shape, (rows, cols // 4, 0, 0))
            p, s, b = prepare.quantize_affine(toy_values(rows, cols, seed), 8)
            self.assertEqual(
                rq.read_bytes(out / rq.RESIDENT, entry.offset, entry.size), p.tobytes()
            )
            self.assertEqual(
                rq.read_bytes(out / rq.RESIDENT, entry.scale_offset, entry.scale_size), s.tobytes()
            )
            self.assertEqual(
                rq.read_bytes(out / rq.RESIDENT, entry.bias_offset, entry.bias_size), b.tobytes()
            )
        norm = entries[NORM]
        self.assertEqual(
            rq.read_bytes(out / rq.RESIDENT, norm.offset, norm.size),
            np.ones(256, dtype=ml_dtypes.bfloat16).tobytes(),
        )
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["modelID"], "toy-4bit-8bitrest")
        self.assertEqual(manifest["quant"]["attention"]["weightBits"], 8)
        self.assertEqual(manifest["quant"]["sharedExpert"]["weightBits"], 8)
        self.assertEqual(manifest["quant"]["routedExpert"]["weightBits"], 4)
        self.assertEqual(manifest["files"][rq.RESIDENT]["size"], (out / rq.RESIDENT).stat().st_size)
        self.assertEqual(
            os.stat(out / "big.bin").st_ino, os.stat(self.tmp / "base" / "big.bin").st_ino
        )
        self.assertNotEqual(
            os.stat(out / "tokenizer.json").st_ino,
            os.stat(self.tmp / "base" / "tokenizer.json").st_ino,
        )

    def test_a_source_that_is_not_the_bases_input_writes_nothing(self) -> None:
        write_install(self.tmp / "src", bf16=True, corrupt=True)
        with self.assertRaises(SystemExit) as raised:
            self.run_tool(self.tmp / "src")
        self.assertIn("self-check failed", str(raised.exception))
        self.assertFalse((self.tmp / "out").exists())


if __name__ == "__main__":
    unittest.main()
