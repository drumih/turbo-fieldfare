"""Model-free tests for verify_identity_bisect.py: synthetic dumps in a temp tree."""

from __future__ import annotations

import io
import math
import pathlib
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import verify_identity_bisect as bisect  # noqa: E402


def _f16_bits(value: float) -> int:
    if math.isnan(value):
        return 0x7E00
    if math.isinf(value):
        return 0x7C00 | (0x8000 if value < 0 else 0)
    sign = 0x8000 if math.copysign(1.0, value) < 0 else 0
    value = abs(value)
    if value == 0.0:
        return sign
    mantissa, exponent = math.frexp(value)
    exponent -= 1
    if exponent >= 16:
        return sign | 0x7C00
    if exponent >= -14:
        fraction = int((mantissa - 0.5) * 2048 + 0.5)
        return sign | ((exponent + 15) << 10) | (fraction & 0x3FF)
    return sign | int(value / 2.0**-24 + 0.5)


def write_f16(path: pathlib.Path, values: list[float]) -> None:
    # IEEE 754 half packed by hand: the runner's Python has no half-float
    # `array` typecode (array("e") raises "bad typecode" there).
    path.write_bytes(b"".join(struct.pack(">H", _f16_bits(v)) for v in values))


class BisectTests(unittest.TestCase):
    def make_position(self, root: pathlib.Path, position: int, diverge_at: int | None) -> None:
        directory = root / f"pos{position}"
        directory.mkdir(parents=True)
        for layer in range(3):
            base = [float(layer), 1.0, 2.0]
            write_f16(
                directory / f"L{layer + 1}_entry.f16" if layer < 2 else directory / "stack_out.f16",
                base,
            )
            shifted = list(base)
            if diverge_at is not None and layer >= diverge_at:
                shifted[1] += 0.5
            write_f16(directory / f"L{layer}_rows_after.f16", shifted)
        write_f16(directory / "logits.f16", [0.0, 3.0, 1.0])
        write_f16(
            directory / "verify_logits.f16",
            [0.0, 3.0, 1.0] if diverge_at is None else [4.0, 3.0, 1.0],
        )

    def test_identical_row_has_no_divergent_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.make_position(root, 10, None)
            result = bisect.compare_row(root / "pos10")
            self.assertIsNone(result["first_divergent_layer"])
            self.assertTrue(result["argmax_same"])
            self.assertEqual(result["logits_max_abs_diff"], 0.0)

    def test_names_the_first_layer_that_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.make_position(root, 10, 1)
            result = bisect.compare_row(root / "pos10")
            self.assertEqual(result["first_divergent_layer"], 1)
            self.assertEqual([d for _, d in result["diffs"]], [0.0, 0.5, 0.5])
            self.assertFalse(result["argmax_same"])

    def test_main_reports_every_row_and_refuses_missing_dumps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.make_position(root, 5, None)
            self.make_position(root, 6, 2)
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(bisect.main([str(root), "5", "2"]), 0)
            self.assertIn("row 0: first divergent layer None", out.getvalue())
            self.assertIn("row 1: first divergent layer 2", out.getvalue())
            with redirect_stdout(io.StringIO()):
                self.assertEqual(bisect.main([str(root), "5", "3"]), 2)


if __name__ == "__main__":
    unittest.main()
