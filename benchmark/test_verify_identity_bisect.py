"""Model-free tests for verify_identity_bisect.py: synthetic dumps in a temp tree."""

from __future__ import annotations

import io
import pathlib
import sys
import tempfile
import unittest
from array import array
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import verify_identity_bisect as bisect  # noqa: E402


def write_f16(path: pathlib.Path, values: list[float]) -> None:
    path.write_bytes(array("e", values).tobytes())


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
