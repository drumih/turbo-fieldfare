#!/usr/bin/env python3
"""Tests for the NumPy reference the sidecar verifier judges the graph against.

`selection_mask` builds the QSA-shaped mask that the folded check feeds both the
graph and the reference. It is the yardstick, so a bug in it would not fail —
it would make a wrong graph pass. The rule it mirrors lives in
`QSAIndexer.selectKeysPrefill`.

It imports the verifier, which imports the exporter, which imports coremltools,
so it skips where that is not installed (the CI python does not have it;
`~/.venvs/coreml-py311` does):

    cd benchmark && ~/.venvs/coreml-py311/bin/python -m unittest \
        test_verify_ane_sidecar -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import verify_ane_sidecar as verify
    import numpy as np

    IMPORT_ERROR = ""
except Exception as exc:  # coremltools or numpy absent
    verify = None
    np = None
    IMPORT_ERROR = str(exc)


@unittest.skipIf(verify is None, f"verifier needs coremltools: {IMPORT_ERROR}")
class ActivationDumpTests(unittest.TestCase):
    """`--activation-dump` runs the verdict on a REAL captured normed.

    The synthetic N(0,1) input lands the score dot-products near 1 where the
    fp16 ulp is ~0.001, and structurally cannot reproduce the live input's
    correlated rows: on ornith-1.5_35B_A3B_4Bit the same graph-vs-reference
    comparison measured 0.555% synthetic but 4.245% real (2026-11-19) — which
    is how an fp16 score materialization shipped blessed. An earlier attempt
    gated a x8 scaled Gaussian instead; it PASSED on the defective graph
    (0.162%) because the relative metric divides by a much larger attention
    output, so it was removed rather than shipped as a gate that does not
    gate.
    """

    def test_loads_a_matching_dump(self):
        import numpy as np
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".f16", delete=False) as handle:
            np.arange(64, dtype=np.float16).tofile(handle)
            path = pathlib.Path(handle.name)
        try:
            got = verify.load_activation_dump(path, 8, 8)
            self.assertEqual(got.shape, (8, 8))
            self.assertEqual(int(got.ravel()[0]), 0)
        finally:
            path.unlink(missing_ok=True)

    def test_refuses_a_dump_of_the_wrong_size(self):
        import numpy as np
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".f16", delete=False) as handle:
            np.arange(63, dtype=np.float16).tofile(handle)
            path = pathlib.Path(handle.name)
        try:
            with self.assertRaises(SystemExit):
                verify.load_activation_dump(path, 8, 8)
        finally:
            path.unlink(missing_ok=True)


@unittest.skipIf(verify is None, f"verifier needs coremltools: {IMPORT_ERROR}")
class VerdictTests(unittest.TestCase):
    """The accept/refuse verdict on a graph-vs-reference comparison.

    `nan > threshold` is False in Python, so a 0/0 = nan error used to pass
    the gate — which is how the zeroed 2B sidecar (0/0 both sides) was
    blessed as "ok" (AUD, 2026-11-19). These tests pin the refusal.
    """

    def test_a_finite_error_inside_the_gate_passes(self):
        got = np.full((4, 4), 1.0, dtype=np.float32)
        expected = got + 0.001  # ~0.1 % relative
        passed, stats = verify.compare_outputs(got, expected, max_relative_error=2.0)
        self.assertTrue(passed, f"stats={stats}")
        self.assertLess(stats["relative"], 2.0)

    def test_an_error_above_the_gate_fails(self):
        got = np.full((4, 4), 9.0, dtype=np.float32)
        expected = np.full((4, 4), 1.0, dtype=np.float32)
        passed, _ = verify.compare_outputs(got, expected, max_relative_error=2.0)
        self.assertFalse(passed)

    def test_a_nan_error_fails_instead_of_passing(self):
        # All-zero expected: mean |expected| = 0, so relative = 0/0 = nan.
        # `nan > 2.0` is False; the verdict must refuse, not pass.
        got = np.zeros((4, 4), dtype=np.float32)
        expected = np.zeros((4, 4), dtype=np.float32)
        passed, stats = verify.compare_outputs(got, expected, max_relative_error=2.0)
        self.assertFalse(passed, f"stats={stats}")

    def test_a_nan_graph_output_fails_instead_of_passing(self):
        expected = np.full((4, 4), 1.0, dtype=np.float32)
        got = np.full((4, 4), np.nan, dtype=np.float32)
        passed, stats = verify.compare_outputs(got, expected, max_relative_error=2.0)
        self.assertFalse(passed, f"stats={stats}")

    def test_an_inf_graph_output_fails_instead_of_passing(self):
        expected = np.full((4, 4), 1.0, dtype=np.float32)
        got = np.full((4, 4), np.inf, dtype=np.float32)
        passed, stats = verify.compare_outputs(got, expected, max_relative_error=2.0)
        self.assertFalse(passed, f"stats={stats}")

    def test_an_all_zero_graph_output_over_real_weights_fails(self):
        # The zeroed-2B shape: real reference, graph that computes nothing.
        # |diff| == |expected| so relative is 100 % — but the explicit
        # all-zero refusal names the vacuous-graph cause rather than
        # reporting a mysterious 100 %.
        expected = np.full((4, 4), 1.0, dtype=np.float32)
        got = np.zeros((4, 4), dtype=np.float32)
        passed, stats = verify.compare_outputs(got, expected, max_relative_error=200.0)
        self.assertFalse(passed, f"stats={stats}")


@unittest.skipIf(verify is None, f"verifier needs coremltools: {IMPORT_ERROR}")
class SelectionMaskTests(unittest.TestCase):
    def test_rows_inside_the_dense_window_keep_everything_visible(self):
        # budget 4, ratio 2 -> the dense window is 5 visible keys.
        mask, kept = verify.selection_mask(16, 4, 2, seed=1)
        for row in range(5):
            visible = row + 1
            self.assertEqual(kept[row], visible)
            self.assertTrue((mask[0, 0, row, :visible] == 0.0).all())

    def test_a_row_past_the_window_keeps_the_budget_and_its_tail(self):
        mask, kept = verify.selection_mask(16, 4, 2, seed=1)
        row = 15
        visible = row + 1
        self.assertEqual(kept[row], 5)  # budget + ratio - 1
        # Dropped keys are masked, not merely absent.
        self.assertEqual(int((mask[0, 0, row, :visible] == verify.ex.NEG).sum()), visible - 5)
        # Nothing past the query's own position is ever visible.
        self.assertTrue((mask[0, 0, row, visible:] == verify.ex.NEG).all())

    def test_the_ragged_tail_of_the_query_block_is_always_kept(self):
        mask, _ = verify.selection_mask(16, 4, 2, seed=3)
        for row in range(16):
            visible = row + 1
            complete = (visible // 2) * 2
            tail = mask[0, 0, row, complete:visible]
            self.assertTrue((tail == 0.0).all(), f"row {row} dropped its own block's tail")

    def test_the_kept_count_matches_the_mask(self):
        mask, kept = verify.selection_mask(32, 6, 4, seed=5)
        for row in range(32):
            visible = row + 1
            row_mask = mask[0, 0, row, :visible]
            self.assertEqual(int((row_mask == 0.0).sum()), kept[row])
            self.assertEqual(int((row_mask == verify.ex.NEG).sum()), visible - kept[row])

    def test_the_budget_is_never_exceeded(self):
        budget, ratio = 8, 4
        selection_width = budget + ratio - 1
        _, kept = verify.selection_mask(64, budget, ratio, seed=7)
        self.assertLessEqual(int(kept.max()), selection_width)
        self.assertGreater(int(kept.max()), 0)

    def test_it_is_deterministic_for_a_seed(self):
        first, first_kept = verify.selection_mask(16, 4, 2, seed=11)
        second, second_kept = verify.selection_mask(16, 4, 2, seed=11)
        self.assertTrue((first == second).all())
        self.assertTrue((first_kept == second_kept).all())
        other, _ = verify.selection_mask(16, 4, 2, seed=12)
        self.assertFalse((first == other).all())


if __name__ == "__main__":
    unittest.main()
