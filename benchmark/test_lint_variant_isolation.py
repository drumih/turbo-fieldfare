"""The `variant-isolation` gate: a variant target imports only the generic library.

The 7.0 design contract (docs/plan-tinytitan-7-engine.md §3, isolation rule): a
variant target may import TinyTitanCore and Foundation/Metal only — never
another variant, never the runtime targets. These tests pin both directions the
house way: a cross-variant or runtime import is refused and named, a clean
variant passes, and an empty scan is a failure rather than a pass.
"""

import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINT = ROOT / "tools" / "lint.sh"


def write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class VariantIsolationHarness(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.harness = write(self.root / "releases" / "lint.sh", LINT.read_text(encoding="utf-8"))
        for name in ("tools", "benchmark", "docs", "examples"):
            (self.root / name).mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_mode(self, mode: str) -> tuple[int, str]:
        proc = subprocess.run(
            ["/bin/bash", str(self.harness), mode],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout + proc.stderr

    def variant_file(self, variant: str, text: str) -> pathlib.Path:
        return write(self.root / "sources" / "Engines" / variant / "Probe.swift", text)


class VariantIsolationTests(VariantIsolationHarness):
    def test_cross_variant_import_is_refused_and_named(self):
        self.variant_file("Qwen35Dense2B4Bit", "import Foundation\nimport Qwen38Flash4Bit\n")
        self.variant_file("Qwen38Flash4Bit", "import Foundation\n")
        rc, out = self.run_mode("variant-isolation")
        self.assertNotEqual(rc, 0, f"variant-isolation passed a cross-variant import:\n{out}")
        self.assertIn("Qwen35Dense2B4Bit", out)
        self.assertIn("Qwen38Flash4Bit", out)

    def test_runtime_import_is_refused(self):
        self.variant_file("Qwen35Dense2B4Bit", "import Foundation\nimport TinyTitan\n")
        rc, out = self.run_mode("variant-isolation")
        self.assertNotEqual(rc, 0, f"variant-isolation passed a runtime import:\n{out}")
        self.assertIn("TinyTitan", out)

    def test_clean_variant_passes(self):
        self.variant_file(
            "Qwen35Dense2B4Bit",
            "import Foundation\nimport Metal\nimport TinyTitanCore\nimport TinyTitanFormat\n",
        )
        rc, out = self.run_mode("variant-isolation")
        self.assertEqual(rc, 0, f"variant-isolation refused a clean variant:\n{out}")

    def test_empty_scan_is_a_failure_not_a_pass(self):
        # sources/Engines exists but holds no Swift: a scan that should have
        # read files and read none must refuse, not report ok.
        (self.root / "sources" / "Engines").mkdir(parents=True)
        rc, out = self.run_mode("variant-isolation")
        self.assertNotEqual(rc, 0, f"variant-isolation passed an empty scan:\n{out}")

    def test_no_variant_targets_yet_is_ok_with_a_note(self):
        # Pre-7.0 there is no sources/Engines at all: nothing to isolate, and
        # the gate says so rather than silently passing or falsely failing.
        rc, out = self.run_mode("variant-isolation")
        self.assertEqual(rc, 0, f"variant-isolation failed with no variants yet:\n{out}")
        self.assertIn("no variant targets yet", out)


if __name__ == "__main__":
    unittest.main()
