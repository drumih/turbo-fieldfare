"""The `core-purity` gate: TinyTitanCore must carry no model knowledge.

The 7.0 design contract (docs/plan-tinytitan-7-engine.md §1.3) makes the generic
library a library, not a framework: it may not grow model knowledge to avoid a
copy. This gate is the enforcement, and these tests pin both directions the
house way (the pattern of test_lint_gate_scan_guard.py): a model name in the
core is refused and named, a clean core passes, and an empty scan is a failure
rather than a pass.
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


class CorePurityHarness(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        # `releases/` is one of the harness's own skip directories, keeping the
        # copy out of the set it scans (the lesson recorded in
        # test_lint_gate_scan_guard.py's setUp).
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

    def core_file(self, text: str) -> pathlib.Path:
        return write(self.root / "sources" / "TinyTitanCore" / "Probe.swift", text)


class CorePurityTests(CorePurityHarness):
    def test_model_name_in_core_is_refused_and_named(self):
        self.core_file("import Foundation\nenum Probe { case qwen36 }\n")
        rc, out = self.run_mode("core-purity")
        self.assertNotEqual(rc, 0, f"core-purity passed a model name:\n{out}")
        self.assertIn("Probe.swift", out)
        self.assertIn("qwen36", out)

    def test_family_type_reference_is_refused(self):
        self.core_file("import Foundation\nlet f: ModelFamily = .qwen36\n")
        rc, out = self.run_mode("core-purity")
        self.assertNotEqual(rc, 0, f"core-purity passed a family type:\n{out}")
        self.assertIn("ModelFamily", out)

    def test_clean_core_passes(self):
        self.core_file("import Foundation\nstruct Probe { let value: Int }\n")
        rc, out = self.run_mode("core-purity")
        self.assertEqual(rc, 0, f"core-purity refused a clean core:\n{out}")

    def test_empty_scan_is_a_failure_not_a_pass(self):
        # No sources/TinyTitanCore at all: the gate must refuse, not report ok.
        rc, out = self.run_mode("core-purity")
        self.assertNotEqual(rc, 0, f"core-purity passed an empty scan:\n{out}")

    def test_exemption_comment_is_honoured(self):
        self.core_file(
            "import Foundation\n"
            "// lint:allow-core-name the probe names a family to test the gate itself\n"
            "enum Probe { case qwen36 }\n"
        )
        rc, out = self.run_mode("core-purity")
        self.assertEqual(rc, 0, f"core-purity ignored its exemption comment:\n{out}")


if __name__ == "__main__":
    unittest.main()
