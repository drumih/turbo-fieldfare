"""Gates the exit status of `tools/install_models.sh --all-4bit|--all-8bit`.

Both batch modes loop over `CATALOGUE` and their body ended in
`[[ "$w" == 4 ]] && install_one "$n"`, so the status the script returns is that
last row's *width test*, not the work: the final catalogue row is 8-bit, which
makes `--all-4bit` exit 1 after installing every 4-bit model with nothing
failing, and makes `--all-8bit` exit 0 as long as the last 8-bit model happened
to install — whatever happened to the seven before it. Measured on this
machine's /bin/bash 3.2.57:

    --all-8bit, qwen38flash-8bit failing -> 8 installed,  EXIT 0
    --all-8bit, nothing failing          -> 9 installed,  EXIT 0
    --all-4bit, ornith15 failing         -> 10 installed, EXIT 1
    --all-4bit, nothing failing          -> 12 installed, EXIT 1

`--all-4bit` is the loud half of the defect and `--all-8bit` is the silent one:
a gated checkpoint that refuses without an `HF_TOKEN` is a per-row failure, so
one row can fail and the verdict still read as success. Both spellings are in
the user-facing installation guide.

These tests drive the dispatcher's own branch text — extracted verbatim from the
script, so the instrument cannot drift away from what the script does — inside a
shell that has sourced the real script and then replaced `install_one` with a
stub. Nothing here can download, convert or build: the stub is the only thing the
loop is allowed to call, and the work, models and bin roots point at a temp
directory.

    cd benchmark && python3 -m unittest test_install_models_batch_status -v
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "install_models.sh"

# Four-bit rows: 13 until the Qwen3.8 MTP draft left the catalogue (2026-11-13,
# when the qwen3.8 install types became the two hardcoded recipes only).
WIDTHS = {"4": 12, "8": 9}
ROWS = re.compile(r'^\s*"([a-z0-9-]+)\|[^|]*\|([48])\|', re.M).findall(SCRIPT.read_text())
CATALOGUE_WIDTHS = {name: width for name, width in ROWS}


def branch(mode: str) -> str:
    """The dispatcher's own `--all-<mode>bit` branch, text and all."""
    match = re.search(rf"^  --all-{mode}bit\).*?;;\n", SCRIPT.read_text(), re.M | re.S)
    if match is None:
        raise AssertionError(
            f"the dispatcher in {SCRIPT} no longer holds a --all-{mode}bit branch to drive"
        )
    return match.group(0)


def run_batch(mode: str, fail_on: tuple[str, ...] = ()) -> dict[str, object]:
    """Run the extracted branch with a stub `install_one`; parse what it reports."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="tt-batch-status-"))
    harness = f"""
set -uo pipefail
set --
source "{SCRIPT}" >/dev/null 2>&1 || true
attempted=()
install_one() {{
  attempted+=("$1")
  case " {" ".join(fail_on)} " in
    *" $1 "*) echo "STUB: $1 refused" >&2; return 1 ;;
  esac
  return 0
}}
case "--all-{mode}bit" in
{branch(mode)}esac
echo "STATUS $?"
echo "ATTEMPTED ${{attempted[*]+"${{attempted[*]}}"}}"
"""
    try:
        env = dict(
            os.environ,
            TINYTITAN_WORK_DIR=str(tmp / "work"),
            TINYTITAN_MODELS_DIR=str(tmp / "models"),
            TINYTITAN_BIN_DIR=str(tmp / "bin"),
            TINYTITAN_PYTHON="/usr/bin/true",
        )
        result = subprocess.run(
            ["/bin/bash", "-c", harness],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    output = result.stdout + result.stderr
    status = re.search(r"^STATUS (\d+)$", result.stdout, re.M)
    attempted = re.search(r"^ATTEMPTED (.*)$", result.stdout, re.M)
    if status is None or attempted is None:
        raise AssertionError(f"the harness reported no verdict; output was:\n{output}")
    return {
        "status": int(status.group(1)),
        "attempted": [name for name in attempted.group(1).split() if name],
        "output": output,
    }


class BatchStatusTests(unittest.TestCase):
    """The verdict must be about every install the operator named."""

    def test_a_batch_that_missed_a_model_does_not_report_success(self) -> None:
        # (mode, width): the row chosen is a real one, and it is not the last row
        # of its width, which is the only row whose status used to reach the exit.
        failing = "qwen38flash-8bit"
        result = run_batch("8", (failing,))
        self.assertIn(failing, result["attempted"], "the batch never tried the row")
        self.assertNotEqual(result["status"], 0, result["output"])
        self.assertIn(failing, result["output"], "the verdict never named what it missed")

    def test_a_batch_that_installed_everything_reports_success(self) -> None:
        # Passes before the fix as well: it pins the other direction, that a clean
        # sweep must not turn into a failure once the loop starts reporting the work.
        result = run_batch("8")
        self.assertEqual(result["status"], 0, result["output"])

    def test_a_complete_4bit_sweep_reports_success(self) -> None:
        result = run_batch("4")
        self.assertEqual(result["status"], 0, result["output"])

    def test_a_4bit_batch_that_missed_a_model_does_not_report_success(self) -> None:
        result = run_batch("4", ("ornith15",))
        self.assertNotEqual(result["status"], 0, result["output"])
        self.assertIn("ornith15", result["output"])

    def test_the_filter_still_runs_exactly_the_models_of_its_width(self) -> None:
        for mode in ("4", "8"):
            with self.subTest(width=mode):
                names = run_batch(mode)["attempted"]
                expected = sorted(name for name, width in CATALOGUE_WIDTHS.items() if width == mode)
                self.assertEqual(sorted(names), expected)
                self.assertEqual(len(names), WIDTHS[mode])


if __name__ == "__main__":
    unittest.main()
