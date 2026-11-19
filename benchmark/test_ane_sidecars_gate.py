#!/usr/bin/env python3.13
"""AUD-294: what `tools/ane_sidecars.sh` is willing to call a verified install.

The script's own header says it `exits non-zero if any model it was asked about
is still without a verified sidecar, so it can gate an install`, and
docs/adding-a-model.md names it as step 8 of adding a model. Measured against
HEAD at `952c1be`, with `TINYTITAN_MODELS_DIR` at a temp tree and
`TINYTITAN_COREML_PYTHON` at `/usr/bin/false` so nothing could export, verify or
be reached:

    models directory                      printed        status
    empty                                 verified: 0        0
    holds other-dir/ with no manifest     verified: 0        0
    does not exist at all                 verified: 0        0
    empty, --verify-only                  verified: 0        0

Four shapes, one of which is a fresh checkout, all reporting the same thing as
every install verified -- and the failed glob names nothing, so the reader
cannot tell an absent directory from a family the script declined to export. A
named-but-missing install it does refuse (1), so the empty walk is the finding.
A second defect fell out of building the fixture: with the interpreter present
but unable to read a manifest, the family substitution fails under `set -e` and
the script exits 1 having printed nothing at all -- that is pinned here too.

These tests pin the refusal at 2, the code the script already answers for a
usage refusal, and hold the verdicts that were already right: a directory whose
installs are all skipped families keeps 0 with `verified: 0`, because those
installs *were* asked about and the script says why it will not export them.

No model, no coremltools, no install touched, nothing fetched: the interpreter
is a shim in a temp directory that reads a manifest and writes a marker file,
and every case runs the real script through /bin/bash.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "ane_sidecars.sh"

SHIM = """#!{python}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
if args[:1] == ["-c"]:
    exec(args[1])
    raise SystemExit(0)
if any("export_ane_prefill.py" in a for a in args):
    if os.environ.get("STUB_EXPORT_FAILS"):
        print("coremltools is not installed (stub)", file=sys.stderr)
        raise SystemExit(1)
    model = args[args.index("--model") + 1]
    chunk = args[args.index("--chunk") + 1]
    name = "ane_prefill" if chunk == "4096" else "ane_prefill-" + chunk
    directory = pathlib.Path(model) / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ane_prefill.json").write_text(json.dumps({{"chunk": int(chunk)}}))
    print("exported 28 variants (stub)")
    raise SystemExit(0)
if any("verify_ane_sidecar.py" in a for a in args):
    if os.environ.get("STUB_VERIFY_FAILS"):
        print("max absolute difference 0.9 (stub)", file=sys.stderr)
        raise SystemExit(1)
    print("max absolute difference 0.00098 (stub)")
    raise SystemExit(0)
print("stub interpreter reached an unexpected call: " + repr(args), file=sys.stderr)
raise SystemExit(99)
""".format(python=sys.executable)


def temp_root() -> pathlib.Path:
    return pathlib.Path(tempfile.mkdtemp())


def make_shim(directory: pathlib.Path) -> pathlib.Path:
    shim = directory / "coreml-shim"
    shim.write_text(SHIM, encoding="utf-8")
    shim.chmod(0o755)
    return shim


def install(
    models: pathlib.Path,
    name: str,
    family: str,
    sidecar: str | None = None,
) -> pathlib.Path:
    directory = models / name
    directory.mkdir(parents=True)
    manifest = {"arch": {"family": family}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if sidecar:
        marker = directory / sidecar
        marker.mkdir()
        (marker / "ane_prefill.json").write_text(json.dumps({"chunk": 4096}), encoding="utf-8")
    return directory


def run(
    models: pathlib.Path,
    *args: str,
    shim: pathlib.Path | None = None,
    verify_fails: bool = False,
    export_fails: bool = False,
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["TINYTITAN_MODELS_DIR"] = str(models)
    env["TINYTITAN_COREML_PYTHON"] = str(shim) if shim else "/usr/bin/false"
    if verify_fails:
        env["STUB_VERIFY_FAILS"] = "1"
    if export_fails:
        env["STUB_EXPORT_FAILS"] = "1"
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        check=False,
    )


class EmptyWalkRefused(unittest.TestCase):
    def test_an_empty_models_directory_is_refused_and_names_the_directory(self):
        models = temp_root() / "models"
        models.mkdir()
        proc = run(models)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("no installed model", proc.stderr)
        self.assertIn(str(models), proc.stderr)
        self.assertNotIn("verified: 0", proc.stdout)

    def test_a_directory_whose_subdirectories_have_no_manifest_is_refused(self):
        models = temp_root() / "models"
        (models / "other-dir").mkdir(parents=True)
        proc = run(models)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("manifest", proc.stderr)
        self.assertNotIn("verified: 0", proc.stdout)

    def test_a_models_directory_that_does_not_exist_is_refused_and_says_which(self):
        models = temp_root() / "absent"
        proc = run(models)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn(str(models), proc.stderr)

    def test_verify_only_over_an_empty_directory_is_refused_too(self):
        """The mode a check runs against installs it believes are there."""
        models = temp_root() / "models"
        models.mkdir()
        proc = run(models, "--verify-only")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("verified: 0", proc.stdout)

    def test_the_refusal_is_not_printed_as_a_result(self):
        """stdout stays the summary channel; a refusal on it reads as one."""
        models = temp_root() / "models"
        models.mkdir()
        proc = run(models)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertNotIn("verified", proc.stdout)
        self.assertNotIn("skipped", proc.stdout)


class VerdictsThatWereAlreadyRight(unittest.TestCase):
    def test_a_named_install_that_is_not_there_still_answers_one(self):
        models = temp_root() / "models"
        models.mkdir()
        proc = run(models, "qwen3.5_4B_4Bit")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("not installed", proc.stdout)
        self.assertIn(str(models / "qwen3.5_4B_4Bit"), proc.stdout)

    def test_an_unknown_option_is_refused_before_the_walk(self):
        models = temp_root() / "models"
        models.mkdir()
        proc = run(models, "--nope")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("unknown option", proc.stderr)

    def test_a_chunk_the_runtime_would_not_route_is_refused(self):
        models = temp_root() / "models"
        models.mkdir()
        proc = run(models, "--chunk", "777")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("not a prefill chunk", proc.stderr)

    def test_an_exportable_install_that_verifies_answers_zero(self):
        models = temp_root() / "models"
        made = install(models, "qwen3.5_4B_4Bit", "qwen3_5_dense")
        proc = run(models, shim=make_shim(models.parent))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("verified: 1", proc.stdout)
        self.assertIn("exported: qwen3.5_4B_4Bit", proc.stdout)
        self.assertTrue((made / "ane_prefill" / "ane_prefill.json").is_file())

    def test_a_non_default_chunk_gets_its_own_sidecar_directory(self):
        models = temp_root() / "models"
        made = install(models, "qwen3.5_4B_4Bit", "qwen3_5_dense")
        proc = run(models, "--chunk", "1024", shim=make_shim(models.parent))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("verified: 1", proc.stdout)
        self.assertTrue((made / "ane_prefill-1024" / "ane_prefill.json").is_file())
        self.assertFalse((made / "ane_prefill" / "ane_prefill.json").exists())

    def test_verify_only_over_an_existing_sidecar_answers_zero_without_exporting(self):
        models = temp_root() / "models"
        install(models, "qwen3.5_4B_4Bit", "qwen3_5_dense", sidecar="ane_prefill")
        proc = run(models, "--verify-only", shim=make_shim(models.parent))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("verified: 1", proc.stdout)
        self.assertNotIn("exported:", proc.stdout)

    def test_a_family_the_exporter_skips_keeps_zero_and_says_why(self):
        """The distinction the guard must not erase: asked about, and declined."""
        models = temp_root() / "models"
        install(models, "qwen3.8-flash-next_4-Bit", "qwen38flash")
        proc = run(models, shim=make_shim(models.parent))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("verified: 0", proc.stdout)
        self.assertIn("measured slower", proc.stdout)
        self.assertIn("skipped :", proc.stdout)

    def test_ornith_sidecar_is_refused_with_the_measured_reason(self):
        """TT-099: the sidecar fails the identical-greedy gate and prefills slower."""
        models = temp_root() / "models"
        made = install(models, "ornith-1.5_35B_A3B_4Bit", "qwen36")
        proc = run(models, shim=make_shim(models.parent))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("verified: 0", proc.stdout)
        self.assertIn("skipped :", proc.stdout)
        self.assertIn("TT-099", proc.stdout)
        self.assertIn("quality gate", proc.stdout)
        self.assertFalse((made / "ane_prefill").exists())

    def test_the_ornith_skip_does_not_reach_another_qwen36_install(self):
        """The guard is scoped to the measured install name, not to `qwen36`."""
        models = temp_root() / "models"
        made = install(models, "qwen3.6_35B_A3B_4Bit", "qwen36")
        proc = run(models, shim=make_shim(models.parent))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("verified: 1", proc.stdout)
        self.assertIn("exported: qwen3.6_35B_A3B_4Bit", proc.stdout)
        self.assertTrue((made / "ane_prefill" / "ane_prefill.json").is_file())

    def test_a_verification_failure_answers_one_and_says_which(self):
        models = temp_root() / "models"
        install(models, "qwen3.5_4B_4Bit", "qwen3_5_dense", sidecar="ane_prefill")
        proc = run(
            models,
            "--verify-only",
            shim=make_shim(models.parent),
            verify_fails=True,
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("verification failed", proc.stderr)
        self.assertNotIn("verified: 1", proc.stdout)

    def test_a_failed_export_is_refused_and_says_which_stage_failed(self):
        """`export failed` and `still no sidecar` are different answers to the operator."""
        models = temp_root() / "models"
        made = install(models, "qwen3.5_4B_4Bit", "qwen3_5_dense")
        proc = run(models, shim=make_shim(models.parent), export_fails=True)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("export failed", proc.stdout + proc.stderr)
        self.assertIn("FAILED  :", proc.stderr)
        self.assertFalse((made / "ane_prefill").exists())

    def test_a_manifest_the_interpreter_cannot_read_is_named_rather_than_silent(self):
        """Found by this suite's own fixture: a failed command substitution dies quietly.

        With `TINYTITAN_COREML_PYTHON` present but unable to read the manifest --
        which is what `/usr/bin/false` stands for here -- the family
        substitution fails under `set -e` and the script exits 1 having printed
        nothing at all, so the reader cannot tell a broken interpreter from a
        corrupt install from a model it declined.
        """
        models = temp_root() / "models"
        install(models, "qwen3.5_4B_4Bit", "qwen3_5_dense")
        proc = run(models)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("qwen3.5_4B_4Bit", proc.stderr)
        # Read at the line that names the cause, not at the summary that repeats
        # the model name: the FAILED row says `manifest unreadable` for any cause,
        # and only the refusal at the substitution says which interpreter failed.
        self.assertIn("/usr/bin/false", proc.stderr)
        self.assertIn("could not read the family from", proc.stderr)
        self.assertIn(str(models / "qwen3.5_4B_4Bit" / "manifest.json"), proc.stderr)


class UsageTextIntact(unittest.TestCase):
    def test_help_prints_the_header_and_stops_before_the_script_body(self):
        """`usage()` is a line range, so a header edit can make it print `set -euo`."""
        models = temp_root() / "models"
        models.mkdir()
        proc = run(models, "--help")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("Give every installed GPU-path model the ANE sidecar", proc.stdout)
        self.assertIn("so it can gate an install", proc.stdout)
        self.assertIn("a models directory holding no install", proc.stdout)
        self.assertNotIn("set -euo pipefail", proc.stdout)
        self.assertNotIn("MODELS_DIR=", proc.stdout)


if __name__ == "__main__":
    unittest.main()
