#!/usr/bin/env python3
"""The launcher's port question, its default, and what it does with a bad one.

The port is the one thing the launcher asks about the *server* rather than the
model, so it has three ways in — the interactive question, `--port`, and
`TINYTITAN_PORT` — and exactly one default, `TINYTITAN_DEFAULT_PORT` in
`tools/tinytitan_models.sh`. These tests pin all four, the validation that runs
before anything is started, and that an unattended run neither hangs nor asks.

The model the port is asked about comes from `launcher_fixture`, so these run on
a checkout with no install and no built server.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_launcher_port -v
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import tempfile
import unittest

import launcher_fixture

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"
MODELS_SH = ROOT / "tools/tinytitan_models.sh"

# Flags that answer every other question, so the port is the only prompt left.
# `--concurrency` is one of them: it is asked before the port, so leaving it
# unanswered would consume the newline these tests mean for the port.
QUIET_ANSWERS = (
    "--answers",
    "default",
    "--thinking",
    "off",
    "--ram",
    "9",
    "--engine",
    "gpu",
    "--context",
    "262144",
    "--kv",
    "8",
    "--concurrency",
    "1",
)


def shared_default() -> int:
    """The one default, read from where the scripts read it."""
    match = re.search(r"^TINYTITAN_DEFAULT_PORT=(\d+)$", MODELS_SH.read_text(), re.MULTILINE)
    if match is None:
        raise AssertionError("tinytitan_models.sh no longer declares TINYTITAN_DEFAULT_PORT")
    return int(match.group(1))


def run_launcher(
    installs: launcher_fixture.SyntheticInstalls,
    *args: str,
    env: dict | None = None,
    stdin: str = "",
    interactive: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = installs.env(TINYTITAN_LAUNCHER_DRY_RUN="1")
    if interactive:
        environment["TINYTITAN_LAUNCHER_ASSUME_TTY"] = "1"
    else:
        environment.pop("TINYTITAN_LAUNCHER_ASSUME_TTY", None)
    environment.pop("TINYTITAN_PORT", None)
    if env:
        environment.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", *args],
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=120,
    )


def reported_port(run: subprocess.CompletedProcess[str]) -> str:
    match = re.search(r"\| Port: (\d+) \|", run.stdout)
    if match is None:
        raise AssertionError(f"no port in the summary:\n{run.stdout}\n{run.stderr}")
    return match.group(1)


class PortChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()
        self.base = ("--client", "server", "--model", self.model)

    def test_the_default_is_the_shared_constant(self) -> None:
        run = run_launcher(self.installs, *self.base)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_port(run), str(shared_default()))
        self.assertIn(f"http://127.0.0.1:{shared_default()}/v1", run.stdout)

    def test_a_flagged_port_is_used_and_the_harness_is_told(self) -> None:
        run = run_launcher(self.installs, *self.base, "--port", "9123")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_port(run), "9123")
        self.assertIn("http://127.0.0.1:9123/v1", run.stdout)
        # Anything that assumes the default has to be told; the harness route is
        # the one the launcher does not write itself.
        self.assertIn("TINYTITAN_PORT=9123 tools/dsh_route.sh --write", run.stdout)

    def test_the_environment_port_is_used(self) -> None:
        run = run_launcher(self.installs, *self.base, env={"TINYTITAN_PORT": "9321"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_port(run), "9321")
        self.assertIn("TINYTITAN_PORT=9321 tools/dsh_route.sh --write", run.stdout)

    def test_a_flag_beats_the_environment(self) -> None:
        run = run_launcher(
            self.installs, *self.base, "--port", "9123", env={"TINYTITAN_PORT": "9321"}
        )
        self.assertEqual(reported_port(run), "9123")

    def test_a_port_the_default_does_not_need_no_hint(self) -> None:
        run = run_launcher(self.installs, *self.base)
        self.assertNotIn("dsh_route.sh --write", run.stdout)

    def test_bad_ports_are_refused_before_anything_starts(self) -> None:
        for bad in ("abc", "0", "99999", "-1", "80x"):
            with self.subTest(port=bad):
                run = run_launcher(self.installs, *self.base, "--port", bad)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown port", run.stderr)

    def test_the_launcher_resolves_the_fixture_not_the_checkout(self) -> None:
        # Every suite here names its model through `launcher_fixture`, and the
        # release policy says `models/` may legitimately be empty. So the answer
        # has to come from the tree the environment pointed at: if this printed a
        # path under the checkout, the fixture would be decoration and the suite
        # would be testing whichever install this particular Mac happens to hold.
        run = run_launcher(self.installs, *self.base)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn(f"Model: {self.installs.path_of(self.model)}", run.stdout)
        self.assertNotIn(str(ROOT / "models") + "/", run.stdout)


class PortQuestionTests(unittest.TestCase):
    """The interactive question itself, through the launcher's test seam."""

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        model = self.installs.first_gpu()
        self.base = ("--client", "server", "--model", model, *QUIET_ANSWERS)

    def test_an_unattended_run_takes_the_default_without_asking(self) -> None:
        run = run_launcher(self.installs, *self.base)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("Port for the server?", run.stdout)
        self.assertEqual(reported_port(run), str(shared_default()))

    def test_enter_keeps_the_default(self) -> None:
        run = run_launcher(self.installs, *self.base, stdin="\n", interactive=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Port for the server?", run.stdout)
        self.assertEqual(reported_port(run), str(shared_default()))

    def test_a_typed_port_is_taken(self) -> None:
        run = run_launcher(self.installs, *self.base, stdin="9123\n", interactive=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_port(run), "9123")
        self.assertIn("TINYTITAN_PORT=9123 tools/dsh_route.sh --write", run.stdout)

    def test_a_typed_port_that_is_not_a_number_is_refused(self) -> None:
        run = run_launcher(self.installs, *self.base, stdin="eighty\n", interactive=True)
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("unknown port", run.stderr)

    def test_a_flagged_port_skips_the_question(self) -> None:
        run = run_launcher(
            self.installs, *self.base, "--port", "9123", stdin="\n", interactive=True
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("Port for the server?", run.stdout)
        self.assertEqual(reported_port(run), "9123")


class LeadingZeroPortTests(unittest.TestCase):
    """A port the launcher's own arithmetic cannot read has to be refused.

    AUD-304, measured through `--dry-run` on the shipped guard
    (`tools/server_launcher.sh:1230-1242` at that revision): `--port 08757`
    exited 0 and printed the whole plan, with two
    `((: 08757: value too great for base` lines on stderr. The digit case-glob
    lets it through, and `(( ))` reads a leading zero as octal, so the range
    check and the privileged-port check both error and answer neither branch --
    the guard exists, runs, and evaluates nothing. `01024` is the shape that
    costs something: no range verdict, no privileged note, and a summary telling
    the operator which port to point a client at.

    Three of these tests are the neighbours the repair has to keep, and they
    passed before it: a low legal port still gets its warning, the port-space
    edges still answer, and no other spelling is refused that the readers used to
    accept.
    """

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()
        self.base = ("--client", "server", "--model", self.model)

    def assert_refused_without_arithmetic(self, run: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn("unknown port", run.stderr)
        combined = run.stdout + run.stderr
        # The refusal is the guard's own sentence, not a bash error that happened
        # to make a comparison come out false.
        self.assertNotIn("value too great for base", combined)
        self.assertNotIn("integer expression expected", combined)
        self.assertNotIn("| Port:", run.stdout)

    def test_a_flagged_leading_zero_port_is_refused(self) -> None:
        for port in ("08757", "01024", "00080", "0008080"):
            with self.subTest(port=port):
                run = run_launcher(self.installs, *self.base, "--port", port)
                self.assert_refused_without_arithmetic(run)

    def test_an_environment_leading_zero_port_is_refused(self) -> None:
        run = run_launcher(self.installs, *self.base, env={"TINYTITAN_PORT": "08757"})
        self.assert_refused_without_arithmetic(run)

    def test_a_typed_leading_zero_port_is_refused(self) -> None:
        run = run_launcher(
            self.installs,
            *self.base,
            *QUIET_ANSWERS,
            stdin="08757\n",
            interactive=True,
        )
        self.assert_refused_without_arithmetic(run)

    def test_a_privileged_port_still_gets_its_note(self) -> None:
        # The check that shares the arithmetic with the range guard: a legal low
        # port is not refused, it is warned about, before anything is started.
        run = run_launcher(self.installs, *self.base, "--port", "80")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("privileged", run.stderr)
        self.assertEqual(reported_port(run), "80")

    def test_the_port_space_edges_still_answer(self) -> None:
        # 1 is the lowest legal port, not a refusal: the guard's own message says
        # "a number 1-65535", and measured on the shipped script `--port 1` exits
        # 0 with the privileged note. What has to keep refusing is anything above
        # the port space.
        for port, expected in (
            ("1", 0),
            ("65535", 0),
            ("65536", 2),
            ("99999", 2),
        ):
            with self.subTest(port=port):
                run = run_launcher(self.installs, *self.base, "--port", port)
                self.assertEqual(run.returncode, expected, run.stdout + run.stderr)
        run = run_launcher(self.installs, *self.base, "--port", "1")
        self.assertIn("privileged", run.stderr)


class CalibrationProfileFlagTests(unittest.TestCase):
    """6.0's calibration profile is a server flag; the launcher forwards it.

    The profile itself is validated by the server (machine, memory, context and
    model), so the launcher's contract is narrow: pass the path through
    unchanged, and refuse a path that cannot be read before anything starts.
    """

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()
        self.base = ("--client", "server", "--model", self.model)

    def test_the_profile_is_forwarded_to_the_server(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w", suffix="-profile.json", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("{}")
            path = pathlib.Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        run = run_launcher(self.installs, *self.base, "--calibration-profile", str(path))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--calibration-profile", run.stdout)
        self.assertIn(str(path), run.stdout)

    def test_a_missing_profile_is_refused_before_starting(self) -> None:
        missing = pathlib.Path(tempfile.gettempdir()) / "tinytitan-no-such-calibration.json"
        run = run_launcher(self.installs, *self.base, "--calibration-profile", str(missing))
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("--calibration-profile not found", run.stderr)

    def test_help_documents_the_6_0_flags_and_one_disk_default(self) -> None:
        run = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--calibration-profile", run.stdout)
        self.assertIn("--page-cache", run.stdout)
        # The disk tier became on-by-default; the help must not still say both.
        self.assertIn("On by default", run.stdout)
        self.assertNotIn("Off by default because that state is derived", run.stdout)


if __name__ == "__main__":
    unittest.main()
