#!/usr/bin/env python3
"""The launcher's concurrency choice: one by default, up to four on request.

The batched width is the one server setting that changes what the machine has to
hold *while it works* -- each running sequence keeps its own KV cache, one GPU is
shared between them, and the prompt cache is switched off above one. So the
launcher asks for it, defaults it to one, and warns in red above one. These tests
pin the default, the choice, the refusal, the warning (including that it is
really red on a terminal), the cache report, and the CPU engine's one-generation
rule.

The model they are asked to start comes from `launcher_fixture`, so these run on a
checkout with no install and no built server.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_launcher_concurrency -v
"""

from __future__ import annotations

import os
import pathlib
import pty
import re
import subprocess
import unittest

import launcher_fixture

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"

# Flags that answer every other question, so the concurrency question is the only
# prompt left. `--ram 4` is the smallest target the launcher accepts (1 and 2 GB
# cannot be honoured: the weights plus the minimum expert cache are ~4.7 GB).
# `--engine gpu` goes with the gpu-backed row these classes ask about: a dense
# row refuses that engine, and the CPU arm names one explicitly.
#
# The machine's physical memory is pinned through the launcher's own seam at
# 16 GB, because the host must not decide what these tests read. On the 7 GB
# CI runner the fixture model plus `--ram 4` cross that machine's 3 GB
# recommendation, and the launcher's correct RAM warning failed
# `test_enter_keeps_one`'s blanket "no WARNING" (AUD-312) — the suite's earlier
# "under 30% of any Mac this runs on" was an assumption about hardware, not a
# property of the code under test.
PHYSICAL_RAM_BYTES = 16 * 2**30
QUIET_ANSWERS = (
    "--answers",
    "default",
    "--thinking",
    "off",
    "--ram",
    "4",
    "--engine",
    "gpu",
    "--context",
    "262144",
    "--kv",
    "8",
    "--port",
    "9123",
)


def run_launcher(
    installs: launcher_fixture.SyntheticInstalls,
    *args: str,
    env: dict | None = None,
    stdin: str = "",
    interactive: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = installs.env(
        TINYTITAN_LAUNCHER_DRY_RUN="1",
        TINYTITAN_PHYSICAL_RAM_BYTES=str(PHYSICAL_RAM_BYTES),
    )
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


def run_launcher_pty(
    installs: launcher_fixture.SyntheticInstalls, *args: str, env: dict | None = None
) -> tuple[int, str]:
    """Run with stderr on a terminal, which is the only way warn_red colours."""
    environment = installs.env(
        TINYTITAN_LAUNCHER_DRY_RUN="1",
        TINYTITAN_LAUNCHER_ASSUME_TTY="1",
        TERM="xterm-256color",
        TINYTITAN_PHYSICAL_RAM_BYTES=str(PHYSICAL_RAM_BYTES),
    )
    environment.pop("NO_COLOR", None)
    if env:
        environment.update(env)
    master, slave = pty.openpty()
    process = subprocess.Popen(
        ["bash", str(LAUNCHER), "--dry-run", *args],
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        env=environment,
        close_fds=True,
    )
    os.close(slave)
    chunks: list[bytes] = []
    while True:
        try:
            data = os.read(master, 65536)
        except OSError:  # EIO: the child closed the slave side
            break
        if not data:
            break
        chunks.append(data)
    process.wait(timeout=120)
    os.close(master)
    return process.returncode, b"".join(chunks).decode("utf-8", "replace")


def reported_concurrency(run: subprocess.CompletedProcess[str]) -> str:
    match = re.search(r"\| At once: (\d+) \|", run.stdout)
    if match is None:
        raise AssertionError(f"no concurrency in the summary:\n{run.stdout}\n{run.stderr}")
    return match.group(1)


class ConcurrencyChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.base = ("--client", "server", "--model", self.installs.first_gpu())

    def test_the_default_is_one_generation_at_a_time(self) -> None:
        run = run_launcher(self.installs, *self.base)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_concurrency(run), "1")
        self.assertIn("--max-concurrent-sequences 1", run.stdout)
        self.assertIn("one generation at a time", run.stdout)

    def test_the_default_does_not_warn(self) -> None:
        run = run_launcher(self.installs, *self.base)
        self.assertNotIn("WARNING", run.stderr)
        self.assertIn("cache multi-prefix | MTP", run.stdout)

    def test_a_flagged_width_reaches_the_server_command(self) -> None:
        run = run_launcher(self.installs, *self.base, "--concurrency", "4")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_concurrency(run), "4")
        self.assertIn("--max-concurrent-sequences 4", run.stdout)
        self.assertIn("4 at once (yours)", run.stdout)

    def test_a_raised_width_warns_in_red(self) -> None:
        run = run_launcher(self.installs, *self.base, "--concurrency", "4")
        # Once when the choice is made, and once more in the summary a person
        # sees immediately before the model starts.
        self.assertIn("WARNING: the server will serve 4 generations at once.", run.stderr)
        self.assertIn("4 generations are served at once.", run.stderr)
        self.assertIn("The prompt cache is off above 1.", run.stderr)

    def test_the_cache_is_reported_as_the_server_will_run_it(self) -> None:
        # The server forces the session-wide cache off above one slot. A summary
        # that still advertised `multi-prefix` was a misreport: the operator saw
        # a cache that was not there.
        run = run_launcher(self.installs, *self.base, "--concurrency", "2")
        self.assertIn("cache off (above 1 at once) | MTP", run.stdout)

    def test_bad_widths_are_refused_before_anything_starts(self) -> None:
        # A power of two is the rule; anything else is a typo.
        for bad in ("0", "3", "5", "6", "100", "512", "abc", "-1"):
            with self.subTest(concurrency=bad):
                run = run_launcher(self.installs, *self.base, "--concurrency", bad)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown --concurrency", run.stderr)

    def test_any_power_of_two_up_to_the_ceiling_is_accepted(self) -> None:
        # Asking for more than this Mac can run is allowed on purpose: the server
        # clamps the width it builds and logs it, so 256 is a legitimate request.
        for good in ("8", "16", "256"):
            with self.subTest(concurrency=good):
                run = run_launcher(self.installs, *self.base, "--concurrency", good)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(reported_concurrency(run), good)
                self.assertIn(f"--max-concurrent-sequences {good}", run.stdout)

    def test_a_large_width_says_the_clamp_may_bind(self) -> None:
        run = run_launcher(self.installs, *self.base, "--concurrency", "256")
        self.assertIn("the clamp is likely to bind", run.stderr)
        self.assertIn("'batch width' line", run.stderr)

    def test_a_missing_value_is_refused_by_the_flag_itself(self) -> None:
        # `${2:?}` is the script's shared contract for "this flag needs a
        # value": bash exits 1 and names the flag, before any question is asked.
        run = run_launcher(self.installs, *self.base, "--concurrency", "")
        self.assertEqual(run.returncode, 1, run.stdout)
        self.assertIn("--concurrency needs a power of two", run.stderr)

    def test_a_cpu_model_is_pinned_to_one_generation(self) -> None:
        # The CPU backend's actor runs a generation to completion before the
        # next starts, so a width above one buys nothing there. The launcher
        # says so and pins the server to one rather than pretending. The dense
        # row is the one the CPU engine serves, so this names it instead of the
        # gpu-backed base.
        run = run_launcher(
            self.installs,
            "--client",
            "server",
            "--model",
            self.installs.dense(),
            "--engine",
            "cpu",
            "--concurrency",
            "4",
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--concurrency", run.stderr)
        self.assertIn("do not apply to the CPU engine", run.stderr)
        self.assertIn("--max-concurrent-sequences 1", run.stdout)
        self.assertNotIn("--max-concurrent-sequences 4", run.stdout)
        self.assertNotIn("WARNING: the server will serve", run.stderr)


class ConcurrencyQuestionTests(unittest.TestCase):
    """The interactive question itself, through the launcher's test seam."""

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.base = ("--client", "server", "--model", self.installs.first_gpu(), *QUIET_ANSWERS)

    def test_an_unattended_run_takes_one_without_asking(self) -> None:
        run = run_launcher(self.installs, *self.base)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("serve at once?", run.stdout)
        self.assertEqual(reported_concurrency(run), "1")

    def test_enter_keeps_one(self) -> None:
        run = run_launcher(self.installs, *self.base, stdin="\n", interactive=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("How many generations should the server serve at once?", run.stdout)
        self.assertEqual(reported_concurrency(run), "1")
        self.assertNotIn("WARNING", run.stderr)

    def test_typing_four_gives_four_and_typing_five_gives_eight(self) -> None:
        # The menu offers 1, 2, 4, 8, 16 and a custom entry; four is choice 3.
        run = run_launcher(self.installs, *self.base, stdin="3\n", interactive=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_concurrency(run), "4")
        self.assertIn("WARNING: the server will serve 4 generations at once.", run.stderr)

        eight = run_launcher(self.installs, *self.base, stdin="4\n", interactive=True)
        self.assertEqual(eight.returncode, 0, eight.stderr)
        self.assertEqual(reported_concurrency(eight), "8")

    def test_a_custom_power_of_two_is_taken(self) -> None:
        run = run_launcher(self.installs, *self.base, stdin="6\n256\n", interactive=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_concurrency(run), "256")
        self.assertIn("--max-concurrent-sequences 256", run.stdout)
        self.assertIn("WARNING: the server will serve 256 generations at once.", run.stderr)

    def test_a_custom_number_that_is_not_a_power_of_two_is_refused(self) -> None:
        run = run_launcher(self.installs, *self.base, stdin="6\n100\n", interactive=True)
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("a power of two", run.stderr)

    def test_a_choice_outside_the_menu_is_refused(self) -> None:
        for bad in ("0", "7", "nine"):
            with self.subTest(choice=bad):
                run = run_launcher(self.installs, *self.base, stdin=f"{bad}\n", interactive=True)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("invalid choice", run.stderr)

    def test_the_flag_skips_the_question(self) -> None:
        run = run_launcher(
            self.installs, *self.base, "--concurrency", "2", stdin="\n", interactive=True
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("serve at once?", run.stdout)
        self.assertEqual(reported_concurrency(run), "2")


class ConcurrencyColourTests(unittest.TestCase):
    """The warning has to be red where it lands: on the operator's terminal."""

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.base = (
            "--client",
            "server",
            "--model",
            self.installs.first_gpu(),
            *QUIET_ANSWERS,
            "--concurrency",
            "2",
        )

    def test_the_warning_carries_the_escape_on_a_terminal(self) -> None:
        status, output = run_launcher_pty(self.installs, *self.base)
        self.assertEqual(status, 0, output)
        self.assertIn("\033[1;31m", output)
        self.assertIn("WARNING: the server will serve 2 generations at once.", output)

    def test_no_color_keeps_the_words_and_drops_the_escape(self) -> None:
        status, output = run_launcher_pty(self.installs, *self.base, env={"NO_COLOR": "1"})
        self.assertEqual(status, 0, output)
        self.assertNotIn("\033[1;31m", output)
        self.assertIn("WARNING: the server will serve 2 generations at once.", output)


if __name__ == "__main__":
    unittest.main()
