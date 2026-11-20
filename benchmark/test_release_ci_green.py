"""`tools/ci-green.sh` decides whether a commit may be tagged and published.

The row this pins (AUD-105) is a real one: v5.18 was published an hour after its
own tag commit came back `CI | completed | failure`. Nothing asked CI before
publishing, and `docs/release-process.md` said "a green main" in prose while no
check read it. So the arithmetic here is the whole defence, and every branch of
it is tested against a stub `gh` -- green, red, still running, CodeQL advisory,
and the two shapes
that would let a vacuous pass through: a commit CI never saw, and a short sha the
API matches against nothing.

Run from `benchmark/`:

    python3 -m unittest test_release_ci_green
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "ci-green.sh"
REPO = "Pummelchen/TinyTitan"
FULL_SHA = "0123456789abcdef0123456789abcdef01234567"

GH_STUB = """#!/bin/sh
if [ "${CI_GREEN_GH_FAIL:-0}" = "1" ]; then
    echo "gh: HTTP 502" >&2
    exit 1
fi
if [ -n "${CI_GREEN_FIXTURE:-}" ]; then
    cat "$CI_GREEN_FIXTURE"
fi
exit 0
"""


def run(
    *args: str,
    fixture: str | None = None,
    gh_fails: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the helper with `gh` replaced by the stub and a fixture as its answer."""
    with tempfile.TemporaryDirectory() as work:
        stub = pathlib.Path(work) / "gh"
        stub.write_text(GH_STUB, encoding="utf-8")
        stub.chmod(0o755)
        env = {
            "PATH": f"{work}{os.pathsep}/bin:/usr/bin",
            "CI_GREEN_FIXTURE": str(pathlib.Path(work) / "fixture.tsv"),
        }
        fixture_path = pathlib.Path(work) / "fixture.tsv"
        fixture_path.write_text(fixture or "", encoding="utf-8")
        if gh_fails:
            env["CI_GREEN_GH_FAIL"] = "1"
        return subprocess.run(
            ["/bin/bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


class CiGreenTests(unittest.TestCase):
    def test_all_success_is_green(self) -> None:
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tsuccess\thttps://x/1\nCodeQL\tcompleted\tsuccess\thttps://x/2\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CI green", result.stdout)
        self.assertEqual(result.stdout.splitlines()[-1], "result\tgreen\t-")

    def test_a_skipped_workflow_is_not_a_failure(self) -> None:
        # CodeQL has its own trigger conditions; a run GitHub skipped says nothing
        # about this commit either way, and refusing it would block releases on a
        # workflow that was never meant to run.
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tsuccess\thttps://x/1\nDocs\tcompleted\tskipped\thttps://x/2\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_pending_codeql_does_not_block_the_release(self) -> None:
        # The 2026-11-20 rule: shipping waits for compile+test, not for the
        # security scan. A CodeQL run that is still going is reported, not fatal.
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tsuccess\thttps://x/1\nCodeQL\tin_progress\t-\thttps://x/2\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("advisory (does not gate a release): CodeQL (in_progress/-)", result.stdout)
        self.assertEqual(result.stdout.splitlines()[-1], "result\tgreen\t-")

    def test_a_red_codeql_is_advisory_not_a_release_blocker(self) -> None:
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tsuccess\thttps://x/1\nCodeQL\tcompleted\tfailure\thttps://x/2\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CodeQL (completed/failure)", result.stdout)
        self.assertNotIn("CI is red", result.stderr)

    def test_codeql_alone_is_not_a_ci_run(self) -> None:
        result = run(REPO, FULL_SHA, fixture="CodeQL\tcompleted\tsuccess\thttps://x/2\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no workflow run", result.stderr)

    def test_a_failed_run_is_refused_and_names_the_run(self) -> None:
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tfailure\thttps://x/9\nCodeQL\tcompleted\tsuccess\thttps://x/2\n",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("CI is red", result.stderr)
        self.assertIn("CI (failure)", result.stderr)
        self.assertIn("https://x/9", result.stderr)

    def test_a_cancelled_or_timed_out_run_is_refused(self) -> None:
        # Not `failure`, and not a pass: the workflow did not complete its checks.
        for conclusion in ("cancelled", "timed_out", "action_required", "stale"):
            with self.subTest(conclusion=conclusion):
                result = run(
                    REPO,
                    FULL_SHA,
                    fixture=f"CI\tcompleted\t{conclusion}\thttps://x/9\n",
                )
                self.assertEqual(result.returncode, 1, conclusion)

    def test_a_run_still_going_is_refused(self) -> None:
        # Releasing over an in-flight run is guessing at an answer that is about to
        # arrive, and the override must not be the way out of waiting.
        result = run(REPO, FULL_SHA, fixture="CI\tin_progress\t-\thttps://x/9\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("has not finished", result.stderr)
        self.assertIn("--allow-red", result.stderr)

    def test_no_run_at_all_is_refused(self) -> None:
        # The vacuous pass this row is about: a query that matches nothing must not
        # read as "nothing failed, so go".
        result = run(REPO, FULL_SHA, fixture="")
        self.assertEqual(result.returncode, 1)
        self.assertIn("never saw", result.stderr)

    def test_a_short_sha_is_refused_before_the_query(self) -> None:
        # `head_sha` with a short sha returns no runs, so the answer to a short sha
        # would always be "no run at all" -- an error message that reads like a
        # finding about CI rather than about the argument. Say which it is.
        result = run(REPO, "ea5de8c", fixture="CI\tcompleted\tsuccess\thttps://x/1\n")
        self.assertEqual(result.returncode, 2)
        self.assertIn("full commit sha", result.stderr)

    def test_a_failed_query_is_not_a_pass(self) -> None:
        # No token, no network, a renamed repo: whatever it is, the helper cannot
        # claim CI passed on an answer it did not get.
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tsuccess\thttps://x/1\n",
            gh_fails=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot read", result.stderr)

    def test_missing_arguments_are_a_usage_error(self) -> None:
        result = run(REPO)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_red_ci_passes_only_with_a_reason(self) -> None:
        fixture = "CI\tcompleted\tfailure\thttps://x/9\n"
        refused = run(REPO, FULL_SHA, fixture=fixture)
        self.assertEqual(refused.returncode, 1)

        allowed = run(REPO, FULL_SHA, "--allow-red", "flake in the runner image", fixture=fixture)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertIn("releasing over it: flake in the runner image", allowed.stdout)
        # The caller needs the run URL out of the machine-readable line: it is what
        # the release notes then have to quote.
        self.assertEqual(allowed.stdout.splitlines()[-1], "result\toverridden\thttps://x/9")

    def test_an_empty_reason_is_refused(self) -> None:
        # `--allow-red ""` would make the override as anonymous as having no check.
        result = run(
            REPO, FULL_SHA, "--allow-red", "", fixture="CI\tcompleted\tfailure\thttps://x/9\n"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-empty reason", result.stderr)

    def test_an_unrecognised_option_is_refused(self) -> None:
        result = run(REPO, FULL_SHA, "--force", fixture="")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be --allow-red", result.stderr)


class ReleaseScriptTests(unittest.TestCase):
    """`tools/release.sh` must ask the question and honour both answers."""

    def script(self) -> str:
        return (ROOT / "tools" / "release.sh").read_text(encoding="utf-8")

    def test_the_release_script_calls_the_helper(self) -> None:
        self.assertIn("ci-green.sh", self.script())

    def test_the_override_needs_a_reason_in_the_script_too(self) -> None:
        # The helper enforces a reason for its own argument; the script's env-var
        # path is a second door to the same room and needs the same lock.
        self.assertIn("TINYTITAN_RELEASE_ALLOW_RED_CI_REASON", self.script())

    def test_an_overridden_release_requires_the_run_url_in_the_notes(self) -> None:
        text = self.script()
        self.assertIn('CI_NOTES_REQUIRE="$CI_RED_URL"', text)
        start = text.index("CI_NOTES_REQUIRE=")
        end = text.index('REQUIRE_ARGS+=(--require "$CI_NOTES_REQUIRE")')
        self.assertLess(start, end, "the notes guard must come after the precondition")


PYTHON_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$TT_ARGV"
exit 0
"""

# Six values the notes must carry, and the placeholder each is filled from.
PLACEHOLDERS = {
    "SHA256_PENDING": "e" * 64,
    "ARCHIVE_BYTES_PENDING": "12345678",
    "LIBRARY_SHA256_PENDING": "a" * 64,
    "LIBRARY_BYTES_PENDING": "22334455",
    "TOOLS_SHA256_PENDING": "b" * 64,
    "TOOLS_BYTES_PENDING": "66778899",
}


def run_notes_block(
    ci_report: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the release script's own notes block, under the CI answer it is given.

    Two regions are taken verbatim out of `tools/release.sh` — the lines that read
    the helper's answer, and everything from `# --- notes` to `# --- publish` —
    and executed by factory `/bin/bash` with `set -uo pipefail`, the same way the
    script starts. Nothing is reimplemented here: the only thing the harness adds
    is the variables the earlier part of the script would have set by then, and a
    `python3` that records the arguments the compaction step is handed.

    `extra_env` exists for the caller that wants to know what the *operator's*
    environment does to this block; absent it, the run is exactly as before.
    """
    text = (ROOT / "tools" / "release.sh").read_text(encoding="utf-8")
    status_start = text.index('CI_STATUS="$(printf')
    status_end = text.index("# A skipped baseline needs its reason", status_start)
    notes_start = text.index("# --- notes")
    notes_end = text.index("# --- publish", notes_start)
    with tempfile.TemporaryDirectory() as work:
        prelude = [
            "set -uo pipefail",
            'die() { echo "error: $*" >&2; exit 1; }',
            f'SCRIPT_DIR="{ROOT / "tools"}"',
            f'CI_REPORT="{ci_report}"',
            "GOLDEN_SKIPPED=''",
            "GOLDEN_ABSENT=''",
            f'STAGE_ROOT="{work}"',
            'NOTES="$STAGE_ROOT/notes-fixture.md"',
            f'SHA="{PLACEHOLDERS["SHA256_PENDING"]}"',
            f'BYTES="{PLACEHOLDERS["ARCHIVE_BYTES_PENDING"]}"',
            f'LIB_SHA="{PLACEHOLDERS["LIBRARY_SHA256_PENDING"]}"',
            f'LIB_BYTES="{PLACEHOLDERS["LIBRARY_BYTES_PENDING"]}"',
            f'TOOLS_SHA="{PLACEHOLDERS["TOOLS_SHA256_PENDING"]}"',
            f'TOOLS_BYTES="{PLACEHOLDERS["TOOLS_BYTES_PENDING"]}"',
        ]
        stub = pathlib.Path(work) / "python3"
        stub.write_text(PYTHON_STUB, encoding="utf-8")
        stub.chmod(0o755)
        notes = pathlib.Path(work) / "notes-fixture.md"
        notes.write_text("\n".join(PLACEHOLDERS) + "\n", encoding="utf-8")
        argv_path = pathlib.Path(work) / "argv"
        argv_path.write_text("", encoding="utf-8")
        env = {
            "PATH": f"{work}{os.pathsep}/bin:/usr/bin",
            "TT_ARGV": str(argv_path),
            "LANG": "C",
        }
        if extra_env:
            env.update(extra_env)
        script = (
            "\n".join(prelude) + "\n" + text[status_start:status_end] + text[notes_start:notes_end]
        )
        result = subprocess.run(
            ["/bin/bash", "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        result.argv = argv_path.read_text(encoding="utf-8").splitlines()  # type: ignore[attr-defined]
        return result


class NotesGuardRunsTests(unittest.TestCase):
    """The notes guard has to run on the green path, not only on the override.

    AUD-105 added `CI_NOTES_REQUIRE` inside the `overridden` branch and read it
    bare at the compaction step. `tools/release.sh` sets `set -u`, so on a green
    run the read is of a variable that was never assigned — and the abort lands
    after the archives are built and hashed, which is the last place on the
    publish path to discover it. v5.18 predates the line, so no release has run
    through it yet. These tests execute the script's own lines rather than
    asserting on their text, because the text-only assertion above is what let
    the shape look covered.
    """

    def test_the_harness_runs_the_scripts_own_guard_line(self) -> None:
        # A silent slip of the anchors would make every test below pass over an
        # empty slice, so the extracted region is checked to contain the line.
        text = (ROOT / "tools" / "release.sh").read_text(encoding="utf-8")
        notes_start = text.index("# --- notes")
        notes_end = text.index("# --- publish", notes_start)
        self.assertIn(
            'REQUIRE_ARGS+=(--require "$CI_NOTES_REQUIRE")',
            text[notes_start:notes_end],
        )

    def test_a_green_ci_reaches_the_compactor(self) -> None:
        result = run_notes_block("result\tgreen\t-")
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertIn("--max-chars", result.argv)
        self.assertNotIn("unbound variable", result.stderr)

    def test_a_green_ci_does_not_require_a_placeholder_dash(self) -> None:
        # The tempting fix — assign from CI_RED_URL unconditionally — is wrong on
        # this path, because the helper's green line ends in `-` and the notes
        # would then have to quote a dash to be publishable.
        result = run_notes_block("result\tgreen\t-")
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        requires = [result.argv[i + 1] for i, a in enumerate(result.argv) if a == "--require"]
        self.assertNotIn("-", requires)

    def test_an_overridden_ci_requires_the_red_run_url(self) -> None:
        result = run_notes_block("result\toverridden\thttps://x/9")
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        requires = [result.argv[i + 1] for i, a in enumerate(result.argv) if a == "--require"]
        self.assertIn("https://x/9", requires)


if __name__ == "__main__":
    unittest.main()
