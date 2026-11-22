"""AUD-267: the release must refuse a GitHub answer it did not get.

`tools/release.sh` asked GitHub one question in its precondition block and threw
the answer away:

    gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1 \\
      && die "a Release for $TAG already exists on $REPO"

`&&` acts on status 0 only, and `>/dev/null 2>&1` discards both streams, so the
script can learn one thing from that call: *a release exists*. Every other
outcome -- including every outcome where no question was answered -- reads as
"no release exists, safe to publish". Measured on gh 2.102.0, each with the
release's own options (`set -uo pipefail`, which has no `-e`, so nothing else
looks at the status either; /tmp/aud267-before.sh):

| what actually happened            | gh's answer                       | release.sh concluded |
|-----------------------------------|-----------------------------------|----------------------|
| a release exists                  | rc 0                              | refuse (correct)     |
| no release, repo readable         | rc 1 `release not found`          | publish on           |
| **repo invisible or renamed**     | rc 1 `release not found`          | publish on           |
| **token revoked**                 | rc 1 `gh: Bad credentials (401)`  | publish on           |
| **api unreachable**               | rc 1 `dial tcp ... refused`       | publish on           |
| **gh not installed**              | rc 127                            | publish on           |

The third row is the one gh makes indistinguishable by design: an invisible
repository answers `release not found` -- the identical status *and* the
identical message a missing release gets (measured, /tmp/aud267-m2.txt). So no
reading of that one call can tell "absent" from "I could not ask", and the
precondition block is exactly where the difference matters: it is the last cheap
check before the clean scratch build and the golden baselines, and every row
above is something an operator can fix in a minute.

The fix is two authoritative queries instead of one discarded one: `gh api
repos/$REPO` first, which proves the repository is readable and the credentials
work, and `gh api repos/$REPO/releases/tags/$TAG` second, where a 404 is the
only answer that means absent. `gh api` rather than `gh release view` because it
labels its failures with the HTTP status (`gh: Not Found (HTTP 404)`, measured)
and `gh release view` does not.

The sibling in the same block is:

    git ls-remote --tags origin 2>/dev/null | grep -q "refs/tags/$TAG$"

which has the same shape in a milder form: it fails closed, but `2>/dev/null`
throws away git's reason. A remote that cannot be read answers rc 128 with
`fatal: 'origin' does not appear to be a git repository` (measured) and the
operator is told the tag `is not pushed to origin`, which sends them to
`git push` when the push is not the problem. Its pattern is a second finding:
the tag is interpolated into a regex, so `v5.18` matches a remote carrying only
`v5X18` (measured, /tmp/aud267-match.sh).

These tests drive the script's own verdict functions, extracted verbatim along
with the `die` they call, against stub `gh` and `git` programs on `PATH` that
replay the measured status and body of each shape above through environment
variables. The stubs are the point: no test here opens a socket, needs a token,
or asks GitHub anything, and each answer is a byte-for-byte copy of what gh
really said. Nothing runs `swift test`, builds, or loads a model.

    cd benchmark && python3 -m unittest test_release_precondition_gate -v
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "release.sh"

REPO = "Pummelchen/TinyTitan"
TAG = "v5.19"

# Every body below is measured output of gh 2.102.0 against the real endpoint it
# is named for (/tmp/aud267-m2.txt, /tmp/aud267-m3.txt).
REPO_OK_BODY = "Pummelchen/TinyTitan\n"
RELEASE_OK_BODY = "v5.19\n"
RELEASE_404_ERR = "gh: Not Found (HTTP 404)"
REPO_404_ERR = "gh: Not Found (HTTP 404)"
REPO_401_ERR = "gh: Bad credentials (HTTP 401)"
NETWORK_ERR = (
    'Get "https://127.0.0.1:1/api/v3/repos/Pummelchen/TinyTitan": '
    "dial tcp 127.0.0.1:1: connect: connection refused"
)
SERVER_500_ERR = "gh: Server Error (HTTP 500)"

# git ls-remote's real lines: sha, tab, ref. Peeled tag lines carry `^{}`.
ORIGIN_WITH_TAG = f"aa11bb22cc\trefs/tags/{TAG}\n33dd44ee55\trefs/tags/{TAG}^{{}}\n"
ORIGIN_WITHOUT_TAG = "aa11bb22cc\trefs/tags/v5.18\naa11bb22cc\trefs/tags/v5.18^{}\n"
# What `grep "refs/tags/v5.18$"` accepts for the tag `v5.18`.
ORIGIN_DOT_CONFLATION = "ff66gg77hh\trefs/tags/v5X18\n"
LS_REMOTE_FAILURE = (
    "fatal: 'origin' does not appear to be a git repository\n"
    "fatal: Could not read from remote repository.\n"
)

GH_STUB = """#!/bin/bash
# Dispatch on the endpoint path the way gh routes it: the release query is the
# argument that contains `/releases/`. rc/body come in through the environment.
path=
for a in "$@"; do case "$a" in repos/*) path="$a";; esac; done
case "$path" in
  */releases/*) rc="$STUB_RELEASE_RC"; body="$STUB_RELEASE_BODY";;
  *)            rc="$STUB_REPO_RC";  body="$STUB_REPO_BODY";;
esac
printf '%s\\n' "$body" >&2
printf '%s\\n' "$body"
exit "$rc"
"""

GIT_STUB = """#!/bin/bash
if [ "$1" = ls-remote ]; then
  printf '%s\\n' "$STUB_GIT_OUT" >&2
  printf '%s\\n' "$STUB_GIT_OUT"
  exit "$STUB_GIT_RC"
fi
exec /usr/bin/git "$@"
"""


def _function(name: str) -> str:
    """One function from the script, text and all."""
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", SCRIPT.read_text(), re.M | re.S)
    if match is None:
        raise AssertionError(
            f"{SCRIPT} no longer holds a {name} function to drive -- the release's "
            "GitHub and origin preconditions are decided somewhere else again"
        )
    return match.group(0)


class _StubbedScript(unittest.TestCase):
    """Runs one extracted verdict function with stub `gh` and `git` on PATH.

    `/usr/bin:/bin` is the tail, and it carries no `gh` on this platform
    (`/usr/bin/gh` does not exist), so a test that never writes the gh stub
    really does run with gh not installed.
    """

    def setUp(self) -> None:
        self.bin = pathlib.Path(tempfile.mkdtemp(prefix="aud267-stub-"))
        self.addCleanup(shutil.rmtree, self.bin, True)
        self.env: dict[str, str] = {}

    def _stub(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def _gh(self, *, repo_rc: int, repo_out: str, release_rc: int, release_out: str) -> None:
        self._stub("gh", GH_STUB)
        self.env.update(
            {
                "STUB_REPO_RC": str(repo_rc),
                "STUB_REPO_BODY": repo_out,
                "STUB_RELEASE_RC": str(release_rc),
                "STUB_RELEASE_BODY": release_out,
            }
        )

    def _git(self, *, rc: int, out: str) -> None:
        self._stub("git", GIT_STUB)
        self.env.update({"STUB_GIT_RC": str(rc), "STUB_GIT_OUT": out})

    def _run(self, call: str) -> subprocess.CompletedProcess[str]:
        fn_name = call.split()[0]
        body = "\n".join([_function("die"), _function(fn_name), call])
        env = dict(os.environ)
        env.update(self.env)
        # The stub dir holds the functions' only externals; /usr/bin:/bin is the
        # factory tail, which has no gh. Nothing here reaches GitHub or the
        # repository's own git.
        env["PATH"] = f"{self.bin}:/usr/bin:/bin"
        return subprocess.run(
            ["/bin/bash", "-c", f"set -uo pipefail\n{body}\n"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(self.bin),
        )


class ReleaseExistsVerdictTests(_StubbedScript):
    def test_a_proven_existing_release_is_refused(self) -> None:
        self._gh(repo_rc=0, repo_out=REPO_OK_BODY, release_rc=0, release_out=RELEASE_OK_BODY)
        proc = self._run(f'release_exists_verdict "{REPO}" "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("already exists", proc.stderr, proc.stderr)
        self.assertIn(TAG, proc.stderr)

    def test_a_proven_absent_release_passes_and_names_what_it_read(self) -> None:
        self._gh(repo_rc=0, repo_out=REPO_OK_BODY, release_rc=1, release_out=RELEASE_404_ERR)
        proc = self._run(f'release_exists_verdict "{REPO}" "{TAG}"')
        self.assertEqual(proc.returncode, 0, f"a proven absence was refused: {proc.stderr}")
        self.assertIn(TAG, proc.stdout, f"the pass should name what it proved: {proc.stdout}")
        self.assertIn(REPO, proc.stdout, proc.stdout)

    def test_an_invisible_repository_is_not_read_as_no_release(self) -> None:
        """gh's `release not found` for a repository it cannot see is the defect."""
        self._gh(repo_rc=1, repo_out=REPO_404_ERR, release_rc=1, release_out=RELEASE_404_ERR)
        proc = self._run(f'release_exists_verdict "{REPO}" "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, "the release path accepted a repo it cannot read")
        self.assertNotIn("already exists", proc.stderr, proc.stderr)
        self.assertIn(
            "repositor", proc.stderr.lower(), f"should name the repository query: {proc.stderr}"
        )

    def test_a_revoked_token_is_not_read_as_no_release(self) -> None:
        self._gh(repo_rc=1, repo_out=REPO_401_ERR, release_rc=1, release_out=RELEASE_404_ERR)
        proc = self._run(f'release_exists_verdict "{REPO}" "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, "the release path published on a 401")
        self.assertIn("401", proc.stderr, f"should carry gh's status: {proc.stderr}")

    def test_an_unreachable_api_is_not_read_as_no_release(self) -> None:
        self._gh(repo_rc=1, repo_out=NETWORK_ERR, release_rc=1, release_out=RELEASE_404_ERR)
        proc = self._run(f'release_exists_verdict "{REPO}" "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, "the release path published over a dead network")
        self.assertIn("connection refused", proc.stderr, proc.stderr)

    def test_a_missing_gh_is_refused_where_it_can_say_so(self) -> None:
        """rc 127 is `command not found`; the release learns of it at line 593 today.

        The refusal has to name the install. Falling through to the query refusal
        also exits non-zero -- bash's own `gh: command not found` is even captured
        into its message, so a test that only checks the status and the word `gh`
        passes with the guard deleted (measured, /tmp/aud267-mut.txt M1) while
        telling the operator to fix their credentials.
        """
        self._gh(repo_rc=0, repo_out=REPO_OK_BODY, release_rc=1, release_out=RELEASE_404_ERR)
        (self.bin / "gh").unlink()
        proc = self._run(f'release_exists_verdict "{REPO}" "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, "the release path ran on a host with no gh")
        self.assertIn("not installed", proc.stderr, f"should name the install: {proc.stderr}")
        self.assertNotIn(
            "repository query",
            proc.stderr.lower(),
            f"misdiagnosed as a failed query: {proc.stderr}",
        )
        self.assertNotIn("already exists", proc.stderr, proc.stderr)

    def test_a_release_query_that_gives_no_http_status_is_refused(self) -> None:
        """The repository is readable, then the release query itself dies."""
        self._gh(repo_rc=0, repo_out=REPO_OK_BODY, release_rc=1, release_out=NETWORK_ERR)
        proc = self._run(f'release_exists_verdict "{REPO}" "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, "an answer with no status read as a proven 404")
        self.assertIn("connection refused", proc.stderr, proc.stderr)

    def test_a_release_query_that_fails_with_a_server_error_is_refused(self) -> None:
        self._gh(repo_rc=0, repo_out=REPO_OK_BODY, release_rc=1, release_out=SERVER_500_ERR)
        proc = self._run(f'release_exists_verdict "{REPO}" "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, "a 500 was read as `no release exists`")
        self.assertIn("500", proc.stderr, proc.stderr)

    def test_the_release_path_never_discards_a_gh_answer(self) -> None:
        """`>/dev/null 2>&1` on a query is the shape this finding is about.

        A `command -v gh` probe is not a query -- it has no answer to discard, and
        the guard this finding adds is exactly one of those -- so the scan is on
        the subcommands that speak to GitHub.
        """
        offenders = [
            line.strip()
            for line in SCRIPT.read_text().splitlines()
            if re.search(r"\bgh\s+(?:release|api)\b.*>/dev/null\s+2>&1", line)
        ]
        self.assertEqual(offenders, [], f"gh answers thrown away: {offenders}")


class GitTagPushedVerdictTests(_StubbedScript):
    def test_a_listed_tag_passes_and_says_what_it_saw(self) -> None:
        self._git(rc=0, out=ORIGIN_WITH_TAG)
        proc = self._run(f'git_tag_pushed_verdict "{TAG}"')
        self.assertEqual(proc.returncode, 0, f"a pushed tag was refused: {proc.stderr}")
        self.assertIn(TAG, proc.stdout, proc.stdout)

    def test_a_peeled_ref_line_alone_counts_as_pushed(self) -> None:
        """`--tags` lists the tag object and the peeled commit; either is the tag."""
        self._git(rc=0, out=f"33dd44ee55\trefs/tags/{TAG}^{{}}\n")
        proc = self._run(f'git_tag_pushed_verdict "{TAG}"')
        self.assertEqual(proc.returncode, 0, f"a peeled-only listing was refused: {proc.stderr}")

    def test_an_absent_tag_is_refused_as_unpushed(self) -> None:
        self._git(rc=0, out=ORIGIN_WITHOUT_TAG)
        proc = self._run(f'git_tag_pushed_verdict "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("not pushed", proc.stderr, proc.stderr)

    def test_a_remote_that_lists_nothing_is_refused_as_unpushed(self) -> None:
        """An empty answer is not a match; vacuity is this audit's defect class."""
        self._git(rc=0, out="")
        proc = self._run(f'git_tag_pushed_verdict "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, "an empty ref list was read as containing the tag")

    def test_a_failing_ls_remote_names_git_and_not_the_push(self) -> None:
        """rc 128 means the remote could not be read, not that the tag is missing."""
        self._git(rc=128, out=LS_REMOTE_FAILURE)
        proc = self._run(f'git_tag_pushed_verdict "{TAG}"')
        self.assertNotEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(
            "ls-remote", proc.stderr.lower(), f"should name the command that failed: {proc.stderr}"
        )
        self.assertIn("fatal", proc.stderr, f"should carry git's reason: {proc.stderr}")

    def test_a_dot_in_the_tag_is_not_a_wildcard(self) -> None:
        """`grep "refs/tags/v5.18$"` accepts a remote holding only `v5X18`."""
        self._git(rc=0, out=ORIGIN_DOT_CONFLATION)
        proc = self._run('git_tag_pushed_verdict "v5.18"')
        self.assertNotEqual(proc.returncode, 0, "the tag was matched by a pattern, not by its ref")

    def test_a_shorter_tag_is_not_matched_by_a_longer_ref(self) -> None:
        """The other end of the same conflation: `v5.1` is not `refs/tags/v5.18`.

        A match that stops at the end of the tag rather than the end of the field
        reads "pushed" from a remote that has never heard of the tag, which is what
        the mutation sweep found when the whole-field case pattern was relaxed to a
        prefix.
        """
        self._git(rc=0, out=ORIGIN_WITHOUT_TAG)
        proc = self._run('git_tag_pushed_verdict "v5.1"')
        self.assertNotEqual(proc.returncode, 0, "v5.1 was read off a ref line that says v5.18")


class PreconditionWiringTests(unittest.TestCase):
    def _block(self) -> str:
        text = SCRIPT.read_text()
        start = text.index('step "preconditions"')
        end = text.index("# --- compile/test CI green on the commit being tagged")
        return text[start:end]

    def test_the_preconditions_run_both_verdicts(self) -> None:
        block = self._block()
        for name in ("release_exists_verdict", "git_tag_pushed_verdict"):
            calls = [
                line.strip() for line in block.splitlines() if re.search(rf"{name}\s+\S", line)
            ]
            self.assertEqual(len(calls), 1, f"{name} is called {len(calls)} times: {calls}")

    def test_the_preconditions_no_longer_suppress_the_cause(self) -> None:
        block = self._block()
        self.assertNotIn("2>/dev/null", block, "an error stream discarded before a grep")
        self.assertNotRegex(
            block,
            r"gh\s+release\s+view[^\n]*>/dev/null",
            "the existence query is back, and its answer is still thrown away",
        )


if __name__ == "__main__":
    unittest.main()
