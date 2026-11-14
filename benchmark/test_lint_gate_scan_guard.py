"""AUD-271: seven gates in `tools/lint.sh` report `ok` over a scan that read no files.

`tools/lint.sh` is the harness behind eighteen of this repository's gates, and six
of its checks already refuse to certify an empty scan:

    raise "no Swift files under #{root}" if paths.empty?   # unbounded-read
    if [ -z "$scanned" ] || [ "$scanned" -eq 0 ]; then     # func-length, file-length,
                                                           # test-hollow
    if not scanned:                                        # stdout-clean
    print("SCANNED:0") + the shell guard over it           # library-facade

`AGENTS.md` states the rule for the whole family -- "a gate that did not run reading
as a pass is the exact defect it hunts". The guards are named here by their check, not
by line number: every one of them shifted when this fix added lines above it, and a
citation that drifts is worse than none. Seven checks in the same file had no guard
at all, and measured on this host each exited **0** over a tree holding nothing it can
read:

    force-cast  `grep -rnE ... "$ROOT/sources" 2>/dev/null` -- an absent or
                unreadable `sources/` makes grep fail, its status is dropped by the
                process substitution, its stderr is hidden, and `found` stays 0  -> ok
    arch-path   `Path(".").rglob(ext)`, no counter, no expectation              -> ok (none)
    shell       `find ... 2>/dev/null`; the count is printed but never tested    -> ok (0 scripts)
    test-skip   `Path("tests").rglob(...)` yields nothing and raises nothing,
                measured on the interpreter the gate targets                     -> ok (none)
    sendable    empty glob plus no baseline writes an empty baseline and calls
                it a pass                                                        -> ok (baseline created, 0 entries)
    swift-format  `--recursive sources tests benchmark Package.swift examples`, and
                  the tool reads a path argument that is missing or empty without
                  complaint -- found by the sibling sweep, not by the report       -> ok
    python        `rglob` over benchmark/, tools/, docs/ parses nothing on an empty
                  tree while ruff answers "All checks passed!" to no files         -> ok

The remaining modes were measured on the same empty tree and do refuse: `converter`
names its missing dependency, `shellcheck` has its own "no shell scripts found"
branch, `swiftlint` exits on "No lintable files found", `javascript` counts the
packages it checked, and `docs` cannot open a `docs-facts.py` that is not there.
Fail-closed is not the same as counted, and only the seven above are pinned here.

So "the gates are green" -- the evidence every close in the 2026 pre-production
audit cited -- says nothing about code the harness never opened. The trigger is not
hypothetical: this checkout was renamed once and `AGENTS.md` records that receipts
and `.build` broke, func-length's own comment records the pass-on-nothing that
already happened here, and a wrapper that copies `tools/` somewhere else moves `ROOT`
with it.

The tests build a throwaway tree, put a copy of the harness one directory down so
its derived `ROOT` is the stub and nothing here can read or write the real checkout,
and assert both directions: an empty scan is refused and named, and a non-empty tree
that satisfies the rule still passes. Without the second half a guard that always
refused would score as a fix, which is the same injury in the other direction.

Six of the tests are controls over guards that already existed, and they earned
their place: breaking one guard per mutant, the first sweep killed 19 of 20 and the
survivor was `library-facade` -- named above as already guarded, tested by nothing
here. The suite is what changed after that, not the harness. The sweep now breaks one
guard per mutant across all seven new guards and those six controls, and 27 of 27
mutants die; each guard is broken three ways, so it must fire on an empty scan, stay
silent on a clean one, and actually move the exit status.

Run from `benchmark/`:

    python3 -m unittest test_lint_gate_scan_guard
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINT = ROOT / "tools" / "lint.sh"


def write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class GateHarness(unittest.TestCase):
    """Run one mode of the harness against a tree this test builds."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        # `releases/` is one of arch-path's own skip directories, which keeps the
        # harness out of the set it scans: measured, a copy under `harness/` made
        # arch-path's "empty" tree hold one real file and report
        # `ok (1 files scanned)`, so the fixture had not built the case it claimed.
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

    def run_with_stub(self, mode: str, name: str, script: str) -> tuple[int, str]:
        """Run a mode with one external tool replaced, so the branch under test is
        the scan guard rather than the toolchain pin in front of it. `swift-format`
        and `python` both refuse when their tool is absent or off-pin, and a test
        that passes by reaching that refusal would say nothing about the count."""
        bindir = self.root / "bin"
        bindir.mkdir(exist_ok=True)
        stub = write(bindir / name, script)
        stub.chmod(0o755)
        env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}")
        proc = subprocess.run(
            ["/bin/bash", str(self.harness), mode],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        return proc.returncode, proc.stdout + proc.stderr

    def swift_format_stub(self) -> str:
        return '#!/bin/bash\nif [ "$1" = "--find" ]; then echo "/usr/bin/swift-format"; exit 0; fi\nexit 0\n'

    def ruff_stub(self) -> str:
        # The pin is read from the harness itself rather than hardcoded here, so
        # this fixture follows `RUFF_PIN` when it moves instead of going red on a
        # bump that has nothing to do with the guard under test.
        pin = re.search(r'^RUFF_PIN="([^"]+)"', LINT.read_text(encoding="utf-8"), re.M)
        if not pin:
            self.fail("tools/lint.sh no longer declares RUFF_PIN on its own line")
        return (
            '#!/bin/bash\nif [ "$1" = "--version" ]; then echo "ruff %s"; exit 0; fi\nexit 0\n'
            % pin.group(1)
        )

    def assertRefused(self, mode: str, expect: tuple[str, ...] = ()) -> str:
        rc, out = self.run_mode(mode)
        self.assertNotEqual(rc, 0, f"{mode} exited 0 over a scan that read nothing:\n{out}")
        self.assertRegex(out, r"(?m)^  FAIL", f"{mode} refused without saying why:\n{out}")
        self.assertNotRegex(
            out, r"(?m)^\s+ok\b", f"{mode} printed a pass next to its refusal:\n{out}"
        )
        for token in expect:
            self.assertIn(token, out, f"{mode} does not name {token!r}:\n{out}")
        return out

    def assertPassed(self, mode: str) -> str:
        rc, out = self.run_mode(mode)
        self.assertEqual(rc, 0, f"{mode} exited {rc} over a scan that satisfies it:\n{out}")
        return out


class EmptyScanTests(GateHarness):
    def test_unbounded_read_refuses_a_tree_with_no_swift(self):
        self.assertRefused("unbounded-read")

    def test_func_length_refuses_a_tree_with_no_swift(self):
        self.assertRefused("func-length")

    def test_file_length_refuses_a_tree_with_no_swift(self):
        self.assertRefused("file-length")

    def test_test_hollow_refuses_a_tree_with_no_tests(self):
        self.assertRefused("test-hollow")

    def test_library_facade_refuses_a_tree_with_no_lib(self):
        # The scanner prints `SCANNED:0` and exits **0** when the library holds no
        # Swift file, so only the guard over its output can tell that apart from a
        # library with no public surface -- which is the pass it used to report.
        # The empty tree is built with the directory *present* and the allowlist
        # seeded, because otherwise a broken guard is caught by whichever branch
        # runs next: measured, a missing directory fell to the UNRESOLVED branch
        # and a missing allowlist fell to its own refusal, both rc 1, and the
        # mutant survived this file's first sweep. With both of those satisfied,
        # only the count stands between an empty scan and an `ok`.
        write(self.root / "releases" / "library-facade-baseline.txt", "")
        (self.root / "sources" / "TinyTitanLib").mkdir(parents=True)
        self.assertRefused("library-facade", ("no Swift files were scanned",))

    def test_stdout_clean_refuses_a_closure_with_no_swift(self):
        # A closure of one empty target is the shape that used to read `ok`: the
        # manifest parses, the directory exists, and no file was ever opened.
        write(
            self.root / "Package.swift",
            "// swift-tools-version: 6.4\nlet package = Package(\n"
            '    name: "Stub",\n    targets: [\n        .target(\n'
            '            name: "TinyTitanLib",\n            path: "sources/Lib"\n'
            "        )\n    ]\n)\n",
        )
        (self.root / "sources" / "Lib").mkdir(parents=True)
        self.assertRefused("stdout-clean", ("no Swift file",))

    def test_swift_format_refuses_a_tree_with_no_swift(self):
        # Found by the sibling sweep, not by the report: swift-format reads a path
        # argument that is missing or empty without complaint, so the gate that
        # names `sources tests benchmark Package.swift examples` certifies
        # `--strict clean` over a checkout it never opened. Measured on an empty
        # tree before this fix: `ok (xcrun swift-format, --strict clean)`, exit 0.
        rc, out = self.run_with_stub("swift-format", "xcrun", self.swift_format_stub())
        self.assertNotEqual(rc, 0, f"swift-format exited 0 over 0 Swift files:\n{out}")
        self.assertIn("0 Swift files", out)
        self.assertNotRegex(out, r"(?m)^\s+ok\b", f"swift-format passed next to refusing:\n{out}")

    def test_python_floor_refuses_a_tree_with_no_scripts(self):
        # The same shape one tool down: the parse-floor scan walks
        # `benchmark/ tools/ docs/` and says nothing when that walk finds none,
        # while ruff on an empty tree answers "All checks passed!". Measured:
        # `ok (ruff …, parses under 3.13)`, exit 0 over 0 scripts.
        rc, out = self.run_with_stub("python", "ruff", self.ruff_stub())
        self.assertNotEqual(rc, 0, f"python exited 0 over 0 .py files:\n{out}")
        self.assertIn("0 .py files", out)
        self.assertNotRegex(out, r"(?m)^\s+ok\b", f"python passed next to refusing:\n{out}")

    def test_force_cast_refuses_a_tree_with_no_swift(self):
        # The offender list comes from one grep over sources/. With that directory
        # absent there is no offender because there is no input, and `found` cannot
        # tell the two apart -- so the refusal has to name the directory, not just
        # the absence of hits.
        self.assertRefused("force-cast", ("sources",))

    def test_arch_path_refuses_a_tree_with_no_scripts(self):
        self.assertRefused("arch-path", ("0",))

    def test_shell_portability_refuses_a_tree_with_no_scripts(self):
        # It already prints the count; the guard has to test the number it shows
        # rather than depend on an operator reading the line.
        self.assertRefused("shell", ("0",))

    def test_silent_test_skip_refuses_a_tree_with_no_tests(self):
        self.assertRefused("test-skip", ("tests",))

    def test_unchecked_sendable_refuses_a_tree_with_no_swift(self):
        # The baseline is seeded empty here because a missing baseline is the
        # bootstrap path; this assertion is about the scan, not the ratchet.
        write(self.root / "releases" / "unchecked-sendable-baseline.txt", "")
        self.assertRefused("sendable", ("sources",))

    def test_unchecked_sendable_refuses_before_it_writes_a_baseline(self):
        # The other half of the same defect: an empty scan with no allowlist file
        # used to write an empty baseline and call it `ok (baseline created, 0
        # entries)`, which then blesses whatever the tree really holds. The guard
        # has to read first, so the file must still be absent afterwards.
        rc, out = self.run_mode("sendable")
        baseline = self.root / "releases" / "unchecked-sendable-baseline.txt"
        self.assertNotEqual(rc, 0, f"sendable exited 0 over an empty scan:\n{out}")
        self.assertRegex(out, r"(?m)^  FAIL", out)
        self.assertFalse(baseline.exists(), f"the empty scan wrote a baseline:\n{out}")


class PassDirectionTests(GateHarness):
    """Each gate must still report a pass over a non-empty tree that satisfies its
    rule, so a zero-scan guard cannot be satisfied by refusing everything."""

    def test_force_cast_passes_a_clean_sources_tree(self):
        write(
            self.root / "sources" / "Clean.swift",
            "import Foundation\n\nfunc f(_ v: Any) -> Int {\n    return Int(v) ?? 0\n}\n",
        )
        self.assertPassed("force-cast")

    def test_arch_path_passes_a_tree_with_no_offending_literal(self):
        write(self.root / "tools" / "clean.sh", "#!/bin/bash\nswift build -c release\n")
        self.assertPassed("arch-path")

    def test_shell_portability_passes_a_guarded_script(self):
        write(
            self.root / "tools" / "clean.sh",
            '#!/bin/bash\nset -uo pipefail\nargs=("a" "b")\n'
            'printf "%s\\n" ${args[@]+"${args[@]}"}\n',
        )
        self.assertPassed("shell")

    def test_silent_test_skip_passes_a_body_that_does_not_escape(self):
        write(
            self.root / "tests" / "CleanTests.swift",
            "import Testing\n\n@Test func reads() {\n    let n = 3\n    #expect(n == 3)\n}\n",
        )
        self.assertPassed("test-skip")

    def test_unchecked_sendable_passes_a_declared_conformance(self):
        write(self.root / "releases" / "unchecked-sendable-baseline.txt", "")
        write(
            self.root / "sources" / "Holder.swift",
            "// unchecked-invariant: value is set once during init\n"
            "final class Holder: @unchecked Sendable {\n    let value: Int\n"
            "    init(_ value: Int) { self.value = value }\n}\n",
        )
        self.assertPassed("sendable")

    def test_swift_format_passes_a_tree_with_a_swift_file(self):
        write(self.root / "sources" / "One.swift", "let answer = 42\n")
        rc, out = self.run_with_stub("swift-format", "xcrun", self.swift_format_stub())
        self.assertEqual(rc, 0, f"swift-format refused a tree it can read:\n{out}")

    def test_python_floor_passes_a_tree_with_a_script(self):
        write(self.root / "benchmark" / "one.py", "print('one')\n")
        rc, out = self.run_with_stub("python", "ruff", self.ruff_stub())
        self.assertEqual(rc, 0, f"python refused a tree it can read:\n{out}")


class OffenderTests(GateHarness):
    """And each must still name a real offender, so the empty-scan branch is not the
    only one the harness has ever exercised."""

    def test_force_cast_names_an_unexempt_force_cast(self):
        write(
            self.root / "sources" / "Bad.swift",
            "func f(_ v: Any) -> Int {\n    return v as! Int\n}\n",
        )
        rc, out = self.run_mode("force-cast")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("Bad.swift:2", out)

    def test_arch_path_names_the_hardcoded_triple(self):
        # The sanctioned opt-out, on the line the gate reads: this fixture *is* the
        # offender the gate has to find, and the assertion below is that it does.
        triple = "arm64-apple-macosx"  # lint:allow-arch-path the fixture names the literal to prove the gate finds it
        write(
            self.root / "tools" / "bad.sh",
            f"#!/bin/bash\nls .build/{triple}/release\n",
        )
        rc, out = self.run_mode("arch-path")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("bad.sh:2", out)

    def test_shell_portability_names_a_bash4_builtin(self):
        write(self.root / "tools" / "bad.sh", "#!/bin/bash\nmapfile -t lines < list\n")
        rc, out = self.run_mode("shell")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("bad.sh:2", out)

    def test_silent_test_skip_names_an_env_shaped_early_return(self):
        write(
            self.root / "tests" / "SkipTests.swift",
            "import Testing\n\n@Test func gated() {\n"
            '    if ProcessInfo.processInfo.environment["TT_MODEL"] == nil {\n'
            "    } else { return }\n    #expect(true)\n}\n",
        )
        rc, out = self.run_mode("test-skip")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("SkipTests.swift:5", out)

    def test_unchecked_sendable_names_an_undocumented_conformance(self):
        write(self.root / "releases" / "unchecked-sendable-baseline.txt", "")
        write(
            self.root / "sources" / "Quiet.swift",
            "final class Quiet: @unchecked Sendable {\n    var count = 0\n}\n",
        )
        rc, out = self.run_mode("sendable")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("Quiet", out)


if __name__ == "__main__":
    unittest.main()
