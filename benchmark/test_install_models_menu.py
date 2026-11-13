"""The install tool's contract with the person running it.

Two things `tools/install_models.sh` must not get wrong:

- `--choose` starts a real download on a single keystroke, so "the caller pressed
  Enter" and "the caller's stdin ended" have to mean different things. It was
  measured the wrong way round on 2026-09-24 — closing stdin started the
  recommended 36.9 GB download — so this file pins both directions.
- Every download and conversion is staged under one absolute work root. The
  staging paths were bare `.build/...` until 2026-09-24, which put a factory-new
  install's tens-to-hundreds of GB in `~/.build` — outside the install root —
  because a relative path follows the caller's working directory and the
  installer never changes it.

`install_one` is replaced with a stub before the menu is called: a regression
there prints `INSTALLED <model>` instead of downloading one. The path tests read
the script. Nothing here touches the network or `models/`.

    cd benchmark && python3 -m unittest test_install_models_menu -v
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
SCRIPT = ROOT / "tools/install_models.sh"

# `source` also runs the script's own dispatcher, which with no arguments prints
# the status table; it is silenced, and `install_one` is stubbed straight after.
HARNESS = f"""
set -uo pipefail
set --
source "{SCRIPT}" >/dev/null 2>&1 || true
install_one() {{ echo "INSTALLED $1"; }}
choose_model
echo "rc=$?"
"""


def run_menu(stdin: str | None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", HARNESS],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class InstallMenuAnswerTests(unittest.TestCase):
    def test_an_empty_line_takes_the_default(self) -> None:
        result = run_menu("\n")
        self.assertIn("INSTALLED ornith15-8bit", result.stdout + result.stderr)

    def test_a_number_installs_that_row(self) -> None:
        # Row 14 is Qwen 3.5 2B 4-bit in the current catalogue; the bf16-dense
        # row sits at 3, so this also pins that a row's number moved with its
        # position rather than being read from a stale list.
        result = run_menu("14\n")
        self.assertIn("INSTALLED qwen35-2b", result.stdout + result.stderr)

    def test_end_of_input_installs_nothing(self) -> None:
        # The bug: `read` failing left the reply empty, and the empty reply fell
        # through to the default. stdin closed must refuse, not choose for the
        # caller.
        result = run_menu("")
        output = result.stdout + result.stderr
        self.assertNotIn("INSTALLED", output)
        self.assertIn("rc=2", result.stdout)
        self.assertIn("nothing was installed", output)

    def test_a_choice_outside_the_list_installs_nothing(self) -> None:
        result = run_menu("0\n")
        output = result.stdout + result.stderr
        self.assertNotIn("INSTALLED", output)
        self.assertIn("not a choice", output)


class MenuRowPaddingTests(unittest.TestCase):
    """A zero-padded row number must mean the row its digits name.

    AUD-305, measured on /bin/bash 3.2.57 with the shipped 16-row catalogue. The
    guard and the index both read the reply through arithmetic, and `(( ))` takes
    a leading zero as octal, so three different things go wrong:

    - `010` is octal 8, legitimately in range, and installs row 8 -- a 20.0 GB
      KAT-Coder 4-bit -- for a person who pointed at row 10, the 5.3 GB Qwen 3.5
      4B 8-bit. Nothing errors; the wrong model is downloaded.
    - `08` and `09` make the range test error and answer neither branch. That
      test asks the question in the negative (`below 1, or above the count`), so
      "no verdict" reads as ACCEPTED, and the very next line indexes the
      catalogue with the same unreadable value: the script dies with
      `09: value too great for base` and installs nothing.
    - `01` to `07` land on the right row, because octal and decimal agree on a
      single digit. Those two cases are pinned here as neighbours and they pass
      before the repair; they are in this file because the repair must not lose
      them.

    The repair forces the decimal base rather than refusing a leading zero, and
    that is a measured choice, not a preference: this tree refuses a padded
    number where the value flows outward to readers it cannot all fix (`PORT`,
    which reaches curl URLs, the server's argv and the route writer), and reads
    it in decimal where the value is consumed on the spot (`ram_tier` echoes
    `10#`-forced GB, `valid_concurrency` forces the base at each read). A menu
    row has exactly two readers, both inside these three lines. Refusing instead
    would turn `01` through `07` -- replies that install the right row today --
    into new refusals, which is a rule no reader of this menu has.
    """

    def installed(self, stdin: str) -> tuple[str, str]:
        result = run_menu(stdin)
        match = re.search(r"INSTALLED (\S+)", result.stdout)
        return (match.group(1) if match else "", result.stdout + result.stderr)

    def test_a_zero_padded_row_installs_the_row_its_digits_name(self) -> None:
        for padded, plain in (("01", "1"), ("07", "7"), ("08", "8"), ("010", "10"), ("018", "18")):
            with self.subTest(reply=padded):
                want, _ = self.installed(plain + "\n")
                self.assertTrue(want, f"the plain reply {plain} names no row")
                got, output = self.installed(padded + "\n")
                self.assertEqual(got, want, output)
                self.assertNotIn("value too great for base", output)

    def test_a_padded_row_outside_the_list_still_refuses(self) -> None:
        # Forcing the base is not a pass: 20 and 21 are still not rows (the
        # catalogue has 19), and 0 is still not a row, padded or not.
        for reply in ("00", "020", "021"):
            with self.subTest(reply=reply):
                got, output = self.installed(reply + "\n")
                self.assertEqual(got, "", output)
                self.assertIn("not a choice", output)
                self.assertNotIn("value too great for base", output)


class StagingPathTests(unittest.TestCase):
    """Downloads stage under one absolute root, whatever the caller's cwd is."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text()

    def test_the_root_is_derived_absolutely_from_the_script_path(self) -> None:
        self.assertIn('ROOT="$(cd "$(dirname "$0")/.." && pwd)"', self.script)

    def test_the_work_root_defaults_under_the_install_root(self) -> None:
        self.assertIn('WORK="${TINYTITAN_WORK_DIR:-$ROOT/.build}"', self.script)

    def test_no_staging_path_is_relative_to_the_working_directory(self) -> None:
        # Comments may name `.build/release` (the repack binary default) or the
        # old shape; what matters is that no *code* argument or variable is a
        # bare relative `.build/...` path any more.
        for number, line in enumerate(self.script.splitlines(), start=1):
            if line.lstrip().startswith("#") or ".build/release" in line:
                continue
            with self.subTest(line=number):
                self.assertNotIn(".build/", line)

    def test_download_and_conversion_arguments_use_the_work_root(self) -> None:
        for argument in (
            '--work "$WORK/${preset}-shards"',
            '--output "$WORK/qwen38-affine-${width}bit"',
            '--work "$WORK/qwen38-shards"',
            '--output "$WORK/${preset}-affine"',
        ):
            with self.subTest(argument=argument):
                self.assertIn(argument, self.script)

    def test_qwen38_ships_exactly_two_types_with_hardcoded_recipes(self) -> None:
        # The 4-bit type is the 4E-8R-16N hybrid (experts 4-bit, the rest 8-bit,
        # the n-gram table fp16) and installs under a directory named for the
        # recipe; the 8-bit type is uniform. The MTP draft head is not an
        # install type, and the repacker's no-Python streaming hint must not
        # offer a qwen38 build the installer would not produce.
        for fragment in (
            '"qwen38flash|qwen3.8-flash-next_125B_A6B_4E-8R-16N|4|convert"',
            '"qwen38flash-8bit|qwen3.8-flash-next_125B_A6B_8Bit|8|convert"',
            'rest8_arg="--rest-8bit"',
            'model_id="qwen3.8-flash-next-8bitrest"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)
        self.assertNotIn("convert_qwen38_mtp", self.script)
        self.assertNotIn("qwen3.8-flash-next_125B_A6B_4Bit", self.script)
        self.assertNotIn("qwen38flash-mtp", self.script)
        self.assertNotIn("qwen3.8-flash-next_125B_A6B_MTP_4Bit", self.script)

    def test_relative_staging_paths_are_not_quoted_in_a_command(self) -> None:
        # Belt and braces for a future edit that reintroduces the old spelling.
        self.assertIsNone(re.search(r'"\.build/(?!release)', self.script))


class DiskSpaceAndCleanupTests(unittest.TestCase):
    """A download starts only if it can finish; staging leaves when it cannot help.

    Everything runs against temp volumes with a stubbed `df`, a stubbed repack
    binary and `/usr/bin/true` as the interpreter, so no test here can fetch a
    byte. The stub repack prints `FAKE-REPACK …`, which is how a test sees that
    an install would have started.
    """

    ORNITH_4 = "ornith-1.5_35B_A3B_4Bit"
    ORNITH_8 = "ornith-1.5_35B_A3B_8Bit"

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-install-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.work = self.root / "work"
        self.models = self.root / "models"
        self.bin = self.root / "bin"
        for path in (self.work, self.models, self.bin):
            path.mkdir(parents=True)
        repack = self.bin / "TinyTitanRepack"
        repack.write_text('#!/bin/sh\necho "FAKE-REPACK $*"\n')
        repack.chmod(0o755)
        self.df_dir = self.root / "dfbin"
        self.df_dir.mkdir()

    def stub_df(self, available_gb: int) -> None:
        df = self.df_dir / "df"
        df.write_text(
            f'#!/bin/sh\necho "Filesystem 1G-blocks Used Available Capacity Mounted on"\n'
            f'echo "/dev/test 10000 1 {available_gb} 1% /"\n'
        )
        df.chmod(0o755)

    def run_tool(self, *args: str, available_gb: int | None = None):
        env = dict(
            os.environ,
            TINYTITAN_WORK_DIR=str(self.work),
            TINYTITAN_MODELS_DIR=str(self.models),
            TINYTITAN_BIN_DIR=str(self.bin),
            TINYTITAN_PYTHON="/usr/bin/true",
        )
        if available_gb is not None:
            self.stub_df(available_gb)
            env["PATH"] = f"{self.df_dir}:{env['PATH']}"
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def install(self, name: str) -> None:
        (self.models / name).mkdir(parents=True, exist_ok=True)

    def stage(self, relative: str, megabytes: int) -> pathlib.Path:
        path = self.work / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "blob").write_bytes(b"x" * (megabytes * 1024))
        return path

    def test_a_short_volume_refuses_before_the_download_starts(self) -> None:
        result = self.run_tool("ornith15", available_gb=3)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("but about", output)
        self.assertIn("Not starting ornith15", output)
        self.assertIn("TINYTITAN_WORK_DIR", output)
        self.assertNotIn("FAKE-REPACK", output)
        self.assertFalse((self.models / self.ORNITH_4).exists())

    def test_solo_122b_staging_needs_the_retirement_headroom(self) -> None:
        # The 4-bit row is 66 GB; the install (69 GB including the reserve)
        # fits in 80 GB, but the staging volume needs the snapshot plus the
        # pre-unlink headroom, so 80 must refuse while 100 proceeds. The
        # converter is `/usr/bin/true` here, so no byte is fetched.
        short = self.run_tool("ornith122", available_gb=80)
        output = short.stdout + short.stderr
        self.assertNotEqual(short.returncode, 0, output)
        self.assertIn("Not starting ornith122", output)
        self.assertNotIn("FAKE-REPACK", output)

        enough = self.run_tool("ornith122", available_gb=100)
        output = enough.stdout + enough.stderr
        self.assertEqual(enough.returncode, 0, output)
        self.assertIn("FAKE-REPACK", output)
        # The import consumes its snapshot; the guard's estimate depends on it.
        self.assertIn("--consume-snapshot", output)

    def test_enough_space_lets_the_install_proceed(self) -> None:
        result = self.run_tool("ornith15", available_gb=9999)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("installing ornith15", output)
        self.assertIn("FAKE-REPACK", output)

    def test_the_precheck_can_be_skipped_explicitly(self) -> None:
        env = dict(
            os.environ,
            TINYTITAN_WORK_DIR=str(self.work),
            TINYTITAN_MODELS_DIR=str(self.models),
            TINYTITAN_BIN_DIR=str(self.bin),
            TINYTITAN_PYTHON="/usr/bin/true",
            TINYTITAN_SKIP_DISK_CHECK="1",
        )
        self.stub_df(3)
        env["PATH"] = f"{self.df_dir}:{env['PATH']}"
        result = subprocess.run(
            ["bash", str(SCRIPT), "ornith15"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertIn("FAKE-REPACK", result.stdout + result.stderr)

    def test_staging_for_a_missing_width_is_kept_and_explained(self) -> None:
        self.install(self.ORNITH_4)
        snapshot = self.stage("ornith15-affine-4bit", 1)
        result = self.run_tool("ornith15")
        output = result.stdout + result.stderr
        self.assertTrue(snapshot.exists(), "the reusable snapshot was removed")
        self.assertIn("kept", output)
        self.assertIn("clean", output)

    def test_staging_goes_once_both_widths_are_installed(self) -> None:
        for name in (self.ORNITH_4, self.ORNITH_8):
            self.install(name)
        snapshot = self.stage("ornith15-affine-4bit", 1)
        result = self.run_tool("ornith15")
        self.assertFalse(snapshot.exists())
        self.assertIn("staging cleaned", result.stdout + result.stderr)

    def test_draft_head_staging_is_removed_immediately(self) -> None:
        # Nothing reuses a draft head's source shard, installed or not.
        self.install("ornith-1.5_35B_A3B_MTP_4Bit")
        source = self.stage("ornith-mtp-src", 1)
        result = self.run_tool("ornith15-mtp")
        self.assertFalse(source.exists())
        self.assertIn("staging cleaned", result.stdout + result.stderr)

    def test_clean_keeps_staging_an_uninstalled_width_still_needs(self) -> None:
        self.install(self.ORNITH_4)
        snapshot = self.stage("ornith15-affine-4bit", 1)
        result = self.run_tool("clean")
        self.assertTrue(snapshot.exists())
        self.assertIn("still staged", result.stdout + result.stderr)

        self.install(self.ORNITH_8)
        result = self.run_tool("clean")
        self.assertFalse(snapshot.exists())
        self.assertIn("nothing left staged", result.stdout + result.stderr)

    def test_status_shows_what_is_staged(self) -> None:
        self.stage("ornith15-affine-4bit", 1)
        result = self.run_tool()
        self.assertIn("staging:", result.stdout)


class DownloadWordingTests(unittest.TestCase):
    """A run downloads, and its output has to say so.

    Every catalogue row has always fetched the model's pinned bf16 release
    before converting it, but the line printed when that fetch starts said only
    `converting`, the status table described a row with an internal branch name
    (`convert_qwen35moe`), and the menu's size column was labelled `on disk`.
    Read together, those told a person the release had to be downloaded by hand
    first. The behaviour is unchanged; these pin the words that describe it.
    """

    def run_tool(self, *args: str):
        root = pathlib.Path(tempfile.mkdtemp(prefix="tt-wording-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for name in ("work", "models", "bin"):
            (root / name).mkdir()
        repack = root / "bin" / "TinyTitanRepack"
        repack.write_text('#!/bin/sh\necho "FAKE-REPACK $*"\n')
        repack.chmod(0o755)
        env = dict(
            os.environ,
            TINYTITAN_WORK_DIR=str(root / "work"),
            TINYTITAN_MODELS_DIR=str(root / "models"),
            TINYTITAN_BIN_DIR=str(root / "bin"),
            TINYTITAN_PYTHON="/usr/bin/true",
            TINYTITAN_SKIP_DISK_CHECK="1",
        )
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_a_run_says_it_downloads_before_it_converts(self) -> None:
        result = self.run_tool("ornith15")
        output = result.stdout + result.stderr
        self.assertIn("downloading and converting", output)
        self.assertIn("70 GB", output)

    def test_status_names_the_download_not_the_branch(self) -> None:
        result = self.run_tool()
        output = result.stdout + result.stderr
        self.assertIn("not installed", output)
        self.assertIn("download + convert", output)
        self.assertIn("Nothing needs fetching by hand", output)
        self.assertNotIn("convert_qwen35moe", output)

    def test_help_says_the_command_downloads(self) -> None:
        result = self.run_tool("--help")
        output = result.stdout + result.stderr
        self.assertIn("download, convert and install one width", output)
        self.assertIn("What a run costs", output)

    def test_the_menu_says_its_sizes_are_installed_sizes(self) -> None:
        result = run_menu("\n")
        output = result.stdout + result.stderr
        self.assertIn("Sizes are installed sizes", output)


class ConverterPathTests(unittest.TestCase):
    """A converter is found from any directory, not only the repo root.

    `./install_models.sh` run from `tools/` — a natural place, since the script
    sits there — passed `tools/prepare_agentworld.py` to Python. That resolves
    against the caller's working directory, so it opened
    `tools/tools/prepare_agentworld.py` and died before fetching a byte. The
    release installer cds to the checkout before calling this script, which is
    why the bug survived every release.
    """

    def test_the_converter_path_is_absolute_and_exists_from_tools(self) -> None:
        root = pathlib.Path(tempfile.mkdtemp(prefix="tt-converter-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for name in ("work", "models", "bin"):
            (root / name).mkdir()
        repack = root / "bin" / "TinyTitanRepack"
        repack.write_text("#!/bin/sh\nexit 0\n")
        repack.chmod(0o755)
        # The resolver probes every interpreter with `-c`; only the converter
        # call is recorded, and it fails so nothing reaches the repack.
        argv_log = root / "argv"
        fake_python = root / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            'case "$1" in -c) exit 0 ;; esac\n'
            f'printf "%s\\n" "$@" > "{argv_log}"\n'
            "exit 1\n"
        )
        fake_python.chmod(0o755)
        env = dict(
            os.environ,
            TINYTITAN_WORK_DIR=str(root / "work"),
            TINYTITAN_MODELS_DIR=str(root / "models"),
            TINYTITAN_BIN_DIR=str(root / "bin"),
            TINYTITAN_PYTHON=str(fake_python),
            TINYTITAN_SKIP_DISK_CHECK="1",
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "ornith15"],
            env=env,
            cwd=ROOT / "tools",
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        arguments = argv_log.read_text().splitlines()
        converters = [a for a in arguments if a.endswith("prepare_agentworld.py")]
        self.assertTrue(converters, arguments)
        path = pathlib.Path(converters[0])
        self.assertTrue(path.is_absolute(), f"resolved against the caller's cwd: {path}")
        self.assertTrue(path.is_file(), path)


class DenseBf16InstallTests(unittest.TestCase):
    """The bf16-dense row is installed, not merely listed.

    This is the branch that was once pasted into `row_staging_dirs` instead of
    `install_one`, so the catalogue row, the menu entry and the README all said
    the build existed while selecting it returned success and installed nothing.
    The stub repack records its arguments, so the test pins the converter path,
    the variant's own model id (the catalog id must not collide with the plain
    4-bit row) and that the per-build staging is dropped afterwards.
    """

    def test_the_row_repacks_the_dense_snapshot_and_verifies_it(self) -> None:
        root = pathlib.Path(tempfile.mkdtemp(prefix="tt-dense-install-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for name in ("work", "models", "bin"):
            (root / name).mkdir()
        # The resolver only has to answer its `-c` probe; the snapshot exists,
        # so the converter itself is never called.
        fake_python = root / "python"
        fake_python.write_text("#!/bin/sh\nexit 0\n")
        fake_python.chmod(0o755)
        calls = root / "calls"
        repack = root / "bin" / "TinyTitanRepack"
        repack.write_text(
            "#!/bin/sh\n"
            "{ printf 'CALL\\n'; printf '%s\\n' \"$@\"; } >> \"$STUB_LOG\"\n"
            "while [ $# -gt 0 ]; do\n"
            '  case "$1" in --output) mkdir -p "$2"; shift 2 ;; *) shift ;; esac\n'
            "done\n"
            "exit 0\n"
        )
        repack.chmod(0o755)
        snap = root / "work" / "ornith15-bf16dense-affine"
        snap.mkdir(parents=True)
        (snap / "model.safetensors.index.json").write_text("{}")
        shards = root / "work" / "ornith15-bf16dense-shards"
        shards.mkdir()
        (shards / "model-00001-of-00016.safetensors").write_text("partial")
        env = dict(
            os.environ,
            TINYTITAN_WORK_DIR=str(root / "work"),
            TINYTITAN_MODELS_DIR=str(root / "models"),
            TINYTITAN_BIN_DIR=str(root / "bin"),
            TINYTITAN_PYTHON=str(fake_python),
            TINYTITAN_SKIP_DISK_CHECK="1",
            STUB_LOG=str(calls),
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "ornith15-bf16dense"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        groups = [
            group.splitlines() for group in calls.read_text().split("CALL\n") if group.strip()
        ]
        self.assertEqual(len(groups), 2, output)
        install = groups[0]
        self.assertIn("--input-snapshot", install)
        self.assertEqual(install[install.index("--input-snapshot") + 1], str(snap))
        self.assertEqual(
            install[install.index("--model-id") + 1],
            "ornith-1.5-35b-a3b-bf16dense",
            "the variant must not advertise the plain 4-bit row's catalog id",
        )
        self.assertEqual(
            install[install.index("--output") + 1],
            str(root / "models" / "ornith-1.5_35B_A3B-bf16dense_4Bit"),
        )
        self.assertIn("--verify-install", groups[1], output)
        self.assertFalse(snap.exists(), "the dense snapshot was not cleaned up")
        self.assertFalse(
            (root / "work" / "ornith15-bf16dense-shards").exists(),
            "the shard directory was not cleaned up",
        )


if __name__ == "__main__":
    unittest.main()
