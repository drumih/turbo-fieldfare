#!/usr/bin/env python3
"""The launcher fetching a model that is not on disk yet.

Asking for a model this checkout does not have used to end in an error and a
pointer at another tool. It now ends in the model: the launcher asks once, runs
`tools/install_models.sh` with the catalogue's own key, and carries on with the
freshly read catalog. These tests pin the parts that must not drift or surprise:

- the stem -> install-key map, and that every key it returns is a key the
  installer's `CATALOGUE` actually carries (the two lists cannot disagree);
- that a piped or `--dry-run` invocation never downloads, and says the exact
  command instead — a launcher test harness runs every case with `--dry-run`,
  so an install that ignored it would fetch 70 GB during the suite;
- that a model which is present still takes the normal path.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_launcher_install -v
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"
MODELS_SH = ROOT / "tools/tinytitan_models.sh"
INSTALLER = ROOT / "tools/install_models.sh"
MODELS = ROOT / "models"

# The runtime stems the launcher can name, and the catalogue key each width
# installs under. Spelled out here on purpose: this is the assertion, so reading
# it from the script under test would assert nothing.
STEM_KEYS = {
    "ornith-1.5_35B_A3B": ("ornith15", "ornith15-8bit"),
    "ornith-1.5_122B": ("ornith122", "ornith122-8bit"),
    "qwen3.6_35B_A3B": ("qwen36", "qwen36-8bit"),
    "qwen-agentworld_35B_A3B": ("agentworld", "agentworld-8bit"),
    "kat-coder-v2.5_35B_A3B": ("katcoder", "katcoder-8bit"),
    "qwen3.8-flash-next_125B_A6B": ("qwen38flash", "qwen38flash-8bit"),
    "qwen3.5_2B": ("qwen35-2b", "qwen35-2b-8bit"),
    "qwen3.5_4B": ("qwen35-4b", "qwen35-4b-8bit"),
    "qwen3.5_9B": ("qwen35-9b", "qwen35-9b-8bit"),
}

# The names the launcher's own `tinytitan_resolve_model` accepts, with the
# catalogue key each one's 4-bit install uses. A missing model is chosen from
# these, because they are the canonical keys both scripts share.
LAUNCHER_NAMES = {
    "ornith": "ornith15",
    "ornith122": "ornith122",
    "qwen36": "qwen36",
    "agentworld": "agentworld",
    "katcoder": "katcoder",
    "qwen38": "qwen38flash",
    "qwen35-2b": "qwen35-2b",
    "qwen35-4b": "qwen35-4b",
    "qwen35-9b": "qwen35-9b",
}

# The canonical key list both scripts derive from, with the display name each
# model must print everywhere and the installer key its 4-bit row carries.
# Spelled out as the assertion: the menu (TINYTITAN_MODEL_CHOICES), the
# launcher's resolver and the installer's CATALOGUE are three views of this
# one list, and CatalogueSyncTests holds them equal to it.
CANONICAL = {
    "ornith": ("Ornith 1.5 35B-A3B", "ornith15"),
    "ornith122": ("Ornith 1.5 122B", "ornith122"),
    "qwen36": ("Qwen 3.6 35B-A3B", "qwen36"),
    "agentworld": ("Qwen-AgentWorld 35B-A3B", "agentworld"),
    "katcoder": ("KAT-Coder-V2.5-Dev 35B-A3B", "katcoder"),
    "qwen38": ("Qwen3.8-Flash-Next 125B-A6B", "qwen38flash"),
    "qwen35-2b": ("Qwen 3.5 2B", "qwen35-2b"),
    "qwen35-4b": ("Qwen 3.5 4B", "qwen35-4b"),
    "qwen35-9b": ("Qwen 3.5 9B", "qwen35-9b"),
}

# Single-width (4-bit only) models: their directory is their stem directly.
SINGLE_WIDTH_MODELS = {
    "inkling2-gguf-q4": ("Inkling 2 Mini Q4", "inkling-2-mini_GGUF-Q4", "inkling2-gguf-q4"),
    "inkling2-nvfp4": ("Inkling 2 Mini NVFP4", "inkling-2-mini_NVFP4", "inkling2-nvfp4"),
}

# Removed from the catalogue 2026-11-16: a quant variant and two draft-head
# sidecars. Neither script may resolve, offer or install them again.
REMOVED_KEYS = ("ornith15-bf16dense", "ornith15-mtp", "qwen36-mtp")


def catalogue() -> dict[str, str]:
    """`install_models.sh`'s CATALOGUE as `key -> install directory`."""
    text = INSTALLER.read_text()
    block = re.search(r"^CATALOGUE=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    if block is None:
        raise AssertionError("install_models.sh no longer declares CATALOGUE")
    rows = {}
    for line in block.group(1).splitlines():
        match = re.match(r'\s*"([^|]+)\|([^|]+)\|', line)
        if match:
            rows[match.group(1)] = match.group(2)
    if not rows:
        raise AssertionError("CATALOGUE parsed as empty")
    return rows


def shell_model_function(function: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one `tinytitan_models.sh` function in bash and hand back the result."""
    script = f'source "{MODELS_SH}"; {function} {args[0]} {args[1]}'
    return subprocess.run(
        ["bash", "-c", script], text=True, capture_output=True, check=False, timeout=60
    )


def run_launcher(
    *args: str, models_dir: pathlib.Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the launcher in a dry run, optionally against another `models/`."""
    environment = dict(os.environ)
    environment["TINYTITAN_LAUNCHER_DRY_RUN"] = "1"
    environment.pop("TINYTITAN_LAUNCHER_ASSUME_TTY", None)
    if models_dir is not None:
        environment["TINYTITAN_MODELS_DIR"] = str(models_dir)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", *args],
        input="",
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=120,
    )


def first_missing() -> tuple[str, str, str] | None:
    """A (launcher name, bits, install key) this checkout does not have.

    Used for skipping: on a machine that has everything, there is no
    uninstalled row to test, and a test that pretended otherwise would assert
    against a menu shape that cannot occur.
    """
    known = catalogue()
    for name, key in LAUNCHER_NAMES.items():
        for bits in ("4", "8"):
            catalogue_key = key if bits == "4" else f"{key}-8bit"
            directory = known.get(catalogue_key)
            if directory is None:
                continue
            if not (MODELS / directory).is_dir():
                return name, bits, catalogue_key
    return None


def first_catalogue_model() -> tuple[str, str, str]:
    """A (launcher name, bits, install key) both lists agree on, disk ignored.

    `first_missing` depends on what this machine happens to have installed, so a
    test built on it runs or skips by accident. This one is the same lookup with
    the filesystem taken out of it, for a case that must always run.
    """
    known = catalogue()
    for name, key in LAUNCHER_NAMES.items():
        for bits in ("4", "8"):
            catalogue_key = key if bits == "4" else f"{key}-8bit"
            if catalogue_key in known:
                return name, bits, catalogue_key
    raise AssertionError("no launcher name maps to an installer catalogue key")


class InstallKeyTests(unittest.TestCase):
    def test_every_stem_and_width_maps_to_the_catalogue_key(self) -> None:
        for stem, (four, eight) in STEM_KEYS.items():
            for bits, expected in (("4", four), ("8", eight)):
                with self.subTest(stem=stem, bits=bits):
                    result = shell_model_function("tinytitan_install_key", stem, bits)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_every_mapped_key_is_one_the_installer_carries(self) -> None:
        # The map and the installer are two lists; this is the assertion that
        # keeps a renamed or added model from silently having no fetch path.
        known = catalogue()
        for stem, keys in STEM_KEYS.items():
            for expected in keys:
                with self.subTest(key=expected):
                    self.assertIn(expected, known, f"{stem} maps to a key CATALOGUE lacks")

    def test_an_unknown_stem_has_no_key(self) -> None:
        result = shell_model_function("tinytitan_install_key", "not-a-stem", "4")
        self.assertNotEqual(result.returncode, 0)

    def test_the_removed_bf16_dense_stem_has_no_key(self) -> None:
        # The bf16-dense variant was removed (2026-11-16): neither width may
        # map to an install key, so the launcher cannot offer to fetch it.
        for bits in ("4", "8"):
            with self.subTest(bits=bits):
                result = shell_model_function(
                    "tinytitan_install_key", "ornith-1.5_35B_A3B-bf16dense", bits
                )
                self.assertNotEqual(result.returncode, 0)

    def test_the_install_size_comes_from_the_menu_entry(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{MODELS_SH}"; tinytitan_install_size_gb katcoder',
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r"^\d+(\.\d+)?$")


class MissingModelTests(unittest.TestCase):
    def test_a_pipe_is_told_the_command_and_downloads_nothing(self) -> None:
        candidate = first_missing()
        if candidate is None:
            self.skipTest("every catalogue model is installed here")
        name, bits, key = candidate
        before = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        run = run_launcher("--client", "server", "--model", name, "--bits", bits)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn(f"tools/install_models.sh {key}", run.stderr + run.stdout)
        self.assertNotIn("Starting TinyTitanServer", run.stdout)
        after = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        self.assertEqual(before, after, "a piped launcher installed something")

    def test_install_flag_is_documented_and_refuses_to_start_a_download_in_a_dry_run(self) -> None:
        help_run = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(help_run.returncode, 0)
        self.assertIn("--install", help_run.stdout)

        candidate = first_missing()
        if candidate is None:
            self.skipTest("every catalogue model is installed here")
        name, bits, key = candidate
        before = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        run = run_launcher("--client", "server", "--model", name, "--bits", bits, "--install")
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        # `--install` answers the question; the dry run still starts nothing,
        # and it says the command it would have run.
        self.assertIn(f"tools/install_models.sh {key}", run.stdout + run.stderr)
        after = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        self.assertEqual(before, after, "a dry run installed something")


class MissingOffersTests(unittest.TestCase):
    """`tinytitan_missing_offers`: what the menu appends, row by row."""

    def offers(self, models_dir: pathlib.Path) -> list[list[str]]:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{MODELS_SH}"; tinytitan_missing_offers "{models_dir}"',
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("|") for line in result.stdout.splitlines() if line]
        for row in rows:
            self.assertEqual(
                len(row),
                7,
                f"offer row is not name|bits|key|label|gb|engines|thinking: {row}",
            )
        return rows

    def test_an_empty_install_directory_offers_every_supported_width(self) -> None:
        with tempfile.TemporaryDirectory(prefix="missing-") as directory:
            rows = self.offers(pathlib.Path(directory))
        names = [row[0] for row in rows]
        self.assertIn("ornith", names)
        self.assertIn("katcoder", names)
        self.assertIn("qwen38", names)
        # Nine two-width models (18) plus two single-width Inkling 2 models (2) = 20.
        self.assertEqual(len(rows), 20, "nine two-width models + two single-width Inkling 2 models")
        for row in rows:
            self.assertIn(row[1], ("4", "8"))
            self.assertTrue(row[3], f"offer row has no label: {row}")
            self.assertRegex(row[4], r"^\d+(\.\d+)?$", f"size missing on {row}")
            # Engine and levels are the columns a person compares rows by, so a
            # row without them would be the one row they cannot judge.
            self.assertIn(row[5], ("gpu", "cpu", "gpu,cpu", "cpu,gpu"), f"bad engines: {row}")
            self.assertTrue(row[6], f"offer row has no thinking levels: {row}")

    def test_every_offered_key_is_one_the_installer_carries(self) -> None:
        # The rows are what a pick is installed with, so a key the installer
        # does not know would be a menu entry that cannot work.
        known = catalogue()
        with tempfile.TemporaryDirectory(prefix="missing-") as directory:
            rows = self.offers(pathlib.Path(directory))
        for row in rows:
            self.assertIn(row[2], known, f"{row[0]}({row[1]}) offers a key CATALOGUE lacks")

    def test_the_widths_that_are_there_are_not_offered(self) -> None:
        with tempfile.TemporaryDirectory(prefix="missing-") as directory:
            root = pathlib.Path(directory)
            (root / "ornith-1.5_35B_A3B_8Bit").mkdir()
            (root / "kat-coder-v2.5_35B_A3B_4Bit").mkdir()
            rows = self.offers(root)
        offered = {(row[0], row[1]) for row in rows}
        self.assertNotIn(("ornith", "8"), offered)
        self.assertNotIn(("katcoder", "4"), offered)
        self.assertIn(("ornith", "4"), offered, "the other width is still missing")

    def test_nothing_missing_offers_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="missing-") as directory:
            root = pathlib.Path(directory)
            for stem in (
                "ornith-1.5_35B_A3B",
                "ornith-1.5_122B",
                "qwen3.6_35B_A3B",
                "qwen-agentworld_35B_A3B",
                "kat-coder-v2.5_35B_A3B",
                "qwen3.8-flash-next_125B_A6B",
                "qwen3.5_2B",
                "qwen3.5_4B",
                "qwen3.5_9B",
            ):
                for bits in ("4Bit", "8Bit"):
                    (root / f"{stem}_{bits}").mkdir()
            for _label, stem, _key in SINGLE_WIDTH_MODELS.values():
                (root / stem).mkdir()
            rows = self.offers(root)
        self.assertEqual(rows, [])


class PromptCacheDiskFlagTests(unittest.TestCase):
    """The prompt-cache disk tier is a pair, and the pair is checked while parsing.

    Both refusals run before any model is resolved, which is what lets them be
    tested without an installed model -- and what makes the message arrive
    before the launcher has done a catalogue lookup.
    """

    def test_budget_without_a_directory_is_refused(self) -> None:
        result = run_launcher("--prompt-cache-disk-mib", "512")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--prompt-cache-disk-mib needs --prompt-cache-disk", result.stderr)

    def test_a_non_numeric_budget_is_refused_by_name(self) -> None:
        result = run_launcher(
            "--prompt-cache-disk",
            "/tmp/tinytitan-test-pcache",
            "--prompt-cache-disk-mib",
            "512 MiB",
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("unknown --prompt-cache-disk-mib: 512 MiB", result.stderr)


class MenuSelectionTests(unittest.TestCase):
    """Picking an uninstalled row from the menu fetches it, then continues."""

    def menu(
        self, answers: str, *, extra: tuple[str, ...] = ()
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["TINYTITAN_LAUNCHER_DRY_RUN"] = "1"
        environment["TINYTITAN_LAUNCHER_ASSUME_TTY"] = "1"
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--client",
                "server",
                "--answers",
                "default",
                "--thinking",
                "off",
                "--ram",
                "9",
                "--engine",
                "gpu",
                "--concurrency",
                "1",
                *extra,
            ],
            input=answers,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=120,
        )

    def menu_text(self) -> str:
        return self.menu("\n").stdout

    def first_missing_row(self, text: str) -> tuple[int, str]:
        """The first not-installed row's number and its label."""
        for line in text.splitlines():
            if "not installed" in line and "fetches it first" in line:
                number = int(line.strip().split(")")[0])
                label = line.split(")")[1].strip().split("  ")[0].strip()
                return number, label
        raise AssertionError(f"no uninstalled row in the menu:\n{text}")

    def test_the_menu_offers_the_uninstalled_models_as_rows(self) -> None:
        if first_missing() is None:
            self.skipTest("every catalogue model is installed here")
        text = self.menu_text()
        number, _label = self.first_missing_row(text)
        choice = number
        self.assertIn("fetches it first", text)
        self.assertRegex(text, rf"Rows {choice}-\d+ are not installed yet")

    def test_choosing_one_offers_the_download_and_says_what_it_would_run(self) -> None:
        candidate = first_missing()
        if candidate is None:
            self.skipTest("every catalogue model is installed here")
        number, label = self.first_missing_row(self.menu_text())
        run = self.menu(f"{number}\ny\n")
        self.assertIn(label, run.stdout)
        self.assertIn("Would install:  tools/install_models.sh ", run.stdout)
        self.assertNotIn("Starting TinyTitanServer", run.stdout, "a dry run started a server")

    def test_declining_one_installs_nothing_and_names_the_command(self) -> None:
        if first_missing() is None:
            self.skipTest("every catalogue model is installed here")
        number, _label = self.first_missing_row(self.menu_text())
        before = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        run = self.menu(f"{number}\nn\n")
        self.assertIn("Nothing was installed", run.stdout)
        self.assertIn("tools/install_models.sh ", run.stdout)
        after = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        self.assertEqual(before, after, "declining installed something")

    def test_the_default_row_still_launches_an_installed_model(self) -> None:
        # The offers array must not disturb the path the menu took before it
        # existed: Enter takes the default installed row and the run goes on.
        run = self.menu("\n")
        self.assertRegex(run.stdout, r"Choice \[1-\d+\] \(default \d+\)")
        self.assertNotIn("ERROR", run.stdout)


class EmptyModelsDirTests(unittest.TestCase):
    """A checkout with nothing installed reaches the fetch instead of stopping.

    `models/` is gitignored, so this is the state a fresh clone and every CI run
    are in. The launcher's empty-catalog guard used to answer it with `exit 2`
    before the offer and fetch code, which made the feature these other tests pin
    unreachable on exactly the machines that need it. The directory is created
    here rather than inherited, so the case runs identically on a checkout with
    eight installs and one with none.
    """

    def test_a_named_model_is_offered_not_refused(self) -> None:
        name, bits, key = first_catalogue_model()
        with tempfile.TemporaryDirectory() as empty:
            models = pathlib.Path(empty)
            run = run_launcher(
                "--client", "server", "--model", name, "--bits", bits, models_dir=models
            )
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            combined = run.stdout + run.stderr
            self.assertIn(f"tools/install_models.sh {key}", combined)
            # The old refusal must not be what produced that line.
            self.assertNotIn("no install under", combined)
            self.assertNotIn("Starting TinyTitanServer", run.stdout)
            self.assertEqual(sorted(p.name for p in models.iterdir()), [])

    def test_the_menu_draws_the_fetch_rows(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            run = run_launcher("--client", "server", models_dir=pathlib.Path(empty))
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn("are not installed yet", run.stdout)
        self.assertIn("fetches it first", run.stdout)


class MenuRowPaddingTests(unittest.TestCase):
    """A zero-padded reply names the row its digits spell, not its octal value.

    The pick is guarded with `(( pick >= 1 && pick <= count ))` and indexed with
    `$((pick - 1))`, and bash 3.2 reads a leading zero as octal in both. Measured
    on 2026-10-10 against an empty catalogue, where the menu's 16 rows are the
    catalogue in order, before the fix:

    - `010` picked row 8 -- `katcoder-8bit`, 38.0 GB -- for a person who pointed
      at row 10, `qwen38flash-8bit` at 220.0 GB. Nothing errors.
    - `013` picked row 11 (`qwen35-2b`) for row 13 (`qwen35-4b`), and `016`
      picked row 14 (`qwen35-4b-8bit`) for row 16 (`qwen35-9b-8bit`).
    - `017` and `020` were *accepted* as rows 15 and 16, where the plain replies
      `17` and `20` are refused: padding moved a reply past the end of the list.
    - `08` was refused with `value too great for base` -- the right outcome for
      the wrong reason, and the neighbour `01`-`07` pass unchanged because their
      octal reading equals their decimal one.

    The rows run against a temporary empty `models/` rather than this checkout's,
    so every catalogue row is on the menu in a fixed order on any machine, and the
    run declines the fetch, so nothing is downloaded here.

        cd benchmark && python3 -m unittest test_launcher_install.MenuRowPaddingTests -v
    """

    def pick(self, reply: str, models: pathlib.Path) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["TINYTITAN_LAUNCHER_DRY_RUN"] = "1"
        environment["TINYTITAN_LAUNCHER_ASSUME_TTY"] = "1"
        environment["TINYTITAN_MODELS_DIR"] = str(models)
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--dry-run",
                "--client",
                "server",
                "--answers",
                "default",
                "--thinking",
                "off",
                "--ram",
                "9",
                "--engine",
                "gpu",
                "--concurrency",
                "1",
            ],
            input=f"{reply}\nn\n",
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=120,
        )

    @staticmethod
    def named_key(run: subprocess.CompletedProcess[str]) -> str | None:
        """The catalogue key the pick resolved to, from the command it prints."""
        match = re.search(r"Install it with:\s+tools/install_models\.sh (\S+)", run.stdout)
        return match.group(1) if match else None

    @staticmethod
    def row_label(menu_output: str, number: str) -> str:
        """The label the menu itself prints for a row number.

        Reading the row back out of the drawn menu is what makes the comparison
        below independent of which models the catalogue happens to carry: the
        claim is that the row you pointed at is the row that got picked.
        """
        pattern = rf"^\s+{number}\)\s+(.+?)\s+(4|8)-bit\b"
        match = re.search(pattern, menu_output, re.MULTILINE)
        if match is None:
            raise AssertionError(f"row {number} is not on the drawn menu")
        return f"{match.group(1)} {match.group(2)}-bit"

    def test_a_padded_reply_names_the_row_it_points_at(self) -> None:
        # Each padded reply must reach the same row as its plain digits, and must
        # not do it by an arithmetic error the guard happens to swallow.
        with tempfile.TemporaryDirectory() as empty:
            models = pathlib.Path(empty)
            drawn = self.pick("1", models)
            for plain, padded in (("8", "08"), ("10", "010"), ("13", "013"), ("16", "016")):
                with self.subTest(row=plain, reply=padded):
                    bare = self.pick(plain, models)
                    pad = self.pick(padded, models)
                    combined = pad.stdout + pad.stderr
                    self.assertNotIn(
                        "value too great for base",
                        combined,
                        f"reply {padded} reached the arithmetic guard as an error",
                    )
                    label = self.row_label(drawn.stdout, plain)
                    self.assertIn(
                        f"{label} is not installed",
                        pad.stdout,
                        f"reply {padded} did not pick the row the menu draws as {label}",
                    )
                    self.assertEqual(self.named_key(bare), self.named_key(pad))

    def test_a_padded_reply_past_the_end_still_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            models = pathlib.Path(empty)
            # The catalogue has 20 rows; 21 and 22 are past its end.
            for padded in ("021", "022"):
                with self.subTest(reply=padded):
                    pad = self.pick(padded, models)
                    combined = pad.stdout + pad.stderr
                    self.assertIsNone(self.named_key(pad), f"{padded} reached a row")
                    self.assertIn("invalid choice", combined)
                    self.assertEqual(pad.returncode, 2)


class RemovedModelTests(unittest.TestCase):
    """The removed rows stay removed (2026-11-16).

    ornith15-bf16dense (a quant variant) and the two MTP draft-head sidecars
    (ornith15-mtp, qwen36-mtp) left the catalogue. The launcher must refuse
    them by name, and the shared key resolver must not map them to a fetch.
    OneWidthModelTests lived here while the bf16-dense build was the one
    single-width model; every remaining model ships both widths.
    """

    def test_the_launcher_refuses_the_removed_keys(self) -> None:
        for key in REMOVED_KEYS:
            with self.subTest(model=key), tempfile.TemporaryDirectory(prefix="removed-") as empty:
                run = run_launcher(
                    "--client", "server", "--model", key, models_dir=pathlib.Path(empty)
                )
                combined = run.stdout + run.stderr
                self.assertEqual(run.returncode, 2, combined)
                self.assertIn(f"unknown model: {key}", combined)
                self.assertNotIn("Install it with", combined)

    def test_the_shared_resolver_has_no_target_for_them(self) -> None:
        for key in REMOVED_KEYS:
            with self.subTest(key=key):
                result = subprocess.run(
                    ["bash", "-c", f'source "{MODELS_SH}"; tinytitan_install_target "{key}"'],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=60,
                )
                self.assertNotEqual(result.returncode, 0)


class InstallerSpellingTests(unittest.TestCase):
    """The launcher accepts the installer's key spellings, width included.

    Both scripts read one catalogue (tools/tinytitan_models.sh), so a key the
    installer prints -- ornith15-8bit, qwen38flash -- is a key the launcher
    takes, and a canonical key with an -8bit suffix names its own width.
    """

    def test_installer_keys_and_suffixed_canonical_keys_resolve(self) -> None:
        cases = (
            ("ornith15-8bit", "ornith15-8bit"),
            ("ornith15", "ornith15-8bit"),  # bare key takes the historical 8-bit default
            ("qwen38flash", "qwen38flash-8bit"),
            ("qwen38flash-8bit", "qwen38flash-8bit"),
            ("ornith-8bit", "ornith15-8bit"),
            ("qwen38-8bit", "qwen38flash-8bit"),
            ("qwen35-4b-8bit", "qwen35-4b-8bit"),
        )
        for model, key in cases:
            with (
                self.subTest(model=model),
                tempfile.TemporaryDirectory(prefix="spelling-") as empty,
            ):
                run = run_launcher(
                    "--client", "server", "--model", model, models_dir=pathlib.Path(empty)
                )
                combined = run.stdout + run.stderr
                self.assertEqual(run.returncode, 1, combined)
                self.assertIn(f"tools/install_models.sh {key}", combined)

    def test_a_suffixed_key_conflicting_with_bits_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spelling-") as empty:
            run = run_launcher(
                "--client",
                "server",
                "--model",
                "ornith-8bit",
                "--bits",
                "4",
                models_dir=pathlib.Path(empty),
            )
        combined = run.stdout + run.stderr
        self.assertEqual(run.returncode, 2, combined)
        self.assertIn("names 8-bit, not 4-bit", combined)


def choices() -> dict[str, tuple[str, str]]:
    """`tinytitan_models.sh`'s TINYTITAN_MODEL_CHOICES as `key -> (label, bits)`."""
    text = MODELS_SH.read_text()
    block = re.search(r"^TINYTITAN_MODEL_CHOICES=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    if block is None:
        raise AssertionError("tinytitan_models.sh no longer declares TINYTITAN_MODEL_CHOICES")
    rows = {}
    for line in block.group(1).splitlines():
        match = re.match(r'\s*"([^|]+)\|([^|]+)\|([48])\|', line)
        if match:
            rows[match.group(1)] = (match.group(2), match.group(3))
    if not rows:
        raise AssertionError("TINYTITAN_MODEL_CHOICES parsed as empty")
    return rows


def shell_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{MODELS_SH}"; {snippet}'],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


class CatalogueSyncTests(unittest.TestCase):
    """One catalogue, three views: menu, resolver, installer.

    The installer menu (TINYTITAN_MODEL_CHOICES), the launcher's resolver and
    the installer's CATALOGUE must carry the same keys and print the same
    display names for every model. CANONICAL above is the assertion; these
    tests hold all three lists to it, so a rename or a one-sided addition
    fails here rather than in a user's terminal.
    """

    def test_the_three_lists_carry_exactly_the_same_keys(self) -> None:
        expected = set()
        for _label, key in CANONICAL.values():
            expected.add(key)
            expected.add(f"{key}-8bit")
        for _label, _stem, key in SINGLE_WIDTH_MODELS.values():
            expected.add(key)
        self.assertEqual(set(choices()), expected, "menu keys drifted from the canonical list")
        self.assertEqual(
            set(catalogue()), expected, "installer keys drifted from the canonical list"
        )

    def test_the_menu_and_the_launcher_print_the_same_names(self) -> None:
        menu = choices()
        for name, (label, key) in CANONICAL.items():
            with self.subTest(model=name):
                resolved = shell_snippet(
                    f'tinytitan_resolve_model "{name}" && printf "%s" "$TINYTITAN_MODEL_LABEL"'
                )
                self.assertEqual(resolved.returncode, 0, resolved.stderr)
                self.assertEqual(resolved.stdout, label)
                for width_key in (key, f"{key}-8bit"):
                    self.assertEqual(
                        menu[width_key][0],
                        label,
                        f"menu label for {width_key} differs from the launcher's",
                    )
        for name, (label, _stem, key) in SINGLE_WIDTH_MODELS.items():
            with self.subTest(model=name):
                resolved = shell_snippet(
                    f'tinytitan_resolve_model "{name}" && printf "%s" "$TINYTITAN_MODEL_LABEL"'
                )
                self.assertEqual(resolved.returncode, 0, resolved.stderr)
                self.assertEqual(resolved.stdout, label)
                self.assertEqual(menu[key][0], label)

    def test_install_target_accepts_both_spellings_of_every_model(self) -> None:
        for name, (_label, key) in CANONICAL.items():
            for spelling, expected in (
                (name, key),
                (f"{name}-8bit", f"{key}-8bit"),
                (key, key),
                (f"{key}-8bit", f"{key}-8bit"),
            ):
                with self.subTest(spelling=spelling):
                    result = shell_snippet(f'tinytitan_install_target "{spelling}"')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
