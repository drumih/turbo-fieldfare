"""The launcher's expert-cache rule: 50% of physical memory, warned not capped.

The rule belongs to the launcher, and it is a recommendation rather than a
limit: a larger `--ram` is warned about in red and passed on, because the
machine is the operator's. These tests pin the arithmetic (a half, floored to
whole GB, which is what `--ram` takes), both sides of the boundary, and the
interactive question whose default is that half.

`TINYTITAN_PHYSICAL_RAM_BYTES` is the launcher's seam for exactly this: the mapping
has to be checkable on a machine of any size. The model it is asked about comes
from `launcher_fixture`, so these run on a checkout with no install and no built
server — which is the only kind of checkout a clean clone is.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_launcher_ram -v
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

import launcher_fixture

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"

# Installed memory in bytes, and the launcher's rule for it: half, floored.
MACHINES = {
    8 * 2**30: 4,
    16 * 2**30: 8,
    24 * 2**30: 12,
    32 * 2**30: 16,
    64 * 2**30: 32,
}


def dry_run(
    installs: launcher_fixture.SyntheticInstalls, *args: str, physical_bytes: int
) -> subprocess.CompletedProcess[str]:
    environment = installs.env(TINYTITAN_PHYSICAL_RAM_BYTES=str(physical_bytes))
    return subprocess.run(
        ["bash", str(LAUNCHER), "--client", "server", *args, "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def dry_run_value(
    installs: launcher_fixture.SyntheticInstalls, *args: str, physical_value: str
) -> subprocess.CompletedProcess[str]:
    """`dry_run` with the seam held as the operator wrote it, not as an int."""
    environment = installs.env(TINYTITAN_PHYSICAL_RAM_BYTES=physical_value)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--client", "server", *args, "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def answer_ram(
    installs: launcher_fixture.SyntheticInstalls,
    model: str,
    answer: str,
    *,
    physical_bytes: int,
) -> subprocess.CompletedProcess[str]:
    """Answer the interactive RAM question and nothing else.

    Every other question is pre-answered by a flag, so the piped input is the
    RAM choice; without that the answers would be consumed in some other order
    and this would be testing the question order, not the tiers.
    """
    environment = installs.env(
        TINYTITAN_PHYSICAL_RAM_BYTES=str(physical_bytes),
        TINYTITAN_LAUNCHER_DRY_RUN="1",
        TINYTITAN_LAUNCHER_ASSUME_TTY="1",
    )
    return subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--client",
            "server",
            "--dry-run",
            "--model",
            model,
            "--thinking",
            "off",
            "--answers",
            "default",
            "--engine",
            "gpu",
            "--context",
            "262144",
            "--kv",
            "8",
            "--concurrency",
            "1",
            "--port",
            "8080",
        ],
        input=answer,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=180,
    )


def target_of(run: subprocess.CompletedProcess[str]) -> str:
    """The `--ram-budget <n>G` the launcher would pass, or an empty string."""
    match = re.search(r"--ram-budget (\d+)G", run.stdout)
    return match.group(1) if match else ""


class RamMenuTests(unittest.TestCase):
    """The interactive tiers: 4, 6, 8, 10, 12, 14, 16, Custom, model default."""

    # 24 GB of physical memory, so the rule (and the Custom prompt's default)
    # is 12 GB, which is also tier 5 -- the two must agree, and the equality is
    # the assertion: a Custom default that disagreed with the recommendation
    # would be a second, quieter rule.
    MEMORY = 24 * 2**30
    TIERS = {"1": "4", "2": "6", "3": "8", "4": "10", "5": "12", "6": "14", "7": "16"}

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()

    def menu(self, answer: str) -> subprocess.CompletedProcess[str]:
        return answer_ram(self.installs, self.model, answer, physical_bytes=self.MEMORY)

    def test_the_menu_offers_the_seven_tiers_and_a_custom_value(self) -> None:
        run = self.menu("\n")
        menu = run.stdout
        for label in ("4 GB", "6 GB", "8 GB", "10 GB", "12 GB", "14 GB", "16 GB"):
            self.assertIn(label, menu, f"{label} is not offered")
        self.assertIn("Custom", menu)
        self.assertIn("Model default", menu)
        self.assertIn("Choice [1-9] (default 9)", menu)

    def test_each_tier_passes_its_own_budget(self) -> None:
        for choice, gb in self.TIERS.items():
            with self.subTest(choice=choice, gb=gb):
                self.assertEqual(target_of(self.menu(f"{choice}\n")), gb)

    def test_custom_takes_any_whole_number_of_gb(self) -> None:
        # 20 is above every tier and above the 12 GB rule, so it also exercises
        # the warning path rather than a silent accept.
        run = self.menu("8\n20\n")
        self.assertEqual(target_of(run), "20")

    def test_custom_default_is_the_recommended_half(self) -> None:
        run = self.menu("8\n\n")
        self.assertEqual(target_of(run), "12", "the Custom prompt defaults to the rule")

    def test_custom_refuses_below_the_floor(self) -> None:
        run = self.menu("8\n3\n")
        self.assertEqual(run.returncode, 2)
        self.assertIn("invalid choice: 3", run.stderr)
        self.assertEqual(target_of(run), "")

    def test_enter_takes_the_default_and_passes_half_of_memory(self) -> None:
        # The question's default is the recommendation, and it is passed as an
        # explicit target: an unset budget is clamped to a third by the runtime,
        # so only an explicit one can reach the half the menu advertises.
        run = self.menu("\n")
        self.assertEqual(target_of(run), "12")
        self.assertIn("RAM: 12 GB (default: 50% of this Mac)", run.stdout)
        self.assertIn("Model default (50% of this Mac is 12 GB)", run.stdout)

    def test_the_unattended_path_still_leaves_the_budget_to_the_install(self) -> None:
        # A script gets the shipped profile -- which is what the benchmark
        # protocol measures -- so the half is a choice a person makes, not a
        # silent change to every unattended run.
        run = dry_run(self.installs, "--model", self.model, physical_bytes=self.MEMORY)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(target_of(run), "")
        self.assertIn("model default (measured; 50% of this Mac is 12 GB)", run.stdout)


class RamRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()

    def test_rule_is_half_rounded_down(self) -> None:
        for memory, expected in MACHINES.items():
            with self.subTest(memory_gb=memory // 2**30):
                run = dry_run(self.installs, "--model", self.model, physical_bytes=memory)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(
                    f"RAM: model default (measured; 50% of this Mac is {expected} GB)",
                    run.stdout,
                )

    def test_at_the_rule_is_silent_and_above_it_warns(self) -> None:
        memory = 24 * 2**30
        at_rule = dry_run(
            self.installs, "--model", self.model, "--ram", "12", physical_bytes=memory
        )
        self.assertEqual(at_rule.returncode, 0, at_rule.stderr)
        self.assertNotIn("WARNING", at_rule.stderr)
        self.assertIn("--ram-budget 12G", at_rule.stdout)

        above = dry_run(self.installs, "--model", self.model, "--ram", "14", physical_bytes=memory)
        self.assertEqual(above.returncode, 0, above.stderr)
        self.assertIn("WARNING: the server would hold about 14 GB", above.stderr)
        self.assertIn("50% of this Mac's 24 GB", above.stderr)
        self.assertIn("Starting anyway with 14 GB", above.stderr)
        # Warned, not capped: the requested size is what the server is given.
        self.assertIn("--ram-budget 14G", above.stdout)
        self.assertIn("over 50% of this Mac's RAM", above.stdout)

    def test_default_path_warns_about_nothing(self) -> None:
        run = dry_run(self.installs, "--model", self.model, physical_bytes=24 * 2**30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("WARNING", run.stderr)
        self.assertNotIn("--ram-budget", run.stdout)

    def test_unreadable_memory_means_no_rule(self) -> None:
        # No rule, so nothing to warn about even for a large explicit size…
        run = dry_run(self.installs, "--model", self.model, "--ram", "32", physical_bytes=0)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("WARNING", run.stderr)
        self.assertNotIn("50% of this Mac", run.stdout)
        self.assertIn("--ram-budget 32G", run.stdout)
        # …and the default's note claims no percentage either.
        default = dry_run(self.installs, "--model", self.model, physical_bytes=0)
        self.assertIn("RAM: model default (measured) |", default.stdout)

    def test_boundary_scales_with_the_machine(self) -> None:
        # The rule is 4 GB on an 8 GB Mac -- the floor exactly -- so the
        # smallest accepted target sits on it there, and 6 warns.
        at_floor = dry_run(
            self.installs, "--model", self.model, "--ram", "4", physical_bytes=8 * 2**30
        )
        self.assertEqual(at_floor.returncode, 0, at_floor.stderr)
        self.assertNotIn("WARNING", at_floor.stderr)
        self.assertIn("--ram-budget 4G", at_floor.stdout)

        above = dry_run(
            self.installs, "--model", self.model, "--ram", "6", physical_bytes=8 * 2**30
        )
        self.assertEqual(above.returncode, 0, above.stderr)
        self.assertIn("WARNING: the server would hold about 6 GB", above.stderr)
        self.assertIn("--ram-budget 6G", above.stdout)

        large = dry_run(
            self.installs, "--model", self.model, "--ram", "4", physical_bytes=24 * 2**30
        )
        self.assertEqual(large.returncode, 0, large.stderr)
        self.assertNotIn("WARNING", large.stderr)
        self.assertIn("--ram-budget 4G", large.stdout)

    def test_targets_below_the_floor_are_refused(self) -> None:
        # A streaming install cannot stay under 4 GB: the weights plus the
        # 8-slot minimum cache are ~4.7 GB. The launcher refuses rather than
        # accepting a number the runtime would silently overshoot, and a bare
        # number is refused too instead of being ignored as an unknown word.
        for ram in ("1", "2", "3", "2G"):
            with self.subTest(placement="flag", ram=ram):
                run = dry_run(
                    self.installs, "--model", self.model, "--ram", ram, physical_bytes=24 * 2**30
                )
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown RAM target", run.stderr)
                self.assertNotIn("--ram-budget", run.stdout)
        # A bare number is refused too rather than ignored as an unknown word.
        # 1 and 0 are not tested positionally: they are thinking levels ("on",
        # "off") and the optional-value loop reads those first.
        for ram in ("2", "3", "2G"):
            with self.subTest(placement="positional", ram=ram):
                run = dry_run(self.installs, "--model", self.model, ram, physical_bytes=24 * 2**30)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown RAM target", run.stderr)
                self.assertNotIn("--ram-budget", run.stdout)


class TheSeamIsCheckedWhereItIsRead(unittest.TestCase):
    """A byte count named through the seam is checked before it is used.

    The sanitizer below the read turns a non-digit into 0, which is the documented
    answer for a machine whose size cannot be read -- and the wrong answer for a
    value the operator named, because the ceiling rule then switches off while the
    run still reports a plan. A leading zero escapes the sanitizer entirely: it is
    all digits, so it reaches three arithmetic expansions, where bash reads it as
    octal. Measured on the launcher as it stands:

        TINYTITAN_PHYSICAL_RAM_BYTES=08000  -> three lines of
            `value too great for base` at :931, :935 and :939, then the summary,
            exit 0, and no ceiling rule
        TINYTITAN_PHYSICAL_RAM_BYTES=16g,
        -5, 4000.5                          -> exit 0, sanitized to 0 in silence,
            so the rule the caller asked to measure never ran
        TINYTITAN_PHYSICAL_RAM_BYTES=0      -> exit 0, no rule: the shape two of
            this suite's own tests ask for, and the one that stays accepted

    `0` is a byte count here rather than a switch, because the sysctl read already
    answers with it for "unreadable"; everything else that is not a whole number of
    bytes with no leading zero is refused the way the port and `--ram` are.
    """

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()

    def seam(self, value: str) -> subprocess.CompletedProcess[str]:
        return dry_run_value(self.installs, "--model", self.model, physical_value=value)

    def test_a_zero_padded_byte_count_is_refused_not_read_as_octal(self) -> None:
        run = self.seam("08000")
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn("08000", run.stderr)
        self.assertIn("TINYTITAN_PHYSICAL_RAM_BYTES", run.stderr)

    def test_the_octal_death_is_not_what_the_operator_is_shown(self) -> None:
        run = self.seam("08000")
        output = run.stdout + run.stderr
        self.assertNotIn("value too great for base", output)
        self.assertNotIn("unbound variable", output)

    def test_the_refusal_comes_before_the_run_plans_anything(self) -> None:
        self.assertNotIn("Port:", self.seam("08000").stdout)

    def test_a_unit_suffix_is_refused_rather_than_becoming_no_rule(self) -> None:
        run = self.seam("16g")
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn("16g", run.stderr)
        self.assertIn("TINYTITAN_PHYSICAL_RAM_BYTES", run.stderr)

    def test_a_negative_and_a_fractional_size_are_refused_at_the_seam(self) -> None:
        for value in ("-5", "4000.5"):
            with self.subTest(value=value):
                run = self.seam(value)
                self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
                self.assertIn(value, run.stderr)
                self.assertNotIn("Port:", run.stdout)

    def test_zero_still_means_an_unreadable_machine(self) -> None:
        run = self.seam("0")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Port:", run.stdout)

    def test_a_plain_byte_count_still_reaches_the_rule(self) -> None:
        run = dry_run_value(
            self.installs,
            "--model",
            self.model,
            "--ram",
            "14",
            physical_value=str(24 * 2**30),
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("over 50% of this Mac's RAM", run.stdout)

    def test_a_run_with_no_seam_set_is_left_to_the_machine(self) -> None:
        # The guard is on the seam, not on the read: an ordinary launch must not
        # gain a refusal for a value it never set.
        environment = self.installs.env()
        environment.pop("TINYTITAN_PHYSICAL_RAM_BYTES", None)
        run = subprocess.run(
            ["bash", str(LAUNCHER), "--client", "server", "--model", self.model, "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("TINYTITAN_PHYSICAL_RAM_BYTES", run.stdout + run.stderr)


class ThePromptCacheTierIsLeftToTheServer(unittest.TestCase):
    """The launcher must not pin what the server now derives from the machine.

    `ServerArguments.defaultPromptCacheMemoryMiB` sizes the prompt cache's RAM
    tier from physical memory (256 MiB below 48 GiB, 2 GiB above), the way the
    expert budget is sized. The launcher used to answer that flag with a flat
    256 on every machine, which meant the derivation never reached the users
    this launch path is for — a 128 GiB Mac got the small machine's tier, with
    the pin written *after* the server had already read its own RAM. Not
    passing the flag is the whole fix; the plan line prints the mode, not the
    number, so nothing here needs to know the tier.
    """

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()

    def test_the_default_path_names_no_prompt_cache_tier(self) -> None:
        for memory in (24 * 2**30, 128 * 2**30):
            with self.subTest(memory_gb=memory // 2**30):
                run = dry_run(self.installs, "--model", self.model, physical_bytes=memory)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("--prompt-cache-mode multi-prefix", run.stdout)
                self.assertNotIn("--prompt-cache-memory-mib", run.stdout)

    def test_turning_the_cache_off_still_reaches_the_server(self) -> None:
        run = dry_run(
            self.installs,
            "--model",
            self.model,
            "--prompt-cache",
            "off",
            physical_bytes=24 * 2**30,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--prompt-cache-mode off", run.stdout)

    def test_an_unknown_cache_choice_is_still_refused(self) -> None:
        run = dry_run(
            self.installs,
            "--model",
            self.model,
            "--prompt-cache",
            "sideways",
            physical_bytes=24 * 2**30,
        )
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("unknown --prompt-cache", run.stderr)


class ContextAndKVInteractiveTests(unittest.TestCase):
    """Interactive context-window (presets + custom) and KV-cache precision prompts."""

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()

    def _ask(self, stdin: str, *extra: str) -> subprocess.CompletedProcess[str]:
        env = self.installs.env(
            TINYTITAN_PHYSICAL_RAM_BYTES=str(24 * 2**30),
            TINYTITAN_LAUNCHER_DRY_RUN="1",
            TINYTITAN_LAUNCHER_ASSUME_TTY="1",
        )
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--client",
                "server",
                "--dry-run",
                "--model",
                self.model,
                "--thinking",
                "off",
                "--answers",
                "default",
                "--engine",
                "gpu",
                "--ram",
                "12",
                "--concurrency",
                "1",
                "--port",
                "8080",
                *extra,
            ],
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=120,
        )

    def test_preset_context_and_kv_choices(self) -> None:
        run = self._ask("4\n2\n")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--max-context 131072", run.stdout)
        self.assertIn("--kv-bits 4", run.stdout)

    def test_custom_context_shorthand_and_fp16_kv(self) -> None:
        run = self._ask("9\n64k\n3\n")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--max-context 65536", run.stdout)
        self.assertIn("--kv-bits 16", run.stdout)

    def test_long_context_presets_enable_yarn(self) -> None:
        for choice, expected in (("6", "524288"), ("7", "1048576"), ("8", "2097152")):
            with self.subTest(choice=choice):
                run = self._ask(f"{choice}\n1\n")
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(f"--max-context {expected}", run.stdout)
                self.assertIn("--rope-scaling yarn", run.stdout)

    def test_defaults_on_empty_enter(self) -> None:
        run = self._ask("\n\n")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--max-context 262144", run.stdout)
        self.assertIn("--kv-bits 8", run.stdout)


if __name__ == "__main__":
    unittest.main()
