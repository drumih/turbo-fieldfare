"""`tools/server_launcher.sh --client opencode` writes the checked-in provider.

`plugins/opencode-tinytitan/opencode.json` is the source of truth for OpenCode's
`tinytitan` provider. The launcher used to write its own smaller block (one model,
no variants, another name and output cap), and it never looked at
`disabled_providers`, so a `tinytitan` left there kept the provider hidden. These
tests pin the merge (`tools/client_config_merge.py`) and the launcher's own
`configure_opencode` -- extracted from the script and run against a temporary
HOME -- to that file: the block is the file's, the user's other settings and
models survive, the provider is taken out of `disabled_providers` everywhere
OpenCode reads it, and nothing is rewritten without a backup of what it replaces.

Nothing here starts a server, reads the real ~/.config, or needs an install.

    cd benchmark && python3 -m unittest test_launcher_opencode -v
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

import launcher_fixture

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"
MERGE = ROOT / "tools/client_config_merge.py"
SOURCE = ROOT / "plugins/opencode-tinytitan/opencode.json"
LAUNCHER_LIB = ROOT / "tools/tinytitan_models.sh"
URL = "http://127.0.0.1:8080/v1"


def default_model() -> str:
    """The file's own default id, without the `tinytitan/` prefix.

    Derived, not a literal: the id changes when the checked-in install recipe
    changes (6.3's `qwen3.8-flash-next-8bitrest_4-Bit`), and a literal here
    would fail every such change with a test that is not about the id.
    """
    return source()["model"].split("/", 1)[1]


def installer_menu_rows() -> int:
    """The number of models `tools/install_models.sh --choose` offers.

    The provider file must carry one entry per launcher-menu row — that is the
    coupling TT-066 shipped — so the expected count is read from the menu list
    itself rather than restated here.
    """
    text = LAUNCHER_LIB.read_text()
    block = text.split("TINYTITAN_MODEL_CHOICES=(", 1)[1].split("\n)", 1)[0]
    return sum(1 for line in block.splitlines() if line.strip().startswith('"'))


def source() -> dict:
    return json.loads(SOURCE.read_text())


def merge(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MERGE), *args], text=True, capture_output=True, check=False
    )


def backups(path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path.parent.glob(path.name + ".tinytitan-backup*"))


class SourceOfTruthTests(unittest.TestCase):
    """The checked-in file says what the task and the plugin README say it does."""

    def test_the_provider_block(self) -> None:
        data = source()
        self.assertEqual(data["model"], f"tinytitan/{default_model()}")
        block = data["provider"]["tinytitan"]
        self.assertEqual(block["name"], "TinyTitan")
        self.assertEqual(block["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(block["options"]["baseURL"], URL)
        self.assertIs(block["options"]["timeout"], False)
        self.assertEqual(len(block["models"]), installer_menu_rows())
        self.assertIn(default_model(), block["models"])
        for model_id, model in block["models"].items():
            with self.subTest(model=model_id):
                self.assertEqual(model["limit"], {"context": 262144, "output": 32768})
                for variant in model["variants"].values():
                    self.assertTrue("reasoningEffort" in variant or variant == {"disabled": True})


class MergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        self.file = self.dir / "opencode.jsonc"

    def run_merge(self, *extra: str, model: str | None = None, url: str = URL) -> str:
        run = merge(
            "opencode",
            "--file",
            str(self.file),
            "--source",
            str(SOURCE),
            "--model",
            model or default_model(),
            "--base-url",
            url,
            *extra,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        return run.stdout

    def test_no_config_yet_becomes_the_checked_in_file(self) -> None:
        self.run_merge("--context", "262144")
        self.assertEqual(json.loads(self.file.read_text()), source())
        self.assertEqual(backups(self.file), [])

    def test_existing_config_is_merged_backed_up_and_unhidden(self) -> None:
        original = (
            "{\n"
            "  // mine\n"
            '  "theme": "dark",\n'
            '  "disabled_providers": ["tinytitan", "groq"],\n'
            '  "enabled_providers": ["anthropic"],\n'
            '  "provider": {\n'
            '    "anthropic": {"options": {"apiKey": "x"}},\n'
            '    "tinytitan": {"name": "TinyTitan (local)", "options": {"baseURL": "old"},\n'
            '      "models": {"qwen3.8-flash-next-mtp_4-Bit": {"limit": {"output": 65536}},\n'
            '                 "my-own_4-Bit": {"name": "mine"}}},\n'
            "  },\n"
            "}\n"
        )
        self.file.write_text(original)
        out = self.run_merge()
        self.assertIn("Removed 'tinytitan' from disabled_providers", out)
        [saved] = backups(self.file)
        self.assertEqual(saved.read_text(), original)
        data = json.loads(self.file.read_text())
        self.assertEqual(data["theme"], "dark")
        self.assertNotIn("model", data, "an existing config keeps its own default model")
        self.assertEqual(data["disabled_providers"], ["groq"])
        self.assertEqual(data["enabled_providers"], ["anthropic", "tinytitan"])
        self.assertEqual(data["provider"]["anthropic"], {"options": {"apiKey": "x"}})
        block = data["provider"]["tinytitan"]
        expected = source()["provider"]["tinytitan"]
        self.assertEqual(
            {k: v for k, v in block.items() if k != "models"},
            {k: v for k, v in expected.items() if k != "models"},
        )
        self.assertEqual(block["models"]["my-own_4-Bit"], {"name": "mine"})
        for model_id, model in expected["models"].items():
            self.assertEqual(block["models"][model_id], model)

    def test_a_second_run_changes_nothing_and_backs_up_nothing(self) -> None:
        self.file.write_text('{"disabled_providers": ["tinytitan"]}\n')
        self.run_merge()
        first = self.file.read_text()
        out = self.run_merge()
        self.assertIn("already configured", out)
        self.assertEqual(self.file.read_text(), first)
        self.assertEqual(len(backups(self.file)), 1)

    def test_every_later_change_gets_its_own_backup(self) -> None:
        self.file.write_text("{}\n")
        self.run_merge()
        self.run_merge(url="http://127.0.0.1:9090/v1")
        saved = backups(self.file)
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0].read_text(), "{}\n")
        self.assertIn('"baseURL": "http://127.0.0.1:8080/v1"', saved[1].read_text())

    def test_port_context_and_an_unlisted_model(self) -> None:
        self.run_merge("--context", "524288", url="http://127.0.0.1:9090/v1")
        block = json.loads(self.file.read_text())["provider"]["tinytitan"]
        self.assertEqual(block["options"]["baseURL"], "http://127.0.0.1:9090/v1")
        self.assertEqual(block["models"][default_model()]["limit"]["context"], 524288)
        self.file.unlink()
        self.run_merge(model="katcoder-x_4-Bit")
        data = json.loads(self.file.read_text())
        self.assertEqual(data["model"], "tinytitan/katcoder-x_4-Bit")
        entry = data["provider"]["tinytitan"]["models"]["katcoder-x_4-Bit"]
        self.assertEqual(entry["limit"], {"context": 262144, "output": 32768})
        self.assertNotIn("variants", entry)

    def test_other_config_files_are_unhidden_with_a_backup(self) -> None:
        other = self.dir / "config.json"
        other.write_text('{"disabled_providers": ["tinytitan"], "theme": "x"}\n')
        untouched = self.dir / "opencode.json"
        untouched.write_text('{"theme": "y"}\n')
        self.run_merge("--also", str(other), "--also", str(untouched))
        self.assertEqual(json.loads(other.read_text()), {"disabled_providers": [], "theme": "x"})
        self.assertEqual(len(backups(other)), 1)
        self.assertEqual(untouched.read_text(), '{"theme": "y"}\n')
        self.assertEqual(backups(untouched), [])

    def test_an_unparseable_config_is_left_alone(self) -> None:
        self.file.write_text("{ not json")
        run = merge(
            "opencode", "--file", str(self.file), "--source", str(SOURCE),
            "--model", default_model(), "--base-url", URL,
        )  # fmt: skip
        self.assertEqual(run.returncode, 1)
        self.assertIn("not parseable", run.stderr)
        self.assertEqual(self.file.read_text(), "{ not json")
        self.assertEqual(backups(self.file), [])


def configure_opencode_source() -> str:
    text = LAUNCHER.read_text()
    match = re.search(r"^configure_opencode\(\) \{\n.*?^\}\n", text, re.S | re.M)
    if match is None:
        raise AssertionError("configure_opencode() not found in the launcher")
    return match.group(0)


class LauncherConfigureTests(unittest.TestCase):
    """The launcher's own function, run against a temporary HOME."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = pathlib.Path(self.tmp.name)
        self.dir = self.home / ".config/opencode"

    def configure(self, **env: str) -> subprocess.CompletedProcess[str]:
        script = (
            f"SCRIPT_DIR={str(ROOT / 'tools')!r}; BASE_DIR={str(ROOT)!r}\n"
            f"launch_model={default_model()}; BASE_URL={URL}; max_context=''\n"
            + configure_opencode_source()
            + "configure_opencode\n"
        )
        environment = {"PATH": os.environ["PATH"], "HOME": str(self.home), **env}
        return subprocess.run(
            ["bash", "-c", script], text=True, capture_output=True, check=False, env=environment
        )

    def test_a_fresh_home_gets_the_checked_in_file(self) -> None:
        run = self.configure()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads((self.dir / "opencode.jsonc").read_text()), source())

    def test_an_existing_opencode_json_is_the_one_edited(self) -> None:
        self.dir.mkdir(parents=True)
        (self.dir / "opencode.json").write_text('{"disabled_providers": ["tinytitan"]}')
        custom = self.home / "custom.json"
        custom.write_text('{"disabled_providers": ["tinytitan", "groq"]}')
        run = self.configure(OPENCODE_CONFIG=str(custom))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse((self.dir / "opencode.jsonc").exists())
        data = json.loads((self.dir / "opencode.json").read_text())
        self.assertEqual(data["disabled_providers"], [])
        self.assertIn("tinytitan", data["provider"])
        self.assertEqual(json.loads(custom.read_text())["disabled_providers"], ["groq"])
        self.assertEqual(len(backups(self.dir / "opencode.json")), 1)
        self.assertEqual(len(backups(custom)), 1)


class LauncherWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def test_dry_run_names_the_source_and_disabled_providers(self) -> None:
        run = subprocess.run(
            ["bash", str(LAUNCHER), "--client", "opencode", "--model",
             self.installs.first_gpu(), "--dry-run"],
            text=True, capture_output=True, check=False, env=self.installs.env(),
        )  # fmt: skip
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("plugins/opencode-tinytitan/opencode.json", run.stdout)
        self.assertIn("disabled_providers", run.stdout)

    def test_opencode_starts_on_the_launched_model(self) -> None:
        self.assertIn('opencode) "$BIN" -m "tinytitan/$launch_model" ;;', LAUNCHER.read_text())


if __name__ == "__main__":
    unittest.main()
