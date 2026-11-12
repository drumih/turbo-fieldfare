#!/usr/bin/env python3
"""Tests for the qwen38 converter: n-gram reuse, resume, and downloads.

Three things this file pins, each of which failed or could fail silently in the
field:

- the `--reuse-ngram-table` gate. The PLE table is 102 GB of fp16 and the same
  bytes in every quantization, so a build may hardlink an existing one instead
  of fetching the 128 shards that carry it. What makes that safe is not the
  bytes but the constants the table is addressed by: a table built with
  different multipliers, offsets or vocabulary sizes reads wrongly and produces
  garbage ids, and nothing inside the file says so. So the refusal is the part
  worth pinning.
- resume after an interrupted conversion. Adoption is by whole shard only: a
  file a killed write left behind has a header that parses and a payload that
  does not match it, and adopting one ships a snapshot whose index points at
  bytes that are not there.
- one shard's download. A resume the endpoint refuses (curl exit 33, which is
  what a mirror answering 200 to a Range request looks like) must restart that
  file instead of retrying the request forever.

    cd benchmark && python3.13 -m unittest test_prepare_qwen38 -v

It imports the converter, which imports numpy, ml_dtypes and safetensors, so it
skips where those are absent.
"""

from __future__ import annotations

import functools
import http.server
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import prepare_qwen38 as prepare

    IMPORT_ERROR = ""
except SystemExit as exc:  # the module exits when a dependency is missing
    prepare = None
    IMPORT_ERROR = str(exc)


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class ReuseNgramTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="ngram-reuse-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.constants = {key: f"value-{key}" for key in prepare.REUSE_CONSTANT_KEYS}

    def install(self, constants: dict | None = None, sidecar: bool = True) -> pathlib.Path:
        directory = self.root / "install"
        directory.mkdir()
        if sidecar:
            (directory / "ple_constants.json").write_text(
                json.dumps(self.constants if constants is None else constants)
            )
        (directory / "ngram_table.bin").write_bytes(b"table")
        return directory

    def test_a_matching_install_returns_its_table(self) -> None:
        directory = self.install()
        self.assertEqual(
            prepare.reusable_table_path(directory, self.constants), directory / "ngram_table.bin"
        )

    def test_a_table_file_is_taken_as_given(self) -> None:
        table = self.install() / "ngram_table.bin"
        self.assertEqual(prepare.reusable_table_path(table, self.constants), table)

    def test_every_addressed_constant_is_checked(self) -> None:
        directory = self.install()
        for key in prepare.REUSE_CONSTANT_KEYS:
            with self.subTest(key=key):
                changed = dict(self.constants)
                changed[key] = "something else"
                with self.assertRaises(SystemExit) as caught:
                    prepare.reusable_table_path(directory, changed)
                self.assertIn(key, str(caught.exception))

    def test_a_directory_without_the_sidecar_is_reused_unchecked(self) -> None:
        # The current contract: with no `ple_constants.json` beside the table
        # there is nothing to compare, and the size gate on the Swift side is
        # all that is left. Pinned so a change to it is deliberate.
        directory = self.install(sidecar=False)
        self.assertEqual(
            prepare.reusable_table_path(directory, self.constants), directory / "ngram_table.bin"
        )

    def test_a_missing_table_is_refused(self) -> None:
        directory = self.install()
        (directory / "ngram_table.bin").unlink()
        with self.assertRaises(SystemExit) as caught:
            prepare.reusable_table_path(directory, self.constants)
        self.assertIn("no such file", str(caught.exception))


def write_shard(path: pathlib.Path, tensors: dict) -> bytes:
    """A real safetensors shard on disk, and its bytes."""
    from safetensors.numpy import save_file

    save_file(tensors, str(path))
    return path.read_bytes()


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class ShardHeaderTests(unittest.TestCase):
    """`read_shard_header` is the gate the resume trusts, so its refusals matter
    more than its acceptances."""

    def setUp(self) -> None:
        import numpy as np

        self.root = pathlib.Path(tempfile.mkdtemp(prefix="shard-header-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.path = self.root / "model-00001.safetensors"
        self.raw = write_shard(self.path, {"a.weight": np.zeros((4, 4), np.float32)})

    def test_a_whole_shard_parses(self) -> None:
        header = prepare.read_shard_header(self.path)
        self.assertIsNotNone(header)
        self.assertIn("a.weight", header)

    def test_a_kill_mid_payload_is_refused(self) -> None:
        # The header parses; the payload is half there. This is the shape a
        # killed `save_file` leaves, and the reason the size is checked.
        self.path.write_bytes(self.raw[: len(self.raw) - 20])
        self.assertIsNone(prepare.read_shard_header(self.path))

    def test_a_kill_inside_the_header_is_refused(self) -> None:
        self.path.write_bytes(self.raw[:12])
        self.assertIsNone(prepare.read_shard_header(self.path))

    def test_a_file_shorter_than_its_length_prefix_is_refused(self) -> None:
        self.path.write_bytes(b"\x01\x02\x03")
        self.assertIsNone(prepare.read_shard_header(self.path))

    def test_a_non_json_header_is_refused(self) -> None:
        self.path.write_bytes((8).to_bytes(8, "little") + b"not json")
        self.assertIsNone(prepare.read_shard_header(self.path))

    def test_a_missing_file_is_refused_rather_than_raised(self) -> None:
        self.assertIsNone(prepare.read_shard_header(self.root / "absent.safetensors"))


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class OutputWriterResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        self.root = pathlib.Path(tempfile.mkdtemp(prefix="writer-resume-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.np = np

    def a_writer(self) -> "prepare.OutputWriter":
        return prepare.OutputWriter(self.root)

    def test_a_whole_earlier_shard_is_adopted(self) -> None:
        write_shard(
            self.root / "model-00001.safetensors",
            {"a.weight": self.np.zeros((4, 4), self.np.float32)},
        )
        writer = self.a_writer()
        self.assertEqual(writer.shard_no, 1)
        self.assertIn("a.weight", writer.index)
        self.assertEqual(writer.total, 64)
        self.assertEqual(writer.discarded, 0)

    def test_a_truncated_earlier_shard_is_dropped_not_adopted(self) -> None:
        path = self.root / "model-00001.safetensors"
        write_shard(path, {"a.weight": self.np.zeros((8, 8), self.np.float32)})
        path.write_bytes(path.read_bytes()[:-32])
        writer = self.a_writer()
        self.assertEqual(writer.index, {})
        self.assertEqual(writer.total, 0)
        self.assertEqual(writer.shard_no, 0)
        self.assertEqual(writer.discarded, 1)
        self.assertFalse(path.exists(), "an unusable shard stays discarded")

    def test_a_leftover_partial_is_removed(self) -> None:
        stale = self.root / "model-00002.safetensors.partial"
        stale.write_bytes(b"half a shard")
        writer = self.a_writer()
        self.assertFalse(stale.exists())
        self.assertEqual(writer.shard_no, 0)

    def test_numbering_continues_after_the_adopted_shards(self) -> None:
        write_shard(
            self.root / "model-00001.safetensors",
            {"a.weight": self.np.zeros((4, 4), self.np.float32)},
        )
        writer = self.a_writer()
        writer.add("b.weight", self.np.ones((4, 4), self.np.float32))
        writer.flush()
        self.assertTrue((self.root / "model-00002.safetensors").exists())

    def test_a_flush_leaves_no_partial_behind(self) -> None:
        writer = self.a_writer()
        writer.add("a.weight", self.np.zeros((4, 4), self.np.float32))
        writer.flush()
        self.assertEqual(list(self.root.glob("*.partial")), [])

    def test_finish_names_every_adopted_and_new_shard(self) -> None:
        write_shard(
            self.root / "model-00001.safetensors",
            {"a.weight": self.np.zeros((4, 4), self.np.float32)},
        )
        writer = self.a_writer()
        writer.add("b.weight", self.np.ones((4, 4), self.np.float32))
        writer.finish()
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.assertEqual(index["weight_map"]["a.weight"], "model-00001-of-00002.safetensors")
        self.assertEqual(index["weight_map"]["b.weight"], "model-00002-of-00002.safetensors")
        self.assertEqual(index["metadata"]["total_size"], 128)


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class ConvertedShardTests(unittest.TestCase):
    """The resume decides from names alone, so the name rule has to be the one
    `convert_shard` writes with -- including the quantised triple."""

    shard = "model-00001.safetensors"

    def index_for(self, *names: str) -> dict:
        """The output index `convert_shard` would have built for `names`."""
        index = {}
        for name in names:
            if prepare.quant_bits(name, 4) is None:
                index[name] = self.shard
                continue
            stem = name[: -len(".weight")]
            for suffix in (".weight", ".scales", ".biases"):
                index[stem + suffix] = self.shard
        return index

    def test_the_runtime_names_are_the_ones_convert_shard_writes(self) -> None:
        self.assertEqual(
            prepare.output_names_for("model.language_model.layers.0.mlp.experts.gate_up_proj"),
            [
                "model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight",
                "model.language_model.layers.0.mlp.switch_mlp.up_proj.weight",
            ],
        )
        self.assertEqual(
            prepare.output_names_for(
                "model.language_model.layers.0.self_attn.indexer.index_qk_proj.weight"
            ),
            [
                "model.language_model.layers.0.self_attn.indexer.index_q_proj.weight",
                "model.language_model.layers.0.self_attn.indexer.index_k_proj.weight",
            ],
        )

    def test_a_fused_tensor_needs_both_halves(self) -> None:
        layer = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        gate = "model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight"
        up = "model.language_model.layers.0.mlp.switch_mlp.up_proj.weight"
        self.assertTrue(prepare.checkpoint_shard_is_converted([layer], self.index_for(gate, up), 4))
        self.assertFalse(prepare.checkpoint_shard_is_converted([layer], self.index_for(gate), 4))

    def test_a_quantised_tensor_needs_its_triple(self) -> None:
        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        self.assertTrue(prepare.checkpoint_shard_is_converted([name], self.index_for(name), 4))
        for suffix in (".weight", ".scales", ".biases"):
            with self.subTest(missing=suffix):
                stem = name[: -len(".weight")]
                partial = self.index_for(name)
                del partial[stem + suffix]
                self.assertFalse(prepare.checkpoint_shard_is_converted([name], partial, 4))

    def test_an_unquantised_tensor_needs_only_itself(self) -> None:
        name = "model.language_model.layers.0.input_layernorm.weight"
        self.assertIsNone(prepare.quant_bits(name, 4))
        self.assertTrue(prepare.checkpoint_shard_is_converted([name], self.index_for(name), 4))
        self.assertFalse(prepare.checkpoint_shard_is_converted([name], {}, 4))

    def test_skipped_families_do_not_count_as_written(self) -> None:
        # A shard of nothing but n-gram rows writes no tensor of its own, so it
        # can never be "already converted".
        ngram = "model.language_model.ngram_embedding.shard_0"
        self.assertTrue(prepare.is_ngram(ngram))
        self.assertFalse(prepare.checkpoint_shard_is_converted([ngram], {}, 4))

    def test_the_split_names_match_what_outputs_for_returns(self) -> None:
        for key, shape in (
            ("model.language_model.layers.0.mlp.experts.gate_up_proj", [2, 128, 64]),
            ("model.language_model.layers.0.mlp.experts.down_proj", [2, 64, 64]),
            ("model.language_model.layers.0.self_attn.indexer.index_qk_proj.weight", [640, 2560]),
            ("model.language_model.layers.0.self_attn.q_proj.weight", [256, 256]),
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    prepare.output_names_for(key),
                    [name for name, _ in prepare.outputs_for(key, shape)],
                )


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class RestBf16RecipeTests(unittest.TestCase):
    """`--rest-bf16`: only the routed experts are quantised."""

    def setUp(self) -> None:
        self.addCleanup(setattr, prepare, "REST_BF16", prepare.REST_BF16)
        prepare.REST_BF16 = True

    def test_routed_experts_keep_the_build_width(self) -> None:
        for name in (
            "model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.language_model.layers.47.mlp.switch_mlp.down_proj.weight",
        ):
            self.assertEqual(prepare.quant_bits(name, 4), 4)

    def test_every_other_family_stays_bf16(self) -> None:
        for name in (
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
            "model.language_model.layers.1.ple.key_proj.weight",
        ):
            self.assertIsNone(prepare.quant_bits(name, 4), name)

    def test_the_default_recipe_is_unchanged(self) -> None:
        prepare.REST_BF16 = False
        self.assertEqual(prepare.quant_bits("lm_head.weight", 4), 8)
        self.assertEqual(
            prepare.quant_bits("model.language_model.layers.3.self_attn.q_proj.weight", 4), 4
        )


class LocalTableReuseTests(unittest.TestCase):
    """The decision to reuse the table a previous run finished in the output
    directory: size and constants, both read from disk."""

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="local-table-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.constants = {key: f"value-{key}" for key in prepare.REUSE_CONSTANT_KEYS}

    def table(self, size: int, constants: dict | None) -> None:
        (self.root / "ngram_table.bin").write_bytes(b"x" * size)
        if constants is not None:
            (self.root / "ple_constants.json").write_text(json.dumps(constants))

    def test_a_whole_table_with_matching_constants_is_reused(self) -> None:
        self.table(64, self.constants)
        self.assertEqual(
            prepare.reusable_local_table(self.root, 64, self.constants, self.constants), self.root
        )

    def test_a_short_table_is_refused(self) -> None:
        self.table(63, self.constants)
        self.assertIsNone(
            prepare.reusable_local_table(self.root, 64, self.constants, self.constants)
        )

    def test_different_constants_refuse_it(self) -> None:
        changed = dict(self.constants)
        changed["ple_head_dim"] = "different"
        self.table(64, changed)
        self.assertIsNone(prepare.reusable_local_table(self.root, 64, changed, self.constants))

    def test_no_sidecar_refuses_it(self) -> None:
        # Unlike `reusable_table_path` (which cannot check what is not there),
        # the automatic path has this run's constants in hand, so an absent
        # sidecar is a refusal rather than a shrug.
        self.table(64, None)
        self.assertIsNone(
            prepare.reusable_local_table(
                self.root,
                64,
                prepare.read_json_file(self.root / "ple_constants.json"),
                self.constants,
            )
        )

    def test_a_missing_table_is_none(self) -> None:
        self.assertIsNone(
            prepare.reusable_local_table(self.root, 64, self.constants, self.constants)
        )

    def test_the_adopted_table_is_reused_in_place(self) -> None:
        # The automatic path hands `NgramTable` the table's own directory, so
        # the link source and the destination are one file. Unlinking the
        # destination first would delete the 102 GB source and then fail to
        # link what is gone, which is how this was found.
        self.table(64, self.constants)
        found = prepare.reusable_local_table(self.root, 64, self.constants, self.constants)
        table = self.root / "ngram_table.bin"
        inode = table.stat().st_ino
        reused = prepare.NgramTable(self.root, 4, 8, reuse=found / "ngram_table.bin")
        self.assertTrue(reused.reused)
        reused.finish()
        self.assertTrue(table.exists())
        self.assertEqual(table.stat().st_ino, inode)
        self.assertIn("linked", reused.summary())

    def test_reusing_the_destination_itself_keeps_the_table(self) -> None:
        self.table(64, self.constants)
        table = self.root / "ngram_table.bin"
        inode = table.stat().st_ino
        prepare.NgramTable(self.root, 4, 8, reuse=table)
        self.assertEqual(table.stat().st_ino, inode)


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="download-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.shard = "model-00001-of-00131.safetensors"

    def test_a_clean_run_does_not_retry(self) -> None:
        def run(command, **_kwargs):
            self.assertIn("--max-time", command)
            (self.root / self.shard).write_bytes(b"shard")
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch.object(prepare.subprocess, "run", run),
            mock.patch.object(prepare.time, "sleep") as sleep,
        ):
            self.assertEqual(prepare.download(self.shard, self.root), self.root / self.shard)
        sleep.assert_not_called()

    def test_a_refused_range_restarts_the_file(self) -> None:
        # curl 33 is "the server would not serve the range". The partial file has
        # to go, or every later attempt fails the same way.
        calls = []

        def run(command, **_kwargs):
            calls.append(command)
            if len(calls) == 1:
                return subprocess.CompletedProcess(command, 33)
            (self.root / self.shard).write_bytes(b"whole shard")
            return subprocess.CompletedProcess(command, 0)

        (self.root / self.shard).write_bytes(b"half")
        with (
            mock.patch.object(prepare.subprocess, "run", run),
            mock.patch.object(prepare.time, "sleep") as sleep,
        ):
            prepare.download(self.shard, self.root)
        self.assertEqual(len(calls), 2)
        self.assertEqual((self.root / self.shard).read_bytes(), b"whole shard")
        sleep.assert_called_once()

    def test_an_endless_failure_raises_after_the_bounded_attempts(self) -> None:
        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 22)

        with (
            mock.patch.object(prepare.subprocess, "run", run),
            mock.patch.object(prepare.time, "sleep"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                prepare.download(self.shard, self.root)
        self.assertIn(prepare.DOWNLOAD_ATTEMPTS.__str__(), str(caught.exception))


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class EndpointTests(unittest.TestCase):
    def test_the_weights_base_follows_the_endpoint(self) -> None:
        self.assertEqual(
            prepare.endpoint_base("https://hf-mirror.com"),
            f"https://hf-mirror.com/{prepare.REPO}/resolve/main",
        )
        self.assertEqual(
            prepare.endpoint_base("https://hf-mirror.com/"),
            f"https://hf-mirror.com/{prepare.REPO}/resolve/main",
        )


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class FetchTokenizerTests(unittest.TestCase):
    """`--endpoint` has to cover the tokenizer too: a snapshot fetched from a
    mirror must not reach back to huggingface.co for these five files."""

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tokenizer-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_every_tokenizer_file_comes_from_the_endpoint(self) -> None:
        mirror = prepare.endpoint_base("https://hf-mirror.com")
        seen: list[str] = []

        def run(command, **_kwargs):
            seen.append(command[-1])
            return subprocess.CompletedProcess(command, 0, b"data")

        with (
            mock.patch.object(prepare, "BASE", mirror),
            mock.patch.object(prepare.subprocess, "run", run),
        ):
            prepare.fetch_tokenizer(self.root)

        self.assertEqual(len(seen), len(prepare.TOKENIZER_FILES))
        for url in seen:
            self.assertTrue(url.startswith(f"{mirror}/"), url)
        for name, _required in prepare.TOKENIZER_FILES:
            self.assertTrue((self.root / name).exists(), name)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """A mirror that honours Range, and one that does not, in the two shapes
    that matter: 206 with the slice, and 200 with the slice (which is what
    ModelScope answers and what curl reports as exit 33 on a resume)."""

    status_for_range = 206

    def log_message(self, *_args) -> None:  # keep the test output clean
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = pathlib.Path(path).stat().st_size
        header = self.headers.get("Range")
        if not header or not header.startswith("bytes="):
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            return handle
        start_text, _, end_text = header[len("bytes=") :].partition("-")
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else size - 1
        end = min(end, size - 1)
        handle.seek(start)
        self.send_response(self.status_for_range)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._slice = end - start + 1
        return _Sliced(handle, self._slice)

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_slice", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


class _Sliced:
    def __init__(self, handle, length: int):
        self.handle = handle
        self.length = length

    def read(self, size: int = -1) -> bytes:
        if self.length <= 0:
            return b""
        if size < 0 or size > self.length:
            size = self.length
        data = self.handle.read(size)
        self.length -= len(data)
        return data

    def close(self) -> None:
        self.handle.close()


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class EndpointIntegrationTests(unittest.TestCase):
    """`--plan` and `download` against a local mirror, which is the cheapest way
    to exercise the URL layout, the ranged header read and a resume together."""

    def setUp(self) -> None:
        import numpy as np

        self.root = pathlib.Path(tempfile.mkdtemp(prefix="mirror-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        shard_dir = self.root / prepare.REPO / "resolve" / "main"
        shard_dir.mkdir(parents=True)
        self.shard_name = "model-00001-of-00001.safetensors"
        write_shard(
            shard_dir / self.shard_name,
            {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": np.zeros(
                    (2, 128, 64), np.float32
                )
            },
        )
        self.shard_bytes = (shard_dir / self.shard_name).read_bytes()
        raw_dir = self.root / prepare.REPO / "raw" / "main"
        raw_dir.mkdir(parents=True)
        (raw_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": len(self.shard_bytes)},
                    "weight_map": {
                        "model.language_model.layers.0.mlp.experts.gate_up_proj": self.shard_name
                    },
                }
            )
        )
        self.saved_base = prepare.BASE

    def tearDown(self) -> None:
        prepare.BASE = self.saved_base

    def serve(self, handler) -> str:
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(handler, directory=str(self.root))
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_plan_reads_the_mirror(self) -> None:
        endpoint = self.serve(_RangeHandler)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/prepare_qwen38.py"),
                "--plan",
                "--endpoint",
                endpoint,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("plan validates against the checkpoint's own headers", result.stdout)

    def test_a_partial_download_resumes_against_a_range_endpoint(self) -> None:
        prepare.BASE = f"{self.serve(_RangeHandler)}/{prepare.REPO}/resolve/main"
        work = self.root / "work"
        work.mkdir()
        (work / self.shard_name).write_bytes(self.shard_bytes[:64])
        got = prepare.download(self.shard_name, work)
        self.assertEqual(got.read_bytes(), self.shard_bytes)

    def test_a_refused_range_still_yields_the_whole_shard(self) -> None:
        handler = type("_NoRangeHandler", (_RangeHandler,), {"status_for_range": 200})
        prepare.BASE = f"{self.serve(handler)}/{prepare.REPO}/resolve/main"
        work = self.root / "work"
        work.mkdir()
        (work / self.shard_name).write_bytes(self.shard_bytes[:64])
        with mock.patch.object(prepare.time, "sleep"):
            got = prepare.download(self.shard_name, work)
        self.assertEqual(got.read_bytes(), self.shard_bytes)


@unittest.skipIf(prepare is None, f"prepare_qwen38 unavailable: {IMPORT_ERROR}")
class FinishedOutputGuardTests(unittest.TestCase):
    """A finished snapshot is not a resume target: converting into it would lay a
    second generation of shards beside the first."""

    def test_a_finished_directory_is_refused(self) -> None:
        root = pathlib.Path(tempfile.mkdtemp(prefix="finished-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        index = root / "model.safetensors.index.json"
        index.write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": {}}))
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/prepare_qwen38.py"),
                "--index",
                str(index),
                "--output",
                str(root),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already holds a finished snapshot", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
