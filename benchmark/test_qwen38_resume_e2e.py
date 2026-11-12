#!/usr/bin/env python3
"""End-to-end fault-injection tests for the resumable qwen38 conversion.

`test_prepare_qwen38.py` pins the pieces; this file runs the whole converter
(`main()`) against a synthetic but structurally real checkpoint served over a
real HTTP server, over real `curl` downloads into a test folder, and breaks the
transport and the process underneath it on purpose:

- a transfer dropped or truncated mid-body (curl 18), a stall past `--max-time`
  (curl 28), a 500 (curl 22), and a mirror that answers a `Range` request with
  200 (curl 33) while a partial file is on disk;
- a real `SIGKILL` while the converter is writing output shards, followed by a
  resume that must end byte-identical to a clean run;
- a truncated adopted shard, a leftover `*.partial`, a table from a previous
  run (reused in place), a wrong-size table, changed constants, and a directory
  that is one step away from finished;
- the n-gram table's numeric shard order (shard_10 must not sort before
  shard_2), and a table that has to cross filesystems by copy, not hardlink.

Every fault case ends with `fingerprint(output) == fingerprint(clean run)`: the
recovery paths are allowed to do work again, never to change the bytes. The
synthetic checkpoint carries the shapes that make the converter's decisions
observable -- a fused expert, an indexer split, an 8-bit slot in a 4-bit build,
a unit-offset norm, a PLE buffer, a multimodal tensor, and out-of-order n-gram
shards.

It needs `numpy`, `ml_dtypes` and `safetensors`, the system `curl`, and (for
the cross-filesystem case) `hdiutil`; cases that need a tool that is missing
skip rather than fail. Run it with:

    cd benchmark && python3.13 -m unittest test_qwen38_resume_e2e -v

It never contacts the network: every URL it fetches points at 127.0.0.1.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
import zlib
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError as exc:  # pragma: no cover - environment, not logic
    ml_dtypes = np = safe_open = save_file = None
    DEPS_ERROR = str(exc)
else:
    DEPS_ERROR = ""

try:
    import prepare_qwen38 as prepare

    PREPARE_ERROR = ""
except SystemExit as exc:  # the module exits when a dependency is missing
    prepare = None
    PREPARE_ERROR = str(exc)

REPO = "Qwen/Qwen3.8-Flash-Next"
PREFIX = f"/{REPO}"
HEAD_DIM_DIVISOR = 1  # `make_ngram_vocab_size_divisible_by`, so rows == table rows


def requires_deps(obj):
    reason = DEPS_ERROR or PREPARE_ERROR
    return unittest.skipIf(reason, f"converter or its dependencies unavailable: {reason}")(obj)


# --- the synthetic checkpoint ----------------------------------------------


def text_config() -> dict:
    """The config keys the converter reads, sized to stay a few kB.

    Twelve n-gram heads (six orders x two) is deliberate: it produces shard_10
    and shard_11, whose names sort before shard_2 lexically.
    """
    return {
        "heads_per_ngram": 6,
        "ngram_vocab_size_base": 60,
        "ngram_size": 3,
        "vocab_size": 128,
        "eos_token_id": 2,
        "seed": 1234,
        "ple_embed_dim": 96,
        "make_ngram_vocab_size_divisible_by": HEAD_DIM_DIVISOR,
    }


def _float(tag: str, shape) -> "np.ndarray":
    rng = np.random.default_rng(zlib.crc32(tag.encode()) or 1)
    return (rng.random(shape, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)


def ngram_name(index: int) -> str:
    return f"model.language_model.layers.0.ple.ngram_embedding.shard_{index}.weight"


def ngram_sizes() -> list[int]:
    return prepare.ple_constants(text_config())["ngram_heads_vocab_sizes"]


def head_dim() -> int:
    return prepare.ple_constants(text_config())["ple_head_dim"]


def ngram_block(index: int, size: int, dim: int) -> "np.ndarray":
    """One n-gram table shard, in bf16 like the checkpoint's."""
    rng = np.random.default_rng(1000 + index)
    values = rng.random((size, dim), dtype=np.float32) * 2.0 - 1.0
    return values.astype(ml_dtypes.bfloat16)


def expected_table_bytes() -> bytes:
    """The table the converter must write: numeric shard order, bf16 -> fp16."""
    dim = head_dim()
    out = bytearray()
    for index, size in enumerate(ngram_sizes()):
        block = ngram_block(index, size, dim)
        out += np.ascontiguousarray(block.astype(np.float32).astype(np.float16)).tobytes()
    return bytes(out)


def expected_table_rows() -> int:
    return sum(ngram_sizes())


def checkpoint() -> tuple[dict[str, bytes], dict[str, dict[str, "np.ndarray"]]]:
    """Every file the mirror serves, plus the tensor plan behind them.

    The plan is what the semantic checks compare the converted snapshot
    against; it is built here independently of `convert_shard`.
    """
    sizes = ngram_sizes()
    dim = head_dim()
    shards: dict[str, dict[str, "np.ndarray"]] = {
        "model-00001-of-00005.safetensors": {
            "model.language_model.embed_tokens.weight": _float("embed", (128, 64)),
            "model.language_model.layers.0.input_layernorm.weight": _float("inln", (64,)),
            "model.language_model.layers.0.self_attn.q_norm.weight": _float("qnorm", (64,)),
            "model.language_model.layers.0.mlp.gate.weight": _float("gate", (32, 64)),
        },
        "model-00002-of-00005.safetensors": {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": _float(
                "gateup", (3, 128, 64)
            ),
            "model.language_model.layers.0.mlp.experts.down_proj": _float("down", (3, 64, 64)),
            "model.visual.patch_embed.weight": _float("visual", (8, 8)),
        },
        "model-00003-of-00005.safetensors": {
            "model.language_model.layers.0.self_attn.indexer.index_qk_proj.weight": _float(
                "qk", (prepare.INDEXER_QUERY_ROWS + 64, 64)
            ),
            "model.language_model.layers.0.ple.ple_embedding.layer_multipliers": np.arange(
                3, dtype=np.int64
            ),
        },
        # shard_10 and shard_11 arrive before shard_0: NgramTable must buffer by
        # index, not concatenate in arrival or lexical order.
        "model-00004-of-00005.safetensors": {
            "model.language_model.layers.0.self_attn.o_proj.weight": _float("oproj", (64, 64)),
            ngram_name(10): ngram_block(10, sizes[10], dim),
            ngram_name(11): ngram_block(11, sizes[11], dim),
        },
        # Nothing but n-gram rows: skipped entirely when the table is reused.
        "model-00005-of-00005.safetensors": {
            ngram_name(i): ngram_block(i, sizes[i], dim) for i in range(10)
        },
    }

    stage = pathlib.Path(tempfile.mkdtemp(prefix="tt-checkpoint-"))
    try:
        files: dict[str, bytes] = {}
        weight_map: dict[str, str] = {}
        total = 0
        for shard, tensors in shards.items():
            path = stage / shard
            save_file(tensors, str(path))
            files[shard] = path.read_bytes()
            for name, value in tensors.items():
                weight_map[name] = shard
                total += value.nbytes
        files["model.safetensors.index.json"] = json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1
        ).encode()
        files["config.json"] = json.dumps(
            {"architectures": ["Qwen3.8FlashNextForCausalLM"], "text_config": text_config()},
            indent=1,
        ).encode()
        for name, _required in prepare.TOKENIZER_FILES:
            files[name] = f"{{}}  // {name}\n".encode()
        return files, shards
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# The output the converter must produce, written out by hand from the plan
# above. Quantised tensors carry the `.weight`/`.scales`/`.biases` triple; the
# norm keeps no `.weight` because `rename` strips it for that family.
EXPECTED_QUANTISED = {
    "model.language_model.embed_tokens": 8,
    "model.language_model.layers.0.mlp.gate": 8,
    "model.language_model.layers.0.mlp.switch_mlp.gate_proj": 4,
    "model.language_model.layers.0.mlp.switch_mlp.up_proj": 4,
    "model.language_model.layers.0.mlp.switch_mlp.down_proj": 4,
    "model.language_model.layers.0.self_attn.indexer.index_q_proj": 4,
    "model.language_model.layers.0.self_attn.indexer.index_k_proj": 4,
    "model.language_model.layers.0.self_attn.o_proj": 4,
}
EXPECTED_UNQUANTISED = (
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.self_attn.q_norm",
)


def expected_output_names() -> set[str]:
    names = set(EXPECTED_UNQUANTISED)
    for stem in EXPECTED_QUANTISED:
        names |= {stem + ".weight", stem + ".scales", stem + ".biases"}
    return names


# --- the mirror -------------------------------------------------------------


class FaultMirror(http.server.ThreadingHTTPServer):
    """A Hub-layout mirror that can misbehave on request.

    `faults[name]` is a list consumed one per request for that file, the last
    entry repeating. Behaviours: `ok`, `drop` (correct Content-Length, half the
    body, connection closed -- curl 18), `short` (the same), `refuse_range`
    (200 with the whole body even when a range was asked for -- curl 33 on a
    resume), `stall` (sleeps past the client's `--max-time`), `404` (curl 22
    without the delays a 500 would add through curl's own retry loop).
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, files: dict[str, bytes]):
        super().__init__(("127.0.0.1", 0), _MirrorHandler)
        self.files = files
        self.faults: dict[str, list[str]] = {}
        self.requests: list[dict] = []
        self.counts: dict[str, int] = {}
        self.delay = 0.0
        self.stall_seconds = 2.0
        self.lock = threading.Lock()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def next_behaviour(self, name: str) -> tuple[str, int]:
        with self.lock:
            index = self.counts.get(name, 0)
            self.counts[name] = index + 1
        plan = self.faults.get(name) or []
        if not plan:
            return "ok", index
        return (plan[index] if index < len(plan) else plan[-1]), index

    def record(self, entry: dict) -> None:
        with self.lock:
            self.requests.append(entry)

    def requests_for(self, name: str) -> list[dict]:
        with self.lock:
            return [r for r in self.requests if r["file"] == name]

    def handle_error(self, *_args) -> None:
        # A client that timed out mid-response (the `stall` case) resets the
        # connection; that is the point of the test, not a traceback.
        pass


class _MirrorHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FaultMirror/1"

    def log_message(self, *_args) -> None:  # keep the test output readable
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
        path = urllib.parse.urlparse(self.path).path
        if "/main/" not in path:
            self.send_error(404)
            return
        name = path.split("/main/", 1)[1]
        data = self.server.files.get(name)
        if data is None:
            self.send_error(404)
            return
        if self.server.delay:
            time.sleep(self.server.delay)
        behaviour, _index = self.server.next_behaviour(name)
        header = self.headers.get("Range")
        start = end = None
        if header and header.startswith("bytes="):
            first, _, last = header[len("bytes=") :].split(",")[0].partition("-")
            start = int(first) if first else 0
            end = int(last) if last else len(data) - 1
        self.server.record({"file": name, "path": path, "range": header, "behaviour": behaviour})

        if behaviour == "404":
            self.send_error(404)
            return
        if behaviour == "stall":
            time.sleep(self.server.stall_seconds)
            behaviour = "refuse_range"

        ranged = header is not None and start is not None and behaviour != "refuse_range"
        body = data[start : end + 1] if ranged else data
        status = 206 if ranged else 200
        declared = len(body)
        content_range = f"bytes {start}-{start + declared - 1}/{len(data)}" if ranged else None
        if behaviour in ("drop", "short"):
            body = body[: max(1, declared // 2)]
        self._send(status, body, declared, content_range)

    def _send(self, status: int, body: bytes, declared: int, content_range: str | None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(declared))
        if content_range:
            self.send_header("Content-Range", content_range)
        self.end_headers()
        self.wfile.write(body)
        if len(body) != declared:
            self.close_connection = True


@contextlib.contextmanager
def mirror(files: dict[str, bytes]):
    server = FaultMirror(files)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


# --- running the converter --------------------------------------------------


def run_converter(
    out: pathlib.Path,
    work: pathlib.Path,
    endpoint: str,
    extra: tuple = (),
    patches: dict | None = None,
    patch_sleep: bool = True,
) -> tuple[int, str]:
    """`main()` in this process, against `endpoint`, with output captured.

    `time.sleep` is replaced inside the converter only (a retry backoff is
    5-120 s); the mirror keeps the real clock, so a `stall` is still a stall.
    """
    argv = [
        "prepare_qwen38.py",
        "--output",
        str(out),
        "--work",
        str(work),
        "--endpoint",
        endpoint,
        *[str(item) for item in extra],
    ]
    saved = (prepare.HF_ENDPOINT, prepare.BASE)
    buffer = io.StringIO()
    stack = contextlib.ExitStack()
    try:
        stack.enter_context(mock.patch.object(sys, "argv", argv))
        stack.enter_context(redirect_stdout(buffer))
        stack.enter_context(redirect_stderr(buffer))
        if patch_sleep:
            stack.enter_context(
                mock.patch.object(prepare, "time", mock.Mock(sleep=lambda *_: None))
            )
        # curl writes its diagnostics to a real file descriptor, which
        # `redirect_stderr` does not touch. Capture them into the log so a
        # failure reads in one place (and so the test output stays readable).
        real_run = subprocess.run

        def run_captured(command, **kwargs):
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("capture_output", "stdout", "stderr")
            }
            result = real_run(command, capture_output=True, **kwargs)
            if result.stderr:
                print(result.stderr.decode("utf-8", "replace").rstrip(), file=buffer)
            return result

        stack.enter_context(mock.patch.object(prepare.subprocess, "run", run_captured))
        for name, value in (patches or {}).items():
            stack.enter_context(mock.patch.object(prepare, name, value))
        try:
            code = prepare.main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            print(f"SystemExit: {exc}", file=buffer)
        except BaseException:  # noqa: BLE001 - reported through the log
            if isinstance(sys.exc_info()[1], KeyboardInterrupt):
                raise
            code = 1
            import traceback

            traceback.print_exc(file=buffer)
    finally:
        stack.close()
        prepare.HF_ENDPOINT, prepare.BASE = saved
    return code, buffer.getvalue()


def fingerprint(out: pathlib.Path) -> dict:
    """Everything about a snapshot that a loader would read, content-only."""
    index = json.loads((out / "model.safetensors.index.json").read_text())
    tensors = {}
    for name, shard in index["weight_map"].items():
        with safe_open(out / shard, framework="np") as handle:
            value = handle.get_tensor(name)
        tensors[name] = (
            str(value.dtype),
            tuple(value.shape),
            hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        )
    return {
        "tensors": tensors,
        "table": hashlib.sha256((out / "ngram_table.bin").read_bytes()).hexdigest(),
        "total_size": index["metadata"]["total_size"],
    }


_BASELINE: dict | None = None


def baseline_fingerprint() -> dict:
    """The snapshot a clean run of the same checkpoint produces, cached."""
    global _BASELINE
    if _BASELINE is None:
        files, _plan = checkpoint()
        root = pathlib.Path(tempfile.mkdtemp(prefix="tt-baseline-"))
        try:
            with mirror(files) as server:
                code, log = run_converter(
                    root / "out", root / "work", server.base, patch_sleep=False
                )
            if code != 0:
                raise AssertionError(f"the clean baseline run failed:\n{log}")
            _BASELINE = fingerprint(root / "out")
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return _BASELINE


def dequantise(packed, scales, biases, bits: int, group: int = 64):
    """Unpack an affine tensor back to float, independently of the converter."""
    lanes = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    shifts = np.arange(lanes, dtype=np.uint32) * np.uint32(bits)
    values = (packed.astype(np.uint32)[..., None] >> shifts) & mask
    values = values.reshape(*packed.shape[:-1], packed.shape[-1] * lanes)
    flat = values.astype(np.float32).reshape(*values.shape[:-1], values.shape[-1] // group, group)
    scaled = flat * scales.astype(np.float32)[..., None] + biases.astype(np.float32)[..., None]
    return scaled.reshape(values.shape)


def read_tensor(out: pathlib.Path, name: str):
    index = json.loads((out / "model.safetensors.index.json").read_text())
    with safe_open(out / index["weight_map"][name], framework="np") as handle:
        return handle.get_tensor(name)


# --- cases ------------------------------------------------------------------


@requires_deps
class SnapshotContractTests(unittest.TestCase):
    """One clean run: the snapshot is what the loaders expect, byte for byte."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.files, cls.plan = checkpoint()
        cls.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-e2e-contract-"))
        cls.server_ctx = mirror(cls.files)
        cls.server = cls.server_ctx.__enter__()
        cls.code, cls.log = run_converter(
            cls.root / "out", cls.root / "work", cls.server.base, patch_sleep=False
        )
        cls.out = cls.root / "out"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server_ctx.__exit__(None, None, None)
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_the_run_succeeds(self) -> None:
        self.assertEqual(self.code, 0, self.log)

    def test_every_expected_tensor_is_present_and_nothing_else_is(self) -> None:
        index = json.loads((self.out / "model.safetensors.index.json").read_text())
        self.assertEqual(set(index["weight_map"]), expected_output_names())
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_no_partial_or_orphan_shard_is_left_behind(self) -> None:
        leftovers = [
            p.name
            for p in self.out.iterdir()
            if p.name.endswith(".partial")
            or (
                p.name.startswith("model-")
                and p.name.endswith(".safetensors")
                and "-of-" not in p.name
            )
        ]
        self.assertEqual(leftovers, [])

    def test_the_index_only_names_shards_that_exist(self) -> None:
        index = json.loads((self.out / "model.safetensors.index.json").read_text())
        total = 0
        for name, shard in index["weight_map"].items():
            self.assertTrue((self.out / shard).exists(), shard)
            with safe_open(self.out / shard, framework="np") as handle:
                total += handle.get_tensor(name).nbytes
        self.assertEqual(index["metadata"]["total_size"], total)

    def test_the_fused_experts_are_split_the_right_way_round(self) -> None:
        # Reconstructed values must match the two halves of the checkpoint's
        # fused tensor: gate from the first half, up from the second. A swap
        # keeps every shape and size correct and is otherwise invisible.
        fused = self.plan["model-00002-of-00005.safetensors"][
            "model.language_model.layers.0.mlp.experts.gate_up_proj"
        ]
        half = fused.shape[1] // 2
        for suffix, piece in (("gate_proj", fused[:, :half, :]), ("up_proj", fused[:, half:, :])):
            stem = f"model.language_model.layers.0.mlp.switch_mlp.{suffix}"
            rebuilt = dequantise(
                read_tensor(self.out, stem + ".weight"),
                read_tensor(self.out, stem + ".scales"),
                read_tensor(self.out, stem + ".biases"),
                4,
            )
            error = float(np.abs(rebuilt - piece).max())
            limit = 2 * float(read_tensor(self.out, stem + ".scales").astype(np.float32).max())
            self.assertLessEqual(error, limit + 1e-6, stem)

    def test_the_routed_expert_and_indexer_round_trip(self) -> None:
        cases = [
            (
                "model.language_model.layers.0.mlp.switch_mlp.down_proj",
                self.plan["model-00002-of-00005.safetensors"][
                    "model.language_model.layers.0.mlp.experts.down_proj"
                ],
                4,
            ),
            (
                "model.language_model.layers.0.self_attn.indexer.index_q_proj",
                self.plan["model-00003-of-00005.safetensors"][
                    "model.language_model.layers.0.self_attn.indexer.index_qk_proj.weight"
                ][: prepare.INDEXER_QUERY_ROWS],
                4,
            ),
            (
                "model.language_model.layers.0.self_attn.indexer.index_k_proj",
                self.plan["model-00003-of-00005.safetensors"][
                    "model.language_model.layers.0.self_attn.indexer.index_qk_proj.weight"
                ][prepare.INDEXER_QUERY_ROWS :],
                4,
            ),
            (
                "model.language_model.embed_tokens",
                self.plan["model-00001-of-00005.safetensors"][
                    "model.language_model.embed_tokens.weight"
                ],
                8,
            ),
        ]
        for stem, piece, bits in cases:
            with self.subTest(stem=stem):
                scales = read_tensor(self.out, stem + ".scales")
                rebuilt = dequantise(
                    read_tensor(self.out, stem + ".weight"),
                    scales,
                    read_tensor(self.out, stem + ".biases"),
                    bits,
                )
                self.assertEqual(rebuilt.shape, piece.shape)
                limit = 2 * float(scales.astype(np.float32).max())
                self.assertLessEqual(float(np.abs(rebuilt - piece).max()), limit + 1e-6)

    def test_the_unit_offset_norm_is_folded_and_renamed(self) -> None:
        original = self.plan["model-00001-of-00005.safetensors"][
            "model.language_model.layers.0.self_attn.q_norm.weight"
        ]
        stored = read_tensor(self.out, "model.language_model.layers.0.self_attn.q_norm")
        np.testing.assert_allclose(stored.astype(np.float32), original + 1.0, rtol=0, atol=0)
        # And a plain norm must not be touched.
        plain = self.plan["model-00001-of-00005.safetensors"][
            "model.language_model.layers.0.input_layernorm.weight"
        ]
        np.testing.assert_allclose(
            read_tensor(self.out, "model.language_model.layers.0.input_layernorm.weight"),
            plain,
            rtol=0,
            atol=0,
        )

    def test_the_8_bit_slot_carries_a_per_tensor_override(self) -> None:
        config = json.loads((self.out / "config.json").read_text())
        quant = config["quantization"]
        self.assertEqual(quant["bits"], 4)
        self.assertEqual(quant["model.language_model.embed_tokens"]["bits"], 8)
        self.assertEqual(quant["model.language_model.layers.0.mlp.gate"]["bits"], 8)
        packed = read_tensor(self.out, "model.language_model.embed_tokens.weight")
        self.assertEqual(packed.dtype, np.uint32)
        self.assertEqual(packed.shape[-1], 64 // 4)  # 8 bits -> 4 values per word

    def test_multimodal_and_ple_buffer_tensors_never_reach_the_snapshot(self) -> None:
        index = json.loads((self.out / "model.safetensors.index.json").read_text())
        for name in index["weight_map"]:
            self.assertNotIn("visual", name)
            self.assertNotIn("ple_embedding", name)

    def test_the_ngram_table_is_the_shards_in_numeric_order(self) -> None:
        # shard_10/shard_11 arrive before shard_0 and sort before shard_2
        # lexically; only numeric concatenation produces this byte string.
        table = (self.out / "ngram_table.bin").read_bytes()
        self.assertEqual(table, expected_table_bytes())
        self.assertEqual(len(table), expected_table_rows() * head_dim() * 2)

    def test_everything_was_fetched_from_the_endpoint_it_was_given(self) -> None:
        for request in self.server.requests:
            self.assertTrue(request["path"].startswith(PREFIX), request)
            if request["file"] in ("model.safetensors.index.json", "config.json"):
                self.assertIn("/raw/main/", request["path"], request)
            else:
                self.assertIn("/resolve/main/", request["path"], request)
        fetched = {request["file"] for request in self.server.requests}
        for name, _required in prepare.TOKENIZER_FILES:
            self.assertIn(name, fetched, "tokenizer file was not fetched from the endpoint")
        weights = {name for name in fetched if name.endswith(".safetensors")}
        self.assertEqual(weights, {name for name in self.files if name.endswith(".safetensors")})


@requires_deps
class ParallelFetchTests(unittest.TestCase):
    """`--fetch-workers` changes download order, never the snapshot."""

    def test_parallel_fetch_writes_the_clean_run_snapshot(self) -> None:
        files, _plan = checkpoint()
        root = pathlib.Path(tempfile.mkdtemp(prefix="tt-e2e-parallel-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        with mirror(files) as server:
            code, log = run_converter(
                root / "out",
                root / "work",
                server.base,
                extra=("--fetch-workers", 3),
                patch_sleep=False,
            )
        self.assertEqual(code, 0, log)
        self.assertEqual(fingerprint(root / "out"), baseline_fingerprint())
        leftovers = [p.name for p in (root / "work").glob("*.safetensors")]
        self.assertEqual(leftovers, [])


@requires_deps
class TransportFaultTests(unittest.TestCase):
    """Transport failures during real downloads, and their recovery."""

    def setUp(self) -> None:
        self.files, self.plan = checkpoint()
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-e2e-fault-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        self.out = self.root / "out"
        self.work = self.root / "work"
        self.shard = "model-00002-of-00005.safetensors"

    def run_converter(
        self,
        extra: tuple = (),
        patches: dict | None = None,
        work: pathlib.Path | None = None,
        curl_retries: bool = False,
    ):
        """One conversion with the injected fault reaching `download()`'s loop.

        curl has a retry loop of its own; with it on, a single injected fault is
        usually absorbed there and the layer this file is about never sees it.
        `curl_retries=True` keeps both layers and lets a test observe them.
        """
        layers = {"CURL_RETRY_DELAY_SECONDS": 0}
        if not curl_retries:
            # `--retry N` is N *retries*; 0 makes one curl invocation one
            # request, so each injected fault reaches `download()`'s own loop.
            layers["CURL_RETRY_ATTEMPTS"] = 0
        layers.update(patches or {})
        return run_converter(self.out, work or self.work, self.server.base, extra, layers)

    def assert_baseline(self, log: str) -> None:
        self.assertEqual(fingerprint(self.out), baseline_fingerprint(), log)

    def test_a_transfer_dropped_mid_body_is_retried(self) -> None:
        self.server.faults[self.shard] = ["drop", "ok"]
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertIn("failed (curl", log)
        self.assertEqual(len(self.server.requests_for(self.shard)), 2, log)
        self.assert_baseline(log)

    def test_a_truncated_body_is_retried(self) -> None:
        self.server.faults[self.shard] = ["short", "ok"]
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertIn("failed (curl", log)
        self.assertEqual(len(self.server.requests_for(self.shard)), 2, log)
        self.assert_baseline(log)

    def test_a_missing_shard_is_retried(self) -> None:
        self.server.faults[self.shard] = ["404", "ok"]
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertIn("failed (curl 22", log)
        self.assertEqual(len(self.server.requests_for(self.shard)), 2, log)
        self.assert_baseline(log)

    def test_a_stall_past_the_timeout_is_retried(self) -> None:
        self.server.stall_seconds = 2.0
        self.server.faults[self.shard] = ["stall", "ok"]
        code, log = self.run_converter(patches={"DOWNLOAD_TIMEOUT_SECONDS": 1})
        self.assertEqual(code, 0, log)
        self.assertIn("failed (curl", log)
        self.assertEqual(len(self.server.requests_for(self.shard)), 2, log)
        self.assert_baseline(log)

    def test_a_partial_file_is_resumed_with_a_range_request(self) -> None:
        # Half a shard on disk is what a dropped connection leaves. The next
        # attempt must ask for the remainder, not fetch the file again.
        whole = self.files[self.shard]
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / self.shard).write_bytes(whole[: len(whole) // 2])
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        ranges = [request["range"] for request in self.server.requests_for(self.shard)]
        self.assertIn(f"bytes={len(whole) // 2}-", ranges, log)
        self.assert_baseline(log)

    def test_a_mirror_that_refuses_ranges_restarts_the_file(self) -> None:
        # curl 33: 200 to a range request while a partial file exists. The
        # partial has to be dropped, or every later attempt fails the same way.
        whole = self.files[self.shard]
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / self.shard).write_bytes(whole[: len(whole) // 2])
        self.server.faults[self.shard] = ["refuse_range", "ok"]
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertIn("will not serve a range request", log)
        self.assert_baseline(log)

    def test_endless_failures_stop_after_the_bounded_attempts(self) -> None:
        self.server.faults[self.shard] = ["404"]
        code, log = self.run_converter(patches={"DOWNLOAD_ATTEMPTS": 3})
        self.assertEqual(code, 1, log)
        self.assertIn("failed to download", log)
        # The mirror was asked exactly three times: bounded, not a busy loop.
        self.assertEqual(len(self.server.requests_for(self.shard)), 3, log)

    def test_a_schedule_of_faults_still_produces_the_snapshot(self) -> None:
        schedule = ["404", "drop", "short", "refuse_range", "ok"]
        shards = [
            name
            for name in self.files
            if name.startswith("model-") and name.endswith(".safetensors")
        ]
        for name in shards:
            self.server.faults[name] = list(schedule)
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assert_baseline(log)
        failed = [r for r in self.server.requests if r["behaviour"] != "ok"]
        self.assertEqual(len(failed), 4 * len(shards), log)

    def test_curls_own_retries_can_absorb_a_fault_before_the_loop_sees_it(self) -> None:
        # Both layers on: curl retries inside one attempt, so the shard can end
        # up complete without the outer loop ever logging a failure. The next
        # attempt then finds a whole file and the snapshot is still the baseline.
        self.server.faults[self.shard] = ["drop"] * 4 + ["ok"]
        code, log = self.run_converter(curl_retries=True)
        self.assertEqual(code, 0, log)
        self.assertGreaterEqual(len(self.server.requests_for(self.shard)), 2, log)
        self.assert_baseline(log)


@requires_deps
class ResumeTests(unittest.TestCase):
    """A conversion that stopped, and what the next run does with it."""

    def setUp(self) -> None:
        self.files, self.plan = checkpoint()
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-e2e-resume-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        self.out = self.root / "out"
        self.work = self.root / "work"
        # A second scratch directory for the run that follows an interruption:
        # the interrupted run's fetcher thread can still be writing into the
        # first one, and two curls on one destination is a race, not a resume.
        self.resume_work = self.root / "work-resume"
        # That thread also outlives the capture `run_converter` installs, so its
        # last curl would print to the real stderr after the run it belongs to
        # has ended. Swallow curl's output for the whole test instead.
        real_run = subprocess.run

        def swallow(command, **kwargs):
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("capture_output", "stdout", "stderr")
            }
            return real_run(command, capture_output=True, **kwargs)

        stray_patch = mock.patch.object(prepare.subprocess, "run", swallow)
        stray_patch.start()
        self.addCleanup(stray_patch.stop)

    def run_converter(
        self, extra: tuple = (), patches: dict | None = None, work: pathlib.Path | None = None
    ):
        return run_converter(self.out, work or self.work, self.server.base, extra, patches)

    def interrupt_after_shards(self, count: int) -> None:
        """Stop the run the way a kill does, after `count` converted shards.

        The writer has flushed whole shards by then (the threshold is patched
        down), so what is on disk is exactly what a `SIGKILL` would leave:
        finished output shards, downloaded inputs still in `--work`, no index.
        """
        real_convert = prepare.convert_shard
        real_table = prepare.NgramTable
        state = {"calls": 0}
        tables: list = []

        class RecordingTable(real_table):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                tables.append(self)

        def wrapped(path, writer, ngram, width):
            if state["calls"] >= count:
                raise KeyboardInterrupt("interrupted by the test")
            state["calls"] += 1
            return real_convert(path, writer, ngram, width)

        with self.assertRaises(KeyboardInterrupt):
            self.run_converter(
                patches={
                    "convert_shard": wrapped,
                    "NgramTable": RecordingTable,
                    "OUTPUT_SHARD_BYTES": 512,
                }
            )
        # The in-process interrupt leaves the table's handle open, exactly as a
        # kill would (there is no cleanup path); close the test's copy so it
        # does not leak a descriptor or warn about it. The half-written file
        # stays on disk, which is the state under test.
        for table in tables:
            if table.handle is not None:
                table.handle.close()
                table.handle = None

    def assert_no_duplicate_tensors(self, log: str = "") -> None:
        """Every tensor must live in exactly one output shard.

        Re-converting a checkpoint shard whose output was only partly adopted
        is how a duplicate gets in: the adopted tensors must be dropped, not
        written a second time.
        """
        seen: dict[str, str] = {}
        for shard in sorted(self.out.glob("model-*-of-*.safetensors")):
            with safe_open(shard, framework="np") as handle:
                for name in handle.keys():
                    self.assertNotIn(
                        name, seen, f"{name} is in both {seen.get(name)} and {shard.name}"
                    )
                    seen[name] = shard.name
        self.assertEqual(set(seen), expected_output_names(), log)

    def test_a_killed_conversion_resumes_to_the_clean_snapshot(self) -> None:
        self.interrupt_after_shards(2)
        self.assertFalse((self.out / "model.safetensors.index.json").exists())
        adopted = sorted(self.out.glob("model-[0-9][0-9][0-9][0-9][0-9].safetensors"))
        self.assertTrue(adopted, "the interrupted run left no output shard to adopt")
        code, log = self.run_converter(patches={"OUTPUT_SHARD_BYTES": 512}, work=self.resume_work)
        self.assertEqual(code, 0, log)
        self.assertIn("resuming from", log)
        self.assertIn("already converted", log)
        self.assertFalse(list(self.out.glob("*.partial")), log)
        self.assert_no_duplicate_tensors(log)
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_a_resume_does_not_fetch_the_shards_it_already_converted(self) -> None:
        # The first two checkpoint shards, in download order, are exactly the
        # ones whose output the writer had finished when it was interrupted, so
        # the resume must neither fetch them nor convert them again.
        self.interrupt_after_shards(2)
        mark = len(self.server.requests)
        code, log = self.run_converter(patches={"OUTPUT_SHARD_BYTES": 512}, work=self.resume_work)
        self.assertEqual(code, 0, log)
        second_round = {request["file"] for request in self.server.requests[mark:]}
        self.assertNotIn("model-00001-of-00005.safetensors", second_round, log)
        self.assertNotIn("model-00002-of-00005.safetensors", second_round, log)
        self.assertIn("2 checkpoint shards already converted", log)
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_a_truncated_adopted_shard_is_discarded_and_rebuilt(self) -> None:
        self.interrupt_after_shards(2)
        victim = sorted(self.out.glob("model-[0-9][0-9][0-9][0-9][0-9].safetensors"))[0]
        payload = victim.read_bytes()
        victim.write_bytes(payload[: len(payload) - 8])  # header parses, payload short
        code, log = self.run_converter(patches={"OUTPUT_SHARD_BYTES": 512}, work=self.resume_work)
        self.assertEqual(code, 0, log)
        self.assertIn("discarding an incomplete output shard", log)
        self.assert_no_duplicate_tensors(log)
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_the_discarded_shard_number_is_reused_not_left_as_a_hole(self) -> None:
        # `finish` renames 1..N, so a discarded shard must not leave a gap: the
        # survivors are compacted before the new ones are numbered on top.
        self.interrupt_after_shards(2)
        victim = sorted(self.out.glob("model-[0-9][0-9][0-9][0-9][0-9].safetensors"))[0]
        payload = victim.read_bytes()
        victim.write_bytes(payload[: len(payload) - 8])
        code, log = self.run_converter(patches={"OUTPUT_SHARD_BYTES": 512}, work=self.resume_work)
        self.assertEqual(code, 0, log)
        numbers = sorted(
            int(p.name.split("-")[1].split(".")[0])
            for p in self.out.glob("model-*-of-*.safetensors")
        )
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)), log)
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_a_shard_from_before_the_width_marker_is_still_adopted(self) -> None:
        # The marker is new; a partial conversion from a run that predates it
        # has none, and refusing those would strand exactly the interrupted
        # builds the resume path exists for.
        self.interrupt_after_shards(1)
        (self.out / "conversion.json").unlink()
        code, log = self.run_converter(patches={"OUTPUT_SHARD_BYTES": 512}, work=self.resume_work)
        self.assertEqual(code, 0, log)
        self.assertIn("resuming from", log)
        self.assert_no_duplicate_tensors(log)
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_a_resume_at_another_width_is_refused(self) -> None:
        # A safetensors file does not say which width it was written at, so a
        # second run at another --bits must refuse rather than mix two widths.
        self.interrupt_after_shards(1)
        before = sorted(path.name for path in self.out.iterdir())
        code, log = self.run_converter(extra=("--bits", "8"), work=self.resume_work)
        self.assertEqual(code, 1, log)
        self.assertIn("two widths in one snapshot", log)
        self.assertEqual(
            sorted(path.name for path in self.out.iterdir()),
            before,
            f"the refusal still wrote into the snapshot: {log}",
        )

    def test_a_leftover_partial_file_is_removed(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        leftover = self.out / "model-00007.safetensors.partial"
        leftover.write_bytes(b"half a shard")
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertFalse(leftover.exists())
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_a_finished_directory_is_refused(self) -> None:
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        code, log = self.run_converter()
        self.assertNotEqual(code, 0)
        self.assertIn("already holds a finished snapshot", log)

    def test_renamed_shards_without_an_index_are_refused(self) -> None:
        # The other side of `finish`: shards renamed, index not yet written. A
        # second run would orphan the finished generation beside a new one.
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "model-00001-of-00003.safetensors").write_bytes(b"finished")
        code, log = self.run_converter()
        self.assertNotEqual(code, 0)
        self.assertIn("finished output shards but no", log)
        self.assertEqual((self.out / "model-00001-of-00003.safetensors").read_bytes(), b"finished")

    def test_a_completed_local_table_is_reused_in_place(self) -> None:
        # A previous run (or the older tool) left a whole table here. Reusing it
        # must not disturb it -- the source and the destination are one file.
        self.out.mkdir(parents=True, exist_ok=True)
        table = self.out / "ngram_table.bin"
        table.write_bytes(expected_table_bytes())
        (self.out / "ple_constants.json").write_text(
            json.dumps(prepare.ple_constants(text_config()))
        )
        inode = table.stat().st_ino
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertIn("reusing the completed ngram_table.bin", log)
        self.assertTrue(table.exists())
        self.assertEqual(table.stat().st_ino, inode)
        self.assertEqual(table.read_bytes(), expected_table_bytes())
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_reusing_a_table_skips_the_ngram_only_shards(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "ngram_table.bin").write_bytes(expected_table_bytes())
        (self.out / "ple_constants.json").write_text(
            json.dumps(prepare.ple_constants(text_config()))
        )
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        fetched = {request["file"] for request in self.server.requests}
        self.assertNotIn(
            "model-00005-of-00005.safetensors",
            fetched,
            "the n-gram-only shard was fetched despite the reused table",
        )
        # The mixed shard is still fetched, and its n-gram rows are ignored.
        self.assertIn("model-00004-of-00005.safetensors", fetched)
        self.assertIn("already in the linked table", log)
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_a_wrong_size_local_table_is_rebuilt(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "ngram_table.bin").write_bytes(b"short")
        (self.out / "ple_constants.json").write_text(
            json.dumps(prepare.ple_constants(text_config()))
        )
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertIn("ignoring the ngram_table.bin already here", log)
        self.assertEqual((self.out / "ngram_table.bin").read_bytes(), expected_table_bytes())
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_a_table_built_under_other_constants_is_rebuilt(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "ngram_table.bin").write_bytes(expected_table_bytes())
        changed = prepare.ple_constants(text_config())
        changed["layer_multipliers"] = [1, 1, 1]
        (self.out / "ple_constants.json").write_text(json.dumps(changed))
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertIn("ignoring the ngram_table.bin already here", log)
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())

    def test_a_table_never_published_is_rebuilt(self) -> None:
        # A kill during the table build leaves only `*.partial`; a later run
        # must delete that and build again, never read it.
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "ngram_table.bin.partial").write_bytes(b"part of a table")
        code, log = self.run_converter()
        self.assertEqual(code, 0, log)
        self.assertFalse((self.out / "ngram_table.bin.partial").exists())
        self.assertEqual((self.out / "ngram_table.bin").read_bytes(), expected_table_bytes())
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())


@requires_deps
class KilledProcessTests(unittest.TestCase):
    """A real `SIGKILL` in the middle of the conversion, then a resume."""

    def setUp(self) -> None:
        self.files, _plan = checkpoint()
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-e2e-kill-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        # A slow shard makes "kill it while it is still working" reliable.
        self.server.delay = 0.25
        self.out = self.root / "out"
        self.work = self.root / "work"

    def test_a_sigkill_mid_conversion_resumes_to_the_clean_snapshot(self) -> None:
        driver = self.root / "driver.py"
        driver.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
            "import prepare_qwen38 as prepare\n"
            "prepare.OUTPUT_SHARD_BYTES = 512\n"
            "raise SystemExit(prepare.main())\n"
        )
        log_path = self.root / "killed.log"
        with log_path.open("w") as log_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(driver),
                    "--output",
                    str(self.out),
                    "--work",
                    str(self.work),
                    "--endpoint",
                    self.server.base,
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            deadline = time.time() + 60
            while time.time() < deadline:
                if (self.out / "model-00001.safetensors").exists():
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.005)
            still_running = process.poll() is None
            if still_running:
                process.send_signal(signal.SIGKILL)
            process.wait(timeout=15)
        killed_log = log_path.read_text()
        self.assertTrue(
            still_running, f"the conversion finished before it could be killed:\n{killed_log}"
        )
        self.assertEqual(process.returncode, -signal.SIGKILL)
        self.assertFalse(
            (self.out / "model.safetensors.index.json").exists(),
            "a killed run left a finished index",
        )
        adopted = sorted(self.out.glob("model-[0-9][0-9][0-9][0-9][0-9].safetensors"))
        self.assertTrue(adopted, killed_log)

        code, log = run_converter(
            self.out, self.work, self.server.base, patches={"OUTPUT_SHARD_BYTES": 512}
        )
        self.assertEqual(code, 0, log)
        self.assertIn("resuming from", log)
        self.assertFalse(list(self.out.glob("*.partial")), log)
        self.assertEqual(fingerprint(self.out), baseline_fingerprint())


@requires_deps
class CrossFilesystemTests(unittest.TestCase):
    """`--reuse-ngram-table` across filesystems: copy, do not fail."""

    def setUp(self) -> None:
        for tool in ("hdiutil",):
            if shutil.which(tool) is None:
                self.skipTest(f"{tool} is not available")
        self.files, _plan = checkpoint()
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-e2e-cross-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)

        self.mount = self.root / "volume"
        self.mount.mkdir()
        self.image = self.root / "volume.dmg"
        try:
            subprocess.run(
                [
                    "hdiutil",
                    "create",
                    "-size",
                    "64m",
                    "-fs",
                    "APFS",
                    "-volname",
                    "TinyTitanTest",
                    str(self.image),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "hdiutil",
                    "attach",
                    str(self.image),
                    "-mountpoint",
                    str(self.mount),
                    "-nobrowse",
                    "-quiet",
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            self.skipTest(f"cannot attach a disk image here: {exc}")
        self.addCleanup(
            subprocess.run, ["hdiutil", "detach", str(self.mount), "-quiet"], capture_output=True
        )

    def test_a_table_on_another_filesystem_is_copied(self) -> None:
        source = self.root / "source-table.bin"
        source.write_bytes(expected_table_bytes())
        self.assertNotEqual(
            source.stat().st_dev,
            self.mount.stat().st_dev,
            "the mounted image is not a different filesystem",
        )
        out = self.mount / "out"
        code, log = run_converter(
            out,
            self.root / "work",
            self.server.base,
            extra=("--reuse-ngram-table", str(source)),
            patch_sleep=False,
        )
        self.assertEqual(code, 0, log)
        self.assertIn("another filesystem", log)
        table = out / "ngram_table.bin"
        self.assertEqual(table.read_bytes(), expected_table_bytes())
        self.assertEqual(table.stat().st_dev, self.mount.stat().st_dev)
        self.assertEqual(source.stat().st_nlink, 1, "the source was hardlinked")
        self.assertEqual(fingerprint(out), baseline_fingerprint())
