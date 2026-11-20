#!/usr/bin/env python3
"""TT-098: the 2B converter refuses a size-correct shard with no content.

The 2026-11-16 2B installs shipped zeroed payloads. The download path trusted
curl's exit status and the file size, and both are satisfied by a hole or a
zeroed write. `content_probe` samples the shard before it is renamed into
place; these tests pin the refusal and the accepted shape, and drive the
download with a stubbed curl so nothing reaches the network.
"""

from __future__ import annotations

import pathlib
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import prepare_qwen35 as q  # noqa: E402


def safetensors_bytes(payload: bytes) -> bytes:
    header = b'{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,1]}}'
    return struct.pack("<Q", len(header)) + header + payload


class ContentProbeTests(unittest.TestCase):
    def test_a_zeroed_shard_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.safetensors"
            path.write_bytes(b"\x00" * (1024 * 1024))
            with self.assertRaises(RuntimeError) as raised:
                q.content_probe(path, "model.safetensors", path.stat().st_size)
            self.assertIn("no safetensors header", str(raised.exception))

    def test_a_header_with_a_zeroed_payload_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.safetensors"
            header = b'{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,1]}}'
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * (1024 * 1024))
            with self.assertRaises(RuntimeError) as raised:
                q.content_probe(path, "model.safetensors", path.stat().st_size)
            self.assertIn("nonzero", str(raised.exception))

    def test_a_real_shaped_shard_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.safetensors"
            path.write_bytes(safetensors_bytes(b"\x01" * (1024 * 1024)))
            q.content_probe(path, "model.safetensors", path.stat().st_size)


class _ZeroedCurl:
    """A curl that exits 0 after writing `size` zero bytes."""

    size = 0

    def __init__(self, argv, **_kwargs):
        self.argv = argv
        self.returncode = 0

    def wait(self):
        target = pathlib.Path(self.argv[self.argv.index("-o") + 1])
        target.write_bytes(b"\x00" * self.size)
        return 0

    def poll(self):
        return 0


class DownloadRefusalTests(unittest.TestCase):
    def test_a_size_correct_zeroed_download_is_refused_and_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            fake = type("FakeCurl", (_ZeroedCurl,), {"size": 4096})
            with mock.patch.object(q.subprocess, "Popen", fake):
                with self.assertRaises(RuntimeError) as raised:
                    q.download("model.safetensors", work, 4096)
            self.assertIn("content check", str(raised.exception))
            self.assertFalse((work / "model.safetensors").exists())
            self.assertFalse((work / "model.safetensors.part").exists())

    def test_an_existing_zeroed_shard_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            (work / "model.safetensors").write_bytes(b"\x00" * 4096)
            with self.assertRaises(RuntimeError):
                q.download("model.safetensors", work, 4096)


if __name__ == "__main__":
    unittest.main()
