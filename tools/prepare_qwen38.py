#!/usr/bin/env python3
"""Convert Qwen/Qwen3.8-Flash-Next into an affine snapshot TinyTitanRepack imports.

Why this exists: the installed 4-bit model came from a third-party MLX repack.
TinyTitan does not need one. It needs an affine group-64 layout, and it can produce
that from Qwen's own bf16 release, which is both better provenance and better
numerics -- quantizing once from the original weights beats inheriting somebody
else's quantization error.

The checkpoint is 360 GB across 131 shards. Nothing here holds the whole thing:
one shard is fetched, converted and deleted before the next is needed, with a
background thread fetching shard N+1 while shard N converts. Peak disk is the
output plus one shard.

Three things the official checkpoint does differently from the MLX repack the
runtime was built against. Each is a silent mis-load if assumed rather than
checked, so `--plan` asserts all three against the checkpoint's own headers:

1. Routed experts are *stacked and fused*: one `mlp.experts.gate_up_proj` of
   shape [512, 1280, 2560] per layer, where 1280 is gate and up concatenated,
   plus one `mlp.experts.down_proj`. TinyTitanRepack wants three separately named
   tensors matching `.mlp.switch_mlp.{gate,up,down}_proj.`, each of which may
   stay stacked over the 512 experts because the planner slices per expert.
2. Norms carry `.weight` here and do not in the repack the runtime reads, so
   `hc_norm.weight` becomes `hc_norm`. `linear_attn.norm.weight` keeps its
   suffix -- the schema asks for that one *with* `.weight`.
3. `lm_head` sits at the archive root, not under `model.language_model.`.

`ple_constants.json` is derived here rather than copied, using the algorithm in
the transformers reference (`_build_layer_multipliers`, `_find_nth_prime_after`).
The derivation reproduces the previously shipped constants exactly -- multipliers,
the sixteen prime head vocabularies and their offsets all match -- which is what
lets this run without any dependency on the third-party repack.

Usage:
    tools/prepare_qwen38.py --plan                    # validate, download nothing
    tools/prepare_qwen38.py --output DIR --work DIR   # convert
"""

from __future__ import annotations

import argparse
import concurrent.futures
import errno
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

# The progress line lives beside the shell helpers, not in this file, because
# prepare_agentworld.py draws the identical line: a whole install has to read as
# one story, and two renderers would drift.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from progress import Progress  # noqa: E402

try:
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError as exc:  # pragma: no cover - environment, not logic
    sys.exit(
        f"missing dependency: {exc}\n"
        f"  install them for the interpreter running this file: {sys.executable}\n"
        "    -m pip install safetensors numpy ml_dtypes\n"
        "  (or run this file with an interpreter that has them; TINYTITAN_PYTHON\n"
        "   selects the interpreter that tools/install_models.sh runs these\n"
        "   files with, and this file does not read it)"
    )
REPO = "Qwen/Qwen3.8-Flash-Next"
# `HF_ENDPOINT` (the Hub's own variable) and `--endpoint` point every fetch at a
# mirror. The path below the host is the Hub's -- `/<repo>/resolve/main/<file>`
# for weights and `/<repo>/raw/main/<file>` for the small JSON ones -- so a
# mirror that serves that layout works unchanged, and no provider-specific code
# is needed for one.
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
BASE = f"{HF_ENDPOINT}/{REPO}/resolve/main"


def raw_url(remote: str) -> str:
    """The mirror's URL for one of the repo's small text files."""
    return f"{HF_ENDPOINT}/{REPO}/raw/main/{remote}"


def endpoint_base(endpoint: str) -> str:
    """The weights base URL for a Hub endpoint or mirror."""
    return f"{endpoint.rstrip('/')}/{REPO}/resolve/main"


GROUP_SIZE = 64
BITS_4, BITS_8 = 4, 8
# indexer_n_heads * indexer_head_dim from the config; the remainder of
# index_qk_proj's rows are the key projection.
INDEXER_QUERY_ROWS = 512
OUTPUT_SHARD_BYTES = 4 << 30

# --- PLE constants, per the transformers reference -------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def ple_constants(text_config: dict) -> dict:
    """Reproduce what the reference computes at init, as shippable data.

    `ple_layer_index` is the position *within* `ple_layer_ids`, not the layer
    number: the reference reads it as `ple_layer_ids.index(layer_idx + 1)`, so
    for the single PLE layer it is 0. Getting that wrong changes every hash
    multiplier and every n-gram id, silently.
    """
    ple_layer_index = 0
    heads = text_config["heads_per_ngram"] * 2  # two n-gram orders
    vocab_base = text_config["ngram_vocab_size_base"]
    sizes, offsets, total = [], [], 0
    for head in range(heads):
        size = _find_nth_prime_after(vocab_base - 1, ple_layer_index * heads + head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(text_config["vocab_size"], 1)) // 2)
    base_seed = text_config.get("seed", 1234) + _PRIME_1 * ple_layer_index
    multipliers = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(text_config["ngram_size"])
    ]
    eos = text_config["eos_token_id"]
    return {
        "layer_multipliers": multipliers,
        "ngram_heads_offsets": offsets,
        "ngram_heads_vocab_sizes": sizes,
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "ngram_size": text_config["ngram_size"],
        "heads_per_ngram": text_config["heads_per_ngram"],
        "ple_n_heads": heads,
        "ple_head_dim": text_config["ple_embed_dim"] // heads,
        "table_file": "ngram_table.bin",
        "table_dtype": "float16",
    }


# --- quantisation ----------------------------------------------------------


def quantize_affine(value: np.ndarray, bits: int) -> tuple[np.ndarray, ...]:
    """Identical to prepare_ornith_mtp.quantize_affine, deliberately.

    Duplicated rather than imported so neither converter can drift silently;
    a change to one must be made in the other.
    """
    value = value.astype(np.float32)
    if value.shape[-1] % GROUP_SIZE:
        raise ValueError(f"last dimension {value.shape[-1]} is not group-aligned")
    shape = (*value.shape[:-1], value.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    grouped = value.reshape(shape)
    bias = grouped.min(axis=-1)
    high = grouped.max(axis=-1)
    levels = (1 << bits) - 1
    scale = np.where(high == bias, np.float32(1), (high - bias) / levels)
    scale = scale.astype(ml_dtypes.bfloat16)
    bias = bias.astype(ml_dtypes.bfloat16)
    quantized = (
        np.rint(
            (grouped - bias.astype(np.float32)[..., None]) / scale.astype(np.float32)[..., None]
        )
        .clip(0, levels)
        .astype(np.uint32)
        .reshape(value.shape)
    )
    lanes = 32 // bits
    words = quantized.reshape(*quantized.shape[:-1], quantized.shape[-1] // lanes, lanes)
    packed = np.zeros(words.shape[:-1], dtype=np.uint32)
    for lane in range(lanes):
        packed |= words[..., lane] << np.uint32(bits * lane)
    return packed, scale, bias


# --- naming ----------------------------------------------------------------


def is_multimodal(name: str) -> bool:
    return ".visual." in name or name.startswith("model.visual.")


def is_ngram(name: str) -> bool:
    return ".ngram_embedding.shard_" in name


# The checkpoint carries the PLE hash constants as int64 buffers beside the
# table. They are not weights, the repacker has no dtype for them, and this
# converter derives the same values into ple_constants.json from the algorithm
# in the transformers reference -- verified against these very buffers, which
# match exactly. Carrying them would only give the repacker something it must
# reject.
PLE_BUFFERS = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")


def is_ple_buffer(name: str) -> bool:
    return (
        name.startswith("model.language_model.layers.")
        and ".ple.ple_embedding." in name
        and name.rsplit(".", 1)[-1] in PLE_BUFFERS
    )


# Names the runtime's schema spells without `.weight`, where the checkpoint
# has it. Not a family property -- it is what the repack the runtime was built
# against happened to write -- so each one is listed rather than guessed at
# from a pattern. `linear_attn.conv1d.weight` deliberately keeps its suffix:
# the schema asks for that one with it, and stripping it by pattern would
# break the GDN path.
STRIP_WEIGHT_SUFFIXES = (
    ".hc_norm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    ".self_attn.indexer.q_layernorm",
    ".self_attn.indexer.k_layernorm",
    ".ple.conv1d",
    ".ple.norm_conv",
    ".ple.norm_key",
    ".ple.norm_query",
    ".pre_fc_norm_embedding",
    ".pre_fc_norm_hidden",
)


# RMSNorm weights the model reads as `1 + weight`.
#
# `Qwen3_5RMSNorm` initialises its parameter to *zeros* and computes
# `normalized * (1.0 + weight)`, so the checkpoint stores the offset from one,
# not the gain. The runtime multiplies by the stored value, so the +1 has to be
# folded in here -- which is exactly what MLX's conversion does, and why a
# build made from the MLX repack answered while one made from the original did
# not.
#
# Two absences are deliberate. `linear_attn.norm` is `Qwen3NextRMSNormGated`,
# which uses `weight` directly with no offset. `ple.conv1d` is a convolution
# kernel, not a norm. Folding either would be as wrong as not folding these.
#
# Getting this wrong is silent. Every tensor still matches the checkpoint
# byte for byte, every parity harness still agrees -- both sides read the same
# stored value -- and the model generates fluent nonsense. It cost a full
# investigation; see docs/qwen38-flash-next-port.md.
UNIT_OFFSET_NORM_SUFFIXES = (
    ".hc_norm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    ".self_attn.indexer.q_layernorm",
    ".self_attn.indexer.k_layernorm",
    ".ple.norm_conv",
    ".ple.norm_key",
    ".ple.norm_query",
    ".pre_fc_norm_embedding",
    ".pre_fc_norm_hidden",
)


def fold_unit_offset(out_name: str, value: np.ndarray) -> np.ndarray:
    """Fold the implicit +1 into a zero-centred RMSNorm weight."""
    stem = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
    if stem.endswith(UNIT_OFFSET_NORM_SUFFIXES):
        return (value.astype(np.float32) + 1.0).astype(value.dtype)
    return value


def rename(name: str) -> str:
    for suffix in STRIP_WEIGHT_SUFFIXES:
        if name.endswith(suffix + ".weight"):
            return name[: -len(".weight")]
    return name


# Families kept at the checkpoint's own bf16 in an 8-bit build.
#
# The 8-bit build exists for work where correctness outranks speed -- an
# overnight coding run on a machine doing nothing else -- so it is worth
# spending resident memory that a 4-bit build could not.
#
# Chosen by measurement, not by intuition. Every family measures 0.57-1.0%
# relative error at 8 bits, so the selection is by what the error *does*, not
# how large it is: the router and the indexer produce rankings, and a ranking
# either matches or it does not (the router's top-10 agrees with bf16 on 98.8%
# of tokens at 8 bits). The rest are the smallest tensors in the model, where
# group-64 quantization has the fewest values to amortise over and the error is
# correspondingly highest -- a 1-row gate has nothing to average.
#
# The cost is bounded and lands in the right place: ~108 MB of resident memory,
# 2.0% of the active parameters per token, and about 0.3% of decode. Resident
# weights come from RAM; the routed experts, which are 23x the parameters and
# would double SSD traffic, are deliberately not here.
PROMOTE_TO_BF16_AT_8BIT = (
    ".mlp.gate",  # router: picks which experts run
    ".mlp.shared_expert_gate",  # 1 row of D; highest measured error
    ".block_inject_weight",  # 4 rows; the write gate for every layer
    ".linear_attn.in_proj_a",
    ".linear_attn.in_proj_b",
    ".ple.key_proj",
    ".self_attn.indexer.index_q_proj",
    ".self_attn.indexer.index_k_proj",
)


# `--rest-bf16`: every tensor except the routed experts stays at the
# checkpoint's bf16. An evaluation build (requested 2026-11-12), not an
# installer preset: on the 24 GB M3 it holds ~9.2 GiB resident instead of
# ~3.2, so it cannot start under `--ram-budget 10G` and was projected at
# about 2.5 tok/s against 4.5 at the default budget. It exists to measure
# what 16-bit dense weights buy in quality against that cost.
REST_BF16 = False

# `--rest-8bit`: routed experts at the build width, every other quantised
# tensor at 8 bits -- the attention, GDN, hyper-connection, PLE, indexer and
# shared-expert families move from 4 to 8, and the embedding, LM head and
# router stay at the 8 they already have. Requested 2026-11-13 as the middle
# ground between the 4-bit install and `--rest-bf16`: about +2.6 GB resident
# against +6 GB for bf16. Because every attention-slot tensor is 8-bit, the
# repacker derives an 8-bit attention slot, so the GDN a/b pair sits at the
# slot's own width, which `Model.validateRoleUniformity` accepts. Norms and
# conv1d stay bf16 as in every build.
REST_8BIT = False


def is_routed_expert(name: str) -> bool:
    return ".mlp.switch_mlp." in name


def promoted_to_bf16(name: str, width: int) -> bool:
    """Is this tensor kept unquantized in a build of this width?"""
    if width != BITS_8:
        return False
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return stem.endswith(PROMOTE_TO_BF16_AT_8BIT)


def quant_bits(name: str, width: int = BITS_4) -> int | None:
    """Bits for a tensor, or None to copy it through unquantised.

    This mirrors the runtime's slot model *plus its per-tensor overrides*. `Model`
    derives a tensor's expected size from `manifest.quant.<slot>.weightBits` and
    builds one GEMV per *role*, so a role shares one width — but a tensor whose
    width differs from its slot carries an override keyed by stem, which the
    repacker writes (`SSDAIJSON.quantObject`), the format validates
    (`SSDAIManifestV1`), and the runtime resolves per tensor
    (`ManifestQuant.slot(forTensorNamed:overrides:fallback:)`). The dense
    Qwen 3.5 installs are built exactly that way: `mlp.*` at 4 bits against an
    8-bit slot, full-attention `k_proj`/`v_proj` at 8 against a 4-bit one.

    One pair is the exception, and it is the pair this converter promotes: the
    GDN `a`/`b` projections. The fused QKV+Z+A+B kernel takes a *bf16-or-slot*
    flag rather than a width and its quantized branch is int4, so the runtime
    refuses a quantized override for them by name
    (`Model.validateRoleUniformity`). A 16-bit override is honoured, which is
    the only width asked for here.

    The slots, matching what a working install carries:

        embedding     8   embed_tokens, lm_head
        router        8   mlp.gate
        attention   `width`   q/k/v/o, GDN, hyper-connection, indexer, PLE
        sharedExpert `width`  shared_expert.*, shared_expert_gate
        routedExpert `width`  switch_mlp.*

    This is why the router already sits at 8 bits in a 4-bit build: it has its
    own slot. The measurement in tools/precision_probe.py says the QSA indexer,
    the hyper-connection write gate and the PLE key projection deserve more
    precision -- the indexer picks the same keys 0.0% of the time at 4 bits
    against 49.5% at 8, for about 10 MB. Taking the whole attention slot to 8
    bits instead is measured and *works*, but it is 61% of the active parameters
    and +2.10 GB resident (Engineering Notes, "8-bit attention slot"); these
    tensors are ~10 MB. A per-tensor override is how they get the precision
    cheaply, and the runtime resolves one for all three families --
    `Model.hyperConnectionWeightBits`, `pleKeyWeightBits`,
    `qsaIndexerWeightBits`, uniform across the family because one kernel serves
    every layer. What this converter does not yet do is ask for it, and the
    8-bit indexer was measured and rejected on its own (2.4 points on marginal
    keys), so the promotion is a decision per tensor rather than a switch.
    """
    if not name.endswith(".weight"):
        return None
    if REST_BF16 and not is_routed_expert(name):
        return None
    if REST_8BIT and not is_routed_expert(name):
        if name.endswith("conv1d.weight") or name.endswith("norm.weight"):
            return None
        return BITS_8
    if name.endswith("conv1d.weight"):
        return None
    if name.endswith("norm.weight"):
        return None
    # Checked before the slot rules below: promotion overrides the slot, which
    # is the whole point of it. The runtime reads these tensors' width from
    # their own dtype rather than from the slot they nominally belong to.
    if promoted_to_bf16(name, width):
        return None
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        return BITS_8  # embedding slot
    # Both of these validate against `quant.router` in Model.swift. The scalar
    # gate does so explicitly and against expectation -- "quantized at the
    # ROUTER's bit width ... independent of the sharedExpert slot" -- which is
    # why grouping it with the shared expert produced a tensor half the size
    # the loader wanted.
    if name.endswith(".mlp.gate.weight") or name.endswith(".shared_expert_gate.weight"):
        return BITS_8  # router slot
    return width  # attention / sharedExpert / routedExpert


def output_names_for(name: str) -> list[str]:
    """The tensor names `convert_shard` writes for one checkpoint tensor.

    Shapes are `outputs_for`'s business; this is the name half, shared with the
    resume check so a shard is judged converted by the same rule that wrote it.
    """
    new = rename(name)
    if new.endswith(".mlp.experts.gate_up_proj"):
        stem = new[: -len("experts.gate_up_proj")] + "switch_mlp."
        return [stem + "gate_proj.weight", stem + "up_proj.weight"]
    if new.endswith(".mlp.experts.down_proj"):
        stem = new[: -len("experts.down_proj")] + "switch_mlp."
        return [stem + "down_proj.weight"]
    if new.endswith(".self_attn.indexer.index_qk_proj.weight"):
        # Fused query and key in the checkpoint, separate in the runtime.
        # [640, 2560] is 4 heads x 128 of query followed by 1 kv head x 128 of
        # key, which the config's indexer_n_heads/indexer_kv_heads confirm.
        stem = new[: -len("index_qk_proj.weight")]
        return [stem + "index_q_proj.weight", stem + "index_k_proj.weight"]
    return [new]


def outputs_for(name: str, shape: list[int]) -> list[tuple[str, list[int]]]:
    names = output_names_for(name)
    new = rename(name)
    if new.endswith(".mlp.experts.gate_up_proj"):
        experts, fused, hidden = shape
        return [
            (names[0], [experts, fused // 2, hidden]),
            (names[1], [experts, fused // 2, hidden]),
        ]
    if new.endswith(".mlp.experts.down_proj"):
        return [(names[0], list(shape))]
    if new.endswith(".self_attn.indexer.index_qk_proj.weight"):
        rows, hidden = shape
        q = INDEXER_QUERY_ROWS
        return [(names[0], [q, hidden]), (names[1], [rows - q, hidden])]
    return [(names[0], list(shape))]


def checkpoint_shard_is_converted(names: list[str], index: dict[str, str], width: int) -> bool:
    """Whether every tensor this checkpoint shard would produce is already in
    `index` (the tensors a previous run's output shards hold).

    Mirrors `convert_shard`: skipped families produce nothing, an unquantised
    tensor is stored under its own name, and a quantised one carries the
    `.weight`/`.scales`/`.biases` triple under its stem.
    """
    wrote_anything = False
    for name in names:
        if is_multimodal(name) or is_ngram(name) or is_ple_buffer(name):
            continue
        for out_name in output_names_for(name):
            wrote_anything = True
            if quant_bits(out_name, width) is None:
                if out_name not in index:
                    return False
                continue
            stem = out_name[: -len(".weight")]
            for suffix in (".weight", ".scales", ".biases"):
                if stem + suffix not in index:
                    return False
    return wrote_anything


def write_config(config: dict, out: Path, tensor_names, width: int) -> dict:
    """Emit config.json with the `quantization` block TinyTitanRepack reads.

    The repacker resolves each tensor's width from `config.json -> quantization`:
    a base `bits`/`group_size`/`mode`, plus per-tensor overrides keyed by the
    tensor name with `.weight` stripped. Qwen's own config has no such block --
    it describes an unquantised model -- so copying it verbatim leaves the
    repacker with nothing to read and it refuses the snapshot.

    Every tensor whose width differs from the base is listed explicitly rather
    than relying on the repacker to re-derive the policy, so the snapshot
    records what it actually contains.
    """
    overrides = {}
    for name in tensor_names:
        if not name.endswith(".weight"):
            continue
        bits = quant_bits(name, width)
        if bits is None or bits == width:
            continue
        overrides[name[: -len(".weight")]] = {"bits": bits, "group_size": GROUP_SIZE}
    config = dict(config)
    config["quantization"] = {
        "bits": width,
        "group_size": GROUP_SIZE,
        "mode": "affine",
        **overrides,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    return config


# --- transport -------------------------------------------------------------


def fetch_header(shard: str) -> dict:
    """One shard's safetensors header, by two ranged reads.

    A server that ignores `Range` answers both reads with the whole file, which
    turns the second one into a JSON parse over binary payload. That is worth a
    clear error rather than a confusing one: a mirror has to serve ranges for
    this tool at all, and the first read's length is how that is detected.
    """
    url = f"{BASE}/{shard}"
    raw = subprocess.run(
        ["curl", "-sfL", "--max-time", "60", "-r", "0-7", url], capture_output=True, check=True
    ).stdout
    if len(raw) != 8:
        raise RuntimeError(
            f"{url}: asked for 8 bytes and got {len(raw)}; the endpoint does not "
            "honour range requests, and this converter reads shard headers and "
            "resumes partial downloads with them"
        )
    size = struct.unpack("<Q", raw[:8])[0]
    body = subprocess.run(
        ["curl", "-sfL", "--max-time", "180", "-r", f"8-{8 + size - 1}", url],
        capture_output=True,
        check=True,
    ).stdout
    if len(body) != size:
        raise RuntimeError(
            f"{url}: asked for the {size}-byte header and got {len(body)}; the "
            "endpoint does not honour range requests"
        )
    return json.loads(body)


# One shard is up to ~2 GB; 20 minutes with no byte moving is a dead connection,
# not a slow one, and without a cap the retry loop below can never fire.
DOWNLOAD_TIMEOUT_SECONDS = 1200
# Six attempts, doubling from five seconds: ~5 s, 10, 20, 40, 80, 120. The Hub
# already retries inside curl; this is for the failures that outlast that (a
# dropped connection mid-shard, a mirror that answers a range request with 200).
DOWNLOAD_ATTEMPTS = 6
# curl's own retry loop, inside one `download()` attempt. Constants rather than
# literals so the test suite can put them at 0 and let a fault reach the loop
# above, which is the layer that decides how a shard is resumed.
CURL_RETRY_ATTEMPTS = 3
CURL_RETRY_DELAY_SECONDS = 3


def remote_size(shard: str) -> int | None:
    """The shard's size from the server's own `Content-Length`, or `None`.

    A percentage needs a total, and the safetensors index does not carry file
    sizes -- only tensor names. One HEAD request per shard is the cheap way to
    get them (131 of them, probed in parallel before the fetch); a mirror that
    does not answer with a length simply leaves the line without a percentage
    rather than guessing one.
    """
    url = f"{BASE}/{shard}"
    try:
        result = subprocess.run(
            ["curl", "-sIL", "--http1.1", "--max-time", "60", url],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    for line in reversed(result.stdout.splitlines()):
        if not line.lower().startswith("content-length:"):
            continue
        _, _, value = line.partition(":")
        value = value.strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    return None


def watch_size(dest: Path, progress: Progress, base: float, stop: threading.Event) -> None:
    """Report a shard's growing file size while curl writes it.

    curl writes the destination itself, so the file's size is the only live
    signal a download has. A watcher thread polls it while the (blocking) curl
    runs, which keeps `subprocess.run` as the seam the tests inject a failure
    through. Never raises: a line is not worth failing an install over.
    """
    while not stop.wait(0.5):
        try:
            progress.set(base + dest.stat().st_size)
        except OSError:  # curl has not created the file yet, or it is gone
            continue


def download(shard: str, work: Path, progress: Progress | None = None, base: float = 0.0) -> Path:
    """Fetch one shard, resuming a partial file rather than restarting it.

    A resume that the endpoint will not serve is not retried as a resume: curl
    reports exit 33 when a server answers a range request without honouring it,
    and that answer will never change, so the partial file is dropped and the
    shard is fetched from the start. Without that, one interrupted download on
    such an endpoint is permanently stuck (measured against ModelScope's file
    API: `-C -` exits 33, and an open-ended `-r 100-` returns a 4 KiB chunk with
    status 200 instead of the remainder).

    `progress` is the caller's line for the whole stage, and `base` the bytes
    the shards before this one already landed: together they turn a 2 GB shard
    from one silent hour into a percentage and an ETA. With no line the watcher
    is not started and the call is exactly what it always was.
    """

    def tell(text: str) -> None:
        if progress is not None:
            progress.note(text)
        else:
            print(text, file=sys.stderr, flush=True)

    dest = work / shard
    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = 5
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        url = f"{BASE}/{shard}"
        stop_watch = threading.Event()
        watcher = None
        if progress is not None:
            watcher = threading.Thread(
                target=watch_size, args=(dest, progress, base, stop_watch), daemon=True
            )
            watcher.start()
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-fL",
                    "--retry",
                    str(CURL_RETRY_ATTEMPTS),
                    "--retry-delay",
                    str(CURL_RETRY_DELAY_SECONDS),
                    "--retry-connrefused",
                    "--retry-all-errors",
                    "-C",
                    "-",
                    "--max-time",
                    str(DOWNLOAD_TIMEOUT_SECONDS),
                    "--silent",
                    "--show-error",
                    "-o",
                    str(dest),
                    url,
                ],
                check=False,
            )
        finally:
            stop_watch.set()
            if watcher is not None:
                watcher.join(timeout=2)
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            if progress is not None:
                progress.set(base + dest.stat().st_size)
            return dest
        if result.returncode == 33 and dest.exists():
            tell(
                f"    {shard}: the endpoint will not serve a range request; "
                "downloading it from the start"
            )
            dest.unlink()
        tell(
            f"    [download {attempt}/{DOWNLOAD_ATTEMPTS}] {shard} failed "
            f"(curl {result.returncode}); retrying in {delay}s"
        )
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"failed to download {shard} after {DOWNLOAD_ATTEMPTS} attempts")


# --- conversion ------------------------------------------------------------


def read_shard_header(path: Path) -> dict | None:
    """The safetensors header of `path`, or None when it is not a whole shard.

    A safetensors file is `8-byte header length | JSON header | payload`. A
    process killed during the write leaves a file whose header parses while the
    payload is short, so the declared offsets are checked against the file size:
    trusting the header alone is how a resume decides a truncated shard is
    finished, keeps it, and ships a snapshot whose index points at bytes that
    are not there. Anything unreadable is reported as "not a shard" rather than
    raised -- the caller's move is to drop the file and convert its tensors
    again, which is always safe.
    """
    try:
        size = path.stat().st_size
        if size < 8:
            return None
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            if header_len == 0 or header_len > size - 8:
                return None
            payload = handle.read(header_len)
    except OSError:
        return None
    if len(payload) != header_len:
        return None
    try:
        header = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    end = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        offsets = meta.get("data_offsets") if isinstance(meta, dict) else None
        if not (isinstance(offsets, list) and len(offsets) == 2):
            return None
        try:
            end = max(end, int(offsets[1]))
        except (TypeError, ValueError):
            return None
    return header if size == 8 + header_len + end else None


class OutputWriter:
    """Accumulates converted tensors and flushes them as safetensors shards."""

    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.block: dict[str, np.ndarray] = {}
        self.bytes = 0
        self.index: dict[str, str] = {}
        self.total = 0
        self.shard_no = 0
        self.discarded = 0
        self._resume()

    def _resume(self) -> None:
        """Adopt the shards a previous run left behind.

        The run this exists for is one that died mid-conversion: its output
        shards are finished files numbered from 1, and the checkpoint shards
        they came from can be skipped. A shard that is not whole (a kill during
        its write) is deleted instead of adopted, so its tensors are converted
        again; a leftover `*.partial` from an interrupted flush is removed the
        same way.
        """
        for partial in sorted(self.out.glob("model-*.safetensors.partial")):
            partial.unlink()
        survivors: list[tuple[Path, dict]] = []
        for shard_path in sorted(self.out.glob("model-[0-9][0-9][0-9][0-9][0-9].safetensors")):
            header = read_shard_header(shard_path)
            if header is None:
                print(f"  discarding an incomplete output shard: {shard_path.name}", flush=True)
                shard_path.unlink()
                self.discarded += 1
                continue
            survivors.append((shard_path, header))
        # A discarded shard leaves a hole in the 1..N numbering that `finish`
        # renames over (it walks 1..shard_no), so the survivors are compacted
        # into a contiguous run before anything else happens. Numbers only move
        # down, and each destination was just vacated by the shard before it, so
        # no rename can collide.
        for number, (shard_path, header) in enumerate(survivors, start=1):
            target = self.out / f"model-{number:05d}.safetensors"
            if shard_path != target:
                os.replace(shard_path, target)
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                self.index[key] = target.name
                offsets = meta["data_offsets"]
                self.total += offsets[1] - offsets[0]
        self.shard_no = len(survivors)
        if survivors or self.discarded:
            print(
                f"  resuming from {len(survivors)} output shards "
                f"({self.shard_no:05d} last, {self.total / 1e9:.2f} GB, "
                f"{len(self.index)} tensors); {self.discarded} discarded",
                flush=True,
            )

    def add(self, name: str, value: np.ndarray) -> None:
        if name in self.index:
            # Adopted from a previous run. Its checkpoint shard is converted
            # again when not every tensor of that shard was adopted, so the
            # ones that were must be dropped here -- writing them twice puts a
            # duplicate in the snapshot and counts its bytes twice in the
            # index's total_size.
            return
        self.block[name] = value
        self.bytes += value.nbytes
        self.total += value.nbytes
        if self.bytes >= OUTPUT_SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.block:
            return
        self.shard_no += 1
        name = f"model-{self.shard_no:05d}.safetensors"
        # Written beside the destination and renamed into place: a kill during
        # the write then leaves `*.partial`, which `_resume` deletes, instead of
        # a file that looks like a finished shard.
        partial = self.out / f"{name}.partial"
        save_file(self.block, str(partial))
        os.replace(partial, self.out / name)
        for key in self.block:
            self.index[key] = name
        print(
            f"    wrote {name} ({self.bytes / 1e9:.2f} GB, {len(self.block)} tensors)", flush=True
        )
        self.block.clear()
        self.bytes = 0

    def finish(self) -> None:
        self.flush()
        # Rename to the N-of-M form the loaders expect now that M is known.
        final = {}
        for old_key, old_name in self.index.items():
            n = int(old_name.split("-")[1].split(".")[0])
            final[old_key] = f"model-{n:05d}-of-{self.shard_no:05d}.safetensors"
        for n in range(1, self.shard_no + 1):
            src = self.out / f"model-{n:05d}.safetensors"
            src.rename(self.out / f"model-{n:05d}-of-{self.shard_no:05d}.safetensors")
        (self.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": self.total}, "weight_map": final}, indent=1)
        )


def convert_shard(
    path: Path, writer: OutputWriter, ngram: "NgramTable", width: int = BITS_4
) -> None:
    with safe_open(path, framework="np") as src:
        for name in src.keys():
            if is_multimodal(name):
                continue
            if is_ngram(name):
                ngram.add(name, src.get_tensor(name))
                continue
            if is_ple_buffer(name):
                continue
            value = src.get_tensor(name)
            for out_name, _ in outputs_for(name, list(value.shape)):
                if out_name.endswith("switch_mlp.gate_proj.weight"):
                    piece = value[:, : value.shape[1] // 2, :]
                elif out_name.endswith("switch_mlp.up_proj.weight"):
                    piece = value[:, value.shape[1] // 2 :, :]
                elif out_name.endswith("indexer.index_q_proj.weight"):
                    piece = value[:INDEXER_QUERY_ROWS]
                elif out_name.endswith("indexer.index_k_proj.weight"):
                    piece = value[INDEXER_QUERY_ROWS:]
                else:
                    piece = value
                bits = quant_bits(out_name, width)
                if bits is None:
                    writer.add(out_name, np.ascontiguousarray(fold_unit_offset(out_name, piece)))
                    continue
                stem = out_name[: -len(".weight")]
                packed, scales, biases = quantize_affine(np.ascontiguousarray(piece), bits)
                writer.add(stem + ".weight", packed)
                writer.add(stem + ".scales", scales)
                writer.add(stem + ".biases", biases)


class NgramTable:
    """Assembles ngram_table.bin from the checkpoint's 128 table shards.

    The shards must be concatenated in *numeric* order. Their names sort
    lexically as shard_0, shard_1, shard_10, shard_100 ..., so sorting the
    strings would interleave the table and produce a model that loads, runs,
    and is quietly wrong.
    """

    def __init__(self, out: Path, expected_rows: int, dim: int, reuse: Path | None = None):
        self.path = out / "ngram_table.bin"
        self.expected_rows = expected_rows
        self.dim = dim
        self.pending: dict[int, np.ndarray] = {}
        self.next_index = 0
        self.rows = 0
        self.reused = reuse is not None
        self.skipped = 0
        if reuse is None:
            # Written beside the destination and renamed in `finish`: the table
            # is 102 GB and takes minutes, so a kill during it must not leave a
            # file at the final name that a later run reuses (or, worse, that an
            # installed model already hardlinks to this inode).
            self.temporary = self.path.parent / (self.path.name + ".partial")
            if self.temporary.exists():
                self.temporary.unlink()
            self.handle = self.temporary.open("wb")
            return
        # Hardlink, not copy and not symlink. Both directories then hold a
        # reference to one inode: deleting either leaves the other intact,
        # which a symlink would not, and the runtime's F_NOCACHE reads do not
        # care about the extra link.
        self.handle = None
        expected_bytes = expected_rows * dim * 2
        actual = reuse.stat().st_size
        if actual != expected_bytes:
            raise ValueError(
                f"{reuse}: {actual} bytes, expected {expected_bytes} "
                f"({expected_rows} rows x {dim} x fp16). A table of the wrong "
                "size is a different model's, or a truncated copy."
            )
        # A reuse that resolves to the table already at the destination -- the
        # automatic reuse of a table a previous run finished here, or
        # `--reuse-ngram-table` aimed at this same output -- has nothing to
        # link: unlinking the destination first would delete the 102 GB source
        # and then fail to link what is no longer there.
        if self.path.resolve() == reuse.resolve():
            return
        if self.path.exists():
            self.path.unlink()
        try:
            os.link(reuse, self.path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # A hardlink cannot cross filesystems -- staging on one volume and
            # the install on another is a normal layout (`TINYTITAN_MODELS_DIR`
            # on an external disk). Reusing the table is still what the caller
            # asked for, so copy it: that is the 102 GB the flag exists to save,
            # but a copy is correct where a link is impossible.
            print(
                f"  {reuse} is on another filesystem; copying the table "
                f"({expected_bytes / 1e9:.1f} GB) instead of hardlinking it"
            )
            shutil.copyfile(reuse, self.path)

    @staticmethod
    def index_of(name: str) -> int:
        return int(name.rsplit(".shard_", 1)[1].split(".")[0])

    def add(self, name: str, value: np.ndarray) -> None:
        if self.reused:
            # Two of the 131 checkpoint shards carry n-gram rows alongside
            # ordinary tensors, so they are fetched even when the table is
            # reused and their n-gram names still arrive here. The linked
            # table already contains those rows -- the size check in __init__
            # is what guarantees it holds all of them -- so drop them.
            self.skipped += 1
            return
        self.pending[self.index_of(name)] = value
        while self.next_index in self.pending:
            block = self.pending.pop(self.next_index)
            if block.dtype == ml_dtypes.bfloat16:
                as_f32 = block.astype(np.float32)
                finite = np.isfinite(as_f32.astype(np.float16))
                if not finite.all():
                    raise ValueError(
                        f"{name}: bf16 -> fp16 overflows on "
                        f"{(~finite).sum()} values; the table format is fp16"
                    )
                block = as_f32.astype(np.float16)
            self.handle.write(np.ascontiguousarray(block).tobytes())
            self.rows += block.shape[0]
            self.next_index += 1

    def summary(self) -> str:
        """What the table cost this build, in the mode it actually ran."""
        if self.reused:
            return (
                f"linked, {self.expected_rows} rows "
                f"({self.expected_rows * self.dim * 2 / 1e9:.1f} GB not written)"
            )
        return f"{self.rows} rows"

    def finish(self) -> None:
        if self.reused:
            if self.pending:
                raise ValueError("n-gram shards arrived while reusing a table")
            if self.skipped:
                print(
                    f"  {self.skipped} n-gram tensors in mixed shards ignored "
                    f"(already in the linked table)"
                )
            return
        self.handle.close()
        if self.pending:
            raise ValueError(
                f"n-gram shards never became contiguous: {sorted(self.pending)[:5]} still pending"
            )
        if self.rows != self.expected_rows:
            raise ValueError(f"n-gram table has {self.rows} rows, expected {self.expected_rows}")
        # Published only once it is whole: a run that dies earlier leaves
        # `ngram_table.bin.partial`, which the next run deletes and rebuilds.
        os.replace(self.temporary, self.path)


# --- driver ----------------------------------------------------------------


# The repacker refuses a snapshot without these: a model imported from one has
# no other source for them, and an install that loads but cannot be prompted is
# worse than one that fails. `config.json` is written separately by
# `write_config`, which adds the quantization block the repacker reads.
TOKENIZER_FILES = (
    ("tokenizer.json", True),
    ("tokenizer_config.json", True),
    ("special_tokens_map.json", False),
    ("chat_template.jinja", False),
    ("chat_template.json", False),
)


def fetch_tokenizer(out: Path) -> None:
    """Copy the tokenizer beside the weights, so the snapshot stands alone."""
    for name, required in TOKENIZER_FILES:
        url = f"{BASE}/{name}"
        result = subprocess.run(
            ["curl", "-sfL", "--max-time", "300", url], capture_output=True, check=False
        )
        if result.returncode != 0 or not result.stdout:
            if required:
                raise SystemExit(
                    f"cannot fetch {name} from {REPO}; the snapshot would be rejected at repack"
                )
            continue
        (out / name).write_bytes(result.stdout)
        print(f"  {name} ({len(result.stdout) / 1e6:.2f} MB)")


def plan(index: dict, width: int = BITS_4) -> None:
    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    text = {n: s for n, s in wm.items() if not is_multimodal(n)}
    ngram = [n for n in text if is_ngram(n)]
    print(f"repo     : {REPO}")
    print(
        f"shards   : {len(shards)}   tensors: {len(wm)} ({len(wm) - len(text)} multimodal skipped)"
    )
    print(f"declared : {index['metadata']['total_size'] / 1e9:.1f} GB")
    print(f"n-gram   : {len(ngram)} table shards")
    probe = [s for s in shards if any(".mlp.experts." in n for n, sh in text.items() if sh == s)][
        :1
    ]
    problems: list[str] = []
    for shard in probe:
        for name, meta in fetch_header(shard).items():
            if name == "__metadata__" or is_multimodal(name) or is_ngram(name):
                continue
            for out_name, out_shape in outputs_for(name, meta["shape"]):
                bits = quant_bits(out_name, width)
                if bits and out_shape[-1] % GROUP_SIZE:
                    problems.append(f"{out_name} last dim {out_shape[-1]} unaligned")
                print(
                    f"  {name} {meta['shape']}\n     -> {out_name} {out_shape} "
                    f"{'q' + str(bits) if bits else 'passthrough'}"
                )
    print(
        "\nPROBLEMS: " + "; ".join(problems)
        if problems
        else "\nplan validates against the checkpoint's own headers"
    )


# The PLE constants a table is addressed by. The file is a hash table: its
# contents are meaningless under different multipliers, offsets or vocabulary
# sizes, and none of that is recoverable from the file itself, so a build may
# only reuse a table whose constants match its own.
REUSE_CONSTANT_KEYS = (
    "layer_multipliers",
    "ngram_heads_offsets",
    "ngram_heads_vocab_sizes",
    "ngram_size",
    "heads_per_ngram",
    "ple_head_dim",
)


def read_json_file(path: Path) -> dict | None:
    """The JSON object in `path`, or None when it is absent or unreadable."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def constants_match(previous: dict | None, constants: dict) -> bool:
    """Whether a previous run's constants can address this build's table."""
    if previous is None:
        return False
    return all(previous.get(key) == constants.get(key) for key in REUSE_CONSTANT_KEYS)


def reusable_local_table(
    out: Path, expected_bytes: int, previous_constants: dict | None, constants: dict
) -> Path | None:
    """The table a previous run finished in `out`, or None.

    Three things have to hold: the file is there, it is exactly the size this
    build's constants address (`padded x ple_head_dim x fp16`), and the
    constants beside it are this build's -- a table read under different
    constants produces garbage n-gram ids rather than an error.
    """
    table = out / "ngram_table.bin"
    if not table.exists():
        return None
    if table.stat().st_size != expected_bytes:
        return None
    if not constants_match(previous_constants, constants):
        return None
    return out


def reusable_table_path(reuse: Path, constants: dict) -> Path:
    """The table to hardlink for `--reuse-ngram-table`, or SystemExit.

    `reuse` is an install directory (whose `ple_constants.json` is checked
    against this build's) or the table file itself. Exits rather than returning
    None on a bad one: silently falling back to fetching 128 shards and writing
    102 GB is the wrong default for a flag whose whole purpose is to save them,
    and a table read under the wrong constants produces garbage ids rather than
    an error.
    """
    if reuse.is_dir():
        sibling = reuse / "ple_constants.json"
        if sibling.exists():
            have = json.loads(sibling.read_text())
            for key in REUSE_CONSTANT_KEYS:
                if have.get(key) != constants.get(key):
                    raise SystemExit(
                        f"--reuse-ngram-table: {key} differs between "
                        f"{sibling} and this build; the table is addressed "
                        "by those constants and would be read wrongly"
                    )
        reuse = reuse / "ngram_table.bin"
    if not reuse.exists():
        raise SystemExit(f"--reuse-ngram-table: no such file: {reuse}")
    return reuse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument(
        "--bits",
        type=int,
        choices=(4, 8),
        default=4,
        help="routed-expert width of the install being built",
    )
    ap.add_argument(
        "--fetch-workers",
        type=int,
        default=1,
        help="checkpoint shards downloaded concurrently (default 1). The Hub "
        "serves one connection at a few MB/s on some links, so 4 keeps a "
        "slow link busy; each in-flight shard costs ~3.4 GB of --work space",
    )
    ap.add_argument(
        "--rest-bf16",
        action="store_true",
        help="keep every tensor except the routed experts at bf16 "
        "(evaluation build; cannot start under --ram-budget 10G)",
    )
    ap.add_argument(
        "--rest-8bit",
        action="store_true",
        help="quantise every tensor except the routed experts at 8 bits "
        "(experts stay at --bits; ~+2.6 GB resident against the 4-bit install)",
    )
    ap.add_argument(
        "--rewrite-config", type=Path, help="regenerate config.json for an existing snapshot"
    )
    ap.add_argument("--output", type=Path)
    ap.add_argument("--work", type=Path, help="scratch for in-flight shards")
    ap.add_argument(
        "--reuse-ngram-table",
        type=Path,
        metavar="DIR_OR_FILE",
        help="hardlink ngram_table.bin from an existing install or "
        "snapshot instead of fetching the table shards. The "
        "table is fp16 in every quantization and is fully "
        "determined by the checkpoint, so two builds of the "
        "same model cannot differ in it.",
    )
    ap.add_argument("--index", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hub endpoint or mirror, e.g. https://hf-mirror.com "
        "(default: HF_ENDPOINT or https://huggingface.co). The "
        "mirror must serve the Hub's layout: "
        "/<repo>/resolve/main/<file> for weights and "
        "/<repo>/raw/main/<file> for the small JSON files",
    )
    args = ap.parse_args()

    global HF_ENDPOINT, BASE, REST_BF16, REST_8BIT
    if args.rest_bf16 and args.rest_8bit:
        ap.error("--rest-bf16 and --rest-8bit are exclusive")
    REST_BF16 = args.rest_bf16
    REST_8BIT = args.rest_8bit
    HF_ENDPOINT = args.endpoint.rstrip("/")
    BASE = endpoint_base(HF_ENDPOINT)

    def fetch_json(path: Path | None, remote: str) -> dict:
        if path and path.exists():
            return json.loads(path.read_text())
        return json.loads(
            subprocess.run(
                ["curl", "-sfL", "--max-time", "60", raw_url(remote)],
                capture_output=True,
                check=True,
            ).stdout
        )

    index = fetch_json(args.index, "model.safetensors.index.json")
    if args.rewrite_config:
        snap = args.rewrite_config
        names = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        cfg = (
            json.loads(
                subprocess.run(
                    ["curl", "-sfL", "--max-time", "60", raw_url("config.json")],
                    capture_output=True,
                    check=True,
                ).stdout
            )
            if not (snap / "config.json").exists()
            else json.loads((snap / "config.json").read_text())
        )
        cfg.pop("quantization", None)
        out = write_config(cfg, snap, names.keys(), args.bits)
        n = len(out["quantization"]) - 3
        print(f"wrote {snap}/config.json: base {args.bits}-bit, {n} overrides")
        return 0

    if args.plan or not args.output:
        plan(index, args.bits)
        return 0

    # A finished snapshot is not a resume target: `finish` renames its shards to
    # the N-of-M form and writes the index, so converting into that directory
    # again would lay a second generation of shards beside the first and orphan
    # the older one. The installer skips this case; refuse it here as well.
    if (args.output / "model.safetensors.index.json").exists():
        raise SystemExit(
            f"{args.output} already holds a finished snapshot "
            "(model.safetensors.index.json is present). Converting into it would "
            "leave two generations of shards behind; delete the directory or "
            "point --output at a new one."
        )
    # The same hazard one step earlier: `finish` renames the shards before it
    # writes the index, so a process killed (or a disk that filled) between the
    # two leaves finished N-of-M shards and no index. Nothing adopts those --
    # `_resume` looks for the un-renamed form -- so a run into that directory
    # would convert everything again and leave the finished generation orphaned
    # beside the new one, which for a real snapshot is hundreds of GB.
    renamed = sorted(args.output.glob("model-[0-9]*-of-[0-9]*.safetensors"))
    if renamed:
        raise SystemExit(
            f"{args.output} holds {len(renamed)} finished output shards but no "
            "model.safetensors.index.json: a run stopped between finishing its "
            "shards and writing the index. Converting into it again would "
            "orphan those shards beside a second generation of the same "
            "tensors; delete the directory or point --output at a new one."
        )

    config = fetch_json(args.config, "config.json")
    text_config = config["text_config"]
    work = args.work or (args.output.parent / "qwen38-shards")
    work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    constants = ple_constants(text_config)
    # Read before overwriting: a run that died after writing the table left the
    # constants that built it, and a table reused under different constants
    # yields garbage n-gram ids rather than an error.
    previous_constants = read_json_file(args.output / "ple_constants.json")
    (args.output / "ple_constants.json").write_text(json.dumps(constants, indent=1))
    rows = sum(constants["ngram_heads_vocab_sizes"])
    divisor = text_config.get("make_ngram_vocab_size_divisible_by", 128)
    padded = math.ceil(rows / divisor) * divisor
    table_bytes = padded * constants["ple_head_dim"] * 2

    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    by_shard: dict[str, list[str]] = {}
    for name, shard in wm.items():
        by_shard.setdefault(shard, []).append(name)

    # A resume adopts output shards without looking inside them, and a
    # safetensors file does not record the quantization width it was written
    # at, so a run at a different --bits would mix two widths in one snapshot
    # and the index would not say so. The marker is what makes that refusal
    # possible; a run from before the marker existed has none and is assumed to
    # match, which is the migration case the resume path exists for.
    marker = args.output / "conversion.json"
    adoptable = list(args.output.glob("model-[0-9][0-9][0-9][0-9][0-9].safetensors"))
    previous_bits = read_json_file(marker)
    recipe = {"bits": args.bits, "rest_bf16": REST_BF16, "rest_8bit": REST_8BIT}
    previous_recipe = (
        {
            "bits": previous_bits.get("bits"),
            "rest_bf16": previous_bits.get("rest_bf16", False),
            "rest_8bit": previous_bits.get("rest_8bit", False),
        }
        if previous_bits is not None
        else None
    )
    if adoptable and previous_recipe is not None and previous_recipe != recipe:
        raise SystemExit(
            f"{args.output} holds a partial conversion with recipe {previous_recipe} "
            f"and this run is {recipe}. Adopting those shards would put two "
            "widths in one snapshot without recording it; delete the directory "
            "or point --output at a new one."
        )
    marker.write_text(json.dumps(recipe, indent=1))

    writer = OutputWriter(args.output)

    if args.reuse_ngram_table is None:
        # A previous run may have finished the table before the weights.
        found = reusable_local_table(args.output, table_bytes, previous_constants, constants)
        if found is not None:
            print(
                f"reusing the completed ngram_table.bin already here ({table_bytes / 1e9:.1f} GB)",
                flush=True,
            )
            args.reuse_ngram_table = found
        elif (args.output / "ngram_table.bin").exists():
            print(
                "ignoring the ngram_table.bin already here (wrong size for "
                "this build's constants, or a constants mismatch); it will "
                "be rebuilt",
                flush=True,
            )

    reuse = (
        None
        if args.reuse_ngram_table is None
        else reusable_table_path(args.reuse_ngram_table, constants)
    )

    ngram = NgramTable(args.output, padded, constants["ple_head_dim"], reuse)

    if reuse is not None:
        # Skip the shards that carry nothing else. Two of the 131 hold n-gram
        # rows alongside ordinary tensors and are still fetched; convert_shard
        # already ignores the n-gram names inside them.
        skippable = {shard for shard, names in by_shard.items() if all(is_ngram(n) for n in names)}
        shards = [s for s in shards if s not in skippable]
        print(f"reusing n-gram table: {reuse}")
        print(f"  linked, {table_bytes / 1e9:.1f} GB not written")
        print(f"  {len(skippable)} of {len(by_shard)} checkpoint shards not fetched")

    if writer.index:
        # The shards a previous run finished are converted already; the
        # checkpoint sources of those can be skipped. This is the resume.
        converted = {
            s for s in shards if checkpoint_shard_is_converted(by_shard[s], writer.index, args.bits)
        }
        if converted:
            shards = [s for s in shards if s not in converted]
            print(
                f"  {len(converted)} checkpoint shards already converted; "
                f"{len(shards)} left to fetch and convert",
                flush=True,
            )

    # Fetch shard N+1 while shard N converts.
    #
    # One progress line covers both, because both are happening: the bytes the
    # fetcher lands (with a real percentage, from the sizes the server reports,
    # probed in parallel before the first byte) and the shards the converter
    # finishes. Two bars would fight over the same terminal row.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(remote_size, shards))
    total = sum(size for size in sizes if size) if all(sizes) else None
    progress = Progress("converting", total)
    workers = max(1, args.fetch_workers)
    queue: Queue = Queue(maxsize=workers)
    # Bounds the shards on disk that are fetched but not yet converted, so a
    # link faster than the converter cannot fill --work.
    on_disk = threading.Semaphore(2 * workers)

    def fetch_bounded(shard: str) -> Path:
        on_disk.acquire()
        return download(shard, work, None, 0.0)

    def fetcher() -> None:
        base = 0.0
        for shard, size in zip(shards, sizes, strict=True):
            try:
                queue.put(download(shard, work, progress, base))
            except Exception as exc:  # noqa: BLE001
                queue.put(exc)
                return
            base += size or 0
        queue.put(None)

    def parallel_fetcher() -> None:
        # Shards are converted in completion order: each checkpoint shard's
        # tensors land in output shards the index names, so order does not
        # change what the snapshot holds. The live byte counter assumes one
        # download at a time, so the parallel path reports converted shards.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as fetch_pool:
            futures = [fetch_pool.submit(fetch_bounded, shard) for shard in shards]
            for future in concurrent.futures.as_completed(futures):
                try:
                    queue.put(future.result())
                except Exception as exc:  # noqa: BLE001
                    queue.put(exc)
                    return
        queue.put(None)

    threading.Thread(target=fetcher if workers == 1 else parallel_fetcher, daemon=True).start()
    done = 0
    while True:
        item = queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            progress.finish()
            raise item
        done += 1
        progress.show(f"{done}/{len(shards)} shards  {progress.amount}")
        convert_shard(item, writer, ngram, args.bits)
        item.unlink()
        if workers > 1:
            on_disk.release()
    progress.finish(f"  fetched and converted {done} of {len(shards)} shards")

    writer.finish()
    ngram.finish()
    # Written last: the quantization block lists every tensor whose width
    # differs from the base, which is only known once they have all been seen.
    write_config(config, args.output, writer.index.keys(), args.bits)
    print("tokenizer:")
    fetch_tokenizer(args.output)
    print(f"\naffine snapshot written to {args.output}")
    print(f"  {writer.shard_no} shards, {writer.total / 1e9:.1f} GB")
    print(f"  ngram_table.bin {ngram.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
