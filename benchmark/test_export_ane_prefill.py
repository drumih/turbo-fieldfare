"""The ANE sidecar exporter's geometry and width rules.

Two things decide whether a sidecar is right, and both are pure functions of
the model's manifest and its weight index:

- **the geometry** (`geometry_for`) — what the Core ML graph is built for. A
  graph built for the wrong width or head split computes a *different*
  attention and produces fluent nonsense, so every deviation the graph cannot
  reproduce must be refused rather than approximated;
- **the width of each tensor** (`tensor_weight_bits`) — the manifest's slot is
  not authoritative. The dense Qwen 3.5 installs declare a 4-bit attention
  slot but store `k_proj`/`v_proj` at 8 bits, recorded as per-stem `quant`
  entries; reading those as nibbles is a documented failure mode.

These import the exporter, which imports coremltools, so they skip where that
is not installed (the CI python does not have it; `~/.venvs/coreml-py311` does):

    cd benchmark && ~/.venvs/coreml-py311/bin/python -m unittest \
        test_export_ane_prefill -v
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys
import unittest
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import export_ane_prefill as exporter

    IMPORT_ERROR = ""
except Exception as exc:  # coremltools or numpy absent
    exporter = None
    IMPORT_ERROR = str(exc)


def arch(**overrides) -> dict:
    """The 35B-A3B row, which every supported family only varies from."""
    base = {
        "family": "qwen36",
        "hiddenSize": 2048,
        "numHeads": 16,
        "numKVHeads": 2,
        "numFullKVHeads": 2,
        "headDim": 256,
        "fullHeadDim": 256,
        "attentionScale": 0.0625,
        "attentionKEqV": False,
        "ropeNeoxSubdim": True,
        "slidingWindow": 0,
        "fullRopeTheta": 10_000_000,
        "partialRotaryFactor": 0.25,
        "numLayers": 40,
        "fullAttentionLayerMask": [2, 2, 2, 1] * 10,
    }
    base.update(overrides)
    return {"arch": base}


PREFIX = "language_model.model.layers."
ENTRIES = {f"{PREFIX}3.self_attn.q_proj.weight": {}}


@unittest.skipIf(exporter is None, f"exporter needs coremltools: {IMPORT_ERROR}")
class GeometryTests(unittest.TestCase):
    def test_reads_the_geometry_and_the_full_attention_layers(self):
        geom = exporter.geometry_for(arch(), ENTRIES)
        self.assertEqual(
            (
                geom.family,
                geom.hidden,
                geom.q_heads,
                geom.kv_heads,
                geom.head_dim,
                geom.rotary,
                geom.layers,
            ),
            ("qwen36", 2048, 16, 2, 256, 64, (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)),
        )

    def test_a_dense_shape_is_derived_not_assumed(self):
        # The 4B: hidden 2560, 16 q heads, 4 kv heads, 8 full-attention layers.
        geom = exporter.geometry_for(
            arch(
                hiddenSize=2560,
                numKVHeads=4,
                numFullKVHeads=4,
                numLayers=32,
                fullAttentionLayerMask=[2, 2, 2, 1] * 8,
            ),
            {f"{PREFIX}3.self_attn.q_proj.weight": {}},
        )
        self.assertEqual(
            (geom.hidden, geom.q_heads, geom.kv_heads, geom.q_dim, geom.kv_dim, len(geom.layers)),
            (2560, 16, 4, 4096, 1024, 8),
        )

    def test_the_prefix_is_discovered_from_the_index(self):
        # The 3.8 family spells it the other way round.
        entries = {"model.language_model.layers.3.self_attn.q_proj.weight": {}}
        geom = exporter.geometry_for(arch(family="qwen3_5_dense"), entries)
        self.assertEqual(geom.prefix, "model.language_model.layers.")

    def test_qwen38_is_served_because_only_its_mask_was_missing(self):
        # The indexer selects keys; it does not change the arithmetic. The
        # geometry is the same kind of block as the other families, and the
        # runtime folds the selection into the mask the graph already takes.
        geom = exporter.geometry_for(arch(family="qwen38flash"), ENTRIES)
        self.assertEqual(
            (geom.family, geom.hidden, geom.q_heads, geom.kv_heads, geom.head_dim, geom.rotary),
            ("qwen38flash", 2048, 16, 2, 256, 64),
        )

    def test_the_mtp_draft_is_refused(self):
        # The runtime verifies the one-layer draft rather than prefilling it on
        # the ANE, so a sidecar for it would never be loaded.
        with self.assertRaises(SystemExit) as caught:
            exporter.geometry_for(arch(family="qwen38flash_mtp"), ENTRIES)
        self.assertIn("MTP draft", str(caught.exception))

    def test_a_geometry_the_graph_cannot_reproduce_is_refused(self):
        for override, fragment in (
            ({"attentionKEqV": True}, "K and V separately"),
            ({"ropeNeoxSubdim": False}, "NeoX"),
            ({"slidingWindow": 4096}, "sliding-window"),
        ):
            with self.subTest(override=override):
                with self.assertRaises(SystemExit) as caught:
                    exporter.geometry_for(arch(**override), ENTRIES)
                self.assertIn(fragment, str(caught.exception))

    def test_a_model_with_no_full_attention_layer_is_refused(self):
        with self.assertRaises(SystemExit):
            exporter.geometry_for(arch(fullAttentionLayerMask=[2] * 40), ENTRIES)

    def test_an_unknown_attention_layout_is_refused(self):
        with self.assertRaises(SystemExit):
            exporter.geometry_for(arch(), {"something.else.weight": {}})

    def test_the_recorded_metadata_carries_what_the_runtime_validates(self):
        meta = exporter.geometry_for(arch(), ENTRIES).as_metadata()
        for key in (
            "family",
            "hiddenSize",
            "numHeads",
            "numKVHeads",
            "headDim",
            "chunkTokens",
            "fullAttentionLayers",
        ):
            self.assertIn(key, meta)
        self.assertEqual(meta["chunkTokens"], exporter.CHUNK)
        self.assertEqual(meta["family"], "qwen36")


@unittest.skipIf(exporter is None, f"exporter needs coremltools: {IMPORT_ERROR}")
class TensorWidthTests(unittest.TestCase):
    def test_a_per_tensor_override_beats_the_slot(self):
        # The dense installs' shape: a 4-bit slot, an 8-bit k_proj.
        manifest = {
            "quant": {
                "attention": {"weightBits": 4},
                f"{PREFIX}3.self_attn.k_proj": {"weightBits": 8},
            }
        }
        self.assertEqual(
            exporter.tensor_weight_bits(manifest, f"{PREFIX}3.self_attn.k_proj.weight", 4), 8
        )
        self.assertEqual(
            exporter.tensor_weight_bits(manifest, f"{PREFIX}3.self_attn.q_proj.weight", 4), 4
        )

    def test_the_slot_is_the_fallback(self):
        manifest = {"quant": {"attention": {"weightBits": 8}}}
        self.assertEqual(
            exporter.tensor_weight_bits(manifest, f"{PREFIX}3.self_attn.k_proj.weight", 8), 8
        )


@unittest.skipIf(exporter is None, f"exporter needs coremltools: {IMPORT_ERROR}")
class CompileMarkerTests(unittest.TestCase):
    """The stderr scan that made issue #7's export fail loudly.

    Core ML reports an ANE compile refusal on the native stderr and still
    returns, so the exporter has to read it back and refuse rather than let the
    sidecar claim `aneCompileVerified`.
    """

    def test_a_compile_error_marker_fails_the_step(self):
        def refuses():
            os.write(
                2, b"MILCompilerForANE error: failed to compile ANE model\nANECCompile() FAILED.\n"
            )
            return "ok"

        with self.assertRaises(exporter.ANEExportError) as caught:
            exporter.run_checked("layer 3 h32768 convert", refuses)
        self.assertIn("refused to compile", str(caught.exception))

    def test_a_clean_step_returns_its_result(self):
        self.assertEqual(exporter.run_checked("layer 3 h0 convert", lambda: 41 + 1), 42)


@unittest.skipIf(exporter is None, f"exporter needs coremltools: {IMPORT_ERROR}")
class ANEAssignmentTests(unittest.TestCase):
    """What `aneCompileVerified` claims: the ANE is assigned the graph.

    Issue #7's silent half is a variant the ANE refuses that still loads and
    runs on the CPU at ~38x the GPU cost. Measured on the real 3.8 h12288
    standalone package: 0 of 173 operations assigned to the Neural Engine, while
    a healthy variant of the same graph reports 74 of 173. Only the compute plan
    distinguishes those.
    """

    def setUp(self):
        self.addCleanup(setattr, exporter, "MLComputePlan", exporter.MLComputePlan)
        self.addCleanup(
            setattr,
            exporter.ct.models.utils,
            "compile_model",
            exporter.ct.models.utils.compile_model,
        )
        exporter.ct.models.utils.compile_model = lambda path: "/tmp/fake.mlmodelc"

    @staticmethod
    def fakePlan(on_ane: int, total: int = 8, function: str = "main"):
        class NeuralEngineComputeDevice:
            pass

        class Operation:
            pass

        class Usage:
            def __init__(self, device):
                self.preferred_compute_device = device

        device = NeuralEngineComputeDevice()
        operations = [Operation() for _ in range(total)]
        decided = {id(op): index < on_ane for index, op in enumerate(operations)}
        function_object = type(
            "Function", (), {"block": type("Block", (), {"operations": operations})()}
        )()
        program = type("Program", (), {"functions": {function: function_object}})()

        class Plan:
            model_structure = type("Structure", (), {"program": program})()

            @staticmethod
            def get_compute_device_usage_for_mlprogram_operation(operation):
                return Usage(device if decided[id(operation)] else None)

        return Plan

    def install(self, on_ane: int, total: int = 8):
        exporter.MLComputePlan = type(
            "MLComputePlan",
            (),
            {
                "load_from_path": staticmethod(
                    lambda path, compute_units=None: self.fakePlan(on_ane, total)
                )
            },
        )

    def test_a_variant_assigned_to_the_ane_reports_its_count(self):
        self.install(on_ane=6)
        self.assertEqual(
            exporter.verify_variant_reaches_the_ane(pathlib.Path("h4096.mlpackage"), 4096, 3), 6
        )

    def test_a_variant_the_ane_refuses_fails_the_export(self):
        self.install(on_ane=0, total=173)
        with self.assertRaises(exporter.ANEExportError) as caught:
            exporter.verify_variant_reaches_the_ane(pathlib.Path("h12288.mlpackage"), 12288, 3)
        message = str(caught.exception)
        self.assertIn("h12288", message)
        self.assertIn("173", message)
        self.assertIn("38x", message)

    def test_an_empty_graph_is_refused_rather_than_counted(self):
        self.install(on_ane=0, total=0)
        with self.assertRaises(exporter.ANEExportError) as caught:
            exporter.verify_variant_reaches_the_ane(pathlib.Path("h0.mlpackage"), 0, 3)
        self.assertIn("no operations", str(caught.exception))

    def test_no_compute_plan_api_warns_once_and_does_not_fail(self):
        exporter.MLComputePlan = None
        exporter._warnedNoComputePlan = False
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            first = exporter.verify_variant_reaches_the_ane(pathlib.Path("h0.mlpackage"), 0, 3)
            second = exporter.verify_variant_reaches_the_ane(
                pathlib.Path("h4096.mlpackage"), 4096, 3
            )
        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertIn("MLComputePlan", stderr.getvalue())
        self.assertEqual(stderr.getvalue().count("warning:"), 1)


@unittest.skipIf(exporter is None, f"exporter needs coremltools: {IMPORT_ERROR}")
class VariantLoadTests(unittest.TestCase):
    """A recorded history the runtime cannot load is announced coverage.

    The converter accepting a graph is not Core ML being able to load it, and a
    sidecar that records a variant it cannot serve fails the request that
    reaches it instead of falling back. Qwen 3.8's h12288 is the measured case.
    """

    def setUp(self):
        self.real = exporter.ct.models.MLModel
        self.addCleanup(setattr, exporter.ct.models, "MLModel", self.real)

    def test_a_variant_that_does_not_load_fails_the_export(self):
        seen = []

        def fake(path, compute_units=None, function_name=None):
            seen.append(function_name)
            if function_name == "h12288":
                raise RuntimeError(
                    "`.functionName` property must be nil unless the model type is ML Program."
                )
            return object()

        exporter.ct.models.MLModel = fake
        with self.assertRaises(exporter.ANEExportError) as caught:
            exporter.verify_variants_load(
                pathlib.Path("layer_3.mlpackage"), [0, 4096, 8192, 12288], 3
            )
        self.assertIn("h12288", str(caught.exception))
        # It reaches the failing variant rather than stopping early, so the
        # error names the history that has to change.
        self.assertEqual(seen, ["h0", "h4096", "h8192", "h12288"])

    def test_a_variant_that_loads_but_cannot_run_fails_the_export(self):
        # The Python API does not raise for a variant the ANE refused; it warns
        # that predict() will not work and returns a model. The runtime's
        # Objective-C call *does* raise, so an exception-only check here would
        # pass and the sidecar would still advertise coverage it cannot serve.
        def fake(path, compute_units=None, function_name=None):
            if function_name == "h12288":
                warnings.warn(
                    "You will not be able to run predict() on this Core ML "
                    "model. Underlying exception message was: `.functionName` "
                    "property must be nil unless the model type is ML Program.",
                    stacklevel=2,
                )
            return object()

        exporter.ct.models.MLModel = fake
        with self.assertRaises(exporter.ANEExportError) as caught:
            exporter.verify_variants_load(pathlib.Path("layer_3.mlpackage"), [0, 12288], 3)
        self.assertIn("h12288", str(caught.exception))
        self.assertIn("cannot run", str(caught.exception))

    def test_every_recorded_variant_is_loaded(self):
        seen = []

        def fake(path, compute_units=None, function_name=None):
            seen.append(function_name)
            return object()

        exporter.ct.models.MLModel = fake
        exporter.verify_variants_load(pathlib.Path("layer_7.mlpackage"), [0, 4096], 7)
        self.assertEqual(seen, ["h0", "h4096"])


@unittest.skipIf(exporter is None, f"exporter needs coremltools: {IMPORT_ERROR}")
class LoadTensorTests(unittest.TestCase):
    def entry(self, rows, cols, size, dtype=0):
        return {
            "shape": (rows, cols, 0, 0),
            "size": size,
            "dtype": dtype,
            "offset": 0,
            "scale": (0, 2 * rows * (cols // 64)),
            "bias": (0, 2 * rows * (cols // 64)),
        }

    def test_a_manifest_that_lies_about_its_width_is_refused(self):
        # 8-bit payload declared as 4-bit: 1,048,576 bytes for 512x2048.
        with self.assertRaises(SystemExit) as caught:
            exporter.load_tensor(
                io.BytesIO(b"\x00" * (1 << 21)),
                self.entry(512, 2048, 1 << 20),
                weight_bits=4,
                name="k_proj",
            )
        self.assertIn("refusing to guess", str(caught.exception))

    def test_an_unsupported_width_is_refused(self):
        with self.assertRaises(SystemExit):
            exporter.load_tensor(
                io.BytesIO(b"\x00" * 32), self.entry(8, 8, 32), weight_bits=3, name="k_proj"
            )


if __name__ == "__main__":
    unittest.main()
