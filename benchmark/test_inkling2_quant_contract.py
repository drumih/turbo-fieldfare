import json
import pathlib
import tempfile
import unittest

from inkling2_quant_contract import ContractError, contract


def config() -> dict:
    return {
        "text_config": {
            "num_hidden_layers": 24,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "hidden_size": 2304,
            "model_max_length": 1048576,
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3},
        },
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }


def quant() -> dict:
    nvfp4 = [
        f"model.llm.layers.{layer}.mlp.experts.{suffix}"
        for layer in range(20)
        for suffix in ("w13_weight", "w2_weight")
    ]
    mxfp8 = [
        f"model.llm.layers.{layer}.mlp.experts.{suffix}"
        for layer in range(20, 24)
        for suffix in ("w13_weight", "w2_weight")
    ]
    return {
        "model": {
            "parameter_groups": [
                {"compute_dtype": "bfloat16", "tensors": ["model.llm.embed.weight"]},
                {"compute_dtype": "float32", "tensors": ["model.llm.attn_res.output_query"]},
                {"compute_dtype": "nvfp4", "tensors": nvfp4},
                {"compute_dtype": "mxfp8", "tensors": mxfp8},
            ]
        },
        "quant_method": "inkling2",
    }


class ContractTests(unittest.TestCase):
    def test_expected_split_is_parsed(self) -> None:
        result = contract(config(), quant())
        self.assertEqual(result["layers"], 24)
        self.assertEqual(result["per_layer"][0]["w13_weight"], "nvfp4")
        self.assertEqual(result["per_layer"][19]["w2_weight"], "nvfp4")
        self.assertEqual(result["per_layer"][20]["w13_weight"], "mxfp8")
        self.assertEqual(result["per_layer"][23]["w2_weight"], "mxfp8")
        self.assertEqual(result["routed_experts"], 128)
        self.assertEqual(result["experts_per_token"], 8)

    def test_a_missing_layer_is_refused(self) -> None:
        broken = quant()
        broken["model"]["parameter_groups"][2]["tensors"].remove(
            "model.llm.layers.7.mlp.experts.w13_weight"
        )
        with self.assertRaises(ContractError):
            contract(config(), broken)

    def test_a_wrong_format_for_a_layer_is_refused(self) -> None:
        broken = quant()
        broken["model"]["parameter_groups"][3]["tensors"].append(
            "model.llm.layers.19.mlp.experts.w13_weight"
        )
        with self.assertRaises(ContractError):
            contract(config(), broken)

    def test_a_duplicate_tensor_is_refused(self) -> None:
        broken = quant()
        broken["model"]["parameter_groups"][2]["tensors"].append("model.llm.embed.weight")
        with self.assertRaises(ContractError):
            contract(config(), broken)

    def test_the_cli_reads_local_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            c = pathlib.Path(work) / "config.json"
            q = pathlib.Path(work) / "hf_quant_config.json"
            c.write_text(json.dumps(config()))
            q.write_text(json.dumps(quant()))
            from inkling2_quant_contract import main

            self.assertEqual(main(["--config", str(c), "--quant-config", str(q)]), 0)


if __name__ == "__main__":
    unittest.main()
