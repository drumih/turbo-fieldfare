import Foundation
import Testing

@testable import TinyTitanRepackCore

@Suite struct Inkling2ContractTests {
    private func config() -> Data {
        Data(
            """
            {"text_config":{"num_hidden_layers":24,"n_routed_experts":128,
             "num_experts_per_tok":8,"model_max_length":1048576},
             "quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}}
            """.utf8)
    }

    private func quantObject() throws -> [String: Any] {
        let nvfp4 = (0..<20).flatMap { layer in
            ["w13_weight", "w2_weight"].map {
                "model.llm.layers.\(layer).mlp.experts.\($0)"
            }
        }
        let mxfp8 = (20..<24).flatMap { layer in
            ["w13_weight", "w2_weight"].map {
                "model.llm.layers.\(layer).mlp.experts.\($0)"
            }
        }
        let object: [String: Any] = [
            "model": [
                "parameter_groups": [
                    ["compute_dtype": "bfloat16", "tensors": ["model.llm.embed.weight"]],
                    ["compute_dtype": "float32", "tensors": ["model.llm.attn_res.output_query"]],
                    ["compute_dtype": "nvfp4", "tensors": nvfp4],
                    ["compute_dtype": "mxfp8", "tensors": mxfp8],
                ]
            ],
            "quant_method": "inkling2",
        ]
        return object
    }

    private func quant() throws -> Data {
        try JSONSerialization.data(withJSONObject: try quantObject())
    }

    private func mutatedQuant(
        _ mutate: (inout [[String: Any]]) -> Void
    ) throws -> Data {
        var object = try quantObject()
        var model = try #require(object["model"] as? [String: Any])
        var groups = try #require(model["parameter_groups"] as? [[String: Any]])
        mutate(&groups)
        model["parameter_groups"] = groups
        object["model"] = model
        return try JSONSerialization.data(withJSONObject: object)
    }

    @Test func readsTheTwentyFourLayerSplit() throws {
        let contract = try Inkling2Contract.parse(config: config(), quant: try quant())
        #expect(contract.layers == 24)
        #expect(contract.routedExperts == 128)
        #expect(contract.expertsPerToken == 8)
        #expect(contract.contextTokens == 1_048_576)
        #expect(contract.perLayer[0].w13 == .nvfp4)
        #expect(contract.perLayer[19].w2 == .nvfp4)
        #expect(contract.perLayer[20].w13 == .mxfp8)
        #expect(contract.perLayer[23].w2 == .mxfp8)
    }

    @Test func refusesAWrongFormatForALayer() throws {
        let broken = try mutatedQuant { groups in
            var nvfp4 = groups[2]["tensors"] as? [String] ?? []
            nvfp4.append("model.llm.layers.20.mlp.experts.w13_weight")
            groups[2]["tensors"] = nvfp4
        }
        #expect(throws: Inkling2Contract.Error.self) {
            try Inkling2Contract.parse(config: config(), quant: broken)
        }
    }

    @Test func refusesADuplicateTensor() throws {
        let broken = try mutatedQuant { groups in
            var nvfp4 = groups[2]["tensors"] as? [String] ?? []
            nvfp4.append("model.llm.embed.weight")
            groups[2]["tensors"] = nvfp4
        }
        #expect(throws: Inkling2Contract.Error.self) {
            try Inkling2Contract.parse(config: config(), quant: broken)
        }
    }

    @Test func refusesAMissingLayer() throws {
        let broken = try mutatedQuant { groups in
            var nvfp4 = groups[2]["tensors"] as? [String] ?? []
            nvfp4.removeAll { $0 == "model.llm.layers.7.mlp.experts.w13_weight" }
            groups[2]["tensors"] = nvfp4
        }
        #expect(throws: Inkling2Contract.Error.self) {
            try Inkling2Contract.parse(config: config(), quant: broken)
        }
    }
}
