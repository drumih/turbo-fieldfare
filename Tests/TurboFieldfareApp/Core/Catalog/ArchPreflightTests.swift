import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ArchPreflightTests {
    private let gemmaConfig = Data("""
    {
      "model_type": "gemma4",
      "architectures": ["Gemma4ForConditionalGeneration"],
      "text_config": { "model_type": "gemma4_text", "num_hidden_layers": 30 }
    }
    """.utf8)

    private let qwenNextConfig = Data("""
    {
      "model_type": "qwen3_next",
      "architectures": ["Qwen3NextForCausalLM"],
      "num_hidden_layers": 48
    }
    """.utf8)

    private let qwen36Config = Data("""
    {
      "model_type": "qwen3_6_moe",
      "architectures": ["Qwen36MoeForConditionalGeneration"],
      "num_hidden_layers": 40
    }
    """.utf8)

    @Test func acceptsTheSupportedGemmaArchitecture() throws {
        #expect(try ArchPreflight.evaluate(configJSON: gemmaConfig) == .supported(modelType: "gemma4"))
    }

    @Test func rejectsQwenNext() throws {
        #expect(try ArchPreflight.evaluate(configJSON: qwenNextConfig)
            == .unsupported(modelType: "qwen3_next"))
    }

    @Test func rejectsQwen36() throws {
        #expect(try ArchPreflight.evaluate(configJSON: qwen36Config)
            == .unsupported(modelType: "qwen3_6_moe"))
    }

    @Test func fallsBackToArchitecturesWhenModelTypeIsAbsent() throws {
        let config = Data("""
        { "architectures": ["Gemma4ForConditionalGeneration"] }
        """.utf8)
        #expect(try ArchPreflight.evaluate(configJSON: config)
            == .supported(modelType: "Gemma4ForConditionalGeneration"))
    }

    @Test func throwsWhenNeitherFieldIsPresent() {
        let config = Data("{ \"num_hidden_layers\": 30 }".utf8)
        #expect(throws: ArchPreflight.MalformedConfig.self) {
            try ArchPreflight.evaluate(configJSON: config)
        }
    }

    @Test func throwsOnUndecodableJSON() {
        #expect(throws: (any Error).self) {
            try ArchPreflight.evaluate(configJSON: Data("{ not json".utf8))
        }
    }
}
