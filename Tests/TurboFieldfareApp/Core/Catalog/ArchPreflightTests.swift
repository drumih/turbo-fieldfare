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

    /// Verbatim from Qwen/Qwen3.6-35B-A3B's published config.json — the model
    /// type really is `qwen3_5_moe`, not a `3_6` variant.
    private let qwen36Config = Data("""
    {
      "model_type": "qwen3_5_moe",
      "architectures": ["Qwen3_5MoeForConditionalGeneration"],
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
            == .unsupported(modelType: "qwen3_5_moe"))
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
