import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2QualityTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func answersShortFactualPrompts() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let config = try Inkling2Config.load(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let tokenizer = try Inkling2Tokenizer(directory: directory)
        let model = try Inkling2TextModel(
            directory: directory, config: config, store: store)
        let cases: [(String, [String])] = [
            ("The capital of France is", ["paris"]),
            ("The chemical symbol for water is", ["h2o", "h₂o"]),
            ("Two plus two equals", ["4", "four"]),
        ]
        for (question, expected) in cases {
            var generator = Inkling2Generator(model: model, tokenizer: tokenizer)
            let prompt =
                "<|message_user|><|content_text|>\(question)<|end_message|>"
                + "<|message_model|><|content_text|>"
            let result = try generator.generate(prompt: prompt, maxTokens: 12)
            print("Q=\(question) A=\(result.text)")
            #expect(
                expected.contains { result.text.lowercased().contains($0) })
        }
    }
}
