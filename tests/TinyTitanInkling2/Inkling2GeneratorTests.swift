import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2GeneratorTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func generatesAContinuation() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let config = try Inkling2Config.load(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let tokenizer = try Inkling2Tokenizer(directory: directory)
        let model = try Inkling2TextModel(
            directory: directory, config: config, store: store)
        var generator = Inkling2Generator(model: model, tokenizer: tokenizer)
        let prompt =
            "<|message_user|><|content_text|>The capital of France is"
            + "<|end_message|><|message_model|><|content_text|>"
        let result = try generator.generate(prompt: prompt, maxTokens: 8)
        print("generated=\(result.tokens) text=\(result.text)")
        #expect(!result.tokens.isEmpty)
        #expect(result.tokens.count <= 8)
    }
}
