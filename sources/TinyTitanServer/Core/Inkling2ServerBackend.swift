import Foundation
import TinyTitanCore
import TinyTitanInkling2
import TinyTitanLib

/// Single-model server backend for the text-only Inkling 2 Mini directory.
package actor Inkling2ServerBackend: ServerInferenceBackend {
    private let model: Inkling2TextModel
    private let tokenizer: Inkling2Tokenizer
    package nonisolated let maximumContext: Int
    package nonisolated let samplingDefaults: SamplingProfile = .house

    package init(directory: URL) throws {
        let config = try Inkling2Config.load(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        self.model = try Inkling2TextModel(
            directory: directory, config: config, store: store)
        self.tokenizer = try Inkling2Tokenizer(directory: directory)
        self.maximumContext = config.text.contextTokens
    }

    package static func isInkling2(_ directory: URL) -> Bool {
        guard
            let data = try? BoundedMetadataRead.read(
                fileAt: directory.appendingPathComponent("config.json"),
                maxBytes: 1 << 20),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = root["model_type"] as? String
        else { return false }
        return type.hasPrefix("inkling2")
    }

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let prompt = render(request.messages)
        let promptIDs = try tokenizer.encode(prompt)
        var generator = Inkling2Generator(model: model, tokenizer: tokenizer)
        let maximum = min(max(request.generationConfig.maxNewTokens, 1), 64)
        onEvent(.promptProcessed(tokens: promptIDs.count, cachedTokens: 0))
        let result = try generator.generate(prompt: prompt, maxTokens: maximum) { delta in
            onEvent(.content(delta))
        }
        let usage = OpenAIUsage(
            promptTokens: promptIDs.count,
            completionTokens: result.tokens.count,
            totalTokens: promptIDs.count + result.tokens.count)
        return ServerCompletion(
            content: result.text,
            toolCalls: [],
            finishReason: "stop",
            usage: usage)
    }

    package func tokenizeRawPrompt(_ text: String) async throws -> [Int32] {
        try tokenizer.encode(text).map(Int32.init)
    }

    private func render(_ messages: [GFTokenizer.Message]) -> String {
        var out = ""
        for message in messages {
            let body = message.content ?? ""
            switch message.role {
            case .system, .developer:
                out += "<|message_system|><|content_text|>\(body)<|end_message|>"
            case .user:
                out += "<|message_user|><|content_text|>\(body)<|end_message|>"
            case .assistant:
                out += "<|message_model|><|content_text|>\(body)<|end_message|>"
            case .tool:
                out += "<|message_tool|><|content_text|>\(body)<|end_message|>"
            }
        }
        return out + "<|message_model|><|content_text|>"
    }
}
