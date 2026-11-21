import Foundation

/// Greedy continuation over the cached text model.
public struct Inkling2Generator {
    private let model: Inkling2TextModel
    private let tokenizer: Inkling2Tokenizer
    private var caches: [Inkling2DecoderCache]

    public init(model: Inkling2TextModel, tokenizer: Inkling2Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
        self.caches = model.makeCaches()
    }

    public mutating func generate(
        prompt: String, maxTokens: Int
    ) throws -> (tokens: [Int], text: String) {
        var promptIds = try tokenizer.encode(prompt)
        guard !promptIds.isEmpty else { return ([], "") }
        for id in promptIds.dropLast() {
            _ = try autoreleasepool {
                try model.forward(token: id, caches: &caches)
            }
        }
        var next = promptIds.removeLast()
        var generated: [Int] = []
        let stops = Set(
            [
                try tokenizer.encode("<|end_message|>").first,
                try tokenizer.encode("<|content_model_end_sampling|>").first,
            ]
            .compactMap { $0 })
        for _ in 0..<maxTokens {
            let logits = try autoreleasepool {
                try model.forward(token: next, caches: &caches)
            }
            next = Inkling2TextModel.greedyToken(logits)
            if stops.contains(next) { break }
            generated.append(next)
        }
        return (generated, tokenizer.decode(generated))
    }
}
