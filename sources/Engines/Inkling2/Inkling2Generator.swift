import Foundation

/// Greedy continuation over the cached text model.
public struct Inkling2Generator {
    private let model: Inkling2TextModel
    private let tokenizer: Inkling2Tokenizer
    private var caches: [Inkling2DecoderCache]
    public private(set) var lastPrefillSeconds: Double = 0
    public private(set) var lastDecodeSeconds: Double = 0

    public init(model: Inkling2TextModel, tokenizer: Inkling2Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
        self.caches = model.makeCaches()
    }

    /// Greedy decode of up to `maxTokens` tokens. `onDelta` receives the
    /// answer's text as it decodes (TT-128: every route streams), never the
    /// whole answer at the end; the returned text is the same concatenation.
    public mutating func generate(
        prompt: String, maxTokens: Int, onDelta: (String) -> Void = { _ in }
    ) throws -> (tokens: [Int], text: String) {
        var promptIds = try tokenizer.encode(prompt)
        guard !promptIds.isEmpty else { return ([], "") }
        let prefillStart = Date()
        for id in promptIds.dropLast() {
            _ = try autoreleasepool {
                try model.forward(token: id, caches: &caches)
            }
        }
        self.lastPrefillSeconds = Date().timeIntervalSince(prefillStart)
        var next = promptIds.removeLast()
        var generated: [Int] = []
        let decodeStart = Date()
        let stops = Set(
            [
                try tokenizer.encode("<|end_message|>").first,
                try tokenizer.encode("<|content_model_end_sampling|>").first,
            ]
            .compactMap { $0 })
        var stream = Inkling2TextDelta()
        for _ in 0..<maxTokens {
            let logits = try autoreleasepool {
                try model.forward(token: next, caches: &caches)
            }
            next = Inkling2TextModel.greedyToken(logits)
            if stops.contains(next) { break }
            generated.append(next)
            let delta = stream.advance(to: tokenizer.decode(generated))
            if !delta.isEmpty { onDelta(delta) }
        }
        self.lastDecodeSeconds = Date().timeIntervalSince(decodeStart)
        let text = tokenizer.decode(generated)
        let tail = stream.finish(text)
        if !tail.isEmpty { onDelta(tail) }
        return (generated, text)
    }
}
