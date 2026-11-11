import Foundation
import Testing

@testable import TinyTitan

/// The generation loop the CLI's GGUF-as-is branch runs: sidecar tokenizer,
/// token-at-a-time prefill, then greedy decode with the argmax fed back. This
/// is the first test that generates from the model rather than feeding fixed
/// ids, and it pins the loop's contract: the same prompt decodes to the same
/// tokens twice (fresh decoder, fresh state), every id is in vocabulary, and
/// the streamed text decodes to something non-empty.
@Suite struct GGUFGenerationLoopTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    /// One greedy generation, mirroring `runGGUFAsIs` step for step.
    private static func generate(
        modelPath: String, tokenizer: GFTokenizer, prompt: String, maxNew: Int
    ) throws -> (tokens: [Int32], text: String, prefillSeconds: Double, decodeSeconds: Double) {
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: modelPath))
        let context = try MetalContext()
        let decoder = try GGUFDecoder(
            context: context, arch: .qwen38FlashNext, model: model, maxSeq: 2_048)
        let promptIDs = tokenizer.encode(prompt, addBOS: false)
        let prefill = try decoder.prefill(tokens: promptIDs)
        let sampler = CPUSampler()
        let generator = sampler.makeGenerator()

        var generated: [Int32] = []
        var logits = prefill.logits
        var decodeSeconds = 0.0
        for index in 0..<maxNew {
            let picked = sampler.pick(logits, history: promptIDs + generated, using: generator)
            let token = Int32(picked)
            if tokenizer.stopTokenIDs.contains(token) { break }
            generated.append(token)
            let stepStart = DispatchTime.now().uptimeNanoseconds
            logits = try decoder.decode(
                sequence: promptIDs + generated, position: promptIDs.count + index)
            decodeSeconds += Double(DispatchTime.now().uptimeNanoseconds - stepStart) / 1e9
        }
        return (
            generated, tokenizer.decode(generated, skipSpecialTokens: true),
            prefill.seconds, decodeSeconds
        )
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func greedyGenerationIsDeterministicAndProducesText() async throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-generate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let sidecarFolder = temporary.appendingPathComponent("tokenizer")
        try GGUFTokenizerSidecar.write(from: model, to: sidecarFolder)
        let tokenizer = try await GFTokenizer.load(from: sidecarFolder)

        let first = try Self.generate(
            modelPath: path, tokenizer: tokenizer,
            prompt: "The capital of France is", maxNew: 8)
        #expect(!first.tokens.isEmpty)
        #expect(first.tokens.allSatisfy { $0 >= 0 && $0 < 248_320 })
        #expect(!first.text.isEmpty)

        // A fresh decoder carries fresh KV/GDN/PLE state, so the second run of
        // the same greedy prompt must land on the same tokens; anything else is
        // state leaking between runs.
        let second = try Self.generate(
            modelPath: path, tokenizer: tokenizer,
            prompt: "The capital of France is", maxNew: 8)
        #expect(second.tokens == first.tokens)
    }
}
