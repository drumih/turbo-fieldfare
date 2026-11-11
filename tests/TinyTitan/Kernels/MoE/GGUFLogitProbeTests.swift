import Foundation
import Testing

@testable import TinyTitan

/// The as-is decoder against llama.cpp on the same GGUF (model-gated).
///
/// Reference: llama.cpp `7c2653a`, `llama-debug -ngl 0 -t 8 --temp 0`, CPU
/// backend, prompt "The capital of France is" (ids 760 6511 314 9338 369),
/// `result_output` of the last prompt token. Its top five ids and logits are
/// pinned below. The engines differ inside the noise of their arithmetic (llama
/// quantises activations to 8 bits on the CPU; this decoder keeps them in
/// half), so the check is the decision, not the bits: the same top-1, the same
/// top-5 set, and every top-5 logit within 1.0 of the reference.
@Suite struct GGUFLogitProbeTests {
    private static let modelPath = ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]

    private static let referenceTop5: [(id: Int, logit: Float)] = [
        (11751, 16.21), (271, 14.271), (198, 13.392), (264, 13.008), (1076, 12.793),
    ]

    @Test(.enabled(if: modelPath != nil))
    func lastPromptTokenAgreesWithLlamaCpp() async throws {
        let path = try #require(Self.modelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-logit-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let sidecar = temporary.appendingPathComponent("tokenizer")
        try GGUFTokenizerSidecar.write(from: model, to: sidecar)
        let tokenizer = try await GFTokenizer.load(from: sidecar)
        let promptIDs = tokenizer.encode("The capital of France is", addBOS: false)
        #expect(promptIDs == [760, 6511, 314, 9338, 369])

        let decoder = try GGUFDecoder(
            context: try MetalContext(), arch: .qwen38FlashNext, model: model, maxSeq: 64)
        let logits = try decoder.prefill(tokens: promptIDs).logits
        #expect(logits.allSatisfy { $0.isFinite })
        let ranked = logits.indices.sorted { logits[$0] > logits[$1] }
        #expect(ranked[0] == Self.referenceTop5[0].id, "top-1 \(ranked[0])")
        #expect(Set(ranked.prefix(5)) == Set(Self.referenceTop5.map(\.id)))
        for reference in Self.referenceTop5 {
            let difference = abs(logits[reference.id] - reference.logit)
            #expect(difference < 1.0, "id \(reference.id): \(logits[reference.id])")
        }
    }
}
