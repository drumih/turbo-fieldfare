import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The prefill loop over the as-is decoder: a prompt walks through the
/// token-at-a-time path and produces the last token's logits. This is the GPU
/// reference the ANE sidecar has to beat, so it also reports the wall time.
@Suite struct GGUFDecoderPrefillTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func prefillsAPromptAndReportsTheWallTime() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let decoder = try GGUFDecoder(
            context: context, arch: .qwen38FlashNext, model: model, maxSeq: 32)
        let prompt: [Int32] = [100, 200, 300, 400, 500, 600, 700, 800]
        let measurement = try decoder.prefill(tokens: prompt)

        #expect(measurement.tokenCount == prompt.count)
        #expect(measurement.logits.count == ArchConfig.qwen38FlashNext.vocabSize)
        #expect(measurement.logits.allSatisfy { $0.isFinite })
        #expect(measurement.seconds > 0)
        print(
            "GGUF prefill (GPU): \(measurement.tokenCount) tokens in "
                + "\(String(format: "%.2f", measurement.seconds)) s -> "
                + "\(String(format: "%.2f", measurement.tokensPerSecond)) tok/s "
                + "(\(String(format: "%.0f", measurement.seconds * 1000 / Double(measurement.tokenCount))) ms/token)"
        )
    }
}
