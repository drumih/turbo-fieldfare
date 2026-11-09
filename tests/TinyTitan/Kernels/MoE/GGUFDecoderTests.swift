import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The whole as-is forward pass on the real checkpoint: embedding, 48 layers
/// (GDN or full attention, hyper-connections, routed plus shared MoE, PLE) and
/// the output mixer plus head. This is the first end-to-end decode in the tree.
@Suite struct GGUFDecoderTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func decodesTwoTokensEndToEnd() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let built = DispatchTime.now().uptimeNanoseconds
        let decoder = try GGUFDecoder(
            context: context, arch: arch, model: model, maxSeq: 64)
        let loadSeconds = Double(DispatchTime.now().uptimeNanoseconds - built) / 1e9
        print("decoder load: \(String(format: "%.1f", loadSeconds)) s")

        var timings: [Double] = []
        var tokenLogits: [[Float]] = []
        for (position, token) in [Int32(100), Int32(200)].enumerated() {
            var history: [Int32] = [token]
            if position > 0 { history.append(100) }
            let start = DispatchTime.now().uptimeNanoseconds
            let logits = try decoder.decode(
                token: token, position: position, history: history)
            timings.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
            tokenLogits.append(logits)
        }
        for (index, logits) in tokenLogits.enumerated() {
            #expect(logits.count == arch.vocabSize)
            let nonFinite = logits.filter { !$0.isFinite }.count
            let maximum = logits.max() ?? -.infinity
            let argmax = logits.firstIndex(of: maximum) ?? -1
            print(
                "token \(index): \(nonFinite) non-finite, max \(maximum) at \(argmax)")
            #expect(nonFinite == 0)
            #expect(argmax >= 0 && argmax < arch.vocabSize)
        }
        #expect(tokenLogits[0] != tokenLogits[1])
        let total = timings.reduce(0, +)
        let perToken = total / Double(timings.count)
        print(
            "decode: \(String(format: "%.1f", perToken * 1000)) ms/token -> "
                + "\(String(format: "%.2f", 1 / perToken)) tok/s (with a host "
                + "router readback per layer)")
        #expect(perToken > 0)
    }
}
