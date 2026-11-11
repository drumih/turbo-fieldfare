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

        func faults() -> (minor: Int64, major: Int64) {
            var usage = rusage()
            getrusage(RUSAGE_SELF, &usage)
            return (Int64(usage.ru_minflt), Int64(usage.ru_majflt))
        }
        var timings: [Double] = []
        var tokenLogits: [[Float]] = []
        let tokens: [Int32] = [100, 200, 300, 400, 500, 600]
        for position in tokens.indices {
            let before = faults()
            let start = DispatchTime.now().uptimeNanoseconds
            let logits = try decoder.decode(sequence: tokens, position: position)
            timings.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
            let after = faults()
            print(
                "  faults on token \(position): minor \(after.minor - before.minor), "
                    + "major \(after.major - before.major)")
            tokenLogits.append(logits)
        }
        for (index, seconds) in timings.enumerated() {
            print("  token \(index): \(String(format: "%.0f", seconds * 1000)) ms")
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
        // Routing overlap between tokens: a larger expert cache only pays if
        // the same (layer, expert) pairs recur.
        var routings: [Set<Int>] = []
        // Re-run the first two tokens through the decoder is not possible
        // without disturbing state, so this reports the last token's routing
        // count only; the pair study lives in its own test.
        routings.append(Set(decoder.lastRouting.flatMap { $0 }))
        print("last token routed \(routings[0].count) distinct experts across 48 layers")
        #expect(tokenLogits[0] != tokenLogits[1])
        let total = timings.reduce(0, +)
        let perToken = total / Double(timings.count)
        // The first token pays the cold expert reads; the steady state is what
        // a long run would see.
        let warm = timings.dropFirst().reduce(0, +) / Double(max(timings.count - 1, 1))
        print(
            "steady state: \(String(format: "%.0f", warm * 1000)) ms/token -> "
                + "\(String(format: "%.2f", 1 / warm)) tok/s")
        print(
            "decode: \(String(format: "%.1f", perToken * 1000)) ms/token -> "
                + "\(String(format: "%.2f", 1 / perToken)) tok/s (with a host "
                + "router readback per layer)")
        #expect(perToken > 0)
    }
}
