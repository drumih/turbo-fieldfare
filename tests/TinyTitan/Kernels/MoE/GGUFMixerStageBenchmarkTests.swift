import Foundation
import Metal
import Testing

@testable import TinyTitan

/// GPU time of each non-expert stage of one decode layer, on the decoder's own
/// resident weights: where the attention side's milliseconds go. Opt-in
/// (`TINYTITAN_KERNEL_BENCH=1` plus a model); it asserts every stage ran.
@Suite struct GGUFMixerStageBenchmarkTests {
    private static var enabled: Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment["TINYTITAN_GGUF_MODEL"] != nil
            && environment["TINYTITAN_KERNEL_BENCH"] == "1"
    }

    @Test(.enabled(if: enabled))
    func stageGPUTimes() throws {
        let path = try #require(ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"])
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let decoder = try GGUFDecoder(
            context: context, arch: .qwen38FlashNext, model: model, maxSeq: 64,
            ramBudgetBytes: 10 << 30)
        let repeats = 20
        func time(_ label: String, _ body: (MTLCommandBuffer) throws -> Void) throws -> Double {
            // One warm-up pass, then the timed buffer.
            for pass in 0..<2 {
                guard let commandBuffer = context.queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                for _ in 0..<(pass == 0 ? 1 : repeats) { try body(commandBuffer) }
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                #expect(commandBuffer.error == nil, "\(label)")
                if pass == 1 {
                    let seconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
                    let ms = seconds / Double(repeats) * 1000
                    print(String(format: "%-28@ %.3f ms", label as NSString, ms))
                    return ms
                }
            }
            return 0
        }
        var total = 0.0
        for (layer, kind) in [(0, "gdn"), (3, "full")] {
            total += try time("L\(layer) hc read attn") { cb in
                try decoder.hc.encodeRead(
                    commandBuffer: cb, layer: layer, sublayer: "attn",
                    streamsBuffer: decoder.streams, out: decoder.blockInput)
            }
            total += try time("L\(layer) \(kind) mixer") { cb in
                if kind == "gdn" {
                    try decoder.gdn.encode(
                        commandBuffer: cb, layer: layer, x: decoder.blockInput,
                        out: decoder.attnOut)
                } else {
                    try decoder.fullAttention.encode(
                        commandBuffer: cb, layer: layer, position: 32, x: decoder.blockInput,
                        out: decoder.attnOut)
                }
            }
            total += try time("L\(layer) hc write attn") { cb in
                try decoder.hc.encodeWrite(
                    commandBuffer: cb, layer: layer, sublayer: "attn",
                    streamsBuffer: decoder.streams, blockOut: decoder.attnOut)
            }
            total += try time("L\(layer) router") { cb in
                try decoder.router.encode(commandBuffer: cb, layer: layer, x: decoder.blockInput)
            }
            total += try time("L\(layer) shared expert") { cb in
                try decoder.shared.encode(
                    commandBuffer: cb, layer: layer, x: decoder.blockInput, out: decoder.ffnOut)
            }
        }
        _ = try time("output head") { cb in
            try decoder.output.encode(commandBuffer: cb, streamsBuffer: decoder.streams)
        }
        #expect(total > 0)
    }
}
