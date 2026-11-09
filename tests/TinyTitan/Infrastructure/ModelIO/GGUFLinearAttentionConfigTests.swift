import Foundation
import Testing

@testable import TinyTitan

/// The GDN geometry must come from the file, and the runtime preset is wrong
/// for this checkpoint - both facts are pinned here.
@Suite struct GGUFLinearAttentionConfigTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func derivesTheCheckpointsGDNGeometry() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let config = try GGUFLinearAttentionConfig.read(from: model)
        #expect(config.numKHeads == 16)
        #expect(config.numVHeads == 48)
        #expect(config.keyHeadDim == 128)
        #expect(config.valueHeadDim == 128)
        #expect(config.convKernelSize == 4)
        #expect(config.qkvDim == 10_240)
        #expect(config.valueDim == 6_144)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func theRuntimePresetAgreesWithTheFile() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let config = try GGUFLinearAttentionConfig.read(from: model)
        let preset = ArchConfig.qwen38FlashNext.linearAttention
        #expect(
            preset
                == LinearAttentionConfig(
                    numKHeads: config.numKHeads, numVHeads: config.numVHeads,
                    keyHeadDim: config.keyHeadDim, valueHeadDim: config.valueHeadDim,
                    convKernelSize: config.convKernelSize, outputGate: .sigmoid))
    }
}
