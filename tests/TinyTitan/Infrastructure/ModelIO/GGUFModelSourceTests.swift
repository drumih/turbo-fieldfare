import Foundation
import Testing

@testable import TinyTitan

/// The loader's data layer: the architecture the file declares, the resident
/// set the arena will decode, and per-expert access - all on the real install.
@Suite struct GGUFModelSourceTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func describesTheInstalledCheckpoint() throws {
        let path = try #require(Self.realModelPath)
        let source = try GGUFModelSource.open(
            directoryURL: URL(fileURLWithPath: path))

        #expect(source.arch == .qwen38FlashNext)
        #expect(source.layerCount == 48)
        #expect(source.expertsPerLayer == 512)
        // 1,080 role-mapped tensors, minus the PLE table which stays in blocks.
        #expect(source.residentNames.count == 1_079)
        #expect(source.residentNames.allSatisfy { !$0.hasSuffix("_exps.weight") })
        #expect(!source.residentNames.contains("per_layer_token_embd.weight"))

        // The expert formats this checkpoint actually stores.
        #expect(try source.expert(layer: 0, expert: 0).gate.type == .iq2S)
        #expect(try source.expert(layer: 8, expert: 0).down.type == .q2_0)
        #expect(try source.expert(layer: 14, expert: 0).gate.type == .iq3XXS)
        #expect(try source.expert(layer: 47, expert: 0).gate.type == .iq4XS)
        #expect(try source.expert(layer: 0, expert: 511).gate.byteCount == 524_800)

        // The PLE table is addressed, not decoded: 160-wide IQ4_NL rows.
        let ple = source.pleTableRange
        #expect(ple.rowBytes == 90)
        #expect(ple.byteCount == 28_800_138_240)
        #expect(ple.fileOffset > 0)
    }

    /// Geometry that disagrees with the runtime is refused by field, before any
    /// buffer is allocated.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func refusesACheckpointThatIsNotThisArchitecture() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        var metadata = model.metadata
        metadata["qwen4exp.block_count"] = .uint32(47)
        let broken = GGUFModel(shards: model.shards, tensors: model.tensors, metadata: metadata)
        #expect(
            throws: GGUFManifestError.mismatch(
                field: "numLayers", file: "47", preset: "48")
        ) {
            try GGUFManifestBuilder.crossCheck(broken, preset: .qwen38FlashNext)
        }
    }
}
