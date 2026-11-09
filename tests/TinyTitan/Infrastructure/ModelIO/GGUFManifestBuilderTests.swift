import Foundation
import TinyTitanFormat
import Testing

@testable import TinyTitan

@Suite struct GGUFManifestBuilderTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    /// File entries describe bytes that do not exist yet, so the test supplies
    /// them: this is about the manifest's architecture, not about hashing.
    private static func placeholderFiles() -> [String: SSDAIManifestFileV1] {
        let entry = SSDAIManifestFileV1(
            size: 1, sha256: String(repeating: "0", count: 64))
        var files = [
            "model_weights.bin": entry,
            "packed_experts/layout.json": entry,
        ]
        for layer in 0..<48 {
            files[String(format: "packed_experts/layer_%02d.bin", layer)] = entry
        }
        return files
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func buildsAManifestTheRuntimeAccepts() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let wire = try GGUFManifestBuilder.manifest(
            from: model,
            arch: .qwen38FlashNext,
            modelID: "qwen3.8-flash-next-gguf",
            files: Self.placeholderFiles(),
            expertStride: 2_768_896)

        let data = try JSONEncoder().encode(wire)
        let manifest = try ManifestReader.decode(
            data: data, expecting: .qwen38FlashNext)
        #expect(manifest.numLayers == 48)
        #expect(manifest.expertsPerLayer == 512)
        #expect(manifest.expertStride == 2_768_896)
        #expect(manifest.arch.hiddenSize == 2_560)
        #expect(manifest.arch.moeIntermediateSize == 640)
        #expect(manifest.arch.topKExperts == 10)
        #expect(manifest.quant?.routedExpert.weightBits == 4)
        #expect(manifest.quant?.routedExpert.groupSize == 64)
        #expect(manifest.quant?.routedExpert.scheme == "affine")
        #expect(manifest.arch.hcCount == 4)
        #expect(manifest.arch.indexerBudget == 2_048)
        #expect(manifest.arch.pleNgramSize == 3)
    }

    /// The cross-check is the point of the builder: a file whose geometry is
    /// not the preset's must be refused, naming the field.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func refusesAFileWhoseGeometryDisagreesWithTheRuntime() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        var metadata = model.metadata
        metadata["qwen4exp.block_count"] = .uint32(47)
        let broken = GGUFModel(
            shards: model.shards, tensors: model.tensors, metadata: metadata)

        #expect(
            throws: GGUFManifestError.mismatch(
                field: "numLayers", file: "47", preset: "48")
        ) {
            _ = try GGUFManifestBuilder.manifest(
                from: broken, arch: .qwen38FlashNext, modelID: "broken",
                files: Self.placeholderFiles(), expertStride: 2_768_896)
        }
    }
}
