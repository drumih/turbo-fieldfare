import Foundation
import Metal
import Testing
import TinyTitanCore

@testable import TinyTitan

extension ModelLoaderTests {
    @Test func loadsValidDirectory() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: dir, device: device,
            expecting: .qwenToy())
        let embed = try model.embedding()
        // Embedding is int4-packed: 2 values per byte, so length = vocabSize * hiddenSize / 2
        #expect(embed.length == UInt64(1024 * 64 / 2))
        #expect(embed.shape.0 == 1024 && embed.shape.1 == 64)
        let norm = try model.finalNorm()
        #expect(norm.length == UInt64(64 * 2))
        // Qwen carries a separate untied lm_head.
        let lmHead = try model.lmHead()
        #expect(lmHead.offset != embed.offset)
        #expect(lmHead.length == embed.length)
    }

    @Test func residentBytesAreReadableFromBuffer() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: dir, device: device,
            expecting: .qwenToy())
        let norm = try model.finalNorm()
        let contents = norm.buffer.contents()
        // Norm region was patterned 0xC0 | (i & 0x3F).
        for i in 0..<Int(norm.length) {
            let got = contents.load(fromByteOffset: Int(norm.offset) + i, as: UInt8.self)
            #expect(got == UInt8(0xC0 | (i & 0x3F)), "norm byte \(i)")
        }
    }

    @Test func missingManifestFailsPartialInstall() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.removeItem(at: dir.appendingPathComponent("manifest.json"))
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect {
            _ = try Model.load(
                directoryURL: dir, device: device,
                expecting: .qwenToy())
        } throws: { error in
            if case ModelError.partialInstall = error { return true }
            return false
        }
    }

    @Test func mismatchedShaFailsChecksumMismatch() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        // Flip one byte at the very end of the resident region (not inside
        // the index, which the loader reads earlier and would error
        // differently). Manifest sha was computed before this corruption.
        let url = dir.appendingPathComponent("model_weights.bin")
        var data = try Data(contentsOf: url)
        data[data.count - 1] ^= 0xFF
        try data.write(to: url)
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect {
            _ = try Model.load(
                directoryURL: dir, device: device,
                expecting: .qwenToy())
        } throws: { error in
            if case ModelError.checksumMismatch = error { return true }
            return false
        }
    }

    @Test func integrityPoliciesExposeIdenticalResidentAndRoutedBytes() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeVerifiedInstallReceipt(directoryURL: dir)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let full = try Model.load(
            directoryURL: dir,
            device: device,
            expecting: .qwenToy(),
            integrityPolicy: .fullSha256)
        let trusted = try Model.load(
            directoryURL: dir,
            device: device,
            expecting: .qwenToy(),
            integrityPolicy: .sizeCheckTrustedReceipt)

        let fullEmbedding = try full.embedding()
        let trustedEmbedding = try trusted.embedding()
        #expect(fullEmbedding.length == trustedEmbedding.length)
        let fullEmbeddingBytes = fullEmbedding.buffer.contents().advanced(
            by: Int(fullEmbedding.offset))
        let trustedEmbeddingBytes = trustedEmbedding.buffer.contents().advanced(
            by: Int(trustedEmbedding.offset))
        #expect(memcmp(fullEmbeddingBytes, trustedEmbeddingBytes, Int(fullEmbedding.length)) == 0)

        let fullExpert = try full.routedExpert(layer: 0, expert: 0)
        let trustedExpert = try trusted.routedExpert(layer: 0, expert: 0)
        #expect(fullExpert.length == trustedExpert.length)
        let fullExpertBytes = fullExpert.buffer.contents().advanced(by: Int(fullExpert.offset))
        let trustedExpertBytes = trustedExpert.buffer.contents().advanced(
            by: Int(trustedExpert.offset))
        #expect(memcmp(fullExpertBytes, trustedExpertBytes, Int(fullExpert.length)) == 0)
    }

    /// Rewrite one expert's record inside the toy install's `layout.json` and
    /// re-pin the manifest entry for that file, so the loader's size and SHA
    /// checks pass and the schema cross-check is what refuses the install.
    ///
    /// The caller edits a single expert, always one *after* the layer's first:
    /// the reference record stays exactly as the writer produced it, so these
    /// cases only fail if the load path compares every expert rather than
    /// trusting the one it validates against the architecture.
    static func repinToyLayoutExpert(
        directoryURL dir: URL,
        layer: Int,
        expert index: Int,
        mutate: (inout [String: Any]) throws -> Void
    ) throws {
        let layoutURL = dir.appendingPathComponent("packed_experts/layout.json")
        var root = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: layoutURL)) as? [String: Any])
        var layers = try #require(root["layers"] as? [[String: Any]])
        var experts = try #require(layers[layer]["experts"] as? [[String: Any]])
        try mutate(&experts[index])
        layers[layer]["experts"] = experts
        root["layers"] = layers
        let layoutData = try JSONSerialization.data(
            withJSONObject: root, options: [.sortedKeys])
        try layoutData.write(to: layoutURL)

        let manifestURL = dir.appendingPathComponent("manifest.json")
        var manifest = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: manifestURL)) as? [String: Any])
        var files = try #require(manifest["files"] as? [String: [String: Any]])
        var entry = try #require(files["packed_experts/layout.json"])
        entry["size"] = layoutData.count
        entry["sha256"] = Sha256Verifier.hashData(layoutData)
        files["packed_experts/layout.json"] = entry
        manifest["files"] = files
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: manifestURL)
    }

    /// Read one expert's `tensors` map out of its record.
    static func expertTensors(_ expert: inout [String: Any]) throws -> [String: [String: Any]] {
        try #require(expert["tensors"] as? [String: [String: Any]])
    }

    private static func rejectsLaterExpert(
        _ dir: URL, _ phrase: String
    ) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect {
            _ = try Model.load(
                directoryURL: dir, device: device,
                expecting: .qwenToy())
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains(phrase)
            }
            return false
        }
    }

    @Test func laterExpertWidthLieFailsAtLoad() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.repinToyLayoutExpert(directoryURL: dir, layer: 0, expert: 7) { expert in
            var tensors = try Self.expertTensors(&expert)
            tensors["gate"]?["bits"] = 8
            expert["tensors"] = tensors
        }
        // A lying width is caught by the per-tensor extent check: the bytes no
        // longer equal shape * width / 8 for the width it declares.
        try Self.rejectsLaterExpert(dir, "does not match the required schema")
    }

    @Test func laterExpertShapeLieFailsAtLoad() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        // gate is [moeIntermediate, hidden] = [128, 64]; transposed it is the
        // same byte count, so only the role records distinguish it.
        try Self.repinToyLayoutExpert(directoryURL: dir, layer: 2, expert: 3) { expert in
            var tensors = try Self.expertTensors(&expert)
            tensors["up"]?["shape"] = [64, 128]
            expert["tensors"] = tensors
        }
        try Self.rejectsLaterExpert(dir, "does not match the required schema")
    }

    @Test func laterExpertMissingRoleFailsAtLoad() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.repinToyLayoutExpert(directoryURL: dir, layer: 1, expert: 5) { expert in
            var tensors = try Self.expertTensors(&expert)
            tensors.removeValue(forKey: "down_biases")
            expert["tensors"] = tensors
        }
        // The refusal names the expert now that experts are validated one by
        // one rather than compared with expert 0.
        try Self.rejectsLaterExpert(dir, "is missing role down_biases")
    }

    @Test func validToyInstallStillLoadsAfterTheExpertMutationHelper() throws {
        // The helper must not break a correct install on its own: re-hashing
        // and re-pinning alone leave the load path green, so a refusal in the
        // tests above is the schema cross-check and not a size or checksum
        // failure the helper introduced.
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.repinToyLayoutExpert(directoryURL: dir, layer: 0, expert: 1) { _ in }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: dir, device: device,
            expecting: .qwenToy())
        let expert = try model.routedExpert(layer: 0, expert: 1)
        #expect(expert.length > 0)
    }

    @Test func nonPageAlignedExpertStrideFailsAtManifest() throws {
        let dir = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        var root = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: manifestURL)) as? [String: Any])
        root["expertStride"] = 1024
        let data = try JSONSerialization.data(withJSONObject: root)
        try data.write(to: manifestURL)
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect {
            _ = try Model.load(
                directoryURL: dir, device: device,
                expecting: .qwenToy())
        } throws: { error in
            if case ModelError.expertStrideNotPageAligned = error { return true }
            return false
        }
    }

}
