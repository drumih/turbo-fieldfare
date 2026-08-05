import Foundation
import Testing

@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite struct VerifiedInstallTests {
    @Test func acceptsRootSymlinkAndBindsAlias() throws {
        let root = try makeInstall()
        let alias = temporaryURL("verified-install-link.gturbo")
        defer {
            try? FileManager.default.removeItem(at: alias)
            try? FileManager.default.removeItem(at: root)
        }
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: root)

        let result = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: alias.path))
        let receipt = try #require(JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: result.receiptPath)))
            as? [String: Any])
        #expect(receipt["modelDirectoryPath"] as? String == alias.standardizedFileURL.path)
    }

    @Test func concurrentWritersPublishCompleteReceipt() async throws {
        let root = try makeInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        let oldTemporary = root.appendingPathComponent("verified-install.json.tmp")
        try Data("sentinel".utf8).write(to: oldTemporary)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<2 {
                group.addTask {
                    _ = try VerifiedInstallTool.run(
                        options: VerifyInstallOptions(inputGTurbo: root.path))
                }
            }
            try await group.waitForAll()
        }

        let receipt = root.appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        _ = try JSONSerialization.jsonObject(with: Data(contentsOf: receipt))
        #expect(try Data(contentsOf: oldTemporary) == Data("sentinel".utf8))
        let names = try FileManager.default.contentsOfDirectory(atPath: root.path)
        #expect(!names.contains { $0.hasPrefix(".gturbo-") && $0.hasSuffix(".tmp") })
    }

    @Test func verifiedByteAccountingRejectsOverflow() {
        #expect(throws: RepackError.self) {
            _ = try VerifiedInstallTool.addingVerifiedBytes(UInt64.max, 1)
        }
    }

    @Test func verifiesInstallAndReportsUnexpectedEntries() throws {
        let root = try makeInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("stray".utf8).write(to: root.appendingPathComponent("stray.txt"))

        let result = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: root.path))

        #expect(result.fileCount == 4)
        #expect(result.unexpectedEntries == ["stray.txt"])
        #expect(FileManager.default.fileExists(atPath: result.receiptPath))
    }

    @Test func acceptsMetadataExactlyAtCaps() throws {
        let root = try makeInstall(
            manifestBytes: VerifiedInstallTool.manifestMaxBytes,
            layoutBytes: VerifiedInstallTool.layoutMaxBytes)
        defer { try? FileManager.default.removeItem(at: root) }

        let result = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: root.path))
        #expect(result.fileCount == 4)
    }

    @Test func rejectsManifestAboveCapBeforeDecode() throws {
        let root = try makeInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        try truncate(root.appendingPathComponent("manifest.json"),
                     to: VerifiedInstallTool.manifestMaxBytes + 1)

        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: root.path))
        }
    }

    @Test func rejectsLayoutAboveCapBeforeDecode() throws {
        let root = try makeInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        try truncate(root.appendingPathComponent("packed_experts/layout.json"),
                     to: VerifiedInstallTool.layoutMaxBytes + 1)

        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: root.path))
        }
    }

    @Test func rejectsManifestLeafSymlink() throws {
        let root = try makeInstall()
        let outside = temporaryURL("manifest-outside.json")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        let manifest = root.appendingPathComponent("manifest.json")
        try FileManager.default.moveItem(at: manifest, to: outside)
        try FileManager.default.createSymbolicLink(at: manifest,
                                                   withDestinationURL: outside)

        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: root.path))
        }
    }

    @Test func rejectsDeclaredFileLeafSymlink() throws {
        let root = try makeInstall()
        let outside = temporaryURL("layer-outside.bin")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        let layer = root.appendingPathComponent("packed_experts/layer_00.bin")
        try FileManager.default.moveItem(at: layer, to: outside)
        try FileManager.default.createSymbolicLink(at: layer,
                                                   withDestinationURL: outside)

        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: root.path))
        }
    }

    @Test func rejectsDeclaredFileParentSymlink() throws {
        let root = try makeInstall()
        let outside = temporaryURL("packed-outside")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        let packed = root.appendingPathComponent("packed_experts")
        try FileManager.default.moveItem(at: packed, to: outside)
        try FileManager.default.createSymbolicLink(at: packed,
                                                   withDestinationURL: outside)

        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: root.path))
        }
    }

    @Test func receiptSymlinkIsReplacedWithoutFollowingIt() throws {
        let root = try makeInstall()
        let outside = temporaryURL("receipt-outside.json")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        let sentinel = Data("outside-sentinel".utf8)
        try sentinel.write(to: outside)
        let receipt = root.appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        try FileManager.default.createSymbolicLink(at: receipt,
                                                   withDestinationURL: outside)

        _ = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: root.path))

        #expect(try Data(contentsOf: outside) == sentinel)
        let values = try receipt.resourceValues(
            forKeys: [.isSymbolicLinkKey, .isRegularFileKey])
        #expect(values.isSymbolicLink == false)
        #expect(values.isRegularFile == true)
    }

    @Test func rejectsSameSizeCorruption() throws {
        let root = try makeInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        let layer = root.appendingPathComponent("packed_experts/layer_00.bin")
        let handle = try FileHandle(forWritingTo: layer)
        try handle.seek(toOffset: 17)
        try handle.write(contentsOf: Data([1]))
        try handle.close()

        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: root.path))
        }
    }

    private func makeInstall(manifestBytes: UInt64? = nil,
                             layoutBytes: UInt64? = nil) throws -> URL {
        let root = temporaryURL("verified-install.gturbo")
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        let stride = GTurboFormatV1.alignmentBytes
        let modelData = Data([0])
        let layerData = Data(repeating: 0, count: Int(stride))
        var layoutData = try GTurboPackedExpertsLayoutCodec.encode(
            GTurboPackedExpertsLayoutV1(
                expertStride: stride,
                numLayers: 1,
                expertsPerLayer: 1,
                layers: [GTurboLayerV1(
                    layer: 0,
                    file: "layer_00.bin",
                    experts: [GTurboExpertV1(
                        expert: 0,
                        physicalRank: nil,
                        offset: 0,
                        size: stride,
                        tensors: ["gate": GTurboSubTensorV1(
                            offset: 0,
                            size: 32,
                            dtype: "U32",
                            shape: [8, 8],
                            bits: 4)])])]))
        if let layoutBytes { try pad(&layoutData, to: layoutBytes) }

        let files = [
            "model_weights.bin": GTurboManifestFileV1(
                size: UInt64(modelData.count), sha256: hash(modelData)),
            "packed_experts/layout.json": GTurboManifestFileV1(
                size: UInt64(layoutData.count), sha256: hash(layoutData)),
            "packed_experts/layer_00.bin": GTurboManifestFileV1(
                size: UInt64(layerData.count), sha256: hash(layerData)),
        ]
        var manifestData = try GTurboManifestCodec.encode(GTurboManifestV1(
            flags: [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: "fixture/model",
            sourceSnapshotHash: nil,
            arch: fixtureArch(),
            quant: nil,
            files: files,
            expertsPerLayer: 1,
            numLayers: 1,
            expertStride: stride,
            bitWidthOverridesHonored: nil))
        if let manifestBytes { try pad(&manifestData, to: manifestBytes) }

        try modelData.write(to: root.appendingPathComponent("model_weights.bin"))
        try layerData.write(
            to: root.appendingPathComponent("packed_experts/layer_00.bin"))
        try layoutData.write(
            to: root.appendingPathComponent("packed_experts/layout.json"))
        try manifestData.write(to: root.appendingPathComponent("manifest.json"))
        return root
    }

    private func fixtureArch() -> GTurboManifestArchV1 {
        GTurboManifestArchV1(
            hiddenSize: 64,
            ffnIntermediate: 128,
            moeIntermediateSize: 32,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 1_024,
            slidingWindow: 128,
            finalLogitSoftcap: 30,
            ropeTheta: 10_000,
            fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25,
            numLayers: 1,
            numExperts: 1,
            topKExperts: 1,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            hiddenActivation: "gelu_pytorch_tanh",
            fullAttentionLayerMask: [0])
    }

    private func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private func pad(_ data: inout Data, to size: UInt64) throws {
        guard UInt64(data.count) <= size, size <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(detail: "invalid metadata padding")
        }
        data.append(Data(repeating: 0x20, count: Int(size) - data.count))
    }

    private func truncate(_ url: URL, to size: UInt64) throws {
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: size)
        try handle.close()
    }

    private func temporaryURL(_ suffix: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-\(UUID().uuidString)-\(suffix)")
    }
}
