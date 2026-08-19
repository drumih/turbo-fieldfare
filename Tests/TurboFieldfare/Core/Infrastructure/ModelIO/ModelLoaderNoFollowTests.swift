import Foundation
import Metal
import Testing

@testable import TurboFieldfare

extension ModelLoaderTests {
    @Test func acceptsRootSymlinkAndRetainsOpenedDirectory() throws {
        let target = try Self.writeToySynthetic()
        let alias = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-root-\(UUID().uuidString)")
        let replacement = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-replacement-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: alias)
            try? FileManager.default.removeItem(at: target)
            try? FileManager.default.removeItem(at: replacement)
        }
        try FileManager.default.createDirectory(at: replacement,
                                                withIntermediateDirectories: false)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: target)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: alias, device: device, expecting: .gemma4Toy())

        try FileManager.default.removeItem(at: alias)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: replacement)
        _ = try model.routedExpert(layer: 0, expert: 0)
    }

    @Test func trustedReceiptKeepsRootSymlinkPathBinding() throws {
        let target = try Self.writeToySynthetic()
        let alias = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-receipt-root-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: alias)
            try? FileManager.default.removeItem(at: target)
        }
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: target)
        try Self.writeVerifiedInstallReceipt(directoryURL: alias)
        let device = try #require(MTLCreateSystemDefaultDevice())

        _ = try Model.load(
            directoryURL: alias, device: device, expecting: .gemma4Toy(),
            integrityPolicy: .sizeCheckTrustedReceipt)
        #expect(throws: ModelError.self) {
            try Model.load(
                directoryURL: target, device: device, expecting: .gemma4Toy(),
                integrityPolicy: .sizeCheckTrustedReceipt)
        }
    }

    @Test func rejectsManifestLeafSymlink() throws {
        let dir = try Self.writeToySynthetic()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-\(UUID().uuidString).json")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let manifest = dir.appendingPathComponent("manifest.json")
        try FileManager.default.moveItem(at: manifest, to: outside)
        try FileManager.default.createSymbolicLink(at: manifest, withDestinationURL: outside)
        let device = try #require(MTLCreateSystemDefaultDevice())

        #expect(throws: ModelError.self) {
            try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
        }
    }

    @Test func rejectsPackedExpertsParentSymlink() throws {
        let dir = try Self.writeToySynthetic()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-packed-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let packed = dir.appendingPathComponent("packed_experts")
        try FileManager.default.moveItem(at: packed, to: outside)
        try FileManager.default.createSymbolicLink(at: packed, withDestinationURL: outside)
        let device = try #require(MTLCreateSystemDefaultDevice())

        #expect(throws: ModelError.self) {
            try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
        }
    }

    @Test func rejectsTrustedReceiptSymlink() throws {
        let dir = try Self.writeToySynthetic()
        try Self.writeVerifiedInstallReceipt(directoryURL: dir)
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-receipt-\(UUID().uuidString).json")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let receipt = dir.appendingPathComponent(VerifiedInstallReceiptReader.fileName)
        try FileManager.default.moveItem(at: receipt, to: outside)
        try FileManager.default.createSymbolicLink(at: receipt, withDestinationURL: outside)
        let device = try #require(MTLCreateSystemDefaultDevice())

        #expect(throws: ModelError.self) {
            try Model.load(
                directoryURL: dir, device: device, expecting: .gemma4Toy(),
                integrityPolicy: .sizeCheckTrustedReceipt)
        }
    }

    @Test func rejectsRoutedLayerLeafSymlinkOnLazyOpen() throws {
        let dir = try Self.writeToySynthetic()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-layer-\(UUID().uuidString).bin")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let layer = dir.appendingPathComponent("packed_experts/layer_00.bin")
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
        try FileManager.default.moveItem(at: layer, to: outside)
        try FileManager.default.createSymbolicLink(at: layer, withDestinationURL: outside)

        #expect(throws: ModelError.self) {
            try model.routedExpert(layer: 0, expert: 0)
        }
    }
}
