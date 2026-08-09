import Darwin
import Foundation
import TurboFieldfareFormat

/// External, striped expert storage descriptor. It does not alter the pinned
/// `.gturbo` v2 wire format: every shard root has one `expert-shard.json`, and
/// its layer files contain the listed logical experts consecutively in list
/// order. All payloads remain canonical `expertStride`-byte MXFP4 blobs.
public struct K3ExpertShardDescriptor: Codable, Equatable, Sendable {
    public static let formatIdentifier = "turbofieldfare-k3-expert-shard-v1"

    public struct Layer: Codable, Equatable, Sendable {
        public let layer: Int
        public let file: String
        public let experts: [Int]
        public let size: UInt64
        public let sha256: String

        public init(layer: Int, file: String, experts: [Int],
                    size: UInt64, sha256: String) {
            self.layer = layer
            self.file = file
            self.experts = experts
            self.size = size
            self.sha256 = sha256
        }
    }

    public let format: String
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let expertStride: UInt64
    public let expertsPerLayer: Int
    public let shardIndex: Int
    public let shardCount: Int
    public let layers: [Layer]

    public init(modelID: String, sourceSnapshotHash: String?,
                expertStride: UInt64, expertsPerLayer: Int,
                shardIndex: Int, shardCount: Int, layers: [Layer]) {
        self.format = Self.formatIdentifier
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.expertStride = expertStride
        self.expertsPerLayer = expertsPerLayer
        self.shardIndex = shardIndex
        self.shardCount = shardCount
        self.layers = layers
    }
}

/// Validates a complete set of shard roots and opens one file per
/// `(layer, shard)`. `layerFile` maps each logical expert directly to its
/// physical descriptor and offset, allowing the existing bounded pread pool
/// to issue one layer's top-k reads across independent SSDs.
final class K3ExpertShardStore: @unchecked Sendable {
    static let descriptorName = "expert-shard.json"
    static let descriptorMaxBytes: UInt64 = 4 * 1024 * 1024

    private struct Root {
        let directory: GTurboModelDirectory
        let descriptor: K3ExpertShardDescriptor
        let layerByID: [Int: K3ExpertShardDescriptor.Layer]
    }

    private let roots: [Root]
    private let expertStride: UInt64
    private let expertsPerLayer: Int
    private let moeLayers: Set<Int>
    private let integrityPolicy: ModelIntegrityPolicy

    init(rootURLs: [URL], model: K3Model,
         integrityPolicy: ModelIntegrityPolicy) throws {
        guard !rootURLs.isEmpty else {
            throw ModelError.indexCorrupt(detail: "expert shard root list is empty")
        }
        var loaded: [Root] = []
        loaded.reserveCapacity(rootURLs.count)
        for url in rootURLs {
            let directory = try GTurboModelDirectory(rootURL: url)
            let data = try directory.readMetadata(
                Self.descriptorName, maxBytes: Self.descriptorMaxBytes)
            let descriptor: K3ExpertShardDescriptor
            do {
                descriptor = try JSONDecoder().decode(
                    K3ExpertShardDescriptor.self, from: data)
            } catch {
                throw ModelError.indexCorrupt(
                    detail: "\(url.path)/\(Self.descriptorName) is invalid: \(error)")
            }
            try Self.validateCommon(descriptor, model: model, root: url)
            var layerByID: [Int: K3ExpertShardDescriptor.Layer] = [:]
            for layer in descriptor.layers {
                guard layerByID.updateValue(layer, forKey: layer.layer) == nil else {
                    throw ModelError.indexCorrupt(
                        detail: "duplicate layer in \(url.path)/\(Self.descriptorName)")
                }
            }
            loaded.append(Root(
                directory: directory,
                descriptor: descriptor,
                layerByID: layerByID))
        }

        let shardCount = loaded[0].descriptor.shardCount
        guard shardCount == loaded.count,
              Set(loaded.map { $0.descriptor.shardIndex }) == Set(0..<shardCount) else {
            throw ModelError.indexCorrupt(
                detail: "expert shard roots must contain every shard index 0..<\(shardCount)")
        }
        let expectedLayers = Set(model.config.moeLayers0)
        for root in loaded {
            guard root.descriptor.shardCount == shardCount,
                  Set(root.layerByID.keys) == expectedLayers else {
                throw ModelError.indexCorrupt(
                    detail: "expert shard \(root.descriptor.shardIndex) layer set mismatch")
            }
        }
        self.roots = loaded.sorted {
            $0.descriptor.shardIndex < $1.descriptor.shardIndex
        }
        self.expertStride = model.expertsLayout.expertStride
        self.expertsPerLayer = model.expertsLayout.expertsPerLayer
        self.moeLayers = expectedLayers
        self.integrityPolicy = integrityPolicy
    }

    func layerFile(_ layer: Int) throws -> K3ExpertLayerFile {
        guard moeLayers.contains(layer) else {
            throw ModelError.indexCorrupt(
                detail: "striped expert store has no MoE layer \(layer)")
        }
        var locations = [K3ExpertLayerFile.Location](
            repeating: .init(fileDescriptor: -1, offset: 0),
            count: expertsPerLayer)
        var filled = [Bool](repeating: false, count: expertsPerLayer)
        var opened: [Int32] = []
        do {
            for root in roots {
                guard let entry = root.layerByID[layer] else {
                    throw ModelError.indexCorrupt(
                        detail: "expert shard layer \(layer) metadata is missing")
                }
                let expectedSize = UInt64(entry.experts.count) * expertStride
                guard entry.size == expectedSize else {
                    throw ModelError.indexCorrupt(
                        detail: "expert shard \(root.descriptor.shardIndex) layer \(layer) "
                            + "size metadata \(entry.size) != \(expectedSize)")
                }
                let fd = try root.directory.openFile(entry.file)
                opened.append(fd)
                let actualSize = try root.directory.fileSize(
                    fileDescriptor: fd, relativePath: entry.file)
                guard actualSize == entry.size else {
                    throw ModelError.tensorSizeMismatch(
                        name: entry.file, expected: entry.size, actual: actualSize)
                }
                if integrityPolicy == .fullSha256 {
                    try Sha256Verifier.verifyFile(
                        fileDescriptor: fd,
                        named: entry.file,
                        expectedHex: entry.sha256)
                }
                for (rank, expert) in entry.experts.enumerated() {
                    guard (0..<expertsPerLayer).contains(expert), !filled[expert] else {
                        throw ModelError.indexCorrupt(
                            detail: "invalid or duplicate logical expert \(expert) "
                                + "in striped layer \(layer)")
                    }
                    filled[expert] = true
                    locations[expert] = .init(
                        fileDescriptor: fd,
                        offset: UInt64(rank) * expertStride)
                }
            }
            guard filled.allSatisfy({ $0 }) else {
                throw ModelError.indexCorrupt(
                    detail: "striped layer \(layer) does not cover every logical expert")
            }
            return K3ExpertLayerFile(
                layer: layer,
                path: "striped://layer/\(layer)",
                expertsPerLayer: expertsPerLayer,
                expertStride: expertStride,
                expertLocations: locations)
        } catch {
            for fd in opened { close(fd) }
            throw error
        }
    }

    private static func validateCommon(_ descriptor: K3ExpertShardDescriptor,
                                       model: K3Model,
                                       root: URL) throws {
        guard descriptor.format == K3ExpertShardDescriptor.formatIdentifier,
              descriptor.modelID == model.modelID,
              descriptor.sourceSnapshotHash == model.sourceSnapshotHash,
              descriptor.expertStride == model.expertsLayout.expertStride,
              descriptor.expertsPerLayer == model.expertsLayout.expertsPerLayer,
              descriptor.shardCount > 0,
              (0..<descriptor.shardCount).contains(descriptor.shardIndex) else {
            throw ModelError.indexCorrupt(
                detail: "expert shard identity or geometry mismatch at \(root.path)")
        }
        for layer in descriptor.layers {
            guard !layer.experts.isEmpty,
                  layer.size == UInt64(layer.experts.count) * descriptor.expertStride,
                  layer.sha256.count == 64,
                  layer.sha256.utf8.allSatisfy({ byte in
                    (48...57).contains(byte) || (65...70).contains(byte)
                        || (97...102).contains(byte)
                  }) else {
                throw ModelError.indexCorrupt(
                    detail: "invalid expert shard layer \(layer.layer) at \(root.path)")
            }
            do {
                try GTurboPathValidator.validateRelativePath(
                    layer.file, field: "expert-shard.layers.file")
            } catch {
                throw ModelError.indexCorrupt(
                    detail: "unsafe expert shard file \(layer.file): \(error)")
            }
        }
    }
}
