import Foundation

struct SubTensorEntry: Sendable, Equatable {
    let offset: UInt64    // relative to the expert blob's start
}

struct ExpertEntry: Sendable {
    /// Logical routed-expert id used by the model/router.
    let expert: Int
    /// Absolute byte offset of this expert blob's start inside its layer file.
    let offset: UInt64
    /// Total bytes consumed by this expert blob (== `expertStride`).
    let size: UInt64
    /// Sub-tensors keyed by role (gate / up / down / shared) and component
    /// (raw, `_scales`, `_biases`).
    let subTensors: [String: SubTensorEntry]

    init(expert: Int,
                offset: UInt64,
                size: UInt64,
                subTensors: [String: SubTensorEntry]) {
        self.expert = expert
        self.offset = offset
        self.size = size
        self.subTensors = subTensors
    }
}

struct LayerLayout: Sendable {
    let layer: Int
    let file: String          // basename, e.g. "layer_00.bin"
    let experts: [ExpertEntry]
}

struct PackedExpertsLayout: Sendable {
    let expertStride: UInt64
    let numLayers: Int
    let expertsPerLayer: Int
    let layers: [LayerLayout]

    /// Resolve `(layer, expert)` -> `ExpertEntry`. O(1).
    func expert(layer: Int, expert: Int) -> ExpertEntry {
        return layers[layer].experts[expert]
    }
}

enum PackedExpertsLayoutReader {
    static let defaultMaxBytes: UInt64 = 16 * 1024 * 1024

    static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let size = try metadataFileSize(url)
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "layout.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: url)
        let root: [String: Any]
        do {
            root = try (JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        } catch {
            throw ModelError.indexCorrupt(detail: "layout.json: \(error)")
        }
        guard
            let expertStride = (root["expertStride"] as? NSNumber)?.uint64Value,
            let numLayers = root["numLayers"] as? Int,
            let expertsPerLayer = root["expertsPerLayer"] as? Int,
            let layersArr = root["layers"] as? [[String: Any]]
        else {
            throw ModelError.indexCorrupt(detail: "layout.json: missing top-level keys")
        }

        var layers: [LayerLayout] = []
        layers.reserveCapacity(layersArr.count)
        for layerObj in layersArr {
            guard
                let layerIdx = layerObj["layer"] as? Int,
                let file = layerObj["file"] as? String,
                let expertsArr = layerObj["experts"] as? [[String: Any]]
            else {
                throw ModelError.indexCorrupt(detail: "layout.json: malformed layer entry")
            }
            var experts = [ExpertEntry?](repeating: nil, count: expertsPerLayer)
            for expertObj in expertsArr {
                guard
                    let offset = (expertObj["offset"] as? NSNumber)?.uint64Value,
                    let size = (expertObj["size"] as? NSNumber)?.uint64Value,
                    let tensorsObj = expertObj["tensors"] as? [String: [String: Any]]
                else {
                    throw ModelError.indexCorrupt(detail: "layout.json: malformed expert entry")
                }
                var subTensors: [String: SubTensorEntry] = [:]
                for (role, t) in tensorsObj {
                    guard
                        let toff = (t["offset"] as? NSNumber)?.uint64Value,
                        t["size"] is NSNumber,
                        t["dtype"] is String,
                        t["shape"] is [Int]
                    else {
                        throw ModelError.indexCorrupt(detail: "layout.json: malformed tensor \(role)")
                    }
                    if let bits = t["bits"], !(bits is Int) {
                        throw ModelError.indexCorrupt(detail: "layout.json: malformed tensor bits \(role)")
                    }
                    subTensors[role] = SubTensorEntry(offset: toff)
                }
                let expertID = expertObj["expert"] as? Int ?? experts.compactMap { $0 }.count
                guard expertID >= 0 && expertID < expertsPerLayer else {
                    throw ModelError.indexCorrupt(detail: "layout.json: expert id out of range")
                }
                let physicalRank = expertObj["physicalRank"] as? Int
                if let physicalRank,
                   (physicalRank < 0 || physicalRank >= expertsPerLayer) {
                    throw ModelError.indexCorrupt(detail: "layout.json: physicalRank out of range")
                }
                experts[expertID] = ExpertEntry(expert: expertID,
                                                offset: offset,
                                                size: size,
                                                subTensors: subTensors)
            }
            guard experts.allSatisfy({ $0 != nil }) else {
                throw ModelError.indexCorrupt(detail: "layout.json: missing expert entries")
            }
            layers.append(LayerLayout(layer: layerIdx,
                                      file: file,
                                      experts: experts.map { $0! }))
        }

        return PackedExpertsLayout(expertStride: expertStride,
                                   numLayers: numLayers,
                                   expertsPerLayer: expertsPerLayer,
                                   layers: layers)
    }

    private static func metadataFileSize(_ url: URL) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attrs[.size] as? NSNumber else {
            throw ModelError.indexCorrupt(detail: "layout.json: file size unavailable")
        }
        return number.uint64Value
    }
}
