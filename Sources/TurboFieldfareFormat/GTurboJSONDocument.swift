import Foundation

package final class GTurboJSONDocument {
    private var root: [String: Any]
    private let field: String
    private let originalData: Data

    package init(data: Data, field: String) throws {
        self.field = field
        self.originalData = data
        do {
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw GTurboFormatError.invalid(field: field, reason: "expected JSON object")
            }
            self.root = root
        } catch let error as GTurboFormatError {
            throw error
        } catch {
            throw GTurboFormatError.invalid(field: field, reason: "\(error)")
        }
    }

    package func decodeManifest() throws -> GTurboManifestV1 {
        try GTurboManifestCodec.decode(originalData)
    }

    package func decodeLayout() throws -> GTurboPackedExpertsLayoutV1 {
        try GTurboPackedExpertsLayoutCodec.decode(originalData)
    }

    package func applyPhysicalRanks(_ ranksByLayer: [Int: [Int]],
                                    expertStride: UInt64) throws {
        guard var layers = root["layers"] as? [[String: Any]] else {
            throw GTurboFormatError.invalid(field: field, reason: "missing layers")
        }
        for layerIndex in layers.indices {
            guard let logicalLayer = int(layers[layerIndex]["layer"]),
                  let ranks = ranksByLayer[logicalLayer],
                  let experts = layers[layerIndex]["experts"] as? [[String: Any]] else {
                throw GTurboFormatError.invalid(field: field, reason: "missing rank mapping")
            }
            var byLogical: [Int: [String: Any]] = [:]
            for (position, original) in experts.enumerated() {
                let logical = int(original["expert"]) ?? position
                guard logical >= 0, logical < ranks.count else {
                    throw GTurboFormatError.invalid(field: field, reason: "expert id out of range")
                }
                var updated = original
                updated["physicalRank"] = ranks[logical]
                updated["offset"] = try gturboCheckedMultiply(
                    UInt64(ranks[logical]), expertStride, field: "expert.offset")
                byLogical[logical] = updated
            }
            guard byLogical.count == ranks.count else {
                throw GTurboFormatError.invalid(field: field, reason: "missing experts")
            }
            layers[layerIndex]["experts"] = try (0..<ranks.count).map { logical in
                guard let expert = byLogical[logical] else {
                    throw GTurboFormatError.invalid(field: field, reason: "missing expert \(logical)")
                }
                return expert
            }
        }
        root["layers"] = layers
    }

    package func updateManifestFiles(_ updates: [String: GTurboManifestFileV1]) throws {
        guard var files = root["files"] as? [String: Any] else {
            throw GTurboFormatError.invalid(field: field, reason: "missing files")
        }
        for (path, update) in updates {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            var record = files[path] as? [String: Any] ?? [:]
            record["size"] = update.size
            record["sha256"] = update.sha256
            files[path] = record
        }
        root["files"] = files
    }

    package func encoded() throws -> Data {
        do {
            return try JSONSerialization.data(
                withJSONObject: root,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw GTurboFormatError.invalid(field: field, reason: "\(error)")
        }
    }

    private func int(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber else { return nil }
        let raw = number.int64Value
        guard raw >= Int64(Int.min), raw <= Int64(Int.max) else { return nil }
        return Int(raw)
    }

}
