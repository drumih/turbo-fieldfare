import Foundation

enum GTurboLayoutValidator {
    static func validate(path: String, plan: RepackPlan) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let layers = root["layers"] as? [[String: Any]] else {
            throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed root")
        }
        for layerObj in layers {
            guard let layerIndex = layerObj["layer"] as? Int,
                  let experts = layerObj["experts"] as? [[String: Any]],
                  let planLayer = plan.layers.first(where: { $0.layerIndex == layerIndex }) else {
                throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed layer")
            }
            if planLayer.expertsPerLayer == 0 { continue }
            var seenLogical = Set<Int>()
            var seenOffsets = Set<UInt64>()
            for expertObj in experts {
                guard let expert = expertObj["expert"] as? Int,
                      let offset = (expertObj["offset"] as? NSNumber)?.uint64Value,
                      let size = (expertObj["size"] as? NSNumber)?.uint64Value else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed expert")
                }
                guard expert >= 0 && expert < planLayer.expertsPerLayer else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: expert out of range")
                }
                guard seenLogical.insert(expert).inserted else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: duplicate expert \(expert)")
                }
                guard seenOffsets.insert(offset).inserted else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: duplicate offset \(offset)")
                }
                guard offset % Layout.pageBytes == 0 else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: unaligned offset \(offset)")
                }
                let expected = UInt64(expert) * planLayer.expertStride
                guard offset == expected else {
                    throw RepackError.configurationInvalid(detail:
                        "layout.json validation failed: offset \(offset) != expert * stride \(expected)")
                }
                guard size == planLayer.expertStride,
                      offset + size <= planLayer.fileSize else {
                    throw RepackError.configurationInvalid(detail:
                        "layout.json validation failed: expert range outside layer file")
                }
            }
            guard seenLogical.count == planLayer.expertsPerLayer else {
                throw RepackError.configurationInvalid(detail:
                    "layout.json validation failed: missing experts in layer \(layerIndex)")
            }
        }
    }
}
