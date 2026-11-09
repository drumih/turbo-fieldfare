import Foundation
import TinyTitanFormat

package enum GGUFInstallPlanError: Error, Equatable, CustomStringConvertible {
    case missingBlockCount
    case emptyLayer(Int)

    package var description: String {
        switch self {
        case .missingBlockCount:
            return "GGUF metadata has no block_count, so the install's layer count is unknown"
        case .emptyLayer(let layer):
            return "planned layer \(layer) has no experts"
        }
    }
}

package struct GGUFInstallPlan: Sendable {
    package let layout: SSDAIPackedExpertsLayoutV1
    /// Size of the encoded `layout.json`, checked against the reader's 128 MiB
    /// cap by the planner's test rather than assumed to fit.
    package let layoutJSONBytes: Int
    package let expertBytes: UInt64
    package let expertStride: UInt64
}

/// Plans the routed-expert half of a `.ssdai` install from the GGUF's own
/// geometry: one layer record per transformer layer, each carrying its experts
/// and the nine sub-tensor records per expert.
///
/// No weights are read. This is the document a full conversion would write --
/// it proves the layout is constructible and bounded before 68 GB of payload is
/// produced, and it states that bound: `expertBytes` is what the layers will
/// occupy on disk.
package enum GGUFInstallPlanner {
    package static func plan(from model: GGUFModel) throws -> GGUFInstallPlan {
        guard let layers = model.architectureValue("block_count")?.integer, layers > 0 else {
            throw GGUFInstallPlanError.missingBlockCount
        }
        let numLayers = Int(layers)
        let experts = try GGUFExpertLayout.expertsPerLayer(in: model)
        var records: [SSDAILayerV1] = []
        records.reserveCapacity(numLayers)
        for layer in 0..<numLayers {
            records.append(
                try GGUFExpertBlobWriter.layoutRecord(
                    layer: layer, expertCount: experts, in: model))
        }
        guard let first = records.first?.experts.first else {
            throw GGUFInstallPlanError.emptyLayer(0)
        }
        let layout = SSDAIPackedExpertsLayoutV1(
            expertStride: first.size,
            numLayers: numLayers,
            expertsPerLayer: experts,
            compression: nil,
            layers: records)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(layout)
        // Decode through the format's own validator: a plan the runtime would
        // refuse is not a plan, and this is the cheapest place to find out.
        _ = try SSDAIPackedExpertsLayoutCodec.decode(data)
        return GGUFInstallPlan(
            layout: layout,
            layoutJSONBytes: data.count,
            expertBytes: first.size * UInt64(experts) * UInt64(numLayers),
            expertStride: first.size)
    }
}
