import Foundation

/// The routed experts as mapped slices: each (layer, expert) pair's three
/// matrices are mapped on first use and read in place by the fused GEMV, so a
/// token no longer copies ~2.4 GB of expert data into buffers.
package final class GGUFMappedExperts {
    package struct Matrices {
        package let gate: GGUFMappedShards.Weight
        package let up: GGUFMappedShards.Weight
        package let down: GGUFMappedShards.Weight
    }

    private let model: GGUFModel
    private let mapped: GGUFMappedShards

    package init(model: GGUFModel, mapped: GGUFMappedShards) {
        self.model = model
        self.mapped = mapped
    }

    package func matrices(layer: Int, expert: Int) throws -> Matrices {
        let weights = try GGUFExpertLayout.weights(layer: layer, expert: expert, in: model)
        func slice(_ range: GGUFTensorRange, _ role: String, rows: Int, columns: Int) throws
            -> GGUFMappedShards.Weight
        {
            let url = model.shards[range.shardIndex].url
            return try mapped.weight(
                key: "expert.L\(layer).E\(expert).\(role)",
                url: url, fileOffset: range.fileOffset, byteCount: range.byteCount,
                type: range.type, rows: rows, columns: columns)
        }
        return Matrices(
            gate: try slice(weights.gate, "gate", rows: 640, columns: 2_560),
            up: try slice(weights.up, "up", rows: 640, columns: 2_560),
            down: try slice(weights.down, "down", rows: 2_560, columns: 640))
    }
}
