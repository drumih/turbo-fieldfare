import Foundation
import TinyTitanCore

/// The feed-forward paths: the dense SwiGLU of layers 0-2 and the sparse MoE
/// of every other layer.
///
/// The sparse block routes with a sigmoid over the gate logits, adds the gate
/// bias only for *selection*, gathers the un-biased sigmoid scores for the
/// chosen experts, normalizes them over the top-k and scales by 2.5. Each
/// expert is a SwiGLU of width 768; the shared expert (also 768 wide here) is
/// added to the routed sum.
extension K2Model {

    func denseMLP(layer: Int, x: [Float]) throws -> [Float] {
        let stem = layerName(layer, "mlp.")
        let gate = try project(stem + "gate_proj.weight", x)
        let up = try project(stem + "up_proj.weight", x)
        var gated = gate
        for index in gated.indices { gated[index] = silu(gate[index]) * up[index] }
        return try project(stem + "down_proj.weight", gated)
    }

    func sparseMoE(layer: Int, x: [Float]) throws -> [Float] {
        let stem = layerName(layer, "mlp.")
        let logits = try projectFloat(stem + "gate.weight", x)
        let bias = try snapshot.floats(stem + "gate.bias")
        guard bias.count == logits.count else {
            throw ModelError.internalInconsistency(
                detail: "layer \(layer): gate bias \(bias.count) for \(logits.count) experts")
        }
        var scores = logits
        for index in scores.indices { scores[index] = sigmoid(scores[index]) }
        var selection = scores
        for index in selection.indices { selection[index] += bias[index] }
        let chosen = topK(selection, min(configuration.topKExperts, scores.count))
        var weights = chosen.map { scores[$0] }
        let total = weights.reduce(0, +)
        if total > 0 {
            let norm = configuration.normTopK ? 1 / total : 1
            for index in weights.indices { weights[index] *= norm * configuration.routerScaling }
        }

        var out = [Float](repeating: 0, count: configuration.hiddenSize)
        for (slot, expert) in chosen.enumerated() {
            let weight = weights[slot]
            let value = try expertOutput(
                layer: layer, expert: expert,
                gate: stem + "experts.\(expert).gate_proj.weight",
                up: stem + "experts.\(expert).up_proj.weight",
                down: stem + "experts.\(expert).down_proj.weight", x: x)
            for index in out.indices { out[index] += weight * value[index] }
        }
        if configuration.sharedExperts > 0 {
            let width = configuration.moeIntermediateSize * configuration.sharedExperts
            var gate = try project(stem + "shared_experts.gate_proj.weight", x)
            let up = try project(stem + "shared_experts.up_proj.weight", x)
            guard gate.count == up.count, gate.count == width else {
                throw ModelError.internalInconsistency(
                    detail: "layer \(layer): shared expert width \(gate.count) for \(width)")
            }
            for index in gate.indices { gate[index] = silu(gate[index]) * up[index] }
            let shared = try project(stem + "shared_experts.down_proj.weight", gate)
            for index in out.indices { out[index] += shared[index] }
        }
        return out
    }

    /// One SwiGLU expert: `down(silu(gate(x)) * up(x))`.
    private func expertOutput(
        layer: Int, expert: Int, gate: String, up: String, down: String, x: [Float]
    ) throws -> [Float] {
        var gated = try project(gate, x)
        let upValue = try project(up, x)
        guard gated.count == upValue.count else {
            throw ModelError.internalInconsistency(
                detail: "layer \(layer) expert \(expert): gate/up widths differ")
        }
        for index in gated.indices { gated[index] = silu(gated[index]) * upValue[index] }
        return try project(down, gated)
    }
}
