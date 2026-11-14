import Foundation
import TinyTitanKernelsC
import TinyTitanCore

/// The two attention variants of K2-Horizon.
///
/// Dense layers have the usual `v_proj`; sparse layers replace it with MoVA —
/// a router over 64 value experts, the top four of which produce the value
/// states, each activated with SiLU and weighted. Everything after the value
/// states is shared: full RoPE, the KV cache, the softmax and the softplus
/// output gate.
extension K2Model {

    func attention(layer: Int, x: [Float]) throws -> [Float] {
        let stem = layerName(layer, "self_attn.")
        let query = try project(stem + "q_proj.weight", x)
        let key = try project(stem + "k_proj.weight", x)
        let value = try project(stem + "v_proj.weight", x)
        let gate =
            configuration.attentionGate == "none"
            ? [] : try project(stem + "gate_proj.weight", x)
        return try runAttention(layer: layer, query: query, key: key, value: value, gate: gate)
    }

    func movaAttention(layer: Int, x: [Float]) throws -> [Float] {
        let stem = layerName(layer, "self_attn.")
        let query = try project(stem + "q_proj.weight", x)
        let key = try project(stem + "k_proj.weight", x)
        let value = try movaValues(layer: layer, x: x)
        let gate =
            configuration.attentionGate == "none"
            ? [] : try project(stem + "gate_proj.weight", x)
        return try runAttention(layer: layer, query: query, key: key, value: value, gate: gate)
    }

    /// The value states of a sparse layer: top-4 of 64 experts, each
    /// `hidden -> kvHeads * headDim`, activated with SiLU before weighting.
    func movaValues(layer: Int, x: [Float]) throws -> [Float] {
        let stem = layerName(layer, "self_attn.")
        let logits = try projectFloat(stem + "v_router.weight", x)
        let bias = try snapshot.floats(stem + "v_router.bias")
        guard bias.count == logits.count else {
            throw ModelError.internalInconsistency(
                detail: "v_router bias \(bias.count) for \(logits.count) experts")
        }
        var scores = logits
        for index in scores.indices { scores[index] = sigmoid(scores[index]) }
        var selection = scores
        for index in selection.indices { selection[index] += bias[index] }
        let chosen = topK(selection, min(configuration.movaTopK, scores.count))
        var weights = chosen.map { scores[$0] }
        let total = weights.reduce(0, +)
        if total > 0 {
            let norm = configuration.normTopK ? 1 / total : 1
            for index in weights.indices { weights[index] *= norm * configuration.routerScaling }
        }
        let width = configuration.keyValueHeads * configuration.headDim
        var values = [Float](repeating: 0, count: width)
        for (slot, expert) in chosen.enumerated() {
            let output = try project(stem + "v_experts.\(expert).weight", x)
            let weight = weights[slot]
            for index in output.indices {
                values[index] += silu(output[index]) * weight
            }
        }
        return values
    }

    func runAttention(
        layer: Int, query: [Float], key: [Float], value: [Float], gate: [Float]
    ) throws -> [Float] {
        let heads = configuration.heads
        let kvHeads = configuration.keyValueHeads
        let dim = configuration.headDim
        guard query.count == heads * dim, key.count == kvHeads * dim,
            value.count == kvHeads * dim
        else {
            throw ModelError.internalInconsistency(
                detail: "layer \(layer): attention projections do not match the config")
        }
        guard position < configuration.maxPositions else {
            throw ModelError.internalInconsistency(
                detail:
                    "position \(position) is past the checkpoint's \(configuration.maxPositions)")
        }
        var rotatedQuery = query
        var rotatedKey = key
        CPUOps.applyRoPE(
            &rotatedQuery, headDim: dim, rotaryDim: configuration.ropeDim,
            position: position, theta: configuration.ropeTheta)
        CPUOps.applyRoPE(
            &rotatedKey, headDim: dim, rotaryDim: configuration.ropeDim,
            position: position, theta: configuration.ropeTheta)
        keys[layer].append(contentsOf: rotatedKey)
        values[layer].append(contentsOf: value)

        let cached = keys[layer].count / (kvHeads * dim)
        let group = heads / kvHeads
        let scale = 1 / Float(dim).squareRoot()
        var out = [Float](repeating: 0, count: heads * dim)
        var scores = [Float](repeating: 0, count: cached)
        let storedKeys = keys[layer]
        let storedValues = values[layer]
        storedKeys.withUnsafeBufferPointer { keyStore in
            storedValues.withUnsafeBufferPointer { valueStore in
                for head in 0..<heads {
                    let kvHead = head / group
                    let queryBase = head * dim
                    for step in 0..<cached {
                        let base = step * kvHeads * dim + kvHead * dim
                        var total: Float = 0
                        for index in 0..<dim {
                            total += keyStore[base + index] * rotatedQuery[queryBase + index]
                        }
                        scores[step] = total * scale
                    }
                    CPUOps.softmaxInPlace(&scores)
                    for step in 0..<cached {
                        let weight = scores[step]
                        if weight == 0 { continue }
                        let base = step * kvHeads * dim + kvHead * dim
                        for index in 0..<dim {
                            out[queryBase + index] += weight * valueStore[base + index]
                        }
                    }
                }
            }
        }
        if !gate.isEmpty {
            guard gate.count == out.count else {
                throw ModelError.internalInconsistency(
                    detail: "layer \(layer): attention gate \(gate.count) for \(out.count)")
            }
            for index in out.indices { out[index] *= softplusGate(gate[index]) }
        }
        return try project(layerName(layer, "self_attn.o_proj.weight"), out)
    }
}
