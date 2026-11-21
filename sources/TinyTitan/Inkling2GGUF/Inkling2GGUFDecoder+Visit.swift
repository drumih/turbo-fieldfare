import Foundation
import Metal
import TinyTitanCore

extension Inkling2GGUFDecoder {
    /// One layer visit of llama.cpp's inkling2 graph: depth-mixed attention
    /// inputs, optional loop messages, relative attention with sinks and short
    /// convolutions, then the depth-mixed, routed MoE. Returns the new `cur`.
    func encodeVisit(
        visit: Int, layer: Inkling2GGUFLayerWeights, cur: Int, blocks: inout [Int],
        messages: [Int?], _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let inputs = try encodeAttentionInputs(
            layer: layer, cur: cur, blocks: blocks, messages: messages, e)
        let attended = try encodeAttention(visit: visit, layer: layer, inputs: inputs, e)
        return try encodeFeedForward(
            visit: visit, layer: layer, inpSA: cur, attended: attended, blocks: &blocks,
            message: messages[3], e)
    }

    /// The three attention inputs (Q, K, V): depth-mixed, loop-controlled and
    /// normalised. Sliding visits share one input unless loop messages split it.
    private func encodeAttentionInputs(
        layer: Inkling2GGUFLayerWeights, cur: Int, blocks: [Int], messages: [Int?],
        _ e: MTLComputeCommandEncoder
    ) throws -> [Int] {
        let s = state
        let sliding = layer.isSliding
        var inputs = [cur, cur, cur]
        if let query = layer.attnResQuery {
            let count = sliding ? 1 : 3
            let mixed = try mix(blocks + [cur], query: query.operand, count: count, e)
            inputs = sliding ? [mixed, mixed, mixed] : [mixed, mixed + 1, mixed + 2]
        }
        if messages[0] != nil {
            for j in 0..<3 {
                guard let message = messages[j] else { continue }
                let out = try rows(1)
                try ops.control(
                    cur: row(inputs[j]), message: row(message), down: layer.loopDown[j],
                    gain: layer.loopGain[j], forget: layer.loopForget[j],
                    h: scratch(s.controlHidden), out: row(out), width: hidden, encoder: e)
                inputs[j] = out
            }
        }
        if sliding, messages[0] != nil {
            return try inputs.map { try rmsInto($0, layer.inputNorms[0], e) }
        } else if sliding {
            let normed = try rmsInto(inputs[0], layer.inputNorms[0], e)
            return [normed, normed, normed]
        }
        return try (0..<3).map { try rmsInto(inputs[$0], layer.inputNorms[$0], e) }
    }

    /// Relative attention with sinks over this visit's cache, gated and
    /// projected, then the post-attention short convolution. Returns its row.
    private func encodeAttention(
        visit: Int, layer: Inkling2GGUFLayerWeights, inputs: [Int], _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let m = manifest
        let s = state
        let sliding = layer.isSliding
        let heads = m.layers[layer.index].heads
        let kvHeads = m.layers[layer.index].kvHeads
        let headDim = m.headDimension
        let kvWidth = kvHeads * headDim

        // Projections, convolutions and clips.
        let state = s.convStates[visit]
        try ops.gemv(layer.attnQ, x: row(inputs[0]), y: scratch(s.q), encoder: e)
        try ops.gemv(layer.attnK, x: row(inputs[1]), y: scratch(s.k), encoder: e)
        try ops.gemv(layer.attnV, x: row(inputs[2]), y: scratch(s.v), encoder: e)
        var dynamicK: Operand?
        var dynamicV: Operand?
        if let kProj = layer.kConvProj, let vProj = layer.vConvProj {
            try ops.gemv(kProj, x: row(inputs[1]), y: scratch(s.dynamicK), encoder: e)
            try ops.gemv(vProj, x: row(inputs[2]), y: scratch(s.dynamicV), encoder: e)
            dynamicK = scratch(s.dynamicK)
            dynamicV = scratch(s.dynamicV)
        }
        try ops.shortConv(
            x: scratch(s.k), weight: layer.kConv, state: .init(state, offset: 0),
            out: scratch(s.k), dynamic: dynamicK, heads: kvHeads, channels: kvWidth, encoder: e)
        try ops.shortConv(
            x: scratch(s.v), weight: layer.vConv, state: .init(state, offset: kvWidth * 12),
            out: scratch(s.v), dynamic: dynamicV, heads: kvHeads, channels: kvWidth, encoder: e)
        try ops.gemv(layer.attnRel, x: row(inputs[0]), y: scratch(s.r), encoder: e)
        let tau: (floor: Float, alpha: Float)? =
            sliding ? nil : (Float(m.logScaleFloor), m.logScaleAlpha)
        let step = scratch(s.step)
        try ops.clipHeads(
            scratch(s.q), heads: heads, headDim: headDim, scale: layer.qClip,
            threshold: m.qkClip, tau: tau, step: step, encoder: e)
        try ops.clipHeads(
            scratch(s.k), heads: kvHeads, headDim: headDim, scale: nil, threshold: m.qkClip,
            tau: nil, step: step, encoder: e)
        try ops.rmsNorm(
            scratch(s.v), weight: nil, out: scratch(s.v), width: headDim, count: kvHeads,
            epsilon: m.rmsEpsilon, encoder: e)
        try ops.clipHeads(
            scratch(s.r), heads: heads, headDim: m.relativeLength, scale: layer.relClip,
            threshold: m.relativeClip, tau: tau, step: step, encoder: e)
        try ops.gemv(layer.attnSink, x: row(inputs[0]), y: scratch(s.sinks), encoder: e)

        // Attention over the cache, sigmoid gate, output projection and conv.
        let capacity = s.capacities[visit]
        try ops.storeKV(
            k: scratch(s.k), v: scratch(s.v), kCache: s.kCaches[visit], vCache: s.vCaches[visit],
            width: kvWidth, capacity: capacity, ring: sliding, step: step, encoder: e)
        try ops.attention(
            q: scratch(s.q), kCache: s.kCaches[visit], vCache: s.vCaches[visit], r: scratch(s.r),
            relativeBias: layer.relBias, sinks: scratch(s.sinks), out: scratch(s.attention),
            step: step, logits: scratch(s.attentionLogits),
            args: .init(
                heads: UInt32(heads), kvHeads: UInt32(kvHeads), headDim: UInt32(headDim),
                relative: UInt32(m.relativeLength), extent: UInt32(layer.relBias.rows),
                capacity: UInt32(capacity), ring: sliding ? 1 : 0,
                scale: 1 / Float(headDim).squareRoot()),
            encoder: e)
        try ops.gemv(layer.attnGate, x: row(inputs[0]), y: scratch(s.gate), encoder: e)
        try ops.sigmoidMultiply(
            scratch(s.attention), by: scratch(s.gate), count: heads * headDim, encoder: e)
        labelPrefix = "v\(visit).attnProj"
        let projected = try rows(1)
        try ops.gemv(layer.attnOutput, x: scratch(s.attention), y: row(projected), encoder: e)
        labelPrefix = "v\(visit).attnConv"
        let attended = try rows(1)
        try ops.shortConv(
            x: row(projected), weight: layer.attnPostConv,
            state: .init(state, offset: 2 * kvWidth * 12), out: row(attended), dynamic: nil,
            heads: 1, channels: hidden, encoder: e)

        return attended
    }

    /// Residual or new block, then the depth-mixed, loop-controlled, routed
    /// MoE with its post-convolution and residual. Returns the new `cur`.
    private func encodeFeedForward(
        visit: Int, layer: Inkling2GGUFLayerWeights, inpSA: Int, attended: Int,
        blocks: inout [Int], message: Int?, _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let m = manifest
        let s = state
        let kvWidth = m.layers[layer.index].kvHeads * m.headDimension
        let state = s.convStates[visit]
        // A block's first layer starts a new residual stream.
        var next: Int
        if layer.index % m.attentionResidualBlock == 0 {
            blocks.append(inpSA)
            next = attended
        } else {
            next = try rows(1)
            try ops.add(row(inpSA), row(attended), out: row(next), count: hidden, encoder: e)
        }
        let ffnInput = next
        labelPrefix = "v\(visit).ffnMix"

        // Routed MoE on the depth-mixed, controlled, normalised input.
        next = try mix(blocks + [next], query: layer.ffnResQuery.operand, count: 1, e)
        if let message {
            let out = try rows(1)
            try ops.control(
                cur: row(next), message: row(message), down: layer.loopDown[3],
                gain: layer.loopGain[3], forget: layer.loopForget[3],
                h: scratch(s.controlHidden), out: row(out), width: hidden, encoder: e)
            next = out
        }
        labelPrefix = "v\(visit).moeIn"
        let moeInput = try rmsInto(next, layer.ffnNorm, e)
        labelPrefix = "v\(visit).moeOut"
        let moeOutput = try encodeMoE(layer: layer, input: moeInput, e)
        labelPrefix = "v\(visit).ffnConv"
        let convolved = try rows(1)
        try ops.shortConv(
            x: row(moeOutput), weight: layer.ffnPostConv,
            state: .init(state, offset: (2 * kvWidth + hidden) * 12), out: row(convolved),
            dynamic: nil, heads: 1, channels: hidden, encoder: e)
        labelPrefix = "v\(visit).out"
        let out = try rows(1)
        try ops.add(row(ffnInput), row(convolved), out: row(out), count: hidden, encoder: e)
        return out
    }

    private func encodeMoE(
        layer: Inkling2GGUFLayerWeights, input: Int, _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let m = manifest
        let s = state
        let used = m.expertsPerToken
        let feedForward = m.layers[layer.index].expertFeedForward
        try ops.route(
            router: layer.router, bias: layer.routerBias, x: row(input),
            ids: scratch(s.routeIDs), weights: scratch(s.routeWeights),
            args: .init(
                experts: UInt32(m.expertCount), width: UInt32(hidden), used: UInt32(used),
                weightScale: m.expertWeightsScale, normEpsilon: 1e-20),
            encoder: e)
        try ops.gemvIds.encode(
            .init(
                type: layer.gateUp.type, rows: 2 * feedForward, columns: hidden, slots: used,
                expertStride: layer.gateUp.byteCount / m.expertCount, xSlotStride: 0),
            weights: layer.gateUp.operand, x: row(input), y: scratch(s.gateUp),
            ids: scratch(s.routeIDs), scales: layer.gateUpScale?.operand, encoder: e)
        try ops.situ(
            gateUp: scratch(s.gateUp), act: scratch(s.activation),
            args: .init(
                feedForward: UInt32(feedForward), slots: UInt32(used), beta: m.situBeta,
                linearBeta: m.situLinearBeta),
            encoder: e)
        try ops.gemvIds.encode(
            .init(
                type: layer.down.type, rows: hidden, columns: feedForward, slots: used,
                expertStride: layer.down.byteCount / m.expertCount, xSlotStride: feedForward),
            weights: layer.down.operand, x: scratch(s.activation), y: scratch(s.down),
            ids: scratch(s.routeIDs), scales: layer.downScale?.operand, encoder: e)
        let out = try rows(1)
        try ops.weightedSum(
            down: scratch(s.down), weights: scratch(s.routeWeights), out: row(out),
            width: hidden, slots: used, encoder: e)
        return out
    }
}
