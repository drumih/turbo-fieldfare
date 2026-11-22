import Foundation
import Metal
import TinyTitanCore

extension Inkling2GGUFDecoder {
    /// One layer visit of llama.cpp's inkling2 graph for the chunk's tokens:
    /// depth-mixed attention inputs, optional loop messages, relative attention
    /// with sinks and short convolutions, then the depth-mixed, routed MoE.
    /// Returns the new `cur` (a persistent slab).
    func encodeVisit(
        visit: Int, layer: Inkling2GGUFLayerWeights, cur: Int, blocks: inout [Int],
        messages: [Int?], _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let inputs = try encodeAttentionInputs(
            layer: layer, cur: cur, blocks: blocks, messages: messages, e)
        // A block's first layer starts a new residual stream: its attention
        // output stands alone and the stream so far joins the mixed blocks.
        let startsBlock = layer.index % manifest.attentionResidualBlock == 0
        let attended = try encodeAttention(
            visit: visit, layer: layer, inputs: inputs, residual: startsBlock ? nil : cur, e)
        if startsBlock { blocks.append(cur) }
        return try encodeFeedForward(
            visit: visit, layer: layer, ffnInput: attended, blocks: blocks,
            message: messages[3], e)
    }

    /// Mutually independent dispatches (each projection is one dispatch, a
    /// GEMV at decode and a tiled or small-rows GEMM at a chunk): one
    /// concurrent group, so the small K/V/rel/sink GEMMs of a chunk share the
    /// GPU with Q and the gate instead of each running alone.
    private func independent(_ e: MTLComputeCommandEncoder, _ body: () throws -> Void) throws {
        try ops.concurrent(e, body)
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
        if sliding, messages[0] == nil, let query = layer.attnResQuery {
            let normed = try mix(
                blocks + [cur], query: query.operand, count: 1,
                norm: (layer.inputNorms[0].operand, manifest.rmsEpsilon), e)
            return [normed, normed, normed]
        }
        if let query = layer.attnResQuery {
            let count = sliding ? 1 : 3
            let mixed = try mix(blocks + [cur], query: query.operand, count: count, e)
            inputs = sliding ? [mixed, mixed, mixed] : [mixed, mixed + 1, mixed + 2]
        }
        if messages[0] != nil {
            // The attention, K and V controllers are independent: their down
            // projections share one barrier, then their applies another.
            let controlled = (0..<3).filter { messages[$0] != nil }
            let outs = try controlled.map { _ in try rows(1) }
            let sliceBytes = s.maxChunk * manifest.loopGateRank * 4
            func slice(_ j: Int) -> Operand { .init(s.controlHidden, offset: j * sliceBytes) }
            func downs() throws {
                for j in controlled {
                    guard let message = messages[j] else { continue }
                    try ops.controlDown(
                        cur: row(inputs[j]), message: row(message), down: layer.loopDown[j],
                        h: slice(j), joined: scratch(s.controlJoined), width: hidden,
                        tokens: tokens, encoder: e)
                }
            }
            if tokens == 1 { try ops.concurrent(e, downs) } else { try downs() }
            func applies() throws {
                for (k, j) in controlled.enumerated() {
                    guard let message = messages[j] else { continue }
                    try ops.controlApply(
                        cur: row(inputs[j]), message: row(message), h: slice(j),
                        gain: layer.loopGain[j], forget: layer.loopForget[j], out: row(outs[k]),
                        width: hidden, tokens: tokens, encoder: e)
                }
            }
            try ops.concurrent(e, applies)
            for (k, j) in controlled.enumerated() { inputs[j] = outs[k] }
        }
        if sliding, messages[0] != nil {
            var normed: [Int] = []
            try ops.concurrent(e) {
                normed = try inputs.map { try rmsInto($0, layer.inputNorms[0], e) }
            }
            return normed
        } else if sliding {
            let normed = try rmsInto(inputs[0], layer.inputNorms[0], e)
            return [normed, normed, normed]
        }
        var normed: [Int] = []
        try ops.concurrent(e) {
            normed = try (0..<3).map { try rmsInto(inputs[$0], layer.inputNorms[$0], e) }
        }
        return normed
    }

    /// Every projection of the attention inputs; each reads a finished input
    /// and writes its own scratch.
    private func encodeProjections(
        layer: Inkling2GGUFLayerWeights, inputs: [Int], _ e: MTLComputeCommandEncoder
    ) throws {
        let s = state
        let t = tokens
        let jobs: [(Inkling2GGUFTensor, Int, MTLBuffer)] = [
            (layer.attnQ, inputs[0], s.q), (layer.attnK, inputs[1], s.k),
            (layer.attnV, inputs[2], s.v), (layer.attnRel, inputs[0], s.r),
            (layer.attnGate, inputs[0], s.gate), (layer.attnSink, inputs[0], s.sinks),
        ]
        var dynamic: [(Inkling2GGUFTensor, Int, MTLBuffer)] = []
        if let kProj = layer.kConvProj, let vProj = layer.vConvProj {
            dynamic = [(kProj, inputs[1], s.dynamicK), (vProj, inputs[2], s.dynamicV)]
        }
        try independent(e) {
            for (weight, input, out) in jobs + dynamic {
                try ops.gemv(weight, x: row(input), y: scratch(out), tokens: t, encoder: e)
            }
        }
    }

    /// Relative attention with sinks over this visit's cache, gated and
    /// projected, then the post-attention short convolution (+ the residual
    /// stream when the visit continues a block). Returns its slab.
    private func encodeAttention(
        visit: Int, layer: Inkling2GGUFLayerWeights, inputs: [Int], residual: Int?,
        _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let m = manifest
        let s = state
        let sliding = layer.isSliding
        let heads = m.layers[layer.index].heads
        let kvHeads = m.layers[layer.index].kvHeads
        let headDim = m.headDimension
        let kvWidth = kvHeads * headDim
        let t = tokens
        let state = s.convStates[visit]
        try encodeProjections(layer: layer, inputs: inputs, e)
        let dynamicK: Operand? = layer.kConvProj == nil ? nil : scratch(s.dynamicK)
        let dynamicV: Operand? = layer.vConvProj == nil ? nil : scratch(s.dynamicV)
        // K/V convolutions, clip, norm and cache store, with the Q and R clips.
        let tau: (floor: Float, alpha: Float)? =
            sliding ? nil : (Float(m.logScaleFloor), m.logScaleAlpha)
        let step = scratch(s.step)
        let capacity = s.capacities[visit]
        let kvArgs = Inkling2GGUFOps.KVFinishArgs(
            headDim: UInt32(headDim), kvHeads: UInt32(kvHeads),
            hasDynamic: dynamicK == nil ? 0 : 1, capacity: UInt32(capacity),
            ring: sliding ? 1 : 0, threshold: m.qkClip, epsilon: m.rmsEpsilon,
            scale: 1 / Float(dynamicK == nil ? 5 : 9).squareRoot(), tokens: UInt32(t))
        try ops.concurrent(e) {
            try ops.kvFinish(
                k: scratch(s.k), v: scratch(s.v), kWeight: layer.kConv, vWeight: layer.vConv,
                state: .init(state), dynamicK: dynamicK, dynamicV: dynamicV,
                kCache: s.kCaches[visit], vCache: s.vCaches[visit], step: step, args: kvArgs,
                encoder: e)
            try ops.clipHeads(
                scratch(s.q), heads: heads, headDim: headDim, scale: layer.qClip,
                threshold: m.qkClip, tau: tau, step: step, tokens: t, encoder: e)
            try ops.clipHeads(
                scratch(s.r), heads: heads, headDim: m.relativeLength, scale: layer.relClip,
                threshold: m.relativeClip, tau: tau, step: step, tokens: t, encoder: e)
        }

        // Gated attention over the cache, output projection and conv (+ residual).
        let attentionArgs = Inkling2GGUFOps.AttentionArgs(
            heads: UInt32(heads), kvHeads: UInt32(kvHeads), headDim: UInt32(headDim),
            relative: UInt32(m.relativeLength), extent: UInt32(layer.relBias.rows),
            capacity: UInt32(capacity), ring: sliding ? 1 : 0,
            scale: 1 / Float(headDim).squareRoot(),
            window: sliding ? UInt32(m.slidingWindow) : UInt32.max, tokens: UInt32(t))
        try ops.attention(
            q: scratch(s.q), kCache: s.kCaches[visit], vCache: s.vCaches[visit], r: scratch(s.r),
            relativeBias: layer.relBias, sinks: scratch(s.sinks), gate: scratch(s.gate),
            out: scratch(s.attention), step: step, args: attentionArgs, encoder: e)
        let projected = try rows(1)
        try ops.gemv(
            layer.attnOutput, x: scratch(s.attention), y: row(projected), tokens: t, encoder: e)
        let attended = try rows(1)
        try ops.shortConv(
            x: row(projected), weight: layer.attnPostConv,
            state: .init(state, offset: 2 * kvWidth * 12), out: row(attended),
            residual: residual.map { row($0) }, channels: hidden, tokens: t, encoder: e)
        return attended
    }

    /// The depth-mixed, loop-controlled, routed MoE with its post-convolution
    /// and residual. Returns the new `cur` in a persistent slab.
    private func encodeFeedForward(
        visit: Int, layer: Inkling2GGUFLayerWeights, ffnInput: Int, blocks: [Int],
        message: Int?, _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let m = manifest
        let s = state
        let kvWidth = m.layers[layer.index].kvHeads * m.headDimension
        let moeInput: Int
        if let message {
            let mixed = try mix(blocks + [ffnInput], query: layer.ffnResQuery.operand, count: 1, e)
            let controlled = try rows(1)
            try ops.controlDown(
                cur: row(mixed), message: row(message), down: layer.loopDown[3],
                h: scratch(s.controlHidden), joined: scratch(s.controlJoined), width: hidden,
                tokens: tokens, encoder: e)
            try ops.controlApply(
                cur: row(mixed), message: row(message), h: scratch(s.controlHidden),
                gain: layer.loopGain[3], forget: layer.loopForget[3], out: row(controlled),
                width: hidden, tokens: tokens, encoder: e)
            moeInput = try rmsInto(controlled, layer.ffnNorm, e)
        } else {
            moeInput = try mix(
                blocks + [ffnInput], query: layer.ffnResQuery.operand, count: 1,
                norm: (layer.ffnNorm.operand, m.rmsEpsilon), e)
        }
        try encodeMoE(layer: layer, input: moeInput, e)
        // The post-convolution reads the routed experts' weighted sum and adds
        // the FFN input.
        let out = try persistentSlab()
        try ops.shortConv(
            x: scratch(s.down), weight: layer.ffnPostConv,
            state: .init(s.convStates[visit], offset: (2 * kvWidth + hidden) * 12),
            out: row(out), routeWeights: scratch(s.routeWeights), slots: m.expertsPerToken,
            residual: row(ffnInput), channels: hidden, tokens: tokens, encoder: e)
        return out
    }

    /// Router, gate_up, SiTU and down; the pair outputs stay in `state.down`.
    private func encodeMoE(
        layer: Inkling2GGUFLayerWeights, input: Int, _ e: MTLComputeCommandEncoder
    ) throws {
        let m = manifest
        let s = state
        let used = m.expertsPerToken
        let feedForward = m.layers[layer.index].expertFeedForward
        let routerArgs = Inkling2GGUFOps.RouterArgs(
            experts: UInt32(m.expertCount), width: UInt32(hidden), used: UInt32(used),
            weightScale: m.expertWeightsScale, normEpsilon: 1e-20)
        try ops.route(
            router: layer.router, bias: layer.routerBias, x: row(input),
            logits: scratch(s.routerLogits), ids: scratch(s.routeIDs),
            weights: scratch(s.routeWeights), args: routerArgs, tokens: tokens, encoder: e)
        let situArgs = Inkling2GGUFOps.SituArgs(
            feedForward: UInt32(feedForward), slots: UInt32(used * tokens), beta: m.situBeta,
            linearBeta: m.situLinearBeta)
        let grouping = (lists: scratch(s.expertLists), counts: scratch(s.expertCounts))
        try ops.experts(
            gateUp: layer.gateUp, down: layer.down, gateUpScale: layer.gateUpScale?.operand,
            downScale: layer.downScale?.operand, experts: m.expertCount, used: used,
            feedForward: feedForward, hidden: hidden, tokens: tokens, input: row(input),
            ids: scratch(s.routeIDs), gateUpOut: scratch(s.gateUp),
            activation: scratch(s.activation), downOut: scratch(s.down), grouping: grouping,
            situ: situArgs, encoder: e)
    }
}
