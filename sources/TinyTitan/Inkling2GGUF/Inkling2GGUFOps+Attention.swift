import Foundation
import Metal
import TinyTitanCore

extension Inkling2GGUFOps {
    struct ConvArgs {
        var channels: UInt32
        var slots: UInt32
        var hasResidual: UInt32
        var scale: Float
        var tokens: UInt32
    }
    struct KVFinishArgs {
        var headDim: UInt32
        var kvHeads: UInt32
        var hasDynamic: UInt32
        var capacity: UInt32
        var ring: UInt32
        var threshold: Float
        var epsilon: Float
        var scale: Float
        var tokens: UInt32
    }
    struct ClipArgs {
        var headDim: UInt32
        var hasScale: UInt32
        var applyTau: UInt32
        var threshold: Float
        var tauFloor: Float
        var tauAlpha: Float
        var heads: UInt32
    }
    struct AttentionArgs {
        var heads: UInt32
        var kvHeads: UInt32
        var headDim: UInt32
        var relative: UInt32
        var extent: UInt32
        var capacity: UInt32
        var ring: UInt32
        var scale: Float
        var window: UInt32
        var tokens: UInt32
    }
    struct RouterArgs {
        var experts: UInt32
        var width: UInt32
        var used: UInt32
        var weightScale: Float
        var normEpsilon: Float
    }
    struct SituArgs {
        var feedForward: UInt32
        var slots: UInt32
        var beta: Float
        var linearBeta: Float
    }

    /// Short convolution over `channels` for the chunk's tokens in order,
    /// updating `state` ([channels][3]). With `routeWeights`, each token's
    /// input is the weighted sum of its `slots` rows of `x`; with `residual`,
    /// it is added to the output.
    func shortConv(
        x: Operand, weight: Inkling2GGUFTensor, state: Operand, out: Operand,
        routeWeights: Operand? = nil, slots: Int = 0, residual: Operand? = nil, channels: Int,
        tokens: Int, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("conv") { return }
        let chunk = tokens > 1
        if chunk { try requireSequential("chunk convolution") }
        try pipeline(chunk ? "short_conv_chunk" : "short_conv_step", encoder)
        set(x, 0, encoder)
        set(weight.operand, 1, encoder)
        set(state, 2, encoder)
        set(out, 3, encoder)
        set(routeWeights ?? x, 4, encoder)
        set(residual ?? x, 5, encoder)
        var args = ConvArgs(
            channels: UInt32(channels), slots: routeWeights == nil ? 0 : UInt32(slots),
            hasResidual: residual == nil ? 0 : 1, scale: 1 / Float(5).squareRoot(),
            tokens: UInt32(tokens))
        encoder.setBytes(&args, length: MemoryLayout<ConvArgs>.stride, index: 6)
        if !chunk {
            threads(channels, encoder)
            return
        }
        // Every (token, channel) output in parallel, then the state shift once
        // all reads of the old state are done; buffers stay bound.
        threads(channels * tokens, encoder)
        try pipeline("short_conv_state", encoder)
        threads(channels, encoder)
    }

    /// K and V convolutions, the K clip, the V norm and the cache store for
    /// the chunk's tokens, in one dispatch (one threadgroup per K or V head).
    func kvFinish(
        k: Operand, v: Operand, kWeight: Inkling2GGUFTensor, vWeight: Inkling2GGUFTensor,
        state: Operand, dynamicK: Operand?, dynamicV: Operand?, kCache: MTLBuffer,
        vCache: MTLBuffer, step: Operand, args: KVFinishArgs, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("conv") { return }
        // A chunk: every (head, token) in parallel; the state shift is
        // `kvFinishState`, after the group this joins.
        try pipeline(args.tokens > 1 ? "kv_finish_chunk" : "kv_finish", encoder)
        set(k, 0, encoder)
        set(v, 1, encoder)
        set(kWeight.operand, 2, encoder)
        set(vWeight.operand, 3, encoder)
        set(state, 4, encoder)
        set(dynamicK ?? k, 5, encoder)
        set(dynamicV ?? v, 6, encoder)
        encoder.setBuffer(kCache, offset: 0, index: 7)
        encoder.setBuffer(vCache, offset: 0, index: 8)
        set(step, 9, encoder)
        var copy = args
        encoder.setBytes(&copy, length: MemoryLayout<KVFinishArgs>.stride, index: 10)
        groups(
            2 * Int(args.kvHeads) * (args.tokens > 1 ? Int(args.tokens) : 1),
            threads: Int(args.headDim), encoder)
    }

    /// A chunk's K/V convolution state shift (the last three inputs), after
    /// `kvFinish` has read the old state. Nothing to do at one token, where
    /// kv_finish shifts the state itself.
    func kvFinishState(
        k: Operand, v: Operand, state: Operand, args: KVFinishArgs,
        encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("conv") || args.tokens <= 1 { return }
        try pipeline("kv_finish_state", encoder)
        set(k, 0, encoder)
        set(v, 1, encoder)
        set(state, 4, encoder)
        var copy = args
        encoder.setBytes(&copy, length: MemoryLayout<KVFinishArgs>.stride, index: 10)
        threads(2 * Int(args.kvHeads * args.headDim), encoder)
    }

    /// Per-(head, token) L2 clip, optionally scaled, optionally times the
    /// log-position tau.
    func clipHeads(
        _ v: Operand, heads: Int, headDim: Int, scale: Inkling2GGUFTensor?, threshold: Float,
        tau: (floor: Float, alpha: Float)?, step: Operand, tokens: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("clip") { return }
        try pipeline("clip_heads", encoder)
        set(v, 0, encoder)
        set(scale?.operand ?? v, 1, encoder)
        set(step, 2, encoder)
        var args = ClipArgs(
            headDim: UInt32(headDim), hasScale: scale == nil ? 0 : 1,
            applyTau: tau == nil ? 0 : 1, threshold: threshold,
            tauFloor: tau?.floor ?? 1, tauAlpha: tau?.alpha ?? 0, heads: UInt32(heads))
        encoder.setBytes(&args, length: MemoryLayout<ClipArgs>.stride, index: 3)
        groups(heads, tokens: tokens, threads: 32, encoder)
    }

    func attention(
        q: Operand, kCache: MTLBuffer, vCache: MTLBuffer, r: Operand,
        relativeBias: Inkling2GGUFTensor, sinks: Operand, gate: Operand, out: Operand,
        step: Operand, args: AttentionArgs, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("attention") { return }
        // A chunk runs the tiled kernel (16 queries x 16-key blocks on MMA).
        let tiled = args.tokens > 1 && args.headDim == 128
        try pipeline(tiled ? "relative_attention_chunk" : "relative_attention_decode", encoder)
        set(q, 0, encoder)
        encoder.setBuffer(kCache, offset: 0, index: 1)
        encoder.setBuffer(vCache, offset: 0, index: 2)
        set(r, 3, encoder)
        set(relativeBias.operand, 4, encoder)
        set(sinks, 5, encoder)
        set(out, 6, encoder)
        set(step, 7, encoder)
        var copy = args
        encoder.setBytes(&copy, length: MemoryLayout<AttentionArgs>.stride, index: 8)
        set(gate, 9, encoder)
        if tiled {
            groups(Int(args.heads) * ((Int(args.tokens) + 15) / 16), threads: 128, encoder)
            return
        }
        // relative_attention_decode's partial[8][128] assumes eight SIMD groups.
        groups(Int(args.heads), tokens: Int(args.tokens), threads: 256, encoder)
    }

    /// Router logits (GEMV, or GEMM for a chunk), then the sigmoid top-k, one
    /// SIMD group per token. `logits` holds tokens x experts floats.
    func route(
        router: Inkling2GGUFTensor, bias: Inkling2GGUFTensor, x: Operand, logits: Operand,
        ids: Operand, weights: Operand, args: RouterArgs, tokens: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        try requireSequential("route")
        if skipped.contains("router") { return }
        guard args.experts <= 256, args.used <= 32 else {
            throw MetalError.invalidState("router: \(args.experts) experts, top-\(args.used)")
        }
        try gemv(router, x: x, y: logits, tokens: tokens, encoder: encoder)
        try pipeline("sigmoid_topk", encoder)
        set(logits, 0, encoder)
        set(bias.operand, 1, encoder)
        set(ids, 2, encoder)
        set(weights, 3, encoder)
        var copy = args
        encoder.setBytes(&copy, length: MemoryLayout<RouterArgs>.stride, index: 4)
        groups(tokens, threads: 32, encoder)
    }

    /// The routed expert GEMVs at one token; at a chunk, the pairs are grouped
    /// by expert once and both GEMMs read each used expert's weights once per
    /// token tile of its pairs. `pairs` = tokens x used; outputs are pair-major.
    /// Top-k routing picks distinct experts per token, so no expert holds more
    /// than `tokens` pairs: that bounds the grid on the CPU (empty tiles exit),
    /// where a GPU-written indirect grid raced with the dispatch producing it.
    func experts(
        gateUp: Inkling2GGUFTensor, down: Inkling2GGUFTensor, gateUpScale: Operand?,
        downScale: Operand?, experts: Int, used: Int, feedForward: Int, hidden: Int,
        tokens: Int, input: Operand, ids: Operand, gateUpOut: Operand, activation: Operand,
        downOut: Operand, grouping: (lists: Operand, counts: Operand),
        situ situArgs: SituArgs, encoder: MTLComputeCommandEncoder
    ) throws {
        try requireSequential("experts")
        let gateUpStride = gateUp.byteCount / experts
        let downStride = down.byteCount / experts
        let pairs = tokens * used
        let tile = GGUFGemmTiled.expertTileTokens(pairs: pairs, experts: experts)
        if !skipped.contains("experts") {
            if tokens == 1 {
                order(encoder)
                try gemvSlots(
                    .init(
                        type: gateUp.type, rows: 2 * feedForward, columns: hidden, slots: used,
                        expertStride: gateUpStride, xSlotStride: 0),
                    weights: gateUp.operand, x: input, y: gateUpOut, ids: ids,
                    scales: gateUpScale, encoder: encoder)
            } else {
                order(encoder)
                gemm.encodeGrouping(
                    ids: ids, pairs: pairs, experts: experts, lists: grouping.lists,
                    counts: grouping.counts, listStride: pairs,
                    barrier: { encoder.memoryBarrier(scope: .buffers) }, encoder: encoder)
                order(encoder)
                try expertsChunk(
                    type: gateUp.type, weights: gateUp.operand, rows: 2 * feedForward,
                    columns: hidden, experts: experts, expertStride: gateUpStride,
                    tokens: tokens, xDivisor: used, x: input, y: gateUpOut, grouping: grouping,
                    scales: gateUpScale, pairs: pairs, tile: tile, encoder: encoder)
            }
        }
        try situ(gateUp: gateUpOut, act: activation, args: situArgs, encoder: encoder)
        if skipped.contains("experts") { return }
        order(encoder)
        if tokens == 1 {
            try gemvSlots(
                .init(
                    type: down.type, rows: hidden, columns: feedForward, slots: used,
                    expertStride: downStride, xSlotStride: feedForward),
                weights: down.operand, x: activation, y: downOut, ids: ids, scales: downScale,
                encoder: encoder)
        } else {
            try expertsChunk(
                type: down.type, weights: down.operand, rows: hidden, columns: feedForward,
                experts: experts, expertStride: downStride, tokens: tokens, xDivisor: 1,
                x: activation, y: downOut, grouping: grouping, scales: downScale,
                pairs: pairs, tile: tile, encoder: encoder)
        }
    }

    /// One expert matrix over a chunk's grouped pairs on the tiled GEMM. (A
    /// register kernel without threadgroup staging, for experts holding few
    /// pairs, measured 6x slower at pp64: every row group re-read the tokens'
    /// activations; staging up to 8 tokens' activations per 512-column chunk in
    /// threadgroup memory instead was still 5x slower, 54-60 ms vs 11 ms per
    /// routed layer, the per-lane fp32 form losing to the MMA tile.)
    private func expertsChunk(
        type: GGUFType, weights: Operand, rows: Int, columns: Int, experts: Int,
        expertStride: Int, tokens: Int, xDivisor: Int, x: Operand, y: Operand,
        grouping: (lists: Operand, counts: Operand), scales: Operand?, pairs: Int, tile: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        try gemm.encodeExperts(
            type: type, weights: weights, rows: rows, columns: columns, experts: experts,
            expertStride: expertStride, maxPairsPerExpert: tokens, xDivisor: xDivisor, x: x,
            y: y, lists: grouping.lists, counts: grouping.counts, scales: scales,
            listStride: pairs, tileTokens: tile, encoder: encoder)
    }

    func situ(
        gateUp: Operand, act: Operand, args: SituArgs, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("elementwise") { return }
        try pipeline("situ_activation", encoder)
        set(gateUp, 0, encoder)
        set(act, 1, encoder)
        var copy = args
        encoder.setBytes(&copy, length: MemoryLayout<SituArgs>.stride, index: 2)
        threads(Int(args.feedForward * args.slots), encoder)
    }

    /// token = argmax(logits[0..<count]); `partials` holds `groupCount` float2.
    func argmax(
        logits: Operand, count: Int, partials: Operand, groupCount: Int, token: Operand,
        encoder: MTLComputeCommandEncoder
    ) throws {
        try requireSequential("argmax")
        var n = UInt32(count)
        try pipeline("argmax_partial", encoder)
        set(logits, 0, encoder)
        set(partials, 1, encoder)
        encoder.setBytes(&n, length: 4, index: 2)
        groups(groupCount, threads: 256, encoder)
        try pipeline("argmax_final", encoder)
        set(partials, 0, encoder)
        set(token, 1, encoder)
        var g = UInt32(groupCount)
        encoder.setBytes(&g, length: 4, index: 2)
        groups(1, threads: 32, encoder)
    }
}
