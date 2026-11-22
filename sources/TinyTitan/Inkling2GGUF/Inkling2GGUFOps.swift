import Foundation
import Metal
import TinyTitanCore

/// Encoders for the single-token depth-decoder kernels
/// (`depth_decode_ops.metal`, `depth_decode_attention.metal`) and the GGUF
/// GEMVs. Every method appends dispatches to one compute encoder; nothing here
/// waits on the GPU.
final class Inkling2GGUFOps {
    typealias Operand = GGUFGemvIds.Operand

    private struct RMSArgs {
        var width: UInt32
        var inStride: UInt32
        var outStride: UInt32
        var hasWeight: UInt32
        var epsilon: Float
    }
    private struct MixArgs {
        var width: UInt32
        var stateCount: UInt32
        var queryCount: UInt32
        var epsilon: Float
        var normWeighted: UInt32
        var normEpsilon: Float
    }
    private struct ControlArgs {
        var width: UInt32
        var rank: UInt32
        var epsilon: Float
    }
    private struct ConvArgs {
        var channels: UInt32
        var slots: UInt32
        var hasResidual: UInt32
        var scale: Float
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
    }
    private struct ClipArgs {
        var headDim: UInt32
        var hasScale: UInt32
        var applyTau: UInt32
        var threshold: Float
        var tauFloor: Float
        var tauAlpha: Float
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

    let gemvIds: GGUFGemvIds
    /// Profiling ablation: op categories left out of the encoded token, so the
    /// GPU-time difference attributes cost. Tokens produced with any category
    /// skipped are meaningless. Categories: mix, control, conv, clip, attention,
    /// router, smallGemv, projection, rms, elementwise, experts, head.
    var skipped: Set<String> = []
    private var pipelines: [String: MTLComputePipelineState] = [:]
    /// A one-element buffer holding expert id 0, for dense GGUF GEMVs.
    private let zeroID: MTLBuffer

    init(context: MetalContext) throws {
        gemvIds = try GGUFGemvIds(context: context)
        for name in [
            "embed_q8_0_row", "rms_norm_rows", "dense_gemv_f32_bf16", "depth_mix",
            "control_join", "control_apply", "short_conv_step", "clip_heads", "kv_finish",
            "relative_attention_decode", "sigmoid_topk", "situ_activation",
            "argmax_partial", "argmax_final",
        ] {
            pipelines[name] = try context.pipeline(name)
        }
        var zero: Int32 = 0
        guard
            let buffer = context.device.makeBuffer(
                bytes: &zero, length: 16, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("zero expert id") }
        zeroID = buffer
    }

    private func pipeline(_ name: String, _ encoder: MTLComputeCommandEncoder) throws {
        guard let state = pipelines[name] else { throw MetalError.missingFunction(name) }
        encoder.setComputePipelineState(state)
    }

    private func set(_ operand: Operand, _ index: Int, _ encoder: MTLComputeCommandEncoder) {
        encoder.setBuffer(operand.buffer, offset: operand.offset, index: index)
    }

    // MARK: Ordering

    /// Inside `concurrent(_:_:)`; dispatches then share one barrier.
    private var inGroup = false
    private var groupNeedsBarrier = false

    /// The token is encoded with `MTLDispatchType.concurrent`, so every
    /// dispatch is ordered after the previous ones by a buffer barrier unless
    /// it belongs to a group of mutually independent dispatches.
    private func order(_ encoder: MTLComputeCommandEncoder) {
        if !inGroup || groupNeedsBarrier {
            encoder.memoryBarrier(scope: .buffers)
            groupNeedsBarrier = false
        }
    }

    /// Encodes `body`'s dispatches after one barrier and without barriers
    /// between them: they must read only data written before the group and
    /// write disjoint buffers. Multi-dispatch ops (control, route, argmax)
    /// may not appear inside a group.
    func concurrent(
        _ encoder: MTLComputeCommandEncoder, _ body: () throws -> Void
    ) rethrows {
        precondition(!inGroup, "concurrent groups do not nest")
        inGroup = true
        groupNeedsBarrier = true
        defer { inGroup = false }
        try body()
    }

    private func requireSequential(_ name: String) throws {
        guard !inGroup else {
            throw MetalError.invalidState(
                "\(name) has internal dependencies; not in a concurrent group")
        }
    }

    private func threads(_ count: Int, _ encoder: MTLComputeCommandEncoder, width: Int = 256) {
        order(encoder)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(width, max(count, 1)), height: 1, depth: 1))
    }

    private func groups(
        _ count: Int, threads: Int, _ encoder: MTLComputeCommandEncoder
    ) {
        order(encoder)
        encoder.dispatchThreadgroups(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    /// y = W . x for a dense tensor of any supported type.
    func gemv(
        _ weight: Inkling2GGUFTensor, x: Operand, y: Operand, encoder: MTLComputeCommandEncoder
    ) throws {
        let category = [.f32, .bf16].contains(weight.type) ? "smallGemv" : "projection"
        if skipped.contains(category) { return }
        switch weight.type {
        case .f32, .bf16:
            try pipeline("dense_gemv_f32_bf16", encoder)
            set(weight.operand, 0, encoder)
            set(x, 1, encoder)
            set(y, 2, encoder)
            var shape = (UInt32(weight.rows), UInt32(weight.columns))
            encoder.setBytes(&shape, length: 8, index: 3)
            var bf16: UInt32 = weight.type == .bf16 ? 1 : 0
            encoder.setBytes(&bf16, length: 4, index: 4)
            groups((weight.rows + 3) / 4, threads: 128, encoder)
        default:
            order(encoder)
            try gemvIds.encode(
                .init(
                    type: weight.type, rows: weight.rows, columns: weight.columns, slots: 1,
                    expertStride: 0, xSlotStride: 0),
                weights: weight.operand, x: x, y: y, ids: .init(zeroID), scales: nil,
                encoder: encoder)
        }
    }

    /// A routed expert GEMV (`GGUFGemvIds`), ordered like every other dispatch.
    func experts(
        _ shape: GGUFGemvIds.Shape, weights: Operand, x: Operand, y: Operand, ids: Operand,
        scales: Operand?, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("experts") { return }
        order(encoder)
        try gemvIds.encode(
            shape, weights: weights, x: x, y: y, ids: ids, scales: scales, encoder: encoder)
    }

    func embed(
        _ table: Inkling2GGUFTensor, token: Operand, out: Operand,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard table.type == .q8_0 else { throw GGUFDequantError.unsupportedType(table.type.name) }
        try pipeline("embed_q8_0_row", encoder)
        set(table.operand, 0, encoder)
        set(token, 1, encoder)
        set(out, 2, encoder)
        var columns = UInt32(table.columns)
        encoder.setBytes(&columns, length: 4, index: 3)
        threads(table.columns, encoder)
    }

    /// RMS-normalises `count` vectors of `width` (strides in floats).
    func rmsNorm(
        _ x: Operand, weight: Operand?, out: Operand, width: Int, count: Int = 1,
        stride: Int? = nil, epsilon: Float, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("rms") { return }
        try pipeline("rms_norm_rows", encoder)
        set(x, 0, encoder)
        set(weight ?? x, 1, encoder)
        set(out, 2, encoder)
        var args = RMSArgs(
            width: UInt32(width), inStride: UInt32(stride ?? width),
            outStride: UInt32(stride ?? width), hasWeight: weight == nil ? 0 : 1, epsilon: epsilon)
        encoder.setBytes(&args, length: MemoryLayout<RMSArgs>.stride, index: 3)
        groups(count, threads: min(256, max(32, (width + 31) / 32 * 32)), encoder)
    }

    /// out[q] = depth mix of the pool rows `rows` with query q of `queries`.
    /// With `norm` (one query only), the output is RMS-normalised and scaled by
    /// it in the same kernel.
    func depthMix(
        pool: MTLBuffer, rows: [Int], queries: Operand, queryCount: Int, out: Operand,
        width: Int, norm: (weight: Operand, epsilon: Float)? = nil,
        encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("mix") { return }
        guard !rows.isEmpty, rows.count <= 32, norm == nil || queryCount == 1 else {
            throw MetalError.invalidState("depth mix over \(rows.count) states")
        }
        try pipeline("depth_mix", encoder)
        encoder.setBuffer(pool, offset: 0, index: 0)
        set(queries, 1, encoder)
        set(out, 2, encoder)
        var args = MixArgs(
            width: UInt32(width), stateCount: UInt32(rows.count),
            queryCount: UInt32(queryCount), epsilon: 1e-6, normWeighted: norm == nil ? 0 : 1,
            normEpsilon: norm?.epsilon ?? 0)
        encoder.setBytes(&args, length: MemoryLayout<MixArgs>.stride, index: 3)
        var indices = rows.map(UInt32.init)
        encoder.setBytes(&indices, length: indices.count * 4, index: 4)
        set(norm?.weight ?? queries, 5, encoder)
        groups(queryCount, threads: 1_024, encoder)
    }

    /// out = loop message controller(cur, message). `joined` holds 2 * width
    /// floats and `h` the controller rank.
    func control(
        cur: Operand, message: Operand, down: Inkling2GGUFTensor, gain: Inkling2GGUFTensor,
        forget: Inkling2GGUFTensor, joined: Operand, h: Operand, out: Operand, width: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        try requireSequential("control")
        if skipped.contains("control") { return }
        var args = ControlArgs(width: UInt32(width), rank: UInt32(down.rows), epsilon: 1e-6)
        try pipeline("control_join", encoder)
        set(cur, 0, encoder)
        set(message, 1, encoder)
        set(joined, 2, encoder)
        encoder.setBytes(&args, length: MemoryLayout<ControlArgs>.stride, index: 3)
        groups(1, threads: 1_024, encoder)
        try gemv(down, x: joined, y: h, encoder: encoder)
        try pipeline("control_apply", encoder)
        set(cur, 0, encoder)
        set(message, 1, encoder)
        set(h, 2, encoder)
        set(gain.operand, 3, encoder)
        set(forget.operand, 4, encoder)
        set(out, 5, encoder)
        encoder.setBytes(&args, length: MemoryLayout<ControlArgs>.stride, index: 6)
        groups((width + 3) / 4, threads: 128, encoder)
    }

    /// Short convolution step over `channels`, updating `state` ([channels][3]).
    /// With `routeWeights`, the input is the weighted sum of `slots` rows of `x`;
    /// with `residual`, it is added to the output.
    func shortConv(
        x: Operand, weight: Inkling2GGUFTensor, state: Operand, out: Operand,
        routeWeights: Operand? = nil, slots: Int = 0, residual: Operand? = nil, channels: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("conv") { return }
        try pipeline("short_conv_step", encoder)
        set(x, 0, encoder)
        set(weight.operand, 1, encoder)
        set(state, 2, encoder)
        set(out, 3, encoder)
        set(routeWeights ?? x, 4, encoder)
        set(residual ?? x, 5, encoder)
        var args = ConvArgs(
            channels: UInt32(channels), slots: routeWeights == nil ? 0 : UInt32(slots),
            hasResidual: residual == nil ? 0 : 1, scale: 1 / Float(5).squareRoot())
        encoder.setBytes(&args, length: MemoryLayout<ConvArgs>.stride, index: 6)
        threads(channels, encoder)
    }

    /// K and V convolutions, the K clip, the V norm and the cache store, in
    /// one dispatch (one threadgroup per K or V head).
    func kvFinish(
        k: Operand, v: Operand, kWeight: Inkling2GGUFTensor, vWeight: Inkling2GGUFTensor,
        state: Operand, dynamicK: Operand?, dynamicV: Operand?, kCache: MTLBuffer,
        vCache: MTLBuffer, step: Operand, args: KVFinishArgs, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("conv") { return }
        try pipeline("kv_finish", encoder)
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
        groups(2 * Int(args.kvHeads), threads: Int(args.headDim), encoder)
    }

    /// Per-head L2 clip (optionally scaled, optionally times the log-position tau).
    func clipHeads(
        _ v: Operand, heads: Int, headDim: Int, scale: Inkling2GGUFTensor?, threshold: Float,
        tau: (floor: Float, alpha: Float)?, step: Operand, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("clip") { return }
        try pipeline("clip_heads", encoder)
        set(v, 0, encoder)
        set(scale?.operand ?? v, 1, encoder)
        set(step, 2, encoder)
        var args = ClipArgs(
            headDim: UInt32(headDim), hasScale: scale == nil ? 0 : 1,
            applyTau: tau == nil ? 0 : 1, threshold: threshold,
            tauFloor: tau?.floor ?? 1, tauAlpha: tau?.alpha ?? 0)
        encoder.setBytes(&args, length: MemoryLayout<ClipArgs>.stride, index: 3)
        groups(heads, threads: 32, encoder)
    }

    func attention(
        q: Operand, kCache: MTLBuffer, vCache: MTLBuffer, r: Operand,
        relativeBias: Inkling2GGUFTensor, sinks: Operand, gate: Operand, out: Operand,
        step: Operand, logits: Operand, args: AttentionArgs, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("attention") { return }
        try pipeline("relative_attention_decode", encoder)
        set(q, 0, encoder)
        encoder.setBuffer(kCache, offset: 0, index: 1)
        encoder.setBuffer(vCache, offset: 0, index: 2)
        set(r, 3, encoder)
        set(relativeBias.operand, 4, encoder)
        set(sinks, 5, encoder)
        set(out, 6, encoder)
        set(step, 7, encoder)
        set(logits, 8, encoder)
        var copy = args
        encoder.setBytes(&copy, length: MemoryLayout<AttentionArgs>.stride, index: 9)
        set(gate, 10, encoder)
        // relative_attention_decode's partial[8][128] assumes eight SIMD groups.
        groups(Int(args.heads), threads: 256, encoder)
    }

    /// Router logits on the parallel dense GEMV, then the sigmoid top-k in
    /// one SIMD group. `logits` holds `experts` floats.
    func route(
        router: Inkling2GGUFTensor, bias: Inkling2GGUFTensor, x: Operand, logits: Operand,
        ids: Operand, weights: Operand, args: RouterArgs, encoder: MTLComputeCommandEncoder
    ) throws {
        try requireSequential("route")
        if skipped.contains("router") { return }
        guard args.experts <= 256, args.used <= 32 else {
            throw MetalError.invalidState("router: \(args.experts) experts, top-\(args.used)")
        }
        try gemv(router, x: x, y: logits, encoder: encoder)
        try pipeline("sigmoid_topk", encoder)
        set(logits, 0, encoder)
        set(bias.operand, 1, encoder)
        set(ids, 2, encoder)
        set(weights, 3, encoder)
        var copy = args
        encoder.setBytes(&copy, length: MemoryLayout<RouterArgs>.stride, index: 4)
        groups(1, threads: 32, encoder)
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
