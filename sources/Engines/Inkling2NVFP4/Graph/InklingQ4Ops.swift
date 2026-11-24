// Provenance: copied from sources/TinyTitan/Inkling2GGUF/Inkling2GGUFOps.swift
// (TT-121 phase 2) under the variant-isolation rule.
import Foundation
import Metal
import TinyTitanCore

/// Encoders for the depth-decoder kernels (`depth_decode_ops.metal`,
/// `depth_decode_attention.metal`) and the GGUF GEMVs/GEMMs, for a chunk of
/// `tokens` positions: one at decode (bandwidth-bound GEMVs), more at prefill
/// (tiled GEMMs that decode each weight block once per 32 tokens). Every method
/// appends dispatches to one compute encoder; nothing here waits on the GPU.
final class InklingQ4Ops {
    typealias Operand = GGUFGemvIds.Operand

    struct RMSArgs {
        var width: UInt32
        var inStride: UInt32
        var outStride: UInt32
        var hasWeight: UInt32
        var epsilon: Float
    }
    struct MixArgs {
        var width: UInt32
        var stateCount: UInt32
        var queryCount: UInt32
        var epsilon: Float
        var normWeighted: UInt32
        var normEpsilon: Float
        var tokens: UInt32
        var slab: UInt32
    }
    struct ControlArgs {
        var width: UInt32
        var rank: UInt32
        var epsilon: Float
    }

    let gemvIds: GGUFGemvIds
    let gemm: GGUFGemmTiled
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
        gemm = try GGUFGemmTiled(context: context)
        for name in [
            "embed_q8_0_row", "embed_bf16_row", "rms_norm_rows", "dense_gemv_f32_bf16",
            "dense_gemv_wide_rows", "dense_gemm_small_rows", "control_join", "control_apply_chunk",
            "short_conv_chunk", "short_conv_state", "kv_finish_chunk", "kv_finish_state",
            "depth_mix",
            "control_down", "control_apply", "short_conv_step", "clip_heads", "kv_finish",
            "relative_attention_decode", "relative_attention_chunk", "sigmoid_topk",
            "situ_activation",
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

    func pipeline(_ name: String, _ encoder: MTLComputeCommandEncoder) throws {
        guard let state = pipelines[name] else { throw MetalError.missingFunction(name) }
        encoder.setComputePipelineState(state)
    }

    func set(_ operand: Operand, _ index: Int, _ encoder: MTLComputeCommandEncoder) {
        encoder.setBuffer(operand.buffer, offset: operand.offset, index: index)
    }

    // MARK: Ordering

    /// Inside `concurrent(_:_:)`; dispatches then share one barrier.
    private var inGroup = false
    private var groupNeedsBarrier = false

    /// The token is encoded with `MTLDispatchType.concurrent`, so every
    /// dispatch is ordered after the previous ones by a buffer barrier unless
    /// it belongs to a group of mutually independent dispatches.
    func order(_ encoder: MTLComputeCommandEncoder) {
        if !inGroup || groupNeedsBarrier {
            encoder.memoryBarrier(scope: .buffers)
            groupNeedsBarrier = false
        }
    }

    /// Encodes `body`'s dispatches after one barrier and without barriers
    /// between them: they must read only data written before the group and
    /// write disjoint buffers. Multi-dispatch ops (route, the MoE, argmax,
    /// and a chunk's controller down) may not appear inside a group.
    func concurrent(
        _ encoder: MTLComputeCommandEncoder, _ body: () throws -> Void
    ) rethrows {
        precondition(!inGroup, "concurrent groups do not nest")
        inGroup = true
        groupNeedsBarrier = true
        defer { inGroup = false }
        try body()
    }

    func requireSequential(_ name: String) throws {
        guard !inGroup else {
            throw MetalError.invalidState(
                "\(name) has internal dependencies; not in a concurrent group")
        }
    }

    func threads(
        _ count: Int, tokens: Int = 1, _ encoder: MTLComputeCommandEncoder, width: Int = 256
    ) {
        order(encoder)
        encoder.dispatchThreads(
            MTLSize(width: count, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(width, max(count, 1)), height: 1, depth: 1))
    }

    /// `count` threadgroups per token, as one flat grid (token-major): the
    /// kernels decode (item, token) themselves, because Metal will not mix a
    /// vector threadgroup position with the scalar SIMD-group attributes.
    func groups(
        _ count: Int, tokens: Int = 1, threads: Int, _ encoder: MTLComputeCommandEncoder
    ) {
        order(encoder)
        encoder.dispatchThreadgroups(
            MTLSize(width: count * tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    // MARK: Projections

    /// y[t] = W . x[t] for `tokens` token-major rows of `x`: a GEMV at one
    /// token, the tiled GEMM for a chunk.
    func gemv(
        _ weight: InklingQ4Tensor, x: Operand, y: Operand, tokens: Int = 1,
        encoder: MTLComputeCommandEncoder
    ) throws {
        let category = [.f32, .bf16].contains(weight.type) ? "smallGemv" : "projection"
        if skipped.contains(category) { return }
        // Small f32/bf16 matrices stay off the tiled GEMM at a chunk: their
        // few row tiles would starve it (a 128-row router is 2 row tiles).
        let smallDense = [.f32, .bf16].contains(weight.type) && weight.rows < 512
        if tokens > 1, !smallDense {
            // One dispatch, so it may join a concurrent group.
            order(encoder)
            try gemm.encodeDense(
                type: weight.type, weights: weight.operand, rows: weight.rows,
                columns: weight.columns, tokens: tokens, x: x, y: y, encoder: encoder)
            return
        }
        let small = [.f32, .bf16].contains(weight.type)
        switch weight.type {
        case _ where small && tokens > 1 && weight.columns % 4 == 0:
            try smallRowsGemm(weight, x: x, y: y, tokens: tokens, encoder: encoder)
        case .f32, .bf16:
            // Under 256 rows, a threadgroup per row keeps the GPU filled.
            let wide = weight.rows < 256 && weight.columns % 4 == 0
            try pipeline(wide ? "dense_gemv_wide_rows" : "dense_gemv_f32_bf16", encoder)
            set(weight.operand, 0, encoder)
            set(x, 1, encoder)
            set(y, 2, encoder)
            var shape = (UInt32(weight.rows), UInt32(weight.columns))
            encoder.setBytes(&shape, length: 8, index: 3)
            var bf16: UInt32 = weight.type == .bf16 ? 1 : 0
            encoder.setBytes(&bf16, length: 4, index: 4)
            groups(
                wide ? weight.rows : (weight.rows + 3) / 4, tokens: tokens, threads: 128, encoder)
        default:
            order(encoder)
            try gemvSlots(
                .init(
                    type: weight.type, rows: weight.rows, columns: weight.columns, slots: 1,
                    expertStride: 0, xSlotStride: 0),
                weights: weight.operand, x: x, y: y, ids: .init(zeroID), scales: nil,
                encoder: encoder)
        }
    }

    /// y[t] = W . x[t] over a chunk for a small f32/bf16 W (any row count;
    /// columns % 4 == 0): 8-row x 8-token tiles, the columns split across the
    /// threadgroup's SIMD groups, so W is read once per token tile instead of
    /// once per token and small matrices still fill the GPU.
    func smallRowsGemm(
        _ weight: InklingQ4Tensor, x: Operand, y: Operand, tokens: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard [.f32, .bf16].contains(weight.type), weight.columns % 4 == 0 else {
            throw MetalError.invalidState(
                "small-rows GEMM: \(weight.type.name) x \(weight.columns) columns")
        }
        try pipeline("dense_gemm_small_rows", encoder)
        set(weight.operand, 0, encoder)
        set(x, 1, encoder)
        set(y, 2, encoder)
        var shape = (UInt32(weight.rows), UInt32(weight.columns))
        encoder.setBytes(&shape, length: 8, index: 3)
        var bf16: UInt32 = weight.type == .bf16 ? 1 : 0
        encoder.setBytes(&bf16, length: 4, index: 4)
        var count = UInt32(tokens)
        encoder.setBytes(&count, length: 4, index: 5)
        groups((weight.rows + 7) / 8 * ((tokens + 7) / 8), threads: 128, encoder)
    }

    /// The decode GEMV over `shape.slots` expert slots, on the dedicated
    /// NVFP4/Q8_0 kernels (the only block types this install carries).
    func gemvSlots(
        _ shape: GGUFGemvIds.Shape, weights: Operand, x: Operand, y: Operand, ids: Operand,
        scales: Operand?, encoder: MTLComputeCommandEncoder
    ) throws {
        try gemvIds.encode(
            shape, weights: weights, x: x, y: y, ids: ids, scales: scales, encoder: encoder)
    }

    /// Every weight type a decode step and a prefill chunk can run.
    static let supportedTypes = GGUFGemmTiled.supported.intersection(
        GGUFGemvIds.supported.union([.f32, .bf16]))

    func embed(
        _ table: InklingQ4Tensor, token: Operand, out: Operand, tokens: Int = 1,
        encoder: MTLComputeCommandEncoder
    ) throws {
        var columns = UInt32(table.columns)
        switch table.type {
        case .q8_0:
            try pipeline("embed_q8_0_row", encoder)
        case .bf16:
            // The checkpoint's embedding (TT-125).
            try pipeline("embed_bf16_row", encoder)
        default:
            throw GGUFDequantError.unsupportedType("\(table.type.name) token embedding")
        }
        set(table.operand, 0, encoder)
        set(token, 1, encoder)
        set(out, 2, encoder)
        encoder.setBytes(&columns, length: 4, index: 3)
        threads(table.columns, tokens: tokens, encoder)
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

    /// out slab q = depth mix of the pool slabs `rows` with query q, for every
    /// token of the chunk (`slab` floats per pool slab). With `norm` (one query
    /// only), the output is RMS-normalised and scaled in the same kernel.
    func depthMix(
        pool: Operand, rows: [Int], queries: Operand, queryCount: Int, out: Operand,
        width: Int, tokens: Int, slab: Int, norm: (weight: Operand, epsilon: Float)? = nil,
        encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("mix") { return }
        guard !rows.isEmpty, rows.count <= 32, norm == nil || queryCount == 1 else {
            throw MetalError.invalidState("depth mix over \(rows.count) states")
        }
        try pipeline("depth_mix", encoder)
        set(pool, 0, encoder)
        set(queries, 1, encoder)
        set(out, 2, encoder)
        var args = MixArgs(
            width: UInt32(width), stateCount: UInt32(rows.count),
            queryCount: UInt32(queryCount), epsilon: 1e-6, normWeighted: norm == nil ? 0 : 1,
            normEpsilon: norm?.epsilon ?? 0, tokens: UInt32(tokens), slab: UInt32(slab))
        encoder.setBytes(&args, length: MemoryLayout<MixArgs>.stride, index: 3)
        var indices = rows.map(UInt32.init)
        encoder.setBytes(&indices, length: indices.count * 4, index: 4)
        set(norm?.weight ?? queries, 5, encoder)
        groups(queryCount, tokens: tokens, threads: 1_024, encoder)
    }

    /// h = the loop controller's rank-`down.rows` projection of
    /// [rms(cur), rms(message)] for every token (`h` holds tokens x rank).
    /// At one token, one dispatch with no internal dependency (it may join a
    /// group); a chunk joins into `joined` and runs the small-rows GEMM.
    func controlDown(
        cur: Operand, message: Operand, down: InklingQ4Tensor, h: Operand,
        joined: Operand? = nil, width: Int, tokens: Int, encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("control") { return }
        guard down.type == .f32, down.columns == 2 * width else {
            throw GGUFDequantError.unsupportedType("loop controller down \(down.type.name)")
        }
        var args = ControlArgs(width: UInt32(width), rank: UInt32(down.rows), epsilon: 1e-6)
        if tokens > 1 {
            guard let joined else {
                throw MetalError.invalidState("a chunk's controller needs the join scratch")
            }
            try requireSequential("chunk controller")
            try controlJoin(
                cur: cur, message: message, joined: joined, width: width,
                tokens: tokens, encoder: encoder)
            try gemv(down, x: joined, y: h, tokens: tokens, encoder: encoder)
            return
        }
        try pipeline("control_down", encoder)
        set(cur, 0, encoder)
        set(message, 1, encoder)
        set(down.operand, 2, encoder)
        set(h, 3, encoder)
        encoder.setBytes(&args, length: MemoryLayout<ControlArgs>.stride, index: 4)
        groups(down.rows, tokens: tokens, threads: 256, encoder)
    }

    /// joined[t] = [rms(cur[t]), rms(message[t])], a chunk controller's input
    /// to its down projection (`gemv` with `tokens`). One dispatch; may join a
    /// group, as may that GEMM.
    func controlJoin(
        cur: Operand, message: Operand, joined: Operand, width: Int, tokens: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("control") { return }
        var args = ControlArgs(width: UInt32(width), rank: 0, epsilon: 1e-6)
        try pipeline("control_join", encoder)
        set(cur, 0, encoder)
        set(message, 1, encoder)
        set(joined, 2, encoder)
        encoder.setBytes(&args, length: MemoryLayout<ControlArgs>.stride, index: 3)
        groups(tokens, threads: 1_024, encoder)
    }

    /// out = (1 - tanh(F . h)) * cur + tanh(G . h) * message for every token,
    /// reading `controlDown`'s h. One dispatch; may join a group. (A chunk path
    /// of two small-rows GEMMs and a combine matched it to 1e-6 but bought
    /// nothing measurable at pp512 and moved a near-tie greedy token on
    /// Q2_K_XL, so it was rejected.)
    func controlApply(
        cur: Operand, message: Operand, h: Operand, gain: InklingQ4Tensor,
        forget: InklingQ4Tensor, out: Operand, width: Int, tokens: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        if skipped.contains("control") { return }
        var args = ControlArgs(width: UInt32(width), rank: UInt32(gain.columns), epsilon: 1e-6)
        if tokens > 1, gain.columns <= 64 {
            // A chunk: each SIMD group reuses its gain/forget rows across 16
            // tokens; same per-lane order and simd_sum, so bit-identical.
            try pipeline("control_apply_chunk", encoder)
            set(cur, 0, encoder)
            set(message, 1, encoder)
            set(h, 2, encoder)
            set(gain.operand, 3, encoder)
            set(forget.operand, 4, encoder)
            set(out, 5, encoder)
            encoder.setBytes(&args, length: MemoryLayout<ControlArgs>.stride, index: 6)
            var count = UInt32(tokens)
            encoder.setBytes(&count, length: 4, index: 7)
            groups((width + 3) / 4 * ((tokens + 15) / 16), threads: 128, encoder)
            return
        }
        try pipeline("control_apply", encoder)
        set(cur, 0, encoder)
        set(message, 1, encoder)
        set(h, 2, encoder)
        set(gain.operand, 3, encoder)
        set(forget.operand, 4, encoder)
        set(out, 5, encoder)
        encoder.setBytes(&args, length: MemoryLayout<ControlArgs>.stride, index: 6)
        groups((width + 3) / 4, tokens: tokens, threads: 128, encoder)
    }
}
