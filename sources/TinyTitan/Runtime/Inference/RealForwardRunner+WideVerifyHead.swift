import Foundation
import Metal

/// The W-row verify head and the multi-row pipeline prewarm
/// (docs/plan-wide-verify.md, task 7).
///
/// The head used to walk the pass two rows at a time, which read the
/// vocabulary matrix -- the model's largest tensor -- W/2 times per pass, or W
/// times on a hyper-connection stack. Here each row gets scalar decode's own
/// final norm (`encodeFinalNorm`'s kernels, the HC mixer or the RMSNorm) and
/// the lm_head runs once for all rows through `MultiRowGEMV`, which reproduces
/// `encodeHeadGEMV`'s per-row arithmetic. So every row's logits are the bytes
/// scalar decode's logits head gives that row, and the matrix is read once.
extension RealForwardRunner {
    /// What `encodeHeadGEMV` reads the lm_head with.
    var headGEMVFormat: MultiRowGEMV.Format {
        if headIsBF16 { return .bf16 }
        if let affineHead { return .affine(bits: affineHead.weightBits) }
        return .int4
    }

    /// Final norm and lm_head for `rows` residual rows of `hidden` (row
    /// stride `residualWidth`), into `logits` rows of `vocab` FP16 values.
    /// `scratch.normed` holds the normed rows in between.
    func encodeVerifyHeadRows(
        commandBuffer cb: MTLCommandBuffer,
        hidden: MTLBuffer, rows: Int,
        scratch: PrefillChunkScratchBuffers,
        logits: MTLBuffer, logitsOffset: Int = 0
    ) throws {
        precondition(rows <= scratch.layout.chunkTokens, "verify rows exceed the chunk scratch")
        let half = MemoryLayout<Float16>.stride
        let D = cfg.hiddenSize
        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        // Decode's `eps` (RealForwardRunner+Decode) for its `encodeFinalNorm`.
        let eps: Float = 1e-6
        for row in 0..<rows {
            if let hc = hyperConnection {
                try hc.encodeRead(
                    commandBuffer: cb,
                    streamsBuffer: hidden,
                    streamsOffset: row * residualWidth * half,
                    hcNorm: finalNorm.buffer,
                    hcNormOffset: Int(finalNorm.offset),
                    down: gateWeightsPublic(try model.hcMixerDown()),
                    up: gateWeightsPublic(try model.hcMixerUp()),
                    blockInput: scratch.normed,
                    blockInputOffset: row * D * half,
                    eps: eps)
            } else {
                try rms.encodeBF16W(
                    commandBuffer: cb, x: hidden, xOffset: row * residualWidth * half,
                    weight: finalNorm.buffer, weightOffset: Int(finalNorm.offset),
                    out: scratch.normed, outOffset: row * D * half,
                    d: UInt32(D), eps: eps)
            }
        }
        try multiRowGEMV.encode(
            commandBuffer: cb,
            segments: [
                .init(
                    view: lm, format: headGEMVFormat, rows: cfg.vocabSize,
                    y: logits, yOffset: logitsOffset, yRowStride: cfg.vocabSize)
            ],
            x: scratch.normed, xRowStride: D, activationRows: rows, columns: D)
    }

    /// The greedy verify head's `[rows, vocab]` logits, shared for the host
    /// argmax, grown to the widest pass seen.
    func wideVerifyHeadLogits(rows: Int) throws -> MTLBuffer {
        let bytes = rows * cfg.vocabSize * MemoryLayout<Float16>.stride
        if let existing = fusedDecodeState.wideVerifyHeadLogits, existing.length >= bytes {
            return existing
        }
        guard let made = ctx.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        made.label = "wide_verify.head_logits"
        fusedDecodeState.wideVerifyHeadLogits = made
        return made
    }

    /// Compiles every multi-row pipeline a decode-order verify pass of
    /// `rows` rows uses -- GDN and attention projections, shared expert,
    /// hyper-connection and PLE projections and the head -- so a timed pass
    /// does not pay first-use compilation. Reads no weights and runs nothing
    /// on the GPU; pipelines already compiled cost a cache lookup.
    package func prewarmWideVerifyPipelines(
        rows: ClosedRange<Int> = 2...MultiRowGEMV.maxRowsPerDispatch
    ) throws {
        struct Shape: Hashable {
            let formats: [MultiRowGEMV.Format]
            let columns: Int
        }
        let D = cfg.hiddenSize
        var shapes: Set<Shape> = [Shape(formats: [headGEMVFormat], columns: D)]
        for L in 0..<cfg.numLayers {
            if cfg.layerIsLinear(L) {
                let inputs = [
                    try model.linearInProjQKV(layer: L), try model.linearInProjZ(layer: L),
                    try model.linearInProjA(layer: L), try model.linearInProjB(layer: L),
                ]
                shapes.insert(Shape(formats: try gdnInputProjectionFormats(inputs), columns: D))
                shapes.insert(
                    Shape(
                        formats: [primaryGEMVFormat(try model.linearOutProj(layer: L))],
                        columns: cfg.linearAttention.valueDim))
            } else if cfg.attnOutputGate, cfg.fullAttentionLayerMask[L] == 1 {
                let formats = gatedQKVFormats(
                    q: try model.qProj(layer: L), k: try model.kProj(layer: L),
                    v: try model.vProj(layer: L))
                shapes.insert(Shape(formats: formats, columns: D))
                shapes.insert(
                    Shape(
                        formats: [primaryGEMVFormat(try model.oProj(layer: L))],
                        columns: cfg.numHeads * cfg.fullHeadDim))
            }
        }
        for shape in shapes {
            try multiRowGEMV.prewarm(formats: shape.formats, columns: shape.columns, rows: rows)
        }
        let sharedShapes = Set(
            sharedExpertProjections.map { [$0.gate.dtype, Int($0.gate.rows)] })
        for shape in sharedShapes where shape[1] > 0 {
            try shared.prewarmRows(
                dtype: shape[0], hidden: D, intermediate: shape[1], rows: rows)
        }
        try hyperConnection?.prewarmDecodeOrderProjection(rows: rows)
        try pleBlock?.prewarmDecodeOrderProjection(rows: rows)
    }
}
