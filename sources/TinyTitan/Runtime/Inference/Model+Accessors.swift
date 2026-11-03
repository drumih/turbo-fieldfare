import Darwin
import Foundation
import Metal
import TinyTitanFormat

// Resident tensor accessors: the family schema and every named tensor the
// kernels read, plus the bf16 promotion helper they share.
//
// Moved out of the `Model` declaration in `Model.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion:
// the members are unchanged, and `bf16Readable` moved with its only callers.
extension Model {

    // MARK: - Resident accessors
    //
    // Names resolve through the family's TensorSchema (Runtime/Family/): a
    // family with different naming supplies a schema file; these accessors
    // never change.

    var schema: TensorSchema { TensorSchema.schema(for: config.family) }

    /// The family's Flash-only tensor namespace.
    ///
    /// `TensorSchema` covers the shared roles; the hyper-connection, indexer,
    /// GDN and PLE names live in a per-family set because only the Qwen4
    /// lineage has them. Reading the set from the family keeps the accessors
    /// single-bodied while the two contracts stay separate.
    var flashTensors: any FlashTensorSet.Type {
        switch config.family {
        case .qwen4flash, .qwen4flashMTP:
            return Qwen4FlashTensors.self
        default:
            return Qwen38FlashTensors.self
        }
    }

    public func embedding() throws -> TensorView {
        if let sharedTargetWeights { return sharedTargetWeights.embedding }
        return try resident(name: schema.embedding)
    }

    /// Qwen 3.6 carries a separate `lm_head` tensor. The transpose for the
    /// lm_head GEMV path is the kernel's job, not the loader's.
    public func lmHead() throws -> TensorView {
        if let sharedTargetWeights { return sharedTargetWeights.lmHead }
        if config.tieWordEmbeddings { return try embedding() }
        return try resident(name: schema.lmHead)
    }

    public func qProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.qProj(L))
    }
    public func kProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.kProj(L))
    }
    public func vProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.vProj(L))
    }
    public func oProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.oProj(L))
    }

    // MARK: K2-Horizon roles

    /// The separate softplus-gate projection (`hidden -> heads * headDim`).
    /// K2 does not pack it into q_proj the way the Qwen families do.
    public func k2AttentionGate(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.attnGate(L))
    }
    /// The dense layers 0-2 SwiGLU MLP.
    public func k2DenseGate(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.denseGate(L))
    }
    public func k2DenseUp(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.denseUp(L))
    }
    public func k2DenseDown(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.denseDown(L))
    }
    public func k2MoeRouter(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.moeRouter(L))
    }
    public func k2MoeRouterBias(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.moeRouterBias(L))
    }
    public func k2SharedGate(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.sharedGate(L))
    }
    public func k2SharedUp(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.sharedUp(L))
    }
    public func k2SharedDown(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.sharedDown(L))
    }
    public func k2VRouter(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.vRouter(L))
    }
    public func k2VRouterBias(layer L: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.vRouterBias(L))
    }
    public func k2VExpert(layer L: Int, expert: Int) throws -> TensorView {
        try resident(name: K2HorizonTensors.vExpert(L, expert))
    }

    // MARK: Laguna-S-2.1 roles

    /// One softplus gate per query head (`hidden -> heads`), applied to that
    /// head's 128 attention channels before o_proj.
    public func lagunaAttentionGate(layer L: Int) throws -> TensorView {
        try resident(name: LagunaTensors.attnGate(L))
    }
    /// The aux-loss-free router selection bias, bf16, one per expert.
    public func lagunaRouterBias(layer L: Int) throws -> TensorView {
        try resident(name: LagunaTensors.moeRouterBias(L))
    }
    public func lagunaDenseGate(layer L: Int) throws -> TensorView {
        try resident(name: LagunaTensors.denseGate(L))
    }
    public func lagunaDenseUp(layer L: Int) throws -> TensorView {
        try resident(name: LagunaTensors.denseUp(L))
    }
    public func lagunaDenseDown(layer L: Int) throws -> TensorView {
        try resident(name: LagunaTensors.denseDown(L))
    }
    public func lagunaSharedGate(layer L: Int) throws -> TensorView {
        try resident(name: LagunaTensors.sharedGate(L))
    }
    public func lagunaSharedUp(layer L: Int) throws -> TensorView {
        try resident(name: LagunaTensors.sharedUp(L))
    }
    public func lagunaSharedDown(layer L: Int) throws -> TensorView {
        try resident(name: LagunaTensors.sharedDown(L))
    }
    // MARK: Hyper-connection (Gated Residual) weights
    //
    // Present only on families whose `hyperConnections` is enabled; asking for
    // them elsewhere fails at the resident-index lookup with the missing name,
    // which is the right error for a misconfigured install.

    public func hcAttnMixDown(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.attnMixDown(L))
    }
    public func hcAttnMixUp(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.attnMixUp(L))
    }
    public func hcAttnInject(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.attnInject(L))
    }
    public func hcMlpMixDown(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.mlpMixDown(L))
    }
    // MARK: QSA indexer

    public func indexerQProj(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.indexerQProj(L))
    }
    public func indexerKProj(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.indexerKProj(L))
    }
    public func indexerQNorm(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.indexerQNorm(L))
    }
    public func indexerKNorm(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.indexerKNorm(L))
    }

    // MARK: PLE n-gram block

    public func pleKeyProj(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.pleKeyProj(L))
    }
    public func pleValueProj(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.pleValueProj(L))
    }
    public func pleNormKey(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.pleNormKey(L))
    }
    public func pleNormQuery(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.pleNormQuery(L))
    }
    public func pleNormConv(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.pleNormConv(L))
    }
    public func pleConv(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.pleConv(L))
    }

    public func hcMlpMixUp(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.mlpMixUp(L))
    }
    public func hcMlpInject(layer L: Int) throws -> TensorView {
        try resident(name: flashTensors.mlpInject(L))
    }
    /// The model-level mixer that collapses the streams before `lm_head`.
    /// Same read gate as a sublayer's, with no inject.
    public func hcMixerDown() throws -> TensorView {
        try resident(name: flashTensors.mixerDown)
    }
    public func hcMixerUp() throws -> TensorView {
        try resident(name: flashTensors.mixerUp)
    }

    public func router(layer L: Int) throws -> TensorView {
        try resident(name: schema.router(L))
    }
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: schema.sharedExpertGate(L))
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: schema.sharedExpertUp(L))
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: schema.sharedExpertDown(L))
    }
    /// Qwen3.5-MoE scalar gate on the shared-expert branch: a `[1, hidden]`
    /// 8-bit projection whose sigmoid multiplies the shared FFN output.
    public func sharedExpertScalarGate(layer L: Int) throws -> TensorView {
        try resident(name: schema.sharedExpertScalarGate(L))
    }
    public func inputNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.inputNorm(L))
    }
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.postAttnNorm(L))
    }
    public func finalNorm() throws -> TensorView {
        try resident(name: schema.finalNorm)
    }

    /// MTP projection over the normalized next-token embedding followed by the
    /// normalized target hidden state: `[embedding, hidden]`, `[2D] -> [D]`.
    public func mtpProjection() throws -> TensorView {
        return try resident(name: "fc.weight")
    }
    public func mtpEmbeddingNorm() throws -> TensorView {
        return try resident(name: "pre_fc_norm_embedding.weight")
    }
    public func mtpHiddenNorm() throws -> TensorView {
        return try resident(name: "pre_fc_norm_hidden.weight")
    }

    // MARK: Qwen3.8-Flash-Next draft head
    //
    // This family fuses with two projections rather than one over a
    // concatenation: `fc_hidden` takes the target's wide residual and
    // `fc_embedding` takes the next token's embedding, and their outputs are
    // summed. Two [2560, 2560] matrices, not one [2560, 5120] -- the shapes
    // are what say so, and getting it wrong would show up only as a draft
    // that is never accepted.

    /// `[hidden, hc_dim]`: the target's wide residual down to one stream.
    public func mtpHiddenProjection() throws -> TensorView {
        try resident(name: "fc_hidden.weight")
    }
    /// `[hidden, hidden]`: the next token's embedding.
    public func mtpEmbeddingProjection() throws -> TensorView {
        try resident(name: "fc_embedding.weight")
    }
    /// Grouped over the hyper-connection streams, unlike the embedding norm.
    /// Stored already folded (+1) by the MLX conversion, like every other
    /// gamma in this checkpoint; used as-is.
    public func mtpWideNorm() throws -> TensorView {
        try resident(name: "pre_fc_norm_hidden")
    }
    public func mtpTokenNorm() throws -> TensorView {
        try resident(name: "pre_fc_norm_embedding")
    }

    public func qNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.qNorm(L))
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.kNorm(L))
    }

    // MARK: - Gated-DeltaNet linear attention (Qwen 3.6)
    //
    // Layers whose mask value is 2 replace full/sliding attention with the
    // gated delta rule. Projections are 4/6/8-bit affine; the depthwise conv
    // weight, A_log, dt_bias, and the gated output norm are BF16.

    public func linearInProjQKV(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnQKV(L))
    }
    public func linearInProjZ(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnZ(L))
    }
    public func linearInProjA(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnA(L))
    }
    public func linearInProjB(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnB(L))
    }
    public func linearOutProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnOut(L))
    }
    /// Depthwise causal conv weight, source shape `[convDim, kernel, 1]`, BF16.
    public func linearConv1d(layer L: Int) throws -> TensorView {
        try bf16Readable(schema.gdnConv(L))
    }
    /// Per-value-head decay base, shape `[numVHeads]`, BF16.
    public func linearALog(layer L: Int) throws -> TensorView {
        try bf16Readable(schema.gdnALog(L))
    }
    /// Per-value-head dt bias, shape `[numVHeads]`, BF16.
    public func linearDtBias(layer L: Int) throws -> TensorView {
        try bf16Readable(schema.gdnDtBias(L))
    }
    /// Gated RMSNorm weight over the value head dim, shape `[valueHeadDim]`.
    public func linearNorm(layer L: Int) throws -> TensorView {
        try bf16Readable(schema.gdnNorm(L))
    }

    /// Resolve a tensor name to a `TensorView` against the resident buffer.
    /// `fileOffset` (absolute) is converted to a buffer-relative offset by
    /// subtracting the resident region's file offset (which equals
    /// `header.indexSize`).
    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return Self.residentView(
            entry: entry,
            indexSize: residentIndex.header.indexSize,
            buffer: residentBuffer.buffer)
    }

    /// The `TensorView` a resident index entry describes. Shared with the
    /// promotion step, which builds views over buffers it allocates itself and
    /// must handle the same file-to-buffer offset translation.
    static func residentView(
        entry: ResidentIndexEntry,
        indexSize: UInt64,
        buffer: MTLBuffer
    ) -> TensorView {
        let relativeOffset = entry.fileOffset - indexSize
        let scaleRel: UInt64 = entry.scaleSize > 0 ? entry.scaleOffset - indexSize : 0
        let biasRel: UInt64 = entry.biasSize > 0 ? entry.biasOffset - indexSize : 0
        return TensorView(
            buffer: buffer,
            offset: relativeOffset,
            length: entry.sizeBytes,
            scaleOffset: scaleRel, scaleLength: entry.scaleSize,
            biasOffset: biasRel, biasLength: entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype)
    }

    /// Promotes every fp32 tensor the kernels read as bf16 into a bf16 buffer.
    ///
    /// The dense Qwen 3.5 installs keep `A_log` and the gated norm at fp32 (as
    /// their source checkpoints do) while `gdn.metal` declares `device const
    /// bfloat*` for them, so the bytes would be read as the wrong type. The
    /// promotion is the same rounding the MoE installs already ship -- their
    /// `A_log` is bf16 on disk and passes the oracle baselines -- and it keeps
    /// the kernels unchanged. Round-half-to-even, per `Quantization.bf16Bits`.
    ///
    /// Only the small per-head tensors are considered: the quantized
    /// projections keep their own path, and a bf16 tensor is left alone.
    static func buildBF16ReadableViews(
        device: MTLDevice,
        schema: TensorSchema,
        config: ArchConfig,
        residentIndex: ResidentIndex,
        residentBuffer: MTLBuffer
    ) throws -> [String: TensorView] {
        var promoted: [String: TensorView] = [:]
        for layer in 0..<config.numLayers where config.layerIsLinear(layer) {
            let names = [
                schema.gdnALog(layer), schema.gdnDtBias(layer),
                schema.gdnConv(layer), schema.gdnNorm(layer),
            ]
            for name in names {
                guard let entry = residentIndex.entries[name], entry.dtype == 3 else { continue }
                let source = residentView(
                    entry: entry,
                    indexSize: residentIndex.header.indexSize,
                    buffer: residentBuffer)
                let elements = Int(entry.sizeBytes) / MemoryLayout<Float>.size
                guard
                    let converted = device.makeBuffer(
                        length: max(elements * MemoryLayout<UInt16>.size, 4),
                        options: .storageModeShared)
                else {
                    throw ModelError.indexCorrupt(
                        detail: "could not allocate a bf16 promotion buffer for \(name)")
                }
                let sourcePointer = (source.buffer.contents() + Int(source.offset))
                    .assumingMemoryBound(to: Float.self)
                let destination = converted.contents()
                    .assumingMemoryBound(to: UInt16.self)
                for index in 0..<elements {
                    destination[index] = Quantization.bf16Bits(sourcePointer[index])
                }
                promoted[name] = TensorView(
                    buffer: converted, offset: 0,
                    length: UInt64(elements * MemoryLayout<UInt16>.size),
                    scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                    shape: entry.shape, dtype: 1)
            }
        }
        return promoted
    }

    /// A tensor the kernels read as bf16: its promoted view when the checkpoint
    /// stored it in fp32, else the resident bytes themselves.
    private func bf16Readable(_ name: String) throws -> TensorView {
        if let promoted = promotedBF16[name] { return promoted }
        return try resident(name: name)
    }
}
