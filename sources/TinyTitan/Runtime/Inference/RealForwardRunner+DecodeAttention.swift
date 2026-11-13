import Foundation
import Metal

/// Decode-time attention encoding, split out of `RealForwardRunner+Decode.swift`.
///
/// Gated-DeltaNet (linear attention) and gated full attention for a single
/// decoded token, plus the packed query/gate projection they share and the
/// dispatch that chooses between the two branches. Pure code
/// motion: the same encoder bodies, moved so one file holds one concern and no
/// file in this directory passes 1,600 lines.

extension RealForwardRunner {
    /// Gated-DeltaNet linear attention (layer mask 2), one decode step.
    /// Reads `normed`, updates the layer's recurrent state + conv tail in
    /// place, and leaves the attention-branch output in `oOut`.
    func encodeLinearAttentionDecode(
        _ cb: inout MTLCommandBuffer, layer L: Int,
        slot: Int = 0,
        input: MTLBuffer? = nil, inputOffset: Int = 0,
        output: MTLBuffer? = nil, outputOffset: Int = 0
    ) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
            let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut
        else {
            throw ModelError.internalInconsistency(
                detail:
                    "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let xIn = input ?? normed
        let yOut = output ?? oOut
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        // One dispatch over the concatenated qkv/z/a/b row space instead of four
        // separate GEMVs (a and b were 4 threadgroups each). The fused kernel
        // reads all four as packed 4-bit nibbles, so it is only usable when all
        // four *are* affine: the dense Qwen 3.5 installs keep `a` and `b` at
        // bf16, and the per-projection GEMV below routes bf16, 4- and 8-bit.
        // Reading bf16 bytes as nibbles is silent nonsense, not an error.
        let fusedIsUsable = FusedKernelPolicy.mayUseFusedGDNInputProjections(
            attentionWeightBits: model.attentionWeightBits,
            dTypes: [qkvW.dtype, zW.dtype, aW.dtype, bW.dtype].map(Int.init))
        if fusedIsUsable {
            if !ablated("inproj") {
                try gdn.encodeInputProjections(
                    commandBuffer: cb,
                    x: xIn, xOffset: inputOffset,
                    qkv: qkvW, qkvOut: gdnQKVRaw,
                    z: zW, zOut: gdnZ,
                    a: aW, aOut: gdnA,
                    b: bW, bOut: gdnB,
                    hiddenSize: cfg.hiddenSize)
            }
        } else {
            try encodeRoleGEMV(
                commandBuffer: cb, projection: qkvW,
                weightBits: model.gdnProjectionWeightBits,
                x: xIn, xOffset: inputOffset, y: gdnQKVRaw,
                m: UInt32(la.qkvDim), n: D)
            try encodeRoleGEMV(
                commandBuffer: cb, projection: zW,
                weightBits: model.gdnProjectionWeightBits,
                x: xIn, xOffset: inputOffset, y: gdnZ,
                m: UInt32(la.valueDim), n: D)
            try encodeRoleGEMV(
                commandBuffer: cb, projection: aW,
                weightBits: model.gdnProjectionWeightBits,
                x: xIn, xOffset: inputOffset, y: gdnA,
                m: UInt32(la.numVHeads), n: D)
            try encodeRoleGEMV(
                commandBuffer: cb, projection: bW,
                weightBits: model.gdnProjectionWeightBits,
                x: xIn, xOffset: inputOffset, y: gdnB,
                m: UInt32(la.numVHeads), n: D)
        }
        try rotate(&cb, role: "gdn.inproj")

        if !ablated("conv") {
            let tail = gdnState.convTailSlot(layer: L, slot: slot)
            try gdn.encodeConvDecode(
                commandBuffer: cb,
                tail: tail.buffer, tailOffset: tail.offset,
                qkv: gdnQKVRaw,
                convWeight: convW.buffer,
                convWeightOffset: Int(convW.offset),
                out: gdnConvOut)
        }
        try rotate(&cb, role: "gdn.conv")
        if !ablated("qknorm") {
            try gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        }
        try rotate(&cb, role: "gdn.qknorm")
        if !ablated("delta") {
            let state = gdnState.stateSlot(layer: L, slot: slot)
            try gdn.encodeDeltaStepDecode(
                commandBuffer: cb,
                convOut: gdnConvOut,
                aProj: gdnA,
                bProj: gdnB,
                aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                state: state.buffer, stateOffset: state.offset,
                y: gdnY)
        }
        try rotate(&cb, role: "gdn.delta")
        if !ablated("gatednorm") {
            try gdn.encodeGatedNorm(
                commandBuffer: cb,
                y: gdnY,
                z: gdnZ,
                weight: gatedNormW.buffer,
                weightOffset: Int(gatedNormW.offset),
                out: gdnOut)
        }
        try rotate(&cb, role: "gdn.gatednorm")
        if !ablated("outproj") {
            // The projection overload, not the raw one: a promoted out_proj
            // carries no scales or biases, and the raw call reads those
            // companions from offset zero -- the file header -- and produces
            // NaN. The overload dispatches on the tensor's dtype.
            try encodePrimaryGEMV(
                commandBuffer: cb, projection: outW,
                x: gdnOut, y: yOut, yOffset: outputOffset,
                m: D, n: UInt32(la.valueDim))
        }
        try rotate(&cb, role: "gdn.outproj")
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    func encodeGatedFullQKVProjection(
        _ cb: MTLCommandBuffer,
        layer: Int,
        qOutput: MTLBuffer,
        kOutput: (buffer: MTLBuffer, offset: Int),
        vOutput: (buffer: MTLBuffer, offset: Int),
        qDimension: UInt32,
        kvDimension: UInt32
    ) throws {
        let q = try model.qProj(layer: layer)
        let k = try model.kProj(layer: layer)
        let v = try model.vProj(layer: layer)
        let hiddenDimension = UInt32(cfg.hiddenSize)
        // The fused kernel reads packed nibbles for all three, so a promoted
        // (bf16) tensor must not reach it even though its *slot* says 4: the
        // slot is a default, and the tensor's dtype is the truth. Same test the
        // GDN input projections already use.
        let fusedIsUsable = FusedKernelPolicy.mayUseFusedQKV(
            qoProjectionWeightBits: model.qoProjectionWeightBits,
            kvProjectionWeightBits: model.kvProjectionWeightBits,
            qDType: Int(q.dtype), kDType: Int(k.dtype), vDType: Int(v.dtype))
        if fusedIsUsable {
            try fusedQKVGEMV.encode(
                commandBuffer: cb,
                qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                x: normed,
                qOut: qOutput,
                kOut: kOutput.buffer, kOutOffset: kOutput.offset,
                vOut: vOutput.buffer, vOutOffset: vOutput.offset,
                qRows: 2 * qDimension,
                kvRows: kvDimension,
                n: hiddenDimension)
        } else {
            try encodePrimaryGEMV(
                commandBuffer: cb, projection: q,
                x: normed, y: qOutput,
                m: 2 * qDimension, n: hiddenDimension)
            try encodePrimaryGEMV(
                commandBuffer: cb, projection: k,
                x: normed, y: kOutput.buffer,
                yOffset: kOutput.offset,
                m: kvDimension, n: hiddenDimension)
            try encodePrimaryGEMV(
                commandBuffer: cb, projection: v,
                x: normed, y: vOutput.buffer,
                yOffset: vOutput.offset,
                m: kvDimension, n: hiddenDimension)
        }
    }

    func encodeGatedFullAttentionDecode(
        _ cb: inout MTLCommandBuffer,
        layer L: Int,
        position: Int,
        slot: Int = 0,
        seqLen: UInt32,
        selection: DecodeKeySelection? = nil
    ) throws {
        guard let kv else {
            throw ModelError.internalInconsistency(
                detail: "full attention requires a KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let writes = gatedFullKVWrites(kv: kv, layer: L, position: position, slot: slot)
        let o = try model.oProj(layer: L)
        try encodeGatedFullPreAttention(
            &cb, layer: L, position: position,
            kWrite: writes.k, vWrite: writes.v,
            qDim: qDim, kvDim: kvDim,
            headDim: headDim, numKV: numKV, eps: eps)
        try encodeGatedFullAttentionAndGate(
            &cb, kv: kv, layer: L, position: position, slot: slot,
            seqLen: seqLen, selection: selection)
        try encodePrimaryGEMV(
            commandBuffer: cb, projection: o,
            x: attnOut, y: oOut, m: D, n: qDim)
        try rotate(&cb, role: "qsa.oproj")
    }

    /// Attention stage of one decode layer: the gated-DeltaNet branch or the
    /// softmax branch, both writing into `oOut` for the residual add.
    ///
    /// lint:allow-long the two branches are alternatives over the same set of
    /// scratch buffers; splitting them apart again would only re-create the
    /// dispatch this method exists to hold.
    func encodeDecodeAttention(
        attnCB: inout MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
        softmaxCB: inout MTLCommandBuffer?,
        layer L: Int,
        position: Int,
        slot: Int = 0,
        isLinear: Bool,
        rmsEps eps: Float,
        selection: DecodeKeySelection? = nil
    ) throws {
        let D = UInt32(cfg.hiddenSize)
        let isFull = cfg.fullAttentionLayerMask[L] == 1
        let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
        let numKVL = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
        let qDim = UInt32(cfg.numHeads * headDimL)
        let kvDim = UInt32(numKVL * headDimL)
        let seqLen = UInt32(position + 1)
        if isLinear {
            // Gated-DeltaNet linear attention: no KV slots, no RoPE — a
            // fixed-size recurrent state updated in place, one per slot.
            try encodeLinearAttentionDecode(&attnCB, layer: L, slot: slot)
        } else if cfg.attnOutputGate {
            // Qwen full attention: packed [query ; gate] q_proj, real
            // v_proj, no V norm, NeoX sub-dim RoPE, sigmoid output gate.
            try encodeGatedFullAttentionDecode(
                &attnCB, layer: L,
                position: position,
                slot: slot,
                seqLen: seqLen,
                selection: selection)
        } else {
            let kSlot =
                kv?.kSlot(layer: L, position: position, slot: slot)
                ?? (buffer: kStage, offset: 0)
            let vSlot =
                kv?.vSlot(layer: L, position: position, slot: slot)
                ?? (buffer: vStage, offset: 0)
            let quantizedKV = kv?.precision.isQuantized == true
            let kWrite = quantizedKV ? (buffer: kStage, offset: 0) : kSlot
            let vWrite = quantizedKV ? (buffer: vStage, offset: 0) : vSlot
            let q = try model.qProj(layer: L)
            let k = try model.kProj(layer: L)
            // Under the K=V quirk full layers reuse k_proj; otherwise
            // v_proj is a real tensor.
            let vProj = (isFull && cfg.attentionKEqV) ? k : (try model.vProj(layer: L))
            let o = try model.oProj(layer: L)
            let qNorm = try model.qNorm(layer: L)
            let kNorm = try model.kNorm(layer: L)

            // Width-aware, like the gated branch above: the fused kernel is
            // int4-only, so an 8-bit attention install would have had its q/k/v
            // read as packed nibbles. It is also all-or-nothing -- one dispatch
            // reads q, k and v at one width -- so it is usable only while both
            // roles are 4-bit; `encodeRoleGEMV` below routes each projection at
            // its own width, including 8-bit and a promoted bf16.
            if model.qoProjectionWeightBits == 4 && model.kvProjectionWeightBits == 4 {
                try fusedQKVGEMV.encode(
                    commandBuffer: attnCB,
                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                    x: normed,
                    qOut: qScratch,
                    kOut: kWrite.buffer, kOutOffset: kWrite.offset,
                    vOut: vWrite.buffer, vOutOffset: vWrite.offset,
                    qRows: qDim,
                    kvRows: kvDim,
                    n: D)
            } else {
                // Each projection at its own role's width: the dense Qwen 3.5
                // installs keep k/v at 8 bits with q/o at 4, and reading an
                // 8-bit tensor through the 4-bit kernel is nonsense rather than
                // an error.
                try encodeRoleGEMV(
                    commandBuffer: attnCB, projection: q,
                    weightBits: model.qoProjectionWeightBits,
                    x: normed, y: qScratch, m: qDim, n: D)
                try encodeRoleGEMV(
                    commandBuffer: attnCB, projection: k,
                    weightBits: model.kvProjectionWeightBits,
                    x: normed, y: kWrite.buffer,
                    yOffset: kWrite.offset, m: kvDim, n: D)
                try encodeRoleGEMV(
                    commandBuffer: attnCB, projection: vProj,
                    weightBits: model.kvProjectionWeightBits,
                    x: normed, y: vWrite.buffer,
                    yOffset: vWrite.offset, m: kvDim, n: D)
            }

            let rotated =
                isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDimL / 2)
            try fusedQKVEpilogue.encode(
                commandBuffer: attnCB,
                q: qScratch,
                k: kWrite.buffer,
                kOffset: kWrite.offset,
                v: vWrite.buffer,
                vOffset: vWrite.offset,
                qWeight: qNorm.buffer,
                qWeightOffset: Int(qNorm.offset),
                kWeight: kNorm.buffer,
                kWeightOffset: Int(kNorm.offset),
                headDim: UInt32(headDimL),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVL),
                position: UInt32(position),
                theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                rotatedPairs: rotated,
                eps: eps)

            guard let kv else {
                throw ModelError.internalInconsistency(
                    detail: "attention requires a KV cache")
            }
            if quantizedKV {
                try encodeQuantizedKV(
                    commandBuffer: attnCB, kv: kv, layer: L,
                    position: position, slot: slot, keySource: kStage,
                    valueSource: vStage, elementCount: Int(kvDim))
            }
            let keyView = kv.keyView(layer: L, slot: slot, validTokenCount: Int(seqLen))
            let valueView = kv.valueView(layer: L, slot: slot, validTokenCount: Int(seqLen))
            guard let attentionCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            softmaxCB = attentionCB
            if isFull {
                let windowSelection = try decodeWindowSelection(
                    kv: kv, layer: L, slot: slot, position: position,
                    seqLen: seqLen, selection: selection)
                try attention.encodeFull(
                    commandBuffer: attentionCB,
                    q: qScratch,
                    k: keyView.buffer, kOffset: keyView.offset,
                    v: valueView.buffer, vOffset: valueView.offset,
                    out: attnOut,
                    headDim: UInt32(headDimL),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKVL),
                    seqLen: seqLen,
                    scale: Float(cfg.attentionScale),
                    kvFormat: keyView,
                    keepMask: windowSelection == nil ? selection?.mask : nil,
                    windowSelection: windowSelection)
            } else {
                let ringCapacity = kv.ringCapacity(layer: L)
                let activeRingCapacity =
                    ringCapacity > 0 && Int(seqLen) > ringCapacity
                    ? UInt32(ringCapacity)
                    : 0
                try attention.encodeSWA(
                    commandBuffer: attentionCB,
                    q: qScratch,
                    k: kSlot.buffer, kOffset: keyView.offset,
                    v: vSlot.buffer, vOffset: valueView.offset,
                    out: attnOut,
                    headDim: UInt32(headDimL),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKVL),
                    seqLen: seqLen,
                    window: UInt32(cfg.slidingWindow),
                    scale: Float(cfg.attentionScale),
                    ringCapacity: activeRingCapacity,
                    kvFormat: keyView)
            }
            // Same width-awareness as the projections above; `int4` here was the
            // last int4-only call on this branch.
            try encodeRoleGEMV(
                commandBuffer: tailCB, projection: o,
                weightBits: model.qoProjectionWeightBits,
                x: attnOut, y: oOut, m: D, n: qDim)
        }

        // Plain pre-norm residual block: hidden += attention branch,
        // then one post-attention norm feeds router, shared expert,
        // and routed phase 1 (routedX doubles as moeX).
    }
}
