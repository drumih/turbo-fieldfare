import Foundation
import Metal

// Phase two of RealForwardRunner's initializer: the scratch buffers and the
// resident scale tables. Moved verbatim out of the class body (2026-09-28) under the
// 500-line-per-file rule; the statements and their order are unchanged, and the
// only edits are `self.` -> `bp.` plus the optional unwraps the staging type needs.

extension RealForwardRunner {
    /// lint:allow-long a straight-line construction sequence, the second half of the
    /// initializer extraction: every statement builds one scratch buffer in order.
    static func buildScratch(
        _ bp: Builder, model: Model, context: MetalContext, cfg: ArchConfig,
        profile: ModelProfile
    ) throws {
        bp.int8ScalarGate =
            cfg.sharedExpertGated
            ? try DequantInt8GEMV(
                context: context,
                additionalShapes: cfg.decodeInt8GEMVShapes)
            : nil
        bp.bf16ScalarGate =
            cfg.sharedExpertGated
            ? try BF16GEMV(context: context) : nil
        bp.bf16Projection = try BF16GEMV(context: context)

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(
            _ count: Int,
            _ stride: Int = MemoryLayout<Float16>.size,
            label: String
        ) throws -> MTLBuffer {
            guard
                let b = device.makeBuffer(
                    length: max(count, 1) * stride,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label
            return b
        }
        // The residual is the one scratch buffer whose width is not D. A
        // hyper-connection family carries `hc_count` parallel streams of D and
        // reads a single D-wide vector out of them per sublayer, so only this
        // allocation widens -- every downstream buffer stays D. Families
        // without them get exactly D, as before.
        let residualElements =
            cfg.hyperConnections.enabled
            ? D * cfg.hyperConnections.count
            : D
        bp.hidden = try buf(residualElements, label: "decode.hidden")
        bp.normed = try buf(D, label: "decode.normed")
        bp.attnOut = try buf(maxQ, label: "decode.attnOut")
        bp.qScratch = try buf(maxQ, label: "decode.qScratch")
        bp.kStage = try buf(
            max(
                cfg.numKVHeads * cfg.headDim,
                cfg.numFullKVHeads * cfg.fullHeadDim), label: "decode.kStage")
        bp.vStage = try buf(
            max(
                cfg.numKVHeads * cfg.headDim,
                cfg.numFullKVHeads * cfg.fullHeadDim), label: "decode.vStage")
        bp.oOut = try buf(D, label: "decode.oOut")
        bp.h1Buf = try buf(D, label: "decode.h1")
        bp.h2Buf = try buf(D, label: "decode.h2")
        bp.routedX = try buf(D, label: "decode.routedX")
        bp.denseX = try buf(D, label: "decode.denseX")
        bp.denseScratchGate = try buf(F, label: "decode.denseScratchGate")
        bp.denseScratchUp = try buf(F, label: "decode.denseScratchUp")
        bp.denseScratchAct = try buf(F, label: "decode.denseScratchAct")
        bp.routerInput = try buf(D, label: "decode.routerInput")
        bp.zeroResidual = try buf(D, label: "decode.zeroResidual")
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        if let zeroResidual = bp.zeroResidual {
            memset(zeroResidual.contents(), 0, zeroResidual.length)
        }
        bp.outIndices = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.outIndices")
        bp.outWeights = try buf(cfg.topKExperts, label: "decode.outWeights")
        bp.prefetchPredictionIndices = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.prefetchPredictionIndices"
        )
        bp.prefetchPrediction2Indices = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.prefetchPrediction2Indices")
        bp.prefetchPrediction2Weights = try buf(
            cfg.topKExperts, MemoryLayout<Float16>.size,
            label: "decode.prefetchPrediction2Weights")
        bp.prefetchPredictionWeights = try buf(
            cfg.topKExperts, label: "decode.prefetchPredictionWeights")
        bp.moeActs = try buf(
            cfg.topKExperts * cfg.moeIntermediateSize, label: "decode.moeActs")
        bp.moeHitActiveSlots = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.moeHitActiveSlots")
        bp.moeMissActiveSlots = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.moeMissActiveSlots")
        bp.residencyHitCount = try buf(
            1, MemoryLayout<UInt32>.size,
            label: "decode.residencyHitCount")
        bp.residencyHitPositions = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.residencyHitPositions")
        bp.residencyMissCount = try buf(
            1, MemoryLayout<UInt32>.size,
            label: "decode.residencyMissCount")
        bp.residencyMissPositions = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.residencyMissPositions")
        bp.residencyMissExperts = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.residencyMissExperts")
        bp.residencyResolvedSlots = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.residencyResolvedSlots")
        bp.residencyResolvedGenerations = try buf(
            cfg.topKExperts,
            MemoryLayout<UInt64>.size,
            label: "decode.residencyResolvedGenerations")
        guard
            let tok = device.makeBuffer(
                length: MemoryLayout<UInt32>.size,
                options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        tok.label = "decode.greedyToken"
        bp.greedyTokenBuf = tok
        // Two rows of the residual as this family carries it: wide for a
        // hyper-connection model, because the draft's fusion reads all four
        // streams rather than a collapsed one.
        bp.verificationHidden = try buf(
            2 * Self.residualWidthFor(cfg),
            label: "decode.verificationHidden")
        bp.verificationLogits = try buf(2 * cfg.vocabSize, label: "decode.verificationLogits")

        // Qwen 3.6 decode scratch — allocated once here, never in the hot path.
        if cfg.attnOutputGate {
            bp.qPackedScratch = try buf(2 * maxQ, label: "decode.qPackedScratch")
            bp.attnGateScratch = try buf(maxQ, label: "decode.attnGateScratch")
        } else {
            bp.qPackedScratch = .some(nil)
            bp.attnGateScratch = .some(nil)
        }
        if cfg.hasLinearAttentionLayers {
            let la = cfg.linearAttention
            bp.gdnQKVRaw = try buf(la.qkvDim, label: "decode.gdnQKVRaw")
            bp.gdnConvOut = try buf(la.qkvDim, label: "decode.gdnConvOut")
            bp.gdnZ = try buf(la.valueDim, label: "decode.gdnZ")
            bp.gdnA = try buf(la.numVHeads, label: "decode.gdnA")
            bp.gdnB = try buf(la.numVHeads, label: "decode.gdnB")
            bp.gdnY = try buf(la.valueDim, label: "decode.gdnY")
            bp.gdnOut = try buf(la.valueDim, label: "decode.gdnOut")
        } else {
            bp.gdnQKVRaw = .some(nil)
            bp.gdnConvOut = .some(nil)
            bp.gdnZ = .some(nil)
            bp.gdnA = .some(nil)
            bp.gdnB = .some(nil)
            bp.gdnY = .some(nil)
            bp.gdnOut = .some(nil)
        }
        bp.sharedScalarGateBuf =
            cfg.sharedExpertGated ? try buf(1, label: "decode.sharedScalarGate") : nil
        if cfg.family == .qwen36MTP || cfg.family == .qwen38flashMTP || cfg.family == .qwen4flashMTP
        {
            guard
                let tokenBlock = context.device.makeBuffer(
                    length: Self.mtpChunkCapacity * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            bp.mtpTokenBlock = tokenBlock
            bp.mtpEmbeddingBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.embedding")
            bp.mtpNormalizedEmbeddingBlock = try buf(
                Self.mtpChunkCapacity * D, label: "mtp.normalizedEmbedding")
            bp.mtpNormalizedHiddenBlock = try buf(
                Self.mtpChunkCapacity * D, label: "mtp.normalizedHidden")
            // Qwen 3.6 concatenates the two normalized branches and runs one
            // projection; Qwen3.8-Flash-Next projects each separately and adds.
            // The wider block covers the concatenation the first needs and the
            // wide residual the second reads.
            let fuseWidth = max(2 * D, Self.residualWidthFor(cfg))
            bp.mtpConcatBlock = try buf(
                Self.mtpChunkCapacity * fuseWidth,
                label: "mtp.concat")
            bp.mtpProjectedBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.projected")
            bp.mtpTargetHiddenBlock = try buf(
                Self.mtpChunkCapacity * Self.residualWidthFor(cfg),
                label: "mtp.targetHidden")
        } else {
            bp.mtpTokenBlock = .some(nil)
            bp.mtpEmbeddingBlock = .some(nil)
            bp.mtpNormalizedEmbeddingBlock = .some(nil)
            bp.mtpNormalizedHiddenBlock = .some(nil)
            bp.mtpConcatBlock = .some(nil)
            bp.mtpProjectedBlock = .some(nil)
            bp.mtpTargetHiddenBlock = .some(nil)
        }
        bp.mtpPrefillReadback = .some(nil)

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(
                weights: view.buffer,
                scales: view.buffer,
                biases: view.buffer,
                weightsOffset: Int(view.offset),
                scalesOffset: Int(view.scaleOffset),
                biasesOffset: Int(view.biasOffset),
                rows: rows,
                cols: cols,
                dtype: Int(view.dtype))
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(
                LayerSharedExpertProjections(
                    gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                    up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                    down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                    scalarGate: cfg.sharedExpertGated
                        ? try model.sharedExpertScalarGate(layer: L) : nil))
        }
        bp.sharedExpertProjections = sharedViews

        func bf16OnesBuffer(count: Int, label: String) throws -> MTLBuffer {
            // `max(count, 1)`: a dense model has no experts, so its per-expert
            // scale holds nothing -- and Metal needs a non-empty allocation.
            // Nothing reads it on that path (the router stage is skipped), so
            // the one element is a placeholder, not a value.
            guard
                let buf = device.makeBuffer(
                    length: max(count, 1) * MemoryLayout<UInt16>.size,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = 0x3F80 }  // BF16 1.0
            buf.label = label
            return buf
        }

        // Plain linear router (Qwen): one shared BF16 ones buffer keeps
        // the router kernel's effective_scale multiply neutral, and a ones
        // per_expert_scale keeps the top-k weights untouched. (Softmax
        // over top-k then renormalize equals Qwen's softmax over all
        // experts then renormalize the selected top-k.)
        let ones = try bf16OnesBuffer(count: D, label: "effective_scale.ones")
        bp.effectiveScaleBuffers = [MTLBuffer](
            repeating: ones,
            count: cfg.numLayers)
        bp.onesPerExpertScale = try bf16OnesBuffer(
            count: cfg.numExperts,
            label: "per_expert_scale.ones")
        if profile.keepExpertCacheWired {
            model.setKeepExpertCacheWired(true)
            model.setExpertCachePinned(true)
        }
    }
}
