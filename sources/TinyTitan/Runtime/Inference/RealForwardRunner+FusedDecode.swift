import Foundation
import Metal

/// Per-runner fused-decode state: the cached pair scratch, the launch counter
/// and the test override. A class so one stored property on the runner can
/// hold all three without growing that file past its line budget.
final class FusedDecodeState {
    var pairScratch: [PrefillChunkScratchBuffers] = []
    var launches = 0
    var override: Bool?
    /// The wide-verify rollback window (RealForwardRunner+WideVerifyRollback).
    var wideVerifyRollback: WideVerifyGDNRollback?
    /// `[rows, vocab]` FP16 logits of the greedy W-row verify head, grown to
    /// the widest pass seen (RealForwardRunner+WideVerifyHead).
    var wideVerifyHeadLogits: MTLBuffer?
}

// T34: the fused width-2 decode loop.
//
// Attention is batched through `encodeBatchAttentionBlock` (entry norm, one
// QKV/o_proj pair of GEMVs, per-row epilogue/attention/KV write, residual and
// post-attention norm). GDN layers stay per row with the row-sliced linear
// decoder. The FFN is the two-row union-fetch verify pipeline: one router
// readback, one union plan/fetch and per-row phase 1/2 for both rows. Its
// routed command is carried as a pending command across the next layer's
// attention, which is what keeps the GPU busy while the CPU does the router
// readback. The loop is opt-in until the T34 gates pass; T7 flips the default.
extension RealForwardRunner {
    /// Opt-in switch for the fused loop. T7 replaces this with the server's
    /// width selection once the gates are green.
    static var fusedDecodeOptIn: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_FUSED_DECODE"] == "1"
    }

    /// Test hook: force the fused loop on or off for this runner without the
    /// process-global environment switch. Nil follows `TINYTITAN_FUSED_DECODE`.
    var fusedDecodeOverride: Bool? {
        get { fusedDecodeState.override }
        set { fusedDecodeState.override = newValue }
    }

    /// Test-visible: how many decode steps actually took the fused loop.
    var fusedBatchDecodeLaunches: Int { fusedDecodeState.launches }

    /// The families and layouts the fused loop implements today. Anything the
    /// loop does not model -- PLE, QSA, hyper-connections, host-KV streaming,
    /// activation dumps, widths other than two -- falls back to the per-row
    /// path rather than running with a missing stage.
    func canUseFusedBatchDecode(
        _ rows: [(token: Int32, position: Int, slot: Int)]
    ) -> Bool {
        guard fusedDecodeState.override ?? Self.fusedDecodeOptIn else { return false }
        guard rows.count == 2 || rows.count == 4, slots >= rows.count
        else { return false }
        guard cfg.family == .qwen36 else { return false }
        guard cfg.ple.layerIndices.isEmpty else { return false }
        guard !cfg.hyperConnections.enabled else { return false }
        guard qsaIndexer == nil else { return false }
        guard activationDumpDirectory == nil else { return false }
        guard let kv, !kv.streamsKVFromHost else { return false }
        guard cfg.numExperts > 0, cfg.topKExperts > 0 else { return false }
        guard
            model.qoProjectionWeightBits == 4 || model.qoProjectionWeightBits == 8
        else { return false }
        return true
    }

    /// One pair scratch per two-row FFN group: a width-4 batch has two groups
    /// whose in-flight commands must not share the acts or the shared output.
    func ensureBatchPairScratch(index: Int) throws
        -> PrefillChunkScratchBuffers
    {
        while fusedDecodeState.pairScratch.count <= index {
            let layout = PrefillChunkScratchLayout(config: cfg, chunkTokens: 2)
            let made = try PrefillChunkScratchBuffers.allocate(
                device: ctx.device, layout: layout)
            fusedDecodeState.pairScratch.append(made)
        }
        return fusedDecodeState.pairScratch[index]
    }

    private func blitRows(
        _ cb: MTLCommandBuffer,
        from source: MTLBuffer, sourceOffset: Int,
        to destination: MTLBuffer, destinationOffset: Int,
        bytes: Int
    ) throws {
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(
            from: source, sourceOffset: sourceOffset,
            to: destination, destinationOffset: destinationOffset,
            size: bytes)
        blit.endEncoding()
    }

    private func encodeEmbeddingRow(
        token: Int32, into out: MTLBuffer, outOffset: Int,
        commandBuffer cb: MTLCommandBuffer
    ) throws {
        let scale =
            cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot() : 1.0
        try encodeEmbeddingLookup(
            commandBuffer: cb, tokenId: UInt32(bitPattern: token),
            out: out, outOffset: outOffset, outScale: scale)
    }

    /// One decode-shaped union-fetch FFN for both rows.
    ///
    /// lint:allow-long one ordered per-layer pipeline -- router, shared
    /// expert, plan, fetch, per-row phase 1/2, residual -- whose stages share
    /// the union, the blobs and the command buffer; splitting it would thread
    /// those live locals through helpers with no independent test.
    ///
    /// The MTP verify pipeline's router/shared blocks are prefill kernels, and
    /// on the real model their rounding is not the decode kernels': the fused
    /// text diverged. This keeps the decode kernels for every arithmetic stage
    /// (router, shared expert, phase 1/2 and the residual add) and shares only
    /// the work that cannot change numbers -- one union plan, one fetch and
    /// one host readback wait for both rows.
    private func encodeDecodeUnionFFN(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        rows: [(position: Int, slot: Int)],
        scratch: BatchedDecodeScratch,
        pair: PrefillChunkScratchBuffers,
        router: TensorView,
        rowBase: Int
    ) async throws -> PendingRoutedCommand {
        let t = rows.count
        let D = UInt32(cfg.hiddenSize)
        let halfBytes = MemoryLayout<Float16>.stride
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let topK = UInt32(cfg.topKExperts)
        let perExpertScale = try requireOnesPerExpertScale()

        // Router and shared expert for every row, on the attention's buffer.
        for row in 0..<t {
            let offset = (rowBase + row) * cfg.hiddenSize * halfBytes
            let localOffset = row * cfg.hiddenSize * halfBytes
            try moe.encodeRouter(
                commandBuffer: cb,
                weights: router.buffer, weightsOffset: Int(router.offset),
                scales: router.buffer, scalesOffset: Int(router.scaleOffset),
                biases: router.buffer, biasesOffset: Int(router.biasOffset),
                hidden: scratch.routedX, hiddenOffset: offset,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale, perExpertScaleOffset: 0,
                outIndices: pair.routeIDs,
                outIndicesOffset: row * cfg.topKExperts
                    * MemoryLayout<UInt32>.stride,
                outWeights: pair.routeWeights,
                outWeightsOffset: row * cfg.topKExperts * halfBytes,
                numExperts: UInt32(cfg.numExperts), d: D, topK: topK)
            try encodeSharedExpert(
                layer: L,
                x: scratch.routedX, xOffset: offset,
                y: pair.h1, yOffset: localOffset,
                commandBuffer: cb)
        }
        cb.commit()
        // Also drains the previous layer's routed command.
        try waitForCompletion(cb)

        let idPtr = pair.routeIDs.contents().bindMemory(
            to: UInt32.self, capacity: t * cfg.topKExperts)
        var rowExperts = [[Int]](repeating: [], count: t)
        var union: [Int] = []
        var unionIndex: [Int: Int] = [:]
        for row in 0..<t {
            for k in 0..<cfg.topKExperts {
                let expert = min(
                    Int(idPtr[row * cfg.topKExperts + k]),
                    cfg.numExperts - 1)
                rowExperts[row].append(expert)
                if unionIndex[expert] == nil {
                    unionIndex[expert] = union.count
                    union.append(expert)
                }
            }
        }

        let plan = try model.planRoutedExperts(layer: L, experts: union)
        let lease = try plan.map { try model.pinRoutedExperts(for: $0) }
        var leaseTransferred = false
        defer { if !leaseTransferred { lease?.release() } }
        let blobs: [TensorView]
        if let plan {
            if plan.misses.isEmpty {
                blobs = try model.routedExpertBuffers(for: plan)
            } else {
                blobs = try await model.beginFetchRoutedExperts(plan: plan)
                    .completion()
            }
        } else {
            blobs = try await model.fetchRoutedExperts(layer: L, experts: union)
        }
        let routedOffsets = try model.routedExpertOffsets(layer: L)

        while verifyPairActs.count < rowBase + t {
            guard
                let made = ctx.device.makeBuffer(
                    length: cfg.topKExperts * cfg.moeIntermediateSize * halfBytes,
                    options: .storageModePrivate)
            else { throw ModelError.residentBufferWrapFailed }
            made.label = "batch.union.acts.\(verifyPairActs.count)"
            verifyPairActs.append(made)
        }
        while verifyPairY.count < rowBase + t {
            guard
                let made = ctx.device.makeBuffer(
                    length: cfg.hiddenSize * halfBytes,
                    options: .storageModePrivate)
            else { throw ModelError.residentBufferWrapFailed }
            made.label = "batch.union.y.\(verifyPairY.count)"
            verifyPairY.append(made)
        }
        while verifyPairArgBuffers.count < rowBase + t {
            guard let made = moe.makeEmptyRoutedArgumentBuffer(device: ctx.device)
            else { throw ModelError.residentBufferWrapFailed }
            made.label = "batch.union.args.\(verifyPairArgBuffers.count)"
            verifyPairArgBuffers.append(made)
        }

        guard let routedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        for row in 0..<t {
            var rowBufs: [MTLBuffer] = []
            var rowOffsets: [Int] = []
            for expert in rowExperts[row] {
                guard let blobIndex = unionIndex[expert] else { continue }
                let view = blobs[blobIndex]
                rowBufs.append(view.buffer)
                rowOffsets.append(Int(view.offset))
            }
            let argBuf = verifyPairArgBuffers[rowBase + row]
            moe.writeRoutedArgumentBuffer(
                argBuf, routedBlobs: rowBufs, topK: topK,
                routedBufferOffsets: rowOffsets)
            let pairOffset = row * cfg.hiddenSize * halfBytes
            let globalOffset = (rowBase + row) * cfg.hiddenSize * halfBytes
            try moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: rowBufs,
                routedOffsets: routedOffsets,
                x: scratch.routedX, xOffset: globalOffset,
                acts: verifyPairActs[rowBase + row],
                d: D, f: FmoE, topK: topK)
            try moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: rowBufs,
                routedOffsets: routedOffsets,
                acts: verifyPairActs[rowBase + row],
                routingWeights: pair.routeWeights,
                routingWeightsOffset: row * cfg.topKExperts * halfBytes,
                residual: pair.h1, residualOffset: pairOffset,
                y: verifyPairY[rowBase + row],
                d: D, f: FmoE, topK: topK)
            try requireElementwise().encodeResidualAdd(
                commandBuffer: routedCB,
                hidden: scratch.hidden, hiddenOffset: globalOffset,
                delta: verifyPairY[rowBase + row], count: cfg.hiddenSize)
        }
        routedCB.commit()
        leaseTransferred = true
        return PendingRoutedCommand(
            cb: routedCB,
            sharedCB: nil,
            phase1HitCB: nil,
            phase1MissCB: nil,
            expertLease: lease,
            storageOperation: nil,
            overlapCompletionClock: nil,
            expectedOverlapCompletions: 1,
            kernelRole: "fused_union_moe",
            encodeAndCommitNanos: 0)
    }

    /// Runs one width-2 decode step with the fused loop. Returns false when the
    /// batch is not eligible, so `produceBatch` can fall back without having
    /// touched any state.
    ///
    /// lint:allow-long one decode step is a single ordered pass -- embedding,
    /// forty layers of batch attention plus the union-fetch FFN, final norm,
    /// head -- whose stages share the row cursor, the command-buffer chain and
    /// the pair scratch; splitting it would thread a dozen live locals through
    /// helpers that are not independently testable.
    func produceBatchFused(
        _ rows: [(token: Int32, position: Int, slot: Int)],
        logits: [MTLBuffer]
    ) async throws -> Bool {
        guard canUseFusedBatchDecode(rows) else { return false }
        guard let kv else { return false }
        fusedDecodeState.launches += 1
        // Same handover the per-row decode does: wire the routed-expert slot
        // cache for the whole generation before its slots can be reclaimed
        // between the plan and the GPU's read of them.
        model.setExpertCachePinned(true)
        let count = rows.count
        let d = cfg.hiddenSize
        let half = MemoryLayout<Float16>.stride
        let D = UInt32(d)
        let eps: Float = 1e-6

        for row in rows {
            let cursor = kv.position(slot: row.slot)
            guard cursor == row.position else {
                throw PrefillError.prefillCursorMismatch(
                    "fused cursor \(cursor) != position \(row.position) for slot \(row.slot)")
            }
            guard row.position < maxContext else {
                throw PrefillError.prefillCursorMismatch(
                    "fused position \(row.position) exceeds maxContext \(maxContext)")
            }
            try kv.reserve(tokens: row.position + 1, slot: row.slot)
            try requireQSAExact(visibleKeys: row.position + 1)
        }

        let scratch = try ensureBatchedDecodeScratch(rows: count)
        let views = try makeLayerPrefillViews()
        let fNorm = try model.finalNorm()
        let (normed, batchLogits) = try ensureBatchScratch(rows: count)

        guard let embedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        for (index, row) in rows.enumerated() {
            try encodeEmbeddingRow(
                token: row.token, into: scratch.hidden,
                outOffset: index * d * half, commandBuffer: embedCB)
        }
        embedCB.commit()
        try waitForCompletion(embedCB)

        var pending: PendingRoutedCommand?
        defer {
            if let pending {
                try? finishPendingRoutedCommand(pending, waitIfNeeded: true)
            }
        }

        for L in 0..<cfg.numLayers {
            guard var layerCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            if cfg.layerIsLinear(L) {
                for (index, row) in rows.enumerated() {
                    let offset = index * d * half
                    try encodeResidualEntryDecode(
                        commandBuffer: layerCB,
                        hidden: scratch.hidden, hiddenOffset: offset,
                        norm: views[L].inputNorm,
                        out: scratch.normed, outOffset: offset,
                        sublayer: .attention, layer: L, eps: eps)
                    try encodeLinearAttentionDecode(
                        &layerCB, layer: L, slot: row.slot,
                        input: scratch.normed, inputOffset: offset,
                        output: scratch.o, outputOffset: offset)
                    try encodeResidualExitDecode(
                        commandBuffer: layerCB,
                        hidden: scratch.hidden, hiddenOffset: offset,
                        delta: scratch.o, deltaOffset: offset,
                        sublayer: .attention, layer: L)
                    try encodeResidualEntryDecode(
                        commandBuffer: layerCB,
                        hidden: scratch.hidden, hiddenOffset: offset,
                        norm: views[L].postAttention,
                        out: scratch.routedX, outOffset: offset,
                        sublayer: .mlp, layer: L, eps: eps)
                }
            } else {
                try encodeBatchAttentionBlock(
                    layer: L,
                    rows: rows.map { (position: $0.position, slot: $0.slot) },
                    scratch: scratch,
                    postAttnNorm: views[L].postAttention,
                    commandBuffer: &layerCB)
            }

            guard let routerView = views[L].router else {
                throw ModelError.internalInconsistency(
                    detail: "fused loop reached a routed layer without a router")
            }
            // Width 4 runs two two-row union FFNs. Each call commits its own
            // buffer and waits for the router readback; that wait also drains
            // the previously committed routed command, so the GPU was busy
            // with it while the CPU encoded this group.
            for pairIndex in 0..<(count / 2) {
                let pair = try ensureBatchPairScratch(index: pairIndex)
                let rowBase = pairIndex * 2
                let pairCB: MTLCommandBuffer
                if pairIndex == 0 {
                    pairCB = layerCB
                } else {
                    guard let next = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    for local in 0..<2 {
                        let globalOffset = (rowBase + local) * d * half
                        let localOffset = local * d * half
                        try blitRows(
                            next,
                            from: scratch.hidden, sourceOffset: globalOffset,
                            to: pair.hidden, destinationOffset: localOffset,
                            bytes: d * half)
                        try blitRows(
                            next,
                            from: scratch.routedX, sourceOffset: globalOffset,
                            to: pair.routedX, destinationOffset: localOffset,
                            bytes: d * half)
                    }
                    pairCB = next
                }
                var cb = pairCB
                let outcome = try await encodeDecodeUnionFFN(
                    cb: &cb, layer: L,
                    rows: (0..<2).map {
                        (
                            position: rows[rowBase + $0].position,
                            slot: rows[rowBase + $0].slot
                        )
                    },
                    scratch: scratch, pair: pair, router: routerView,
                    rowBase: rowBase)
                if let previous = pending {
                    try finishPendingRoutedCommand(
                        previous, waitIfNeeded: false)
                    pending = nil
                }
                pending = outcome
            }
        }

        if let current = pending {
            try finishPendingRoutedCommand(current, waitIfNeeded: true)
            pending = nil
        }
        guard let headNormCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        for index in rows.indices {
            try rms.encodeBF16W(
                commandBuffer: headNormCB,
                x: scratch.hidden, xOffset: index * d * half,
                weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                out: normed, outOffset: index * d * half,
                d: D, eps: eps)
        }
        headNormCB.commit()
        try waitForCompletion(headNormCB)
        try encodeBatchHead(
            rows: count, logits: logits,
            normed: normed, batchLogits: batchLogits)
        for row in rows {
            kv.advance(slot: row.slot, by: 1)
        }
        return true
    }
}
