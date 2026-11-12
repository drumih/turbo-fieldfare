import Foundation
import Metal

// The routed-MoE stage of the width-2 MTP verify pass.
//
// Split out of `RealForwardRunner+MTP.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion;
// the `lint:allow-long` marker travelled with its declaration.
extension RealForwardRunner {

    /// Routed-MoE stage for the width-2 MTP verify pass (B2 pair schedule).
    ///
    /// Replaces the prefill tile scheduler for exactly this shape. One union
    /// cache plan covers both rows' experts, so a shared expert is read from
    /// SSD once; the miss fetch runs as one parallel batch overlapped with the
    /// shared-expert GPU work instead of per-tile awaits behind a synchronous
    /// shared-expert wait; and the routed math uses the decode phase-1/phase-2
    /// kernels per row, which B1 measured at roughly a third of the grouped
    /// tile kernels' GPU cost at width 2. Numerics are unchanged: phase 2
    /// reduces each row's experts in router order with the shared branch as
    /// its residual, exactly as decode does.
    ///
    /// lint:allow-long one layer's verify-MoE stage is a single ordered
    /// pipeline in the same shape as its decode and tile siblings: route
    /// readback, union plan, overlapped fetch, per-row encode, commit.
    struct VerifyPairOutcome {
        let routedCB: MTLCommandBuffer
        let sharedCB: MTLCommandBuffer
        let expertLease: RoutedExpertLease?
        let nextCB: MTLCommandBuffer?
    }

    /// The verify caller (the MTP pair, or a W-row wide verify): same
    /// pipeline, awaited to completion. `rows` must leave the union plan
    /// within the routed slot cache: `rows * topK <= slots`.
    func encodeRoutedMoEVerifyRows(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        rows: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        hiddenSize D: Int
    ) async throws {
        let outcome = try await encodeVerifyPairBody(
            cb: &cb, layer: L, rows: rows, views: views, scratch: scratch,
            hiddenSize: D)
        try waitForCompletion(outcome.routedCB)
        if let start = wideVerifyDumpStart, L <= dumpLayerLimit {
            for row in 0..<rows where activationDumpActive(position: start + row) {
                dumpActivationPrivate(
                    "L\(L)_mlp_out_rows", verifyPairY[row], count: D, position: start + row)
            }
        }
        recordKernelGPU(role: "prefill_shared_expert", outcome.sharedCB)
        recordKernelGPU(role: "verify_routed_pair", outcome.routedCB)
        outcome.expertLease?.release()
        if let nextCB = outcome.nextCB { cb = nextCB }
    }

    /// One verify-pair FFN stage without the final wait: the routed command,
    /// its lease and the next command buffer are handed back so a decode loop
    /// can carry the GPU work across the next layer.
    ///
    /// lint:allow-long the existing verify-pair pipeline, split out of its
    /// caller as pure code motion; no logic changed.
    func encodeVerifyPairBody(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        rows t: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        hiddenSize D: Int
    ) async throws -> VerifyPairOutcome {
        guard t >= 1, t <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "verify routed stage needs 1...\(scratch.layout.chunkTokens) rows, got \(t)")
        }
        let topK = UInt32(cfg.topKExperts)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let halfBytes = MemoryLayout<Float16>.stride
        let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
            (try requireOnesPerExpertScale(), 0)
        // The MTP draft verifies against a MoE target and has no dense sibling,
        // so its stage always has a router; the view is optional because the
        // dense family shares the view type.
        guard let routerView = views.router else {
            throw ModelError.internalInconsistency(
                detail: "MTP prefill stage reached for a model with no router")
        }
        try encodeVerifyRouter(
            cb: cb, layer: L, routerView: routerView, scratch: scratch,
            rows: t, hiddenSize: D, perExpertScale: perExpertScale)
        cb.commit()
        try waitForCompletion(cb)
        recordKernelGPU(
            role: cfg.layerIsLinear(L)
                ? "prefill_gdn_router"
                : "prefill_attn_router", cb)
        dumpVerifyStageRows(layer: L, rows: t, scratch: scratch, hiddenSize: D)

        let idPtr = scratch.routeIDs.contents()
            .bindMemory(to: UInt32.self, capacity: t * cfg.topKExperts)
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

        // Shared expert for both rows, committed WITHOUT a host wait so its
        // GPU work overlaps the union miss fetch below. The tile path's
        // synchronous wait here was one of B1's three structural findings.
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        // Decode's own shared-expert encode, one row at a time: the prefill
        // block switches to a batched matmul at 8 rows, whose accumulation
        // order differs from decode's GEMV, and every verify row must produce
        // scalar decode's numbers (the wide-verify identity bisect measured
        // the batched block as a source of drift). Rows are encoded in order
        // on one command buffer, so decode's runner-wide scratch is safe.
        let rowBytes = D * halfBytes
        for row in 0..<t {
            try encodeSharedExpert(
                layer: L,
                x: scratch.routedX, xOffset: row * rowBytes,
                y: scratch.h1, yOffset: row * rowBytes,
                commandBuffer: sharedCB)
        }
        sharedCB.commit()

        let blobs: [TensorView]
        if let plan {
            if plan.misses.isEmpty {
                blobs = try model.routedExpertBuffers(for: plan)
            } else {
                let load = try model.beginFetchRoutedExperts(plan: plan)
                blobs = try await load.completion()
            }
        } else {
            blobs = try await model.fetchRoutedExperts(layer: L, experts: union)
        }

        while verifyPairActs.count < t {
            guard
                let made = ctx.device.makeBuffer(
                    length: cfg.topKExperts * cfg.moeIntermediateSize * halfBytes,
                    options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "verify.pair.acts.\(verifyPairActs.count)"
            verifyPairActs.append(made)
        }
        while verifyPairY.count < t {
            guard
                let made = ctx.device.makeBuffer(
                    length: D * halfBytes,
                    options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "verify.pair.y.\(verifyPairY.count)"
            verifyPairY.append(made)
        }
        while verifyPairArgBuffers.count < t {
            guard let made = moe.makeEmptyRoutedArgumentBuffer(device: ctx.device) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "verify.pair.args.\(verifyPairArgBuffers.count)"
            verifyPairArgBuffers.append(made)
        }

        let routedOffsets = try model.routedExpertOffsets(layer: L)
        guard let routedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        var rowBlobBuffers: [[MTLBuffer]] = []
        for row in 0..<t {
            var rowBufs: [MTLBuffer] = []
            var rowOffsets: [Int] = []
            rowBufs.reserveCapacity(cfg.topKExperts)
            rowOffsets.reserveCapacity(cfg.topKExperts)
            for expert in rowExperts[row] {
                guard let blobIndex = unionIndex[expert] else { continue }
                let view = blobs[blobIndex]
                rowBufs.append(view.buffer)
                rowOffsets.append(Int(view.offset))
            }
            rowBlobBuffers.append(rowBufs)
            let argBuf = verifyPairArgBuffers[row]
            moe.writeRoutedArgumentBuffer(
                argBuf,
                routedBlobs: rowBufs,
                topK: topK,
                routedBufferOffsets: rowOffsets)
            try moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: rowBufs,
                routedOffsets: routedOffsets,
                x: scratch.routedX,
                xOffset: row * D * halfBytes,
                acts: verifyPairActs[row],
                d: UInt32(D),
                f: FmoE,
                topK: topK)
        }
        for row in 0..<t {
            try moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: routedCB,
                routedArgBuffer: verifyPairArgBuffers[row],
                routedBlobs: rowBlobBuffers[row],
                routedOffsets: routedOffsets,
                acts: verifyPairActs[row],
                routingWeights: scratch.routeWeights,
                routingWeightsOffset: row * cfg.topKExperts * halfBytes,
                residual: scratch.h1,
                residualOffset: row * D * halfBytes,
                y: verifyPairY[row],
                d: UInt32(D),
                f: FmoE,
                topK: topK)
        }
        // Phase 2 already folded the shared branch (h1 rows as residual), so
        // the tail is the family's residual exit per row. A pre-norm family
        // adds; a hyper-connection family injects through its write gate, and
        // adding there would bypass the gate entirely -- which reads as a
        // model that answers in scrambled fragments, because the verify pass
        // is what produces every emitted token.
        //
        // The gate is chunk-shaped, not row-shaped: it reads the `normed` its
        // matching entry left for all rows and injects with a per-row weight.
        // So the per-row outputs are gathered into one contiguous block first
        // and written once, rather than looped over.
        if hyperConnection != nil {
            guard let blit = routedCB.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            for row in 0..<t {
                blit.copy(
                    from: verifyPairY[row], sourceOffset: 0,
                    to: scratch.h2, destinationOffset: row * D * halfBytes,
                    size: D * halfBytes)
            }
            blit.endEncoding()
            try encodeResidualExitPrefill(
                commandBuffer: routedCB,
                hidden: scratch.hidden,
                delta: scratch.h2,
                sublayer: .mlp, layer: L,
                tokens: t)
        } else {
            for row in 0..<t {
                try requireElementwise().encodeResidualAdd(
                    commandBuffer: routedCB,
                    hidden: scratch.hidden,
                    hiddenOffset: row * D * halfBytes,
                    delta: verifyPairY[row],
                    count: D)
            }
        }
        routedCB.commit()
        leaseTransferred = true
        let nextCB =
            L + 1 < cfg.numLayers
            ? ctx.queue.makeCommandBuffer() : nil
        if L + 1 < cfg.numLayers, nextCB == nil {
            throw ModelError.residentBufferWrapFailed
        }
        return VerifyPairOutcome(
            routedCB: routedCB, sharedCB: sharedCB, expertLease: lease,
            nextCB: nextCB)
    }

    /// Wide-verify identity bisect: each row's attention input and output, MoE
    /// input and post-attention residual, dumped beside scalar decode's
    /// `L{L}_attn_in` / `_attn_out` / `_mlp_in` / `_hidden_post_attn` for the
    /// same position. Called once the router's command buffer has completed, so
    /// every value the attention block and the MLP entry wrote is settled and
    /// the shared expert has not yet overwritten `h1`.
    private func dumpVerifyStageRows(
        layer L: Int, rows t: Int, scratch: PrefillChunkScratchBuffers, hiddenSize D: Int
    ) {
        guard let start = wideVerifyDumpStart, L <= dumpLayerLimit else { return }
        let half = MemoryLayout<Float16>.stride
        for row in 0..<t where activationDumpActive(position: start + row) {
            let position = start + row
            dumpActivationPrivate(
                "L\(L)_attn_in_rows", scratch.normed, count: D, position: position,
                offset: row * D * half)
            dumpActivationPrivate(
                "L\(L)_attn_out_rows", scratch.h1, count: D, position: position,
                offset: row * D * half)
            dumpActivationPrivate(
                "L\(L)_mlp_in_rows", scratch.routedX, count: D, position: position,
                offset: row * D * half)
            dumpActivationPrivate(
                "L\(L)_hidden_post_attn_rows", scratch.hidden, count: residualWidth,
                position: position, offset: row * residualWidth * half)
        }
    }
}
