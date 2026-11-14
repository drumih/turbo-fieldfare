import Foundation
import Metal
import TinyTitanCore

/// Prefill routed MoE: the router readback, expert streaming, tiled
/// phase-1/phase-2 and the residual tail, plus the dense family's FFN.
///
/// Split out of `RealForwardRunner+Prefill.swift` (2026-09-28) under the
/// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
extension RealForwardRunner {
    /// The dense family's prefill FFN.
    ///
    /// The shared-expert block *is* this family's feed-forward network, so this
    /// is that block plus the residual add the MoE tail performs for the shared
    /// branch -- the same arithmetic, without a mixture to route, fetch or
    /// reduce. `sharedExpertGated` is false for the family and its schema has no
    /// gate tensor, so the scalar-gate step the MoE path applies has nothing to
    /// read here.
    func encodeDenseFFNPrefill(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int
    ) throws {
        // Commit and drain the incoming buffer first: it carries this layer's
        // attention and its MLP entry norm. The MoE stage does the same at the
        // top of its pipeline (it needs the routing readback), and without it
        // the layer's attention work is never submitted -- the buffers below
        // stay zero and the stack produces a zero hidden state, which reads as
        // confident nonsense rather than as an error.
        cb.commit()
        try waitForCompletion(cb)
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let sharedProj = sharedExpertProjections[L]
        try prefillSharedExpert.encodeBlock(
            commandBuffer: sharedCB,
            x: scratch.routedX,
            y: scratch.h1,
            gate: sharedProj.gate,
            up: sharedProj.up,
            down: sharedProj.down,
            scratchGate: scratch.sharedGateScratch,
            scratchUp: scratch.sharedUpScratch,
            scratchAct: scratch.sharedActScratch,
            queryCount: t,
            d: D,
            intermediate: cfg.intermediateSize,
            xStrideElements: D,
            yStrideElements: D)
        sharedCB.commit()
        try waitForCompletion(sharedCB)
        recordKernelGPU(role: "prefill_shared_expert", sharedCB)

        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try requireElementwise().encodeResidualAdd(
            commandBuffer: tailCB,
            hidden: scratch.hidden,
            delta: scratch.h1,
            count: t * D)
        tailCB.commit()
        try waitForCompletion(tailCB)
        recordKernelGPU(role: "prefill_dense_ffn_tail", tailCB)
        if L + 1 < cfg.numLayers {
            guard let nextCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = nextCB
        }
    }

    /// Router, routed-expert fetch and the MoE tail for one prefill layer.
    ///
    /// lint:allow-long one layer's MoE stage is a single ordered pipeline:
    /// route readback, expert streaming, tiled phase-1/phase-2, then the
    /// residual tail. It rebinds the command buffer partway through (the
    /// resident buffer wraps between layers), so the stages share mutable
    /// encoding state and cannot be separated without threading it back out.
    func encodeRoutedMoEPrefill(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        startPosition: Int,
        hiddenSize D: Int,
        layerStart prefillLayerStart: UInt64,
        routeNanos prefillRouteNanos: inout UInt64,
        tileNanos prefillTileNanos: inout UInt64,
        tailNanos prefillTailNanos: inout UInt64,
        activeExperts prefillActiveExperts: inout UInt64
    ) async throws {
        // A dense model has no routed mixture: its FFN is the shared-expert
        // block, so the router, the readback and the routed tiles do not exist
        // for it. Taken here, at the top, so none of them is encoded.
        if cfg.numExperts == 0 {
            try encodeDenseFFNPrefill(
                cb: &cb, layer: L, scratch: scratch,
                tokenCount: t, hiddenSize: D)
            return
        }
        guard let routerView = views.router else {
            throw ModelError.internalInconsistency(
                detail: "routed prefill stage reached for a model with no router")
        }
        var prefillRouteEnd = prefillLayerStart
        var prefillTileEnd = prefillLayerStart
        let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
            (try requireOnesPerExpertScale(), 0)
        try prefillRouter.encodeBlock(
            commandBuffer: cb,
            weights: routerView.buffer,
            weightsOffset: Int(routerView.offset),
            scales: routerView.buffer,
            scalesOffset: Int(routerView.scaleOffset),
            biases: routerView.buffer,
            biasesOffset: Int(routerView.biasOffset),
            hidden: scratch.routedX,
            effectiveScale: effectiveScaleBuffers[L],
            perExpertScale: perExpertScale.buffer,
            perExpertScaleOffset: perExpertScale.offset,
            outIndices: scratch.routeIDs,
            outWeights: scratch.routeWeights,
            queryCount: UInt32(t),
            numExperts: UInt32(cfg.numExperts),
            d: UInt32(D),
            topK: UInt32(cfg.topKExperts),
            hiddenStrideElements: UInt32(D))

        cb.commit()
        try waitForCompletion(cb)
        // Prefill had no occupancy instrumentation at all: these buffers
        // never reached recordKernelGPU, so TINYTITAN_KERNEL_STATS reported
        // only the decode tokens of a request and prefill looked idle.
        // Split by layer kind: the Track A go/no-go needs to know how
        // the attention-block time divides between full-attention
        // layers (whole block is ANE-expressible) and Gated-DeltaNet
        // layers (only the dense projections are; the recurrent scan
        // is not representable in a static Core ML graph).
        recordKernelGPU(
            role: cfg.layerIsLinear(L)
                ? "prefill_gdn_router"
                : "prefill_attn_router", cb)

        let routeCount = t * cfg.topKExperts
        let idPtr = scratch.routeIDs.contents()
            .bindMemory(to: UInt32.self, capacity: routeCount)
        let weightPtr = scratch.routeWeights.contents()
            .bindMemory(to: Float16.self, capacity: routeCount)
        // Reused per-chunk host scratch (R38): cleared in place so
        // the routed-tile planner never allocates per chunk.
        routeIDScratch.removeAll(keepingCapacity: true)
        routeWeightScratch.removeAll(keepingCapacity: true)
        routeIDScratch.reserveCapacity(routeCount)
        routeWeightScratch.reserveCapacity(routeCount)
        for i in 0..<routeCount {
            routeIDScratch.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
            routeWeightScratch.append(weightPtr[i])
        }
        if routeTraceFD >= 0 {
            // Same line format as the decode trace, prefixed with
            // the token's absolute position, so tail-window routing
            // can be compared with the response's.
            let k = cfg.topKExperts
            for row in 0..<t {
                recordRouteTrace(
                    layer: L, position: startPosition + row,
                    experts: routeIDScratch[row * k..<(row + 1) * k].map { Int($0) })
            }
        }
        let pairs = PrefillRouter.makeTokenExpertPairs(
            indices: routeIDScratch,
            weights: routeWeightScratch,
            queryCount: t,
            topK: cfg.topKExperts)
        let schedulerConfig: PrefillRoutedTileSchedulerConfig
        let routeTileExpertCount: Int
        if let slotCount = model.routedExpertCacheSlotCount() {
            guard
                let fitted = Self.Defaults.prefillRoutedTileSchedulerConfig.fitting(
                    slotCount: slotCount)
            else {
                throw PrefillError.chunkedUnsupported(
                    "prefill routed tiles cannot fit the \(slotCount)-slot expert cache")
            }
            schedulerConfig = fitted
            routeTileExpertCount = fitted.tileExperts
        } else {
            schedulerConfig = Self.Defaults.prefillRoutedTileSchedulerConfig
            routeTileExpertCount = schedulerConfig.tileExperts
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: t,
            topK: cfg.topKExperts,
            numExperts: cfg.numExperts,
            tileExpertCount: routeTileExpertCount,
            expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))
        prefillRouteEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        prefillRouteNanos &+= prefillRouteEnd - prefillLayerStart
        // One group per *distinct* expert this chunk touches. For a
        // 1-token chunk this is topK; for a speculative 2-token verify
        // it is the union of the two tokens' routes, which is what
        // decides whether the extra row rides along on weights the
        // first row already pulled in or pays for its own.
        prefillActiveExperts &+= UInt64(routes.groups.count)

        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let sharedProj = sharedExpertProjections[L]
        try prefillSharedExpert.encodeBlock(
            commandBuffer: sharedCB,
            x: scratch.routedX,
            y: scratch.h1,
            gate: sharedProj.gate,
            up: sharedProj.up,
            down: sharedProj.down,
            scratchGate: scratch.sharedGateScratch,
            scratchUp: scratch.sharedUpScratch,
            scratchAct: scratch.sharedActScratch,
            queryCount: t,
            d: D,
            intermediate: cfg.intermediateSize,
            xStrideElements: D,
            yStrideElements: D)
        if cfg.sharedExpertGated {
            // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX),
            // per chunk row.
            let gateView = try requireTensorView(sharedProj.scalarGate, "shared-expert scalar gate")
            let halfBytes = MemoryLayout<Float16>.stride
            for row in 0..<t {
                try encodeScalarGate(
                    commandBuffer: sharedCB,
                    view: gateView,
                    x: scratch.routedX,
                    xOffset: row * D * halfBytes,
                    y: scratch.sharedScalarGate,
                    yOffset: row * halfBytes,
                    n: UInt32(D))
            }
            for row in 0..<t {
                try requireElementwise().encodeSigmoidScalarMul(
                    commandBuffer: sharedCB,
                    y: scratch.h1,
                    yOffset: row * D * halfBytes,
                    gate: scratch.sharedScalarGate,
                    gateOffset: row * halfBytes,
                    count: D)
            }
        }
        sharedCB.commit()
        try waitForCompletion(sharedCB)
        recordKernelGPU(role: "prefill_shared_expert", sharedCB)

        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: ctx.device,
            routes: routes)
        let routedOffsets = try model.routedExpertOffsets(layer: L)
        struct PendingPrefillTile {
            let tileIndex: Int
            let commandBuffer: MTLCommandBuffer
            let fetch: PrefillStreamedTileFetchResult
            let argumentBuffer: PrefillStreamedTileArgumentBuffer
        }
        var pendingTiles: [PendingPrefillTile] = []
        var tileLifetime = PrefillStreamedTileSlotLifetime()
        // `withExtendedLifetime` below takes a non-throwing closure,
        // so the wait error is captured here and rethrown after the
        // fetched blobs are released.
        var pendingTileError: Error?
        var tailError: Error?
        func drainOldestPendingTile() throws {
            guard !pendingTiles.isEmpty else { return }
            let pending = pendingTiles.removeFirst()
            withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                do {
                    try waitForCompletion(pending.commandBuffer)
                    recordKernelGPU(
                        role: "prefill_routed_tile",
                        pending.commandBuffer)
                } catch {
                    // Rethrown after the fetched blobs are released.
                    pendingTileError = error
                }
            }
            if let error = pendingTileError {
                pendingTileError = nil
                throw error
            }
            if !pending.fetch.plannedMissSlots.isEmpty {
                try tileLifetime.complete(tileIndex: pending.tileIndex)
            }
        }

        let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
        if ProcessInfo.processInfo.environment["TINYTITAN_TILE_TRACE"] == "1" {
            FileHandle.standardError.write(
                Data(
                    "[tiles] layer=\(L) tiles=\(routes.tiles.count) groups=\(routes.groups.count) pairs=\(routes.sortedPairs.count)\n"
                        .utf8))
        }
        for (tileIndex, tile) in routes.tiles.enumerated() {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                forTile: tileIndex,
                routes: routes)
            var plannedFetch: RoutedExpertFetchPlan?
            if !pendingTiles.isEmpty {
                let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                if !pendingAssignedSlots.isEmpty {
                    let pendingSlots = Set(pendingAssignedSlots)
                    let plan = try model.planRoutedExpertsIfPossible(
                        layer: L,
                        experts: expertIDs,
                        avoidingSlots: pendingSlots)
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: pendingAssignedSlots,
                            avoidingSlotPlanAvailable: plan != nil))
                    switch decision {
                    case .prefetchNext:
                        guard let plan else {
                            throw ModelError.indexCorrupt(
                                detail: "routed tile scheduler requested missing plan")
                        }
                        plannedFetch = plan
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler ignored pending tile")
                    }
                } else {
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: [],
                            avoidingSlotPlanAvailable: false))
                    switch decision {
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending, .prefetchNext:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler failed to drain empty-slot pending tile")
                    }
                }
            } else {
                let decision = routedTileScheduler.decide(
                    PrefillRoutedTileSchedulerInput(
                        hasPendingTile: false,
                        pendingAssignedSlots: [],
                        avoidingSlotPlanAvailable: false))
                switch decision {
                case .issueWithoutPending:
                    break
                case .prefetchNext, .drainBeforeIssue:
                    throw ModelError.indexCorrupt(
                        detail:
                            "routed tile scheduler requested pending action without pending tile")
                }
            }
            let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model,
                layer: L,
                tileIndex: tileIndex,
                routes: routes,
                plannedFetch: plannedFetch,
                avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
            try fetch.binding.validateCoversPairs(
                routes.sortedPairs,
                pairStart: Int(tile.pairStart),
                pairCount: Int(tile.pairCount))
            if !fetch.plannedMissSlots.isEmpty {
                try tileLifetime.begin(
                    tileIndex: tileIndex,
                    plannedSlots: fetch.plannedMissSlots)
            }
            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: ctx.device,
                binding: fetch.binding)
            let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(D),
                routedIntermediate: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts),
                hiddenStrideElements: UInt32(D),
                binding: fetch.binding,
                offsets: routedOffsets)
            guard let tileCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            _ = try prefillGroupedMoE.encodeStreamedBatched(
                commandBuffer: tileCB,
                hidden: scratch.routedX,
                sortedPairs: metadata.sortedPairs,
                routePartials: scratch.routePartials,
                gateUpActScratch: scratch.routedGateUpActScratch,
                downScratch: scratch.routedDownScratch,
                argumentBuffer: argumentBuffer,
                binding: fetch.binding,
                params: streamedParams,
                pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
            tileCB.commit()
            pendingTiles.append(
                PendingPrefillTile(
                    tileIndex: tileIndex,
                    commandBuffer: tileCB,
                    fetch: fetch,
                    argumentBuffer: argumentBuffer))
            while pendingTiles.count > schedulerConfig.maxPendingDepth {
                try drainOldestPendingTile()
            }
        }
        while !pendingTiles.isEmpty {
            try drainOldestPendingTile()
        }
        prefillTileEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        prefillTileNanos &+= prefillTileEnd - prefillRouteEnd
        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try prefillMoE.encodeReduceTokenMajor(
            commandBuffer: tailCB,
            routePartials: scratch.routePartials,
            routeWeights: scratch.routeWeights,
            h2: scratch.h2,
            queryCount: UInt32(t),
            topK: UInt32(cfg.topKExperts),
            d: UInt32(D))
        if hyperConnection != nil {
            // The gated write injects one block output per stream, so
            // the two MLP branches are summed first and written once.
            // Adding them separately would apply the inject gate
            // twice.
            try requireElementwise().encodeResidualAdd(
                commandBuffer: tailCB,
                hidden: scratch.h2,
                delta: scratch.h1,
                count: t * D)
            try encodeResidualExitPrefill(
                commandBuffer: tailCB,
                hidden: scratch.hidden,
                delta: scratch.h2,
                sublayer: .mlp, layer: L,
                tokens: t)
        } else {
            // Plain pre-norm tail: hidden += gated shared branch
            // + routed branch.
            try requireElementwise().encodeResidualAdd(
                commandBuffer: tailCB,
                hidden: scratch.hidden,
                delta: scratch.h1,
                count: t * D)
            try requireElementwise().encodeResidualAdd(
                commandBuffer: tailCB,
                hidden: scratch.hidden,
                delta: scratch.h2,
                count: t * D)
        }
        tailCB.commit()
        withExtendedLifetime(metadata) {
            do {
                try waitForCompletion(tailCB)
                recordKernelGPU(role: "prefill_moe_reduce", tailCB)
            } catch {
                // Rethrown after `metadata` is released.
                tailError = error
            }
        }
        if let error = tailError {
            tailError = nil
            throw error
        }
        if L + 1 < cfg.numLayers {
            guard let nextCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = nextCB
        }
        prefillTailNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - prefillTileEnd
    }
}
