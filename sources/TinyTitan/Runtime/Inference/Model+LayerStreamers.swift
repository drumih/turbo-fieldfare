import Darwin
import Foundation
import Metal
import TinyTitanFormat
import TinyTitanCore

// Lazy routed-expert and per-layer streamer management: opening a layer's
// backend, pinning the cache, and the counters the runner reads.
//
// Moved out of the `Model` declaration in `Model.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion;
// `openLayerLocked` stayed `private` because it moved with its callers.
extension Model {

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        let backend = try openStreamer(for: L)
        // The streamer is per-layer: `openLayerLocked(L)` bound it to layer
        // L's file with `expertOffsets = layers[L].experts.map(\.offset)`, and
        // `StreamLayout.expertOffset(layer: 0, ...)` is the branch that
        // consults that per-layer offset table. Passing the actual layer here
        // would select the dense cross-layer formula and mis-offset every
        // expert on layers above 0 — layer 0 is intentional.
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset: 0, biasLength: 0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: 0)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// The open streamer for a layer, after making sure the layer is open.
    ///
    /// `ensureLayerOpened` either leaves a streamer in the box or throws, so
    /// the lookup below cannot be nil in practice. It is a thrown
    /// `internalInconsistency` rather than a force unwrap so a broken
    /// invariant fails loudly without crashing the process.
    func openStreamer(for layer: Int) throws -> PreadExpertStreamer {
        try ensureLayerOpened(layer)
        guard let streamer = streamersQueue.sync(execute: { streamersBox.streamers[layer] }) else {
            throw ModelError.internalInconsistency(
                detail: "routed-expert streamer for layer \(layer) missing after ensureLayerOpened")
        }
        return streamer
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    /// The open is retried synchronously by `ensureLayerOpened(_:)` before any
    /// expert fetch on the layer, which rethrows the identical error — so a
    /// failure here is never dropped end-to-end.
    /// The capture needs no `nonisolated(unsafe)` anymore: `Model` is
    /// Sendable (implicit, once its handles carry `@unchecked Sendable` for
    /// the TT-097 contract), and the mutable members it does have
    /// (`streamersBox`) stay confined behind the serial `streamersQueue`.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        let model = self
        streamersQueue.async {
            do {
                try model.openLayerLocked(L)
            } catch {
                // Deferred: the synchronous `ensureLayerOpened(L)` that
                // precedes every expert fetch on this layer performs the same
                // idempotent open and rethrows this error to the prefill loop.
                // Nothing is silently lost; the async path only overlaps the
                // SHA-256 verification with the chunk's GPU work.
            }
        }
    }

    /// Drop a layer's expert cache. Safe once that layer's work is finished:
    /// the next use reopens it lazily, which is how it was created.

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let manifestRel = "packed_experts/\(basename)"
        let url =
            directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let layerFD = try modelDirectory.openFile(manifestRel)
        defer { close(layerFD) }
        if !streamersBox.layerVerified[L] {
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let actualSize = try modelDirectory.fileSize(
                fileDescriptor: layerFD, relativePath: manifestRel)
            guard actualSize == entry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: manifestRel, expected: entry.size, actual: actualSize)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(
                    fileDescriptor: layerFD,
                    named: manifestRel,
                    expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                break
            }
            streamersBox.layerVerified[L] = true
        }
        // Checked at the one place this product is formed. `StreamLayout.
        // expertOffset` multiplies `perLayer` again on every expert read
        // (`@inline(always)`, recomputed per call, on the decode path), so the
        // product is validated here rather than there -- a wrapped one would make
        // `streamSize` small, pass the streamer's own file-size check, and leave
        // every derived offset pointing outside the layer file.
        let phase1Bytes = Self.phase1Bytes(
            forLayer: L, layout: packedExpertsLayout)
        let (rans, ransPhases) = try Self.ransLayoutAndPhases(
            forLayer: L, layout: packedExpertsLayout, url: url)
        let compressed = try Self.compressedLayout(
            beside: url,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            phase1Bytes: phase1Bytes)
        let streamSize: UInt64
        if rans != nil {
            // The rANS payload is the layer file itself; its bytes are not at
            // raw offsets, so the stream is sized by the file.
            streamSize =
                try FileManager.default
                .attributesOfItem(atPath: url.path)[.size] as? UInt64 ?? 0
        } else {
            streamSize = try Self.streamSize(
                forLayer: L, layout: packedExpertsLayout,
                compressed: compressed, url: url)
        }
        let layout = try Self.streamLayout(
            url: url, packed: packedExpertsLayout, layer: L,
            streamSize: streamSize, phase1Bytes: phase1Bytes,
            compressed: compressed, rans: rans, ransPhases: ransPhases)
        let slotCount: Int
        switch streamingMode {
        case .pread(let configuredSlotCount):
            slotCount = configuredSlotCount
        }
        let effectiveSlotCount = slotCount
        let metalStagingPool: MetalExpertStagingPool?
        let metalIOService: MetalExpertIOService?
        if try ExpertIOBackend.environmentValue() == .metal {
            if streamersBox.metalStagingPool == nil {
                streamersBox.metalStagingPool = try MetalExpertStagingPool(
                    device: device,
                    byteCount: Int(packedExpertsLayout.expertStride),
                    // One staging slot per routed expert: a layer can miss all
                    // of them, and tryAcquire fails the whole request if the
                    // ring is short. This was hardcoded to 8 for the top-8
                    // families, which made the MTLIO + event path unreachable
                    // on Qwen3.8-Flash-Next -- it routes top-10, so any layer
                    // missing nine or more failed with "staging ring is
                    // unavailable" and the combination could never be measured.
                    slotCapacity: config.topKExperts)
            }
            if streamersBox.metalIOService == nil {
                streamersBox.metalIOService = try MetalExpertIOService(
                    device: device, maximumCommandsInFlight: 4)
            }
            metalStagingPool = streamersBox.metalStagingPool
            metalIOService = streamersBox.metalIOService
        } else {
            metalStagingPool = nil
            metalIOService = nil
        }
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: effectiveSlotCount,
            cachePolicy: expertCachePolicy,
            seedUseCounts: expertUsageProfile?.seedCounts(layer: L),
            eventCoordinator: expertIOTimelines?[L],
            metalStagingPool: metalStagingPool,
            metalIOService: metalIOService,
            pageCachePolicy: pageCachePolicy)
        // A newly opened layer is not wired yet; the next pin walks again.
        // When this run holds the cache wired (`profile.keepExpertCacheWired`,
        // set through `setKeepExpertCacheWired`) it is wired here, so a cache
        // that is never unpinned is never swapped out and the first decode
        // token does not pay to fault it back (measured 1.6-4.7 s per request
        // on Qwen3.8).
        streamersBox.pinnedComplete = false
        if streamersBox.keepWired {
            streamersBox.streamers[L]?.setSlotsPinned(true)
        }
    }

    /// The byte length of one layer's stream.
    ///
    /// A compressed layer is shorter than `expertsPerLayer * expertStride`: its
    /// bytes are the frames its index addresses, so that is its stream size.
    /// Sizing it from the raw product would fail the streamer's own file-size
    /// check on every compressed install -- a compressed file is about 92% of
    /// the raw one, never larger.
    static func streamSize(
        forLayer layer: Int,
        layout: PackedExpertsLayout,
        compressed: ExpertBlobIndex.Layout?,
        url: URL
    ) throws -> UInt64 {
        if let compressed {
            let end = ExpertBlobIndex.maximumEnd(compressed.entries)
            guard end > 0 else {
                throw ModelError.internalInconsistency(
                    detail: "packed expert layer \(layer) has an empty compressed index")
            }
            return end
        }
        let (rawSize, overflow) = UInt64(layout.expertsPerLayer)
            .multipliedReportingOverflow(by: layout.expertStride)
        guard !overflow else {
            throw ModelError.internalInconsistency(
                detail: "packed expert layer \(layer) declares "
                    + "\(layout.expertsPerLayer) experts of "
                    + "\(layout.expertStride) bytes, which overflows the stream size")
        }
        // A tiered layer is compact: its cold experts are shorter than the
        // stride, so the file is smaller than the uniform product. Bound it by
        // the highest expert end and by the product, then stream its real size.
        guard layout.layers.indices.contains(layer) else {
            throw ModelError.internalInconsistency(
                detail: "packed expert layer \(layer) has no layout entry")
        }
        var layerEnd: UInt64 = 0
        for expert in layout.layers[layer].experts {
            let (end, endOverflow) = expert.offset.addingReportingOverflow(expert.size)
            guard !endOverflow else {
                throw ModelError.internalInconsistency(
                    detail: "packed expert layer \(layer) expert ranges overflow")
            }
            layerEnd = max(layerEnd, end)
        }
        guard layerEnd <= rawSize else {
            throw ModelError.internalInconsistency(
                detail:
                    "packed expert layer \(layer) expert ranges exceed the stride product")
        }
        let fileSize =
            (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size]
            as? UInt64 ?? rawSize
        guard fileSize >= layerEnd, fileSize <= rawSize else {
            throw ModelError.internalInconsistency(
                detail: "packed expert layer \(layer) file size \(fileSize) is outside "
                    + "\(layerEnd)...\(rawSize)")
        }
        return fileSize
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

    /// Wire the routed-expert slot cache for decode, or release it for
    /// prefill.
    ///
    /// Decode is the phase where a reclaimed slot page costs an SSD read on
    /// the critical path, so that is the phase worth wiring. Prefill streams
    /// experts in bulk and instead needs the headroom -- holding the cache
    /// wired throughout measurably slowed ANE prefill, which has to place
    /// Core ML arenas alongside it. Called at the phase boundaries; cheap and
    /// idempotent, since each streamer skips a state it is already in.
    /// Wire layers as they open from now on (see `ModelProfile.keepExpertCacheWired`).
    public func setKeepExpertCacheWired(_ keep: Bool) {
        streamersQueue.sync { streamersBox.keepWired = keep }
    }

    public var expertCachePinQueueWaitNanos: UInt64 {
        streamersQueue.sync { streamersBox.pinQueueWaitNanos }
    }

    /// Returns true when every opened streamer is in the requested state.
    @discardableResult
    public func setExpertCachePinned(_ pinned: Bool) -> Bool {
        // Every decode step asks for the cache to be wired. Once every
        // streamer reports it is, there is nothing to do, and walking all of
        // them under the serial queue per token measured 73-91 ms on a
        // 48-layer model (TINYTITAN_RUNNER_STATS pre_pin_ms). The walk only runs
        // again after an unpin or a partial wire.
        let tEnter = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var walked = 0
        var unpinnedAfter = 0
        var earlyReturn = false
        var walkNanos: UInt64 = 0
        streamersQueue.sync {
            streamersBox.pinQueueWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEnter
            if pinned, streamersBox.pinnedComplete {
                earlyReturn = true
                return
            }
            let tWalk = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var complete = pinned
            for streamer in streamersBox.streamers {
                guard let streamer else { continue }
                walked += 1
                streamer.setSlotsPinned(pinned)
                if pinned, !streamer.isPinned {
                    complete = false
                    unpinnedAfter += 1
                }
            }
            walkNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWalk
            streamersBox.pinnedComplete = complete
        }
        let total = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEnter
        if PreadExpertStreamer.wireTraceEnabled, total > 2_000_000 {
            FileHandle.standardError.write(
                Data(
                    "[wire] setExpertCachePinned(\(pinned)) \(Double(total) / 1e6) ms early=\(earlyReturn) walked=\(walked) walk_ms=\(Double(walkNanos) / 1e6) unpinned_after=\(unpinnedAfter)\n"
                        .utf8))
        }
        return streamersQueue.sync { streamersBox.pinnedComplete } == pinned
    }

    /// The frames of a compressed layer, or nil when the layer is raw.
    ///
    /// The sidecar is `layer_NN.idx` beside the layer file. Its geometry must
    /// match the layer's -- expert count, and where the split falls -- so a
    /// sidecar written for another layer or the other split is refused at load
    /// rather than decoded into slots.
    /// The byte layout the streamer opens one layer with.
    /// Per-expert expansion maps for a tiered layer. The target offsets come
    /// from expert 0, which the converter keeps at the wide layout; a cold
    /// expert lists its own compact source offsets against them.
    static func expertTensorMaps(
        forLayer layer: Int, layout: PackedExpertsLayout
    ) throws -> [[ExpertBlobTensor]] {
        let experts = layout.layers[layer].experts
        guard let target = experts.first?.subTensors else { return [] }
        var maps: [[ExpertBlobTensor]] = []
        for expert in experts {
            guard let stored = expert.subTensors["gate"]?.bits,
                stored == 2 || stored == 3
            else {
                maps.append([])
                continue
            }
            let tensors = expert.subTensors.compactMap { name, tensor -> ExpertBlobTensor? in
                guard let wide = target[name] else { return nil }
                return ExpertBlobTensor(
                    sourceOffset: Int(tensor.offset),
                    sourceSize: Int(tensor.size),
                    targetOffset: Int(wide.offset),
                    targetSize: Int(wide.size),
                    bits: stored,
                    expands: name == "gate" || name == "up" || name == "down",
                    rescales: name.hasSuffix("_scales"))
            }
            try TwoBitExpand.validateExpansionLayout(
                tensors, expertStride: Int(layout.expertStride))
            maps.append(tensors)
        }
        return maps
    }

    static func streamLayout(
        url: URL, packed: PackedExpertsLayout, layer: Int,
        streamSize: UInt64, phase1Bytes: UInt64?,
        compressed: ExpertBlobIndex.Layout?, rans: ExpertRANSIndex.Layout?,
        ransPhases: [UInt8]
    ) throws -> StreamLayout {
        StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packed.expertsPerLayer,
            expertStride: packed.expertStride,
            expertOffsets: compressed == nil && rans == nil
                ? packed.layers[layer].experts.map(\.offset) : nil,
            expertSizes: compressed == nil && rans == nil
                ? packed.layers[layer].experts.map(\.size) : nil,
            expertTensors: compressed == nil && rans == nil
                ? try Self.expertTensorMaps(forLayer: layer, layout: packed)
                : nil,
            phase1Bytes: phase1Bytes,
            compressedFrames: compressed?.entries.map(\.frames),
            compressedBlockBytes: compressed?.blockBytes ?? 0,
            rans: rans, ransPhases: ransPhases)
    }

    static func compressedLayout(
        beside url: URL,
        expertsPerLayer: Int,
        expertStride: UInt64,
        phase1Bytes: UInt64?
    ) throws -> ExpertBlobIndex.Layout? {
        let indexName = ExpertBlobIndex.sidecarName(forLayerFile: url.lastPathComponent)
        let indexURL = url.deletingLastPathComponent().appendingPathComponent(indexName)
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return nil }
        guard let cap = ExpertBlobIndex.maximumIndexBytes(expertsPerLayer: expertsPerLayer)
        else {
            throw ModelError.internalInconsistency(
                detail: "\(indexName) cannot be sized: layer declares \(expertsPerLayer) experts")
        }
        // Bounded, and through the model directory's O_NOFOLLOW chain: a 12 KiB
        // metadata file has no business allocating whatever length it declares.
        let indexData = try BoundedMetadataRead.read(fileAt: indexURL, maxBytes: cap)
        let phase0Bytes = phase1Bytes ?? expertStride
        return try ExpertBlobIndex.decodeLayout(
            indexData,
            expectedCount: expertsPerLayer,
            phase0Bytes: phase0Bytes,
            phase1Bytes: phase1Bytes.map { expertStride - $0 })
    }

    /// Where a layer's expert blob splits, or nil when it does not.
    ///
    /// From the manifest, not a fraction of the stride: the blob is gate|up|down
    /// contiguous and the `down` tensor's offset within an expert is exactly the
    /// gate+up byte count. Only an interior boundary splits; anything else (a
    /// manifest without the tensor, or padding that pushed `down` to the end)
    /// reads whole regions.
    static func phase1Bytes(
        forLayer layer: Int,
        layout: PackedExpertsLayout
    ) -> UInt64? {
        guard layer >= 0, layer < layout.layers.count else { return nil }
        let downOffset = layout.layers[layer].experts.first?.subTensors["down"]?.offset
        return ExpertBlobIndex.phase1Bytes(
            downOffset: downOffset, expertStride: layout.expertStride)
    }
}
