import Foundation
import Metal
import TinyTitanCore

// QSA sparse attention: the indexer weights, the selection, the decode and
// prefill encoders, and the snapshot dumps.
//
// Split out of `RealForwardRunner+Residual.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// `writeQSACaches` stayed `private`: both of its callers are in these two
// extensions and moved with it.
// MARK: - QSA sparse attention

extension RealForwardRunner {
    /// The indexer's weights for one full-attention layer.
    func indexerWeights(layer: Int) throws -> QSAIndexer.Weights {
        QSAIndexer.Weights(
            queryProjection: try model.indexerQProj(layer: layer),
            keyProjection: try model.indexerKProj(layer: layer),
            queryNorm: try model.indexerQNorm(layer: layer),
            keyNorm: try model.indexerKNorm(layer: layer))
    }

    /// Whether this layer needs the indexer to choose keys for this query.
    /// Below the dense-exact window every visible key is kept anyway, so the
    /// selection is skipped rather than computed and thrown away.
    func qsaSelectionNeeded(layer: Int, position: Int) -> Bool {
        guard qsaIndexer != nil, cfg.fullAttentionLayerMask[layer] == 1,
            let exactness = qsaExactness
        else { return false }
        return !exactness.isDenseExact(visibleKeys: position + 1)
    }

    /// Runs the residual entry and the indexer for one full-attention layer,
    /// returning the selection the attention should honour.
    ///
    /// This is synchronous, and deliberately so: turning block scores into a
    /// per-key selection reproduces an ordering (score descending, index
    /// ascending, with the cell budget able to cut a block in half) that is
    /// clear on the host and fiddly on the GPU. The sync costs one barrier on
    /// the twelve full-attention layers, against a decode step that is bound
    /// by expert I/O; doing the selection on the GPU is the optimization, and
    /// it should be made against this as the reference.
    /// Decode key selection on the GPU (qsa_select_decode) instead of a
    /// CPU sort behind a command-buffer round trip per full-attention layer.
    /// Off until verified: TINYTITAN_QSA_GPU_SELECT=1 turns it on, =verify runs
    /// both and reports any mask difference.
    var qsaGPUSelectEnabled: Bool { profile.qsaGPUSelect }
    static let qsaSelectVerifyFlag =
        ProcessInfo.processInfo.environment["TINYTITAN_QSA_GPU_SELECT"] == "verify"
    var qsaSelectVerify: Bool { Self.qsaSelectVerifyFlag }

    func encodeQSAEntryAndSelect(
        passthrough: inout MTLCommandBuffer,
        hidden: MTLBuffer,
        norm: TensorView,
        out: MTLBuffer,
        layer: Int,
        position: Int,
        eps: Float
    ) throws -> DecodeKeySelection? {
        guard let indexer = qsaIndexer else { return nil }
        let weights = try indexerWeights(layer: layer)
        let selecting = qsaSelectionNeeded(layer: layer, position: position)
        guard selecting else {
            // Inside the window nothing is selected, so there is nothing to
            // read back and no reason to break the pipeline: the entry and
            // the key append ride the layer's own command buffer.
            try encodeResidualEntryDecode(
                commandBuffer: passthrough,
                hidden: hidden, norm: norm, out: out,
                sublayer: .attention, layer: layer,
                eps: eps)
            try indexer.encodeAppendKey(
                commandBuffer: passthrough, hidden: out,
                weights: weights, layer: layer,
                position: position, eps: eps)
            return DecodeKeySelection(mask: nil, indices: nil, count: position + 1)
        }
        // Host-KV streaming needs the selected keys on the host to gather
        // their rows; the GPU selection deliberately never reads back. Force
        // the CPU reference while the window is engaged.
        let streaming = kv?.streamsLayer(layer) ?? false
        let gpuSelect =
            qsaGPUSelectEnabled && indexer.canSelectOnGPU
            && activationDumpDirectory == nil && !streaming
        if gpuSelect && !qsaSelectVerify {
            // Everything rides the layer's own command buffer: entry, key
            // append, scores, and the selection itself. No readback.
            try encodeResidualEntryDecode(
                commandBuffer: passthrough, hidden: hidden,
                norm: norm, out: out,
                sublayer: .attention, layer: layer,
                eps: eps)
            try indexer.encodeAppendKey(
                commandBuffer: passthrough, hidden: out,
                weights: weights, layer: layer,
                position: position, eps: eps)
            try rotate(&passthrough, role: "qsa.entry_append")
            try indexer.encodeScores(
                commandBuffer: passthrough, hidden: out,
                weights: weights, layer: layer,
                position: position, eps: eps)
            try rotate(&passthrough, role: "qsa.scores")
            let mask = try indexer.encodeSelectKeys(
                commandBuffer: passthrough,
                visibleKeys: position + 1)
            try rotate(&passthrough, role: "qsa.select")
            return DecodeKeySelection(mask: mask, indices: nil, count: 0)
        }
        guard
            try runSync({ cb in
                try encodeResidualEntryDecode(
                    commandBuffer: cb, hidden: hidden,
                    norm: norm, out: out,
                    sublayer: .attention, layer: layer,
                    eps: eps)
                // The key is cached at every position, in or out of the window:
                // crossing the boundary later must not find holes behind it.
                try indexer.encodeAppendKey(
                    commandBuffer: cb, hidden: out,
                    weights: weights, layer: layer,
                    position: position, eps: eps)
                try indexer.encodeScores(
                    commandBuffer: cb, hidden: out,
                    weights: weights, layer: layer,
                    position: position, eps: eps)
                if gpuSelect {
                    _ = try indexer.encodeSelectKeys(
                        commandBuffer: cb,
                        visibleKeys: position + 1)
                }
            }) != nil
        else {
            throw ModelError.residentBufferWrapFailed
        }
        if gpuSelect {
            // Verify mode: the GPU mask is captured, the CPU reference then
            // overwrites the buffer and decides the attention; any difference
            // is reported per (layer, position).
            let gpuMask = indexer.keepMaskBytes(count: position + 1)
            let compact = indexer.selectKeysCompact(visibleKeys: position + 1)
            let cpuMask = indexer.keepMaskBytes(count: position + 1)
            if gpuMask != cpuMask {
                let diff = zip(gpuMask, cpuMask).enumerated().filter {
                    $0.element.0 != $0.element.1
                }
                FileHandle.standardError.write(
                    Data(
                        "TinyTitan qsa_select_verify MISMATCH layer=\(layer) position=\(position) cells=\(diff.count) first=\(diff.prefix(4).map { $0.offset })\n"
                            .utf8))
            } else if position % 64 == 0 {
                FileHandle.standardError.write(
                    Data(
                        "TinyTitan qsa_select_verify ok layer=\(layer) position=\(position)\n".utf8)
                )
            }
            return DecodeKeySelection(
                mask: compact.mask, indices: compact.indices,
                count: compact.count)
        }
        let compact = indexer.selectKeysCompact(visibleKeys: position + 1)
        if layer == Self.qsaSnapshotLayer {
            dumpQSASnapshot(layer: layer, visibleKeys: position + 1)
        }
        return DecodeKeySelection(
            mask: compact.mask, indices: compact.indices, count: compact.count)
    }

    /// The same snapshot from the chunked path, taken from its last row.
    func dumpQSAChunkSnapshot(
        selection: QSASelection,
        lastVisible: Int, rows: Int
    ) {
        guard let directory = activationDumpDirectory,
            let indexer = qsaIndexer
        else { return }
        let base = (rows - 1) * selection.maskStride
        let keep = selection.mask.contents()
            .bindMemory(to: UInt8.self, capacity: base + lastVisible)
        let bytes = (0..<lastVisible).map { keep[base + $0] }
        try? Data(bytes).write(
            to: directory.appendingPathComponent("qsa_keep.bin"))
        let blocks = (lastVisible - 1) / indexer.compressRatio + 1
        let scores = indexer.debugSnapshot(
            cells: 1, blocks: indexer.scoredBlocksPerRow * rows
        ).scores
        let row = Array(
            scores[
                (rows - 1) * indexer.scoredBlocksPerRow..<(rows - 1) * indexer.scoredBlocksPerRow
                    + blocks])
        row.withUnsafeBufferPointer {
            try? Data(buffer: $0).write(
                to: directory.appendingPathComponent("qsa_scores.f32"))
        }
        writeQSACaches(
            directory: directory, indexer: indexer,
            visibleKeys: lastVisible, blocks: blocks)
    }

    /// Writes the indexer's scores and selection for the newest query, so the
    /// decode and prefill paths can be compared as numbers.
    ///
    /// Only the first full-attention layer is snapshotted, and that is the
    /// point: later layers see inputs that have already drifted, so a
    /// disagreement there says nothing about whether the two paths select the
    /// same way. At the first one they have the same input, and any
    /// difference is the selection's own.
    func dumpQSASnapshot(layer: Int, visibleKeys: Int) {
        guard let directory = activationDumpDirectory,
            let indexer = qsaIndexer
        else { return }
        let blocks = (visibleKeys - 1) / indexer.compressRatio + 1
        let snapshot = indexer.debugSnapshot(cells: visibleKeys, blocks: blocks)
        try? Data(snapshot.keep).write(
            to: directory.appendingPathComponent("qsa_keep.bin"))
        snapshot.scores.withUnsafeBufferPointer {
            try? Data(buffer: $0).write(
                to: directory.appendingPathComponent("qsa_scores.f32"))
        }
        writeQSACaches(
            directory: directory, indexer: indexer,
            visibleKeys: visibleKeys, blocks: blocks)
    }

    private func writeQSACaches(
        directory: URL, indexer: QSAIndexer,
        visibleKeys: Int, blocks: Int
    ) {
        if let pooled = indexer.debugPooled(
            layer: Self.qsaSnapshotLayer,
            blocks: blocks)
        {
            pooled.withUnsafeBufferPointer {
                try? Data(buffer: $0).write(
                    to: directory.appendingPathComponent("qsa_pooled.f32"))
            }
        }
        if let raw = indexer.debugRawKeys(
            layer: Self.qsaSnapshotLayer,
            count: visibleKeys)
        {
            raw.withUnsafeBufferPointer {
                try? Data(buffer: $0).write(
                    to: directory.appendingPathComponent("qsa_raw.f32"))
            }
        }
    }
}

/// Fills the indexer's caches for a prefill chunk, so decode can cross the
/// dense-exact boundary later without finding holes behind it.
extension RealForwardRunner {
    func encodeQSAPrefill(
        cb: inout MTLCommandBuffer,
        blockInput: MTLBuffer,
        layer: Int, startPosition: Int, tokens: Int,
        eps: Float
    ) throws -> QSASelection? {
        guard let indexer = qsaIndexer,
            cfg.fullAttentionLayerMask[layer] == 1
        else { return nil }
        let commandBuffer = cb
        let weights = try indexerWeights(layer: layer)
        let destination = try indexer.rawKeyDestination(
            layer: layer, startPosition: startPosition, tokens: tokens)
        let key = weights.keyProjection
        // A promoted projection has no scales or biases, and the quantized QMM
        // cannot read one: it would take the first k bytes of each 2k-byte bf16
        // row as 8-bit codes and multiply by scales read from offset 0.
        // `prepare_qwen38.py` promotes index_q_proj and index_k_proj exactly
        // that way for an 8-bit build, and decode already branches on dtype.
        // One GEMV per row, the same shape the hyper-connection prefill
        // projections use.
        if key.dtype == 1 {
            let half = MemoryLayout<Float16>.stride
            let columns = Int(key.shape.1)
            let rows = indexer.headDim
            for row in 0..<tokens {
                try bf16Projection.encode(
                    commandBuffer: commandBuffer,
                    weights: key.buffer, weightsOffset: Int(key.offset),
                    x: blockInput, xOffset: row * columns * half,
                    y: destination.buffer,
                    yOffset: destination.offset + row * rows * half,
                    m: UInt32(rows), n: UInt32(columns))
            }
        } else {
            try prefillQMM.encode(
                commandBuffer: commandBuffer,
                weights: key.buffer,
                weightsOffset: Int(key.offset),
                scales: key.buffer,
                scalesOffset: Int(key.scaleOffset),
                biases: key.buffer,
                biasesOffset: Int(key.biasOffset),
                x: blockInput,
                y: destination.buffer,
                yOffset: destination.offset,
                t: tokens,
                n: indexer.headDim,
                k: Int(key.shape.1))
        }
        try indexer.encodePoolPrefill(
            commandBuffer: commandBuffer,
            weights: weights, layer: layer,
            startPosition: startPosition,
            tokens: tokens, eps: eps)

        // Nothing to choose while the chunk's longest query still sees
        // everything, so no scoring pass and no barrier.
        guard let exactness = qsaExactness,
            !exactness.isDenseExact(visibleKeys: startPosition + tokens)
        else { return nil }

        try indexer.encodeScoresPrefill(
            commandBuffer: commandBuffer, blockInput: blockInput,
            weights: weights, layer: layer,
            startPosition: startPosition, tokens: tokens, eps: eps,
            project: { cb, view, x, y, rows, columns, count in
                // Same promoted-projection branch as the key path above.
                if view.dtype == 1 {
                    let half = MemoryLayout<Float16>.stride
                    for row in 0..<count {
                        try self.bf16Projection.encode(
                            commandBuffer: cb,
                            weights: view.buffer, weightsOffset: Int(view.offset),
                            x: x, xOffset: row * columns * half,
                            y: y, yOffset: row * rows * half,
                            m: UInt32(rows), n: UInt32(columns))
                    }
                    return
                }
                try self.prefillQMM.encode(
                    commandBuffer: cb,
                    weights: view.buffer,
                    weightsOffset: Int(view.offset),
                    scales: view.buffer,
                    scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer,
                    biasesOffset: Int(view.biasOffset),
                    x: x, y: y,
                    t: count, n: rows, k: columns)
            })
        // The selection is a host computation over the scores, so the chunk's
        // command buffer has to land first. The same barrier the routed MoE
        // already takes for its route readback, one layer earlier.
        commandBuffer.commit()
        try waitForCompletion(commandBuffer)
        recordKernelGPU(role: "prefill_qsa_index", commandBuffer)
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
        let selection = indexer.selectKeysPrefill(
            startPosition: startPosition,
            tokens: tokens,
            layer: layer)
        // The chunk's last row is the one a sequential run's final decode
        // step also produces, so it is the comparable one.
        if activationDumpDirectory != nil, layer == Self.qsaSnapshotLayer,
            let selection
        {
            dumpQSAChunkSnapshot(
                selection: selection,
                lastVisible: startPosition + tokens,
                rows: tokens)
        }
        return selection
    }

    /// The window in which dense attention is exact for this model, or nil
    /// for a family without sparse attention.
    var qsaExactness: QSAExactness? {
        guard cfg.sparseIndexer.enabled else { return nil }
        return QSAExactness(
            budget: Self.qsaBudget(cfg.sparseIndexer),
            compressRatio: cfg.sparseIndexer.compressRatio)
    }

    /// The indexer's key budget, with `TINYTITAN_QSA_BUDGET` able to lower it.
    ///
    /// Verification knob, not a tuning one. At the shipped budget the sparse
    /// path only engages past 2,051 tokens, which makes every check of it a
    /// multi-thousand-token run; lowering the budget moves the boundary down
    /// so the same code runs against a reference at a length that can be
    /// diffed in seconds. It only ever lowers.
    static func qsaBudget(_ config: SparseIndexerConfig) -> Int {
        guard let raw = ProcessInfo.processInfo.environment["TINYTITAN_QSA_BUDGET"],
            let value = Int(raw), value > 0
        else { return config.budget }
        return min(value, config.budget)
    }

    /// Refuses a position the sparse-attention path cannot serve faithfully.
    /// See `QSAIndexerRequired` for why this is an error rather than a note.
    func requireQSAExact(visibleKeys: Int) throws {
        guard qsaIndexer == nil, let exactness = qsaExactness,
            !exactness.isDenseExact(visibleKeys: visibleKeys)
        else { return }
        throw QSAIndexerRequired(
            visibleKeys: visibleKeys,
            exactWindow: exactness.maximumExactVisibleKeys)
    }

    /// The same gate for prefill.
    ///
    /// Both paths select now, so this only fires for a family that declares a
    /// sparse indexer without one being built -- which nothing does today,
    /// and which would otherwise attend densely and say nothing about it.
    func requireQSADensePrefill(visibleKeys: Int) throws {
        try requireQSAExact(visibleKeys: visibleKeys)
    }

    /// The full-attention layer whose indexer state the dumps capture: the
    /// first one, where the decode and prefill paths still share an input.
    static let qsaSnapshotLayer = 3

    /// Forces the one-token-at-a-time prefill for hyper-connection families
    /// (`TINYTITAN_SEQUENTIAL_HC_PREFILL=1`). It is the reference the batched
    /// path is checked against, and far too slow to ship.
    static let sequentialHyperConnectionPrefill =
        ProcessInfo.processInfo.environment["TINYTITAN_SEQUENTIAL_HC_PREFILL"] == "1"

    /// Residual width for a config, before `self` is fully initialized.
    static func residualWidthFor(_ config: ArchConfig) -> Int {
        config.hyperConnections.enabled
            ? config.hiddenSize * config.hyperConnections.count
            : config.hiddenSize
    }
}
