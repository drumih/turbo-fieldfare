import Foundation
import Metal

/// MTP: the speculative checkpoint, the width-2 verify (pair schedule), and the one-layer draft adapter.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes.
extension RealForwardRunner {
    func captureSpeculativeCheckpoint(maximumBytes: Int) throws
        -> SpeculativeInferenceCheckpoint
    {
        guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
        let required = gdnState?.speculativePayloadBytes ?? 0
        guard required <= maximumBytes else {
            throw InferenceStateSnapshotError.exceedsLimit(
                bytes: required,
                limit: maximumBytes)
        }
        return SpeculativeInferenceCheckpoint(position: kv.position)
    }

    func rollbackSpeculativeCheckpoint(_ checkpoint: SpeculativeInferenceCheckpoint) throws {
        guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
        if let gdnState {
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            try gdnState.encodeSpeculativeRestore(commandBuffer: cb)
            cb.commit()
            try waitForCompletion(cb)
        }
        // Row zero was confirmed and is present in the on-GPU checkpoint.
        try kv.rewind(to: checkpoint.position + 1)
        // The verify pass ran two rows; one was accepted. The KV row and the
        // indexer's pooled blocks are recomputed by whatever replaces the
        // rejected row, but the n-gram convolution window is a rolling one and
        // has to be walked back explicitly.
        rewindPLE(acceptedRows: 1, passRows: 2)
        resetTransientState()
    }

    /// Discard an unaccepted native-MTP cache row. The draft contains only
    /// trimmable full-attention KV, so its logical cursor can move back without
    /// copying payload bytes; the next draft pass overwrites the stale row.
    func rewindMTP(to position: Int) throws {
        guard
            cfg.family == .qwen36MTP || cfg.family == .qwen38flashMTP
                || cfg.family == .qwen4flashMTP,
            let kv
        else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        try kv.rewind(to: position)
        resetTransientState()
    }

    var speculativeRollbackBytes: Int {
        gdnState?.speculativePayloadBytes ?? 0
    }

    /// Advance the one-layer MTP sidecar with aligned `(target hidden,
    /// next-token)` pairs. At most 32 rows are admitted so adapter scratch is
    /// fixed and the routed expert cache remains exactly top-k sized.
    func advanceMTP(
        tokens: ArraySlice<Int32>,
        targetHiddenRows: Data,
        startPosition: Int,
        predictNext: Bool
    ) async throws -> Int32? {
        guard cfg.family == .qwen36MTP else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        guard !tokens.isEmpty, tokens.count <= Self.mtpChunkCapacity else {
            throw PrefillError.chunkedUnsupported(
                "MTP adapter accepts 1...\(Self.mtpChunkCapacity) aligned rows")
        }
        let D = cfg.hiddenSize
        let expectedBytes = tokens.count * D * MemoryLayout<Float16>.stride
        guard targetHiddenRows.count == expectedBytes else {
            throw PrefillError.chunkedUnsupported(
                "MTP target hidden payload has \(targetHiddenRows.count) bytes; expected \(expectedBytes)"
            )
        }
        guard let tokenBuffer = mtpTokenBlock,
            let embeddingBlock = mtpEmbeddingBlock,
            let normalizedEmbedding = mtpNormalizedEmbeddingBlock,
            let normalizedHidden = mtpNormalizedHiddenBlock,
            let concat = mtpConcatBlock,
            let projected = mtpProjectedBlock,
            let targetHidden = mtpTargetHiddenBlock,
            let elementwise
        else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        targetHiddenRows.copyBytes(
            to: targetHidden.contents()
                .assumingMemoryBound(to: UInt8.self), count: expectedBytes)
        let ids = tokens.map { UInt32(bitPattern: $0) }
        ids.withUnsafeBytes { bytes in
            // An empty id list has no base address and nothing to copy.
            guard let base = bytes.baseAddress else { return }
            tokenBuffer.contents().copyMemory(from: base, byteCount: bytes.count)
        }
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try encodeEmbeddingBlock(
            commandBuffer: cb, tokens: tokenBuffer,
            out: embeddingBlock, t: tokens.count, outScale: 1)
        let embeddingNorm = try model.mtpEmbeddingNorm()
        let hiddenNorm = try model.mtpHiddenNorm()
        try prefillRMS.encodeBF16W(
            commandBuffer: cb,
            x: embeddingBlock,
            weight: embeddingNorm.buffer,
            weightOffset: Int(embeddingNorm.offset),
            out: normalizedEmbedding,
            t: UInt32(tokens.count),
            d: UInt32(D), eps: 1e-6)
        try prefillRMS.encodeBF16W(
            commandBuffer: cb,
            x: targetHidden,
            weight: hiddenNorm.buffer,
            weightOffset: Int(hiddenNorm.offset),
            out: normalizedHidden,
            t: UInt32(tokens.count),
            d: UInt32(D), eps: 1e-6)
        try elementwise.encodeConcatRows(
            commandBuffer: cb,
            lhs: normalizedEmbedding,
            rhs: normalizedHidden,
            out: concat,
            rows: tokens.count,
            dim: D)
        let projection = try model.mtpProjection()
        try prefillQMM.encode(
            commandBuffer: cb,
            weights: projection.buffer,
            weightsOffset: Int(projection.offset),
            scales: projection.buffer,
            scalesOffset: Int(projection.scaleOffset),
            biases: projection.buffer,
            biasesOffset: Int(projection.biasOffset),
            x: concat,
            y: projected,
            t: tokens.count,
            n: D,
            k: 2 * D)
        cb.commit()
        try waitForCompletion(cb)

        let runtime = PrefillRuntimeConfig.production(chunkTokens: 32)
        let scratch = try ensurePrefillScratch(config: runtime)
        let mode: PrefillOutputMode = useFusedGreedyHead ? .greedyIfAvailable : .logits
        try await executePrefillChunk(
            tokens: tokens,
            startPosition: startPosition,
            outputMode: mode,
            logits: verificationLogits,
            scratch: scratch,
            config: runtime,
            writeFinalHead: predictNext,
            preparedHidden: projected)
        guard predictNext else { return nil }
        if useFusedGreedyHead {
            return Int32(bitPattern: lastGreedyToken)
        }
        let values = verificationLogits.contents()
            .assumingMemoryBound(to: Float16.self)
        var best = 0
        var bestValue = Float(values[0])
        for index in 1..<cfg.vocabSize {
            let value = Float(values[index])
            if value > bestValue {
                best = index
                bestValue = value
            }
        }
        return Int32(best)
    }

    func ensureMTPPrefillReadback(rows: Int) throws -> MTLBuffer {
        let bytes = rows * residualWidth * MemoryLayout<Float16>.stride
        if let existing = mtpPrefillReadback, existing.length >= bytes {
            return existing
        }
        guard
            let buffer = ctx.device.makeBuffer(
                length: bytes,
                options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        buffer.label = "mtp.target-hidden-readback"
        mtpPrefillReadback = buffer
        return buffer
    }

    /// Target prefill with a bounded hidden-state tap that simultaneously
    /// aligns the streaming MTP sidecar. Only one target chunk is exposed at a
    /// time; no prompt-sized hidden-state tensor is retained.
    func prefillChunkedWithMTP(
        tokens: ArraySlice<Int32>,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        mtp: RealForwardRunner,
        onProgress: (Int) -> Void
    ) async throws -> MTPPrefillResult {
        defer { resetExpertUseCountsAfterPrefill() }
        guard cfg.family == .qwen36 || cfg.family == .qwen38flash || cfg.family == .qwen4flash
        else {
            throw StreamingMTPError.targetMustBeQwen36
        }
        guard
            mtp.cfg.family == .qwen36MTP
                || mtp.cfg.family == .qwen38flashMTP
                || mtp.cfg.family == .qwen4flashMTP
        else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        guard !tokens.isEmpty, tokens.count <= mtp.maxContext else {
            throw PrefillError.chunkedUnsupported(
                "MTP prompt must fit its bounded \(mtp.maxContext)-token draft context")
        }
        reset()
        mtp.reset()
        // The target may stream its full-attention KV through a window; the
        // chunk must be small enough that the union of its selections fits.
        let effectiveConfig = effectivePrefillConfig(config)
        let scratch = try ensurePrefillScratch(config: effectiveConfig)
        let spans = PrefillChunkPlanner.spans(
            tokenCount: tokens.count,
            startPosition: 0,
            config: effectiveConfig)
        var carry: Data?
        do {
            for (spanIndex, span) in spans.enumerated() {
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                let chunk = tokens[lower..<upper]
                try await executePrefillChunk(
                    tokens: chunk,
                    startPosition: span.startPosition,
                    outputMode: useFusedGreedyHead
                        ? .greedyIfAvailable : .logits,
                    logits: logits,
                    scratch: scratch,
                    config: effectiveConfig,
                    writeFinalHead: spanIndex == spans.count - 1)

                let readback = try ensureMTPPrefillReadback(rows: span.tokenCount)
                guard let cb = ctx.queue.makeCommandBuffer(),
                    let blit = cb.makeBlitCommandEncoder()
                else {
                    throw ModelError.residentBufferWrapFailed
                }
                // The draft is handed the residual as the target carries it,
                // which for a hyper-connection family is the wide form: its
                // fusion reads all four streams, not a collapsed one.
                let rowBytes = residualWidth * MemoryLayout<Float16>.stride
                blit.copy(
                    from: scratch.hidden, sourceOffset: 0,
                    to: readback, destinationOffset: 0,
                    size: span.tokenCount * rowBytes)
                blit.endEncoding()
                cb.commit()
                try waitForCompletion(cb)
                let chunkHidden = Data(
                    bytes: readback.contents(),
                    count: span.tokenCount * rowBytes)

                var pairTokens: [Int32] = []
                var pairHidden = Data()
                if let carry {
                    pairTokens.reserveCapacity(span.tokenCount)
                    pairTokens.append(contentsOf: chunk)
                    pairHidden.reserveCapacity(span.tokenCount * rowBytes)
                    pairHidden.append(carry)
                    if span.tokenCount > 1 {
                        pairHidden.append(chunkHidden.prefix((span.tokenCount - 1) * rowBytes))
                    }
                } else if span.tokenCount > 1 {
                    pairTokens.append(contentsOf: chunk.dropFirst())
                    pairHidden.append(chunkHidden.prefix((span.tokenCount - 1) * rowBytes))
                }
                var pairOffset = 0
                while pairOffset < pairTokens.count {
                    let count = min(Self.mtpChunkCapacity, pairTokens.count - pairOffset)
                    let hiddenStart = pairOffset * rowBytes
                    let hiddenEnd = hiddenStart + count * rowBytes
                    _ = try await mtp.advanceMTPForFamily(
                        tokens: pairTokens[pairOffset..<(pairOffset + count)],
                        targetHiddenRows: pairHidden.subdata(in: hiddenStart..<hiddenEnd),
                        startPosition: mtp.continuationPosition,
                        predictNext: false)
                    pairOffset += count
                }
                carry = Data(chunkHidden.suffix(rowBytes))
                onProgress(span.completedCount)
            }
        } catch {
            // A failed chunk (cancellation, GPU error, expert-fetch I/O) may
            // have left partial KV rows in both runners; clear both so the
            // next request starts clean.
            reset()
            mtp.reset()
            throw error
        }
        guard let lastTargetHidden = carry else {
            throw StreamingMTPError.draftNotReady
        }
        let seed: PrefillSeed =
            useFusedGreedyHead
            ? .greedyToken(lastGreedyToken) : .logitsWritten
        return MTPPrefillResult(
            target: PrefillResult(newPosition: tokens.count, seed: seed),
            lastTargetHidden: lastTargetHidden)
    }

}
