import Foundation
import Metal
import TinyTitanCore

// Scratch management for the QSA indexer: the per-layer key buffers and the
// grow-on-demand scratch the encode paths call.
//
// Split out of `QSAIndexer.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion. The moved helpers are
// called by the public encode paths that stay behind, so those six widened from
// `private` to internal, and so did the twelve stored properties they read.
extension QSAIndexer {

    /// The layer's key buffers, grown to hold at least `required` cells.
    ///
    /// Grows on demand like the KV cache, starting at
    /// `QSAIndexer.initialCapacityCells` and doubling to the configured
    /// capacity: at the production geometry the full-context reservation is
    /// 84 MB a layer and a short conversation never touches most of it (§5o
    /// measured that reservation deciding whether 96 expert-cache slots fit
    /// on a 24 GB machine). The copy is host-side over shared buffers, and it
    /// is safe for the same reason `KVCacheManager.reserve` is: callers reach
    /// the buffers only through this accessor, growth happens before the
    /// token or chunk that needs the room is encoded, and Metal keeps the old
    /// buffer alive for any command buffer already holding it.
    func layerBuffers(_ layer: Int, hold required: Int) throws -> (
        raw: MTLBuffer, pooled: MTLBuffer
    ) {
        precondition(
            required <= capacity,
            "QSA indexer capacity \(capacity) exceeded: \(required) cells requested")
        let f16 = MemoryLayout<Float16>.stride
        let cellBytes = headDim * f16
        if let raw = rawKeys[layer], let pool = pooled[layer],
            raw.length / cellBytes >= required
        {
            return (raw, pool)
        }
        let current = rawKeys[layer].map { $0.length / cellBytes } ?? 0
        var target = max(current, min(capacity, QSAIndexer.initialCapacityCells))
        while target < required { target *= 2 }
        target = min(target, capacity)
        guard
            let raw = ctx.device.makeBuffer(
                length: target * cellBytes, options: .storageModeShared),
            let pool = ctx.device.makeBuffer(
                length: (target / compressRatio + 1) * cellBytes,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("QSA indexer layer \(layer)")
        }
        raw.label = "qsa.rawKeys.L\(layer)"
        pool.label = "qsa.pooled.L\(layer)"
        if let oldRaw = rawKeys[layer], let oldPool = pooled[layer] {
            memcpy(raw.contents(), oldRaw.contents(), oldRaw.length)
            memcpy(pool.contents(), oldPool.contents(), oldPool.length)
        }
        rawKeys[layer] = raw
        pooled[layer] = pool
        return (raw, pool)
    }

    /// Appends this token's raw indexer key and repools the block it lands
    /// in. `position` is the token's absolute position.
    func encodeAppendKey(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        weights: Weights, layer: Int, position: Int,
        eps: Float
    ) throws {
        precondition(
            position < capacity,
            "QSA indexer capacity \(capacity) exceeded at \(position)")
        let buffers = try layerBuffers(layer, hold: position + 1)
        let key = weights.keyProjection
        try gemv.encode(
            commandBuffer: commandBuffer,
            weights: key.buffer, weightsOffset: Int(key.offset),
            scales: key.buffer, scalesOffset: Int(key.scaleOffset),
            biases: key.buffer, biasesOffset: Int(key.biasOffset),
            x: hidden, xOffset: hiddenOffset,
            y: buffers.raw,
            yOffset: position * headDim * MemoryLayout<Float16>.stride,
            m: UInt32(headDim), n: UInt32(hiddenColumns(key)),
            isBF16: key.dtype == 1)

        // Repool the block this token joined. Members present is what the
        // mean divides by, so a tail block is not diluted by absent members.
        let block = position / compressRatio
        let first = block * compressRatio
        let count = position - first + 1
        try encodePool(
            commandBuffer: commandBuffer, buffers: buffers,
            block: block, first: first, count: count)
        try rms.encodeBF16W(
            commandBuffer: commandBuffer,
            x: buffers.pooled,
            xOffset: block * headDim * MemoryLayout<Float16>.stride,
            weight: weights.keyNorm.buffer,
            weightOffset: Int(weights.keyNorm.offset),
            out: buffers.pooled,
            outOffset: block * headDim * MemoryLayout<Float16>.stride,
            d: UInt32(headDim), eps: eps)
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer,
            data: buffers.pooled,
            dataOffset: block * headDim * MemoryLayout<Float16>.stride,
            position: UInt32(first),
            headDim: UInt32(headDim), numHeads: UInt32(1),
            rotaryDim: UInt32(headDim), theta: ropeTheta)
    }

    /// Columns of a `[rows, columns]` projection, from the recorded shape.
    func hiddenColumns(_ view: TensorView) -> Int {
        Int(view.shape.1)
    }
    func growQueryScratch(rows: Int) throws {
        let needed = rows * heads * headDim * MemoryLayout<Float16>.stride
        if let existing = queryRowsBuf, existing.length >= needed { return }
        guard
            let made = ctx.device.makeBuffer(
                length: needed,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("QSA indexer chunk queries")
        }
        made.label = "qsa.chunkQueries"
        queryRowsBuf = made
    }

    func growScoreScratch(count: Int) throws {
        let needed = count * MemoryLayout<Float>.stride
        if scoresBuf.length >= needed { return }
        guard
            let made = ctx.device.makeBuffer(
                length: needed,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("QSA indexer chunk scores")
        }
        made.label = "qsa.scores"
        scoresBuf = made
    }

    func growCompactScratch(rows: Int, width: Int) throws {
        let idxBytes = rows * width * MemoryLayout<UInt32>.stride
        let cntBytes = rows * MemoryLayout<UInt32>.stride
        if (keepIndexBuf?.length ?? 0) < idxBytes {
            guard
                let made = ctx.device.makeBuffer(
                    length: idxBytes,
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed("QSA selection indices")
            }
            made.label = "qsa.keep.indices"
            keepIndexBuf = made
        }
        if (keepCountBuf?.length ?? 0) < cntBytes {
            guard
                let made = ctx.device.makeBuffer(
                    length: cntBytes,
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed("QSA selection counts")
            }
            made.label = "qsa.keep.counts"
            keepCountBuf = made
        }
    }

    func growKeepScratch(count: Int) throws {
        if keepBuf.length >= count { return }
        guard
            let made = ctx.device.makeBuffer(
                length: count,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("QSA indexer chunk selection")
        }
        made.label = "qsa.keep"
        keepBuf = made
    }

    /// `selectKeys` plus the compacted list the host-KV window needs.
    ///
    /// Inside the dense-exact window every visible key is kept, so there is no
    /// mask and the caller can pass a nil index list (the window path expands
    /// `0..<visible`). Past it the list is ascending, which keeps the gathered
    /// window rows sequential.
    func selectKeysCompact(visibleKeys: Int) -> (
        mask: MTLBuffer?, indices: MTLBuffer?, count: Int
    ) {
        guard visibleKeys > selectionWidth else {
            return (nil, nil, visibleKeys)
        }
        let mask = selectKeys(visibleKeys: visibleKeys)
        guard (try? growCompactScratch(rows: 1, width: selectionWidth)) != nil,
            let indexBuf = keepIndexBuf, let countBuf = keepCountBuf
        else {
            return (mask, nil, 0)
        }
        let keep = keepBuf.contents().bindMemory(
            to: UInt8.self, capacity: capacity)
        let out = indexBuf.contents().bindMemory(
            to: UInt32.self, capacity: selectionWidth)
        let counts = countBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
        var written = 0
        for key in 0..<visibleKeys where keep[key] != 0 {
            if written == selectionWidth { break }
            out[written] = UInt32(key)
            written += 1
        }
        counts[0] = UInt32(written)
        return (mask, indexBuf, written)
    }

    /// Turns the scored blocks into the per-key selection the attention reads.
    ///
    /// The scores have to be on the host for this: the ordering is
    /// (score descending, index ascending) over cells, every cell in a block
    /// shares its block's score, and the budget can cut a block in half. A
    /// GPU top-k that reproduces that tie-break exactly is the obvious next
    /// step; this is the version that is clearly correct.
    ///
    /// Returns nil when everything visible is kept, so the caller can skip
    /// the mask entirely rather than binding an all-ones buffer.
    func selectKeys(visibleKeys: Int) -> MTLBuffer? {
        guard visibleKeys > selectionWidth else { return nil }
        let keep = keepBuf.contents().bindMemory(to: UInt8.self, capacity: capacity)
        memset(keep, 0, visibleKeys)

        // The ragged tail of the query's own block is always kept: the
        // reference biases it above every score rather than ranking it.
        let completeCells = (visibleKeys / compressRatio) * compressRatio
        for cell in completeCells..<visibleKeys { keep[cell] = 1 }
        var remaining = selectionWidth - (visibleKeys - completeCells)
        guard remaining > 0 else { return keepBuf }

        let blocks = completeCells / compressRatio
        let scores = scoresBuf.contents().bindMemory(
            to: Float.self,
            capacity: blocks + 1)
        // Descending score, ties to the lower block index -- which is what
        // the reference's stable cell ordering reduces to, because every cell
        // in a block carries the same score.
        let ranked = (0..<blocks).sorted {
            scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
        }
        for block in ranked {
            if remaining <= 0 { break }
            let take = min(compressRatio, remaining)
            let base = block * compressRatio
            for offset in 0..<take { keep[base + offset] = 1 }
            remaining -= take
        }
        return keepBuf
    }

    var canSelectOnGPU: Bool { selectPSO != nil }

    /// `selectKeys` on the GPU, riding `commandBuffer` after `encodeScores`:
    /// same mask, no readback. Returns nil inside the dense window, exactly
    /// where `selectKeys` does.
    func encodeSelectKeys(commandBuffer: MTLCommandBuffer, visibleKeys: Int) throws -> MTLBuffer? {
        guard visibleKeys > selectionWidth else { return nil }
        guard let pso = selectPSO else { throw MetalError.commandEncoderFailed }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(scoresBuf, offset: 0, index: 0)
        enc.setBuffer(keepBuf, offset: 0, index: 1)
        var v = UInt32(visibleKeys)
        var r = UInt32(compressRatio)
        var w = UInt32(selectionWidth)
        enc.setBytes(&v, length: 4, index: 2)
        enc.setBytes(&r, length: 4, index: 3)
        enc.setBytes(&w, length: 4, index: 4)
        let threads = min(pso.maxTotalThreadsPerThreadgroup, 1024)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        return keepBuf
    }

    /// The keep mask's first `count` bytes, for the verify mode.
    func keepMaskBytes(count: Int) -> [UInt8] {
        let p = keepBuf.contents().bindMemory(to: UInt8.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }
}
