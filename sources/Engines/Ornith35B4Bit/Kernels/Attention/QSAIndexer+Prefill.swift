import Foundation
import Metal
import TinyTitanCore

extension QSAIndexer {
    /// The first block of a chunk may already hold members from the previous
    /// one, and the last may be ragged; repooling from
    /// `startPosition / compressRatio` through the end covers both without a
    /// special case for either.
    /// Repools every block a prefill chunk touched. The caller has already
    /// projected the chunk's raw keys into the cache at
    /// `rawKeyDestination(layer:startPosition:tokens:)`.
    func encodePoolPrefill(
        commandBuffer: MTLCommandBuffer,
        weights: Weights, layer: Int,
        startPosition: Int, tokens: Int,
        eps: Float
    ) throws {
        precondition(
            startPosition + tokens <= capacity,
            "QSA indexer capacity \(capacity) exceeded at "
                + "\(startPosition + tokens)")
        let buffers = try layerBuffers(layer, hold: startPosition + tokens)
        let f16 = MemoryLayout<Float16>.stride
        let endPosition = startPosition + tokens
        let firstBlock = startPosition / compressRatio
        let lastBlock = (endPosition - 1) / compressRatio
        let blockCount = lastBlock - firstBlock + 1
        try encodePoolRange(
            commandBuffer: commandBuffer, buffers: buffers,
            firstBlock: firstBlock, blockCount: blockCount,
            visibleKeys: endPosition)
        try rms.encodeBF16WPerHead(
            commandBuffer: commandBuffer,
            x: buffers.pooled,
            xOffset: firstBlock * headDim * f16,
            weight: weights.keyNorm.buffer,
            weightOffset: Int(weights.keyNorm.offset),
            out: buffers.pooled,
            outOffset: firstBlock * headDim * f16,
            headDim: UInt32(headDim),
            numHeads: blockCount, eps: eps)
        try rope.encodeNeoxSubdimStrided(
            commandBuffer: commandBuffer,
            data: buffers.pooled,
            dataOffset: firstBlock * headDim * f16,
            position: UInt32(firstBlock * compressRatio),
            headDim: UInt32(headDim), numHeads: 1,
            rotaryDim: UInt32(headDim), numTokens: UInt32(blockCount),
            stride: UInt32(compressRatio), theta: ropeTheta)
    }

    /// The raw-key cache slice a prefill chunk's projection should write into.
    func rawKeyDestination(layer: Int, startPosition: Int, tokens: Int) throws
        -> (buffer: MTLBuffer, offset: Int)
    {
        let buffers = try layerBuffers(layer, hold: startPosition + tokens)
        return (
            buffers.raw,
            startPosition * headDim * MemoryLayout<Float16>.stride
        )
    }

    private func encodePoolRange(
        commandBuffer: MTLCommandBuffer,
        buffers: (raw: MTLBuffer, pooled: MTLBuffer),
        firstBlock: Int, blockCount: Int,
        visibleKeys: Int
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(poolRangePSO)
        enc.setBuffer(buffers.raw, offset: 0, index: 0)
        enc.setBuffer(buffers.pooled, offset: 0, index: 1)
        var d = UInt32(headDim)
        var fb = UInt32(firstBlock)
        var bc = UInt32(blockCount)
        var r = UInt32(compressRatio)
        var n = UInt32(visibleKeys)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&fb, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&bc, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)
        let w = min(poolRangePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: headDim * blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Scores every block against every query in a prefill chunk.
    ///
    /// `blockInput` is the chunk's `[tokens, hidden]` block input; the query
    /// projection, its norm and its rope all run over the whole chunk, with
    /// the rope advancing one position per row.
    func encodeScoresPrefill(
        commandBuffer: MTLCommandBuffer,
        blockInput: MTLBuffer,
        weights: Weights, layer: Int,
        startPosition: Int, tokens: Int,
        eps: Float,
        project: (
            MTLCommandBuffer, TensorView, MTLBuffer,
            MTLBuffer, Int, Int, Int
        ) throws -> Void
    ) throws {
        let buffers = try layerBuffers(layer, hold: startPosition + tokens)
        let query = weights.queryProjection
        try growQueryScratch(rows: tokens)
        guard let queryRows = queryRowsBuf else {
            throw MetalError.bufferAllocationFailed("QSA query scratch")
        }
        try project(
            commandBuffer, query, blockInput, queryRows,
            heads * headDim, hiddenColumns(query), tokens)
        try rms.encodeBF16WPerHead(
            commandBuffer: commandBuffer,
            x: queryRows,
            weight: weights.queryNorm.buffer,
            weightOffset: Int(weights.queryNorm.offset),
            out: queryRows,
            headDim: UInt32(headDim),
            numHeads: heads * tokens, eps: eps)
        try rope.encodeNeoxSubdimStrided(
            commandBuffer: commandBuffer, data: queryRows,
            position: UInt32(startPosition),
            headDim: UInt32(headDim), numHeads: UInt32(heads),
            rotaryDim: UInt32(headDim), numTokens: UInt32(tokens),
            stride: 1, theta: ropeTheta)

        let blocks = (startPosition + tokens - 1) / compressRatio + 1
        try growScoreScratch(count: blocks * tokens)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(scoreRowsPSO)
        enc.setBuffer(queryRows, offset: 0, index: 0)
        enc.setBuffer(buffers.pooled, offset: 0, index: 1)
        enc.setBuffer(scoresBuf, offset: 0, index: 2)
        var d = UInt32(headDim)
        var h = UInt32(heads)
        var b = UInt32(blocks)
        var t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&h, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 6)
        let w = min(scoreRowsPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: blocks * tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        lastScoredBlocks = blocks
    }

    /// The chunk's selection, or nil when every query in the chunk is inside
    /// the dense-exact window. Carries both forms: the mask that existing
    /// callers and the activation dump read, and the compacted index list the
    /// attention kernel loops over.
    ///
    /// Every query gets the same treatment the decode path gives one: its own
    /// ragged tail forced in, then complete blocks by score until the cell
    /// budget runs out. A query inside the window keeps everything it can see.
    func selectKeysPrefill(
        startPosition: Int, tokens: Int,
        layer: Int = -1
    ) -> QSASelection? {
        let traceStart =
            ProcessInfo.processInfo.environment["TINYTITAN_QSA_SELECT_TRACE"] == "1"
            ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0
        defer {
            if traceStart != 0 {
                let hostMillis = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- traceStart) / 1e6
                FileHandle.standardError.write(
                    Data(
                        String(
                            format:
                                "[qsa-select] layer=%d start=%d tokens=%d host_ms=%.3f\n",
                            layer, startPosition, tokens, hostMillis
                        ).utf8))
            }
        }
        let lastVisible = startPosition + tokens
        guard lastVisible > selectionWidth else { return nil }
        let stride = lastVisible
        guard (try? growKeepScratch(count: stride * tokens)) != nil else { return nil }
        // One index slot per key the budget can admit. The dense-window rows
        // keep `visible` keys, which is at most selectionWidth by the guard
        // above, so this width covers both branches.
        let indexWidth = selectionWidth
        guard (try? growCompactScratch(rows: tokens, width: indexWidth)) != nil,
            let indexBuf = keepIndexBuf, let countBuf = keepCountBuf
        else { return nil }
        let keep = keepBuf.contents().bindMemory(
            to: UInt8.self,
            capacity: stride * tokens)
        let indices = indexBuf.contents().bindMemory(
            to: UInt32.self,
            capacity: tokens * indexWidth)
        let counts = countBuf.contents().bindMemory(to: UInt32.self, capacity: tokens)
        let scores = scoresBuf.contents().bindMemory(
            to: Float.self, capacity: lastScoredBlocks * tokens)
        for row in 0..<tokens {
            let visible = startPosition + row + 1
            let base = row * stride
            memset(keep + base, 0, stride)
            if visible <= selectionWidth {
                memset(keep + base, 1, visible)
                let out = indices + row * indexWidth
                for key in 0..<visible { out[key] = UInt32(key) }
                counts[row] = UInt32(visible)
                continue
            }
            let completeCells = (visible / compressRatio) * compressRatio
            for cell in completeCells..<visible { keep[base + cell] = 1 }
            var remaining = selectionWidth - (visible - completeCells)
            guard remaining > 0 else { continue }
            let blocks = completeCells / compressRatio
            let rowScores = scores + row * lastScoredBlocks
            let ranked = (0..<blocks).sorted {
                rowScores[$0] == rowScores[$1]
                    ? $0 < $1 : rowScores[$0] > rowScores[$1]
            }
            for block in ranked {
                if remaining <= 0 { break }
                let take = min(compressRatio, remaining)
                let cellBase = base + block * compressRatio
                for offset in 0..<take { keep[cellBase + offset] = 1 }
                remaining -= take
            }
            // Compact ascending. Scanning the row once here is O(visible) on
            // the host and replaces the same scan done once per query *per
            // head* on the GPU; ascending order also keeps the K/V reads
            // sequential, which a score-ordered list would not.
            var written = 0
            let out = indices + row * indexWidth
            for key in 0..<visible where keep[base + key] != 0 {
                if written == indexWidth { break }
                out[written] = UInt32(key)
                written += 1
            }
            counts[row] = UInt32(written)
        }
        return QSASelection(
            mask: keepBuf, maskStride: stride,
            indices: indexBuf, indexStride: indexWidth,
            counts: countBuf)
    }

}
