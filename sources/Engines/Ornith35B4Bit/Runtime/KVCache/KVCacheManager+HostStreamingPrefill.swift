import Darwin
import Foundation
import Metal
import TinyTitanCore

// The write half of host-KV streaming: staging a prefill chunk, remapping its
// selection through the window, committing rows to the arena, and snapshot
// access. Split out of `KVCacheManager+HostStreaming.swift` (2026-10-21) under
// the 500-line-per-file rule.
extension KVCacheManager {

    // MARK: - Prefill staging

    /// Routes a prefill chunk's KV writes into the staging buffer instead of
    /// the window. The rows become addressable through `kRange`/`vRange` until
    /// `advance` moves them to the arena.
    public func beginPrefillWrite(
        layer: Int, slot: Int, startPosition: Int, tokenCount: Int
    ) {
        guard streamsLayer(layer) else { return }
        validateSlot(slot)
        validateRange(start: startPosition, count: tokenCount)
        let index = layer * slots + slot
        if prefillStagingCapacity[index] < tokenCount {
            let stride = strides[layer]
            let length = slots * tokenCount * stride
            guard
                let k = device.makeBuffer(
                    length: length, options: .storageModeShared),
                let v = device.makeBuffer(
                    length: length, options: .storageModeShared)
            else {
                preconditionFailure("KV prefill staging allocation failed")
            }
            k.label = "kv.window.stageK.layer\(layer)"
            v.label = "kv.window.stageV.layer\(layer)"
            prefillStagingK[index] = k
            prefillStagingV[index] = v
            prefillStagingCapacity[index] = tokenCount
        }
        prefillStagingStart[index] = startPosition
        prefillStagingCount[index] = tokenCount
        prefillStagingActive[index] = true
    }

    func stagedRange(
        layer: Int, slot: Int, start: Int, count: Int,
        buffers: [MTLBuffer?]
    ) -> (buffer: MTLBuffer, offset: Int, stride: Int)? {
        let index = layer * slots + slot
        guard prefillStagingActive[index],
            start >= prefillStagingStart[index],
            start + count <= prefillStagingStart[index] + prefillStagingCount[index],
            let buffer = buffers[index]
        else { return nil }
        let row =
            slot * prefillStagingCapacity[index]
            + (start - prefillStagingStart[index])
        return (buffer, row * strides[layer], strides[layer])
    }

    /// Remaps a prefill chunk's selection through the window and queues the
    /// staging -> window copies on `commandBuffer`.
    ///
    /// Rows from earlier chunks are read out of the host arena on the CPU.
    /// They are stable: a previous chunk's `advance` wrote them back. Rows from
    /// the current chunk are still in staging, so those copies ride the GPU.
    func preparePrefillWindow(
        layer: Int, slot: Int, startPosition: Int, tokenCount: Int,
        selection: QSASelection?, commandBuffer: MTLCommandBuffer
    ) throws -> KVPrefillWindow {
        precondition(streamsLayer(layer))
        let window = windowTokens ?? 0
        let visible = startPosition + tokenCount
        let indexStride = selection?.indexStride ?? visible
        precondition(indexStride > 0)
        guard visible <= window || selection != nil else {
            throw ModelError.internalInconsistency(
                detail:
                    "dense prefill of \(visible) keys does not fit the "
                    + "\(window)-row KV window")
        }
        try growPrefillSelectionScratch(
            entries: tokenCount * indexStride, count: tokenCount)
        guard let indicesBuf = prefillIndicesBuf,
            let rowsBuf = prefillRowsBuf, let countsBuf = prefillCountsBuf
        else {
            throw ModelError.residentBufferWrapFailed
        }
        _ = try markPrefillUnion(
            layer: layer, startPosition: startPosition,
            tokenCount: tokenCount, visible: visible,
            selection: selection, window: window)
        assignPrefillRows(
            layer: layer, slot: slot, startPosition: startPosition,
            visible: visible, window: window)
        fillPrefillIndirection(
            layer: layer, slot: slot, startPosition: startPosition,
            tokenCount: tokenCount, indexStride: indexStride,
            visible: visible, selection: selection,
            indicesBuf: indicesBuf, rowsBuf: rowsBuf, countsBuf: countsBuf)
        encodePrefillStagingBlits(
            layer: layer, slot: slot, startPosition: startPosition,
            tokenCount: tokenCount, visible: visible, window: window,
            commandBuffer: commandBuffer)
        guard let mask = windowEmptyMask else {
            throw ModelError.residentBufferWrapFailed
        }
        return KVPrefillWindow(
            indices: indicesBuf, rows: rowsBuf, counts: countsBuf,
            indexStride: indexStride, mask: mask)
    }

    /// Marks every logical position any query in the chunk selects and returns
    /// the union's size. The window is assigned once per distinct row, not once
    /// per (query, key) pair.
    private func markPrefillUnion(
        layer: Int, startPosition: Int, tokenCount: Int, visible: Int,
        selection: QSASelection?, window: Int
    ) throws -> Int {
        windowNeeded.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress, visible > 0 {
                memset(base, 0, visible)
            }
        }
        for row in 0..<tokenCount {
            let count: Int
            if let selection {
                let counts = selection.counts.contents().bindMemory(
                    to: UInt32.self, capacity: tokenCount)
                count = Int(counts[row])
            } else {
                count = startPosition + row + 1
            }
            if let selection {
                let selected = selection.indices.contents().bindMemory(
                    to: UInt32.self,
                    capacity: tokenCount * selection.indexStride)
                for i in 0..<count {
                    let logical = Int(
                        selected[row * selection.indexStride + i])
                    precondition(
                        logical < visible,
                        "prefill selection \(logical) exceeds visible \(visible)")
                    windowNeeded[logical] = 1
                }
            } else {
                for logical in 0..<count { windowNeeded[logical] = 1 }
            }
        }
        var needed = 0
        for position in 0..<visible where windowNeeded[position] != 0 {
            needed += 1
        }
        guard needed <= window else {
            throw ModelError.internalInconsistency(
                detail:
                    "prefill selection union \(needed) exceeds the "
                    + "\(window)-row KV window at layer \(layer) "
                    + "position \(startPosition)")
        }
        return needed
    }

    /// Assigns each needed row, copying history out of the arena now. The union
    /// is bounded by the window, so protecting every row assigned in this sweep
    /// always leaves an eviction victim.
    private func assignPrefillRows(
        layer: Int, slot: Int, startPosition: Int, visible: Int, window: Int
    ) {
        windowProtected.removeAll(keepingCapacity: true)
        defer { windowProtected.removeAll(keepingCapacity: true) }
        for position in 0..<visible where windowNeeded[position] != 0 {
            let alreadyResident =
                windowRow(
                    layer: layer, slot: slot, position: position) != nil
            let row = ensureWindowRow(
                layer: layer, slot: slot, position: position, forWrite: false)
            windowProtected.insert(
                layer * (slots * window) + slot * window + row)
            if !alreadyResident, position < startPosition {
                let stride = strides[layer]
                for key in [true, false] {
                    let pointers = hostRow(
                        layer: layer, slot: slot, position: position,
                        key: key, windowRow: row)
                    memcpy(pointers.destination, pointers.source, stride)
                }
                windowCopyBytes += stride * 2
                windowRowCopies += 1
            }
        }
    }

    /// Fills the kernel's `[token, indexStride]` indirection matrices.
    private func fillPrefillIndirection(
        layer: Int, slot: Int, startPosition: Int, tokenCount: Int,
        indexStride: Int, visible: Int, selection: QSASelection?,
        indicesBuf: MTLBuffer, rowsBuf: MTLBuffer, countsBuf: MTLBuffer
    ) {
        let indexPtr = indicesBuf.contents().bindMemory(
            to: UInt32.self, capacity: tokenCount * indexStride)
        let rowPtr = rowsBuf.contents().bindMemory(
            to: UInt32.self, capacity: tokenCount * indexStride)
        let countPtr = countsBuf.contents().bindMemory(
            to: UInt32.self, capacity: tokenCount)
        for row in 0..<tokenCount {
            let count: Int
            if let selection {
                let counts = selection.counts.contents().bindMemory(
                    to: UInt32.self, capacity: tokenCount)
                count = Int(counts[row])
            } else {
                count = startPosition + row + 1
            }
            countPtr[row] = UInt32(count)
            if let selection {
                let selected = selection.indices.contents().bindMemory(
                    to: UInt32.self,
                    capacity: tokenCount * selection.indexStride)
                for i in 0..<count {
                    let logical = Int(
                        selected[row * selection.indexStride + i])
                    indexPtr[row * indexStride + i] = UInt32(logical)
                    rowPtr[row * indexStride + i] = UInt32(
                        windowRow(
                            layer: layer, slot: slot, position: logical) ?? 0)
                }
            } else {
                for logical in 0..<count {
                    indexPtr[row * indexStride + logical] = UInt32(logical)
                    rowPtr[row * indexStride + logical] = UInt32(
                        windowRow(
                            layer: layer, slot: slot, position: logical) ?? 0)
                }
            }
        }
    }

    /// Current-chunk rows move from staging to the window on the GPU.
    private func encodePrefillStagingBlits(
        layer: Int, slot: Int, startPosition: Int, tokenCount: Int,
        visible: Int, window: Int, commandBuffer: MTLCommandBuffer
    ) {
        guard tokenCount > 0,
            let blit = commandBuffer.makeBlitCommandEncoder()
        else { return }
        let stride = strides[layer]
        let index = layer * slots + slot
        let stagingCapacity = prefillStagingCapacity[index]
        for position in startPosition..<visible
        where windowNeeded[position] != 0 {
            guard
                let row = windowRow(
                    layer: layer, slot: slot, position: position)
            else { continue }
            let sourceRow = slot * stagingCapacity + (position - startPosition)
            let copies: [(MTLBuffer?, MTLBuffer?)] = [
                (prefillStagingK[index], kBuffers[layer]),
                (prefillStagingV[index], vBuffers[layer]),
            ]
            for (staging, windowBuffer) in copies {
                guard let staging, let windowBuffer else { continue }
                blit.copy(
                    from: staging,
                    sourceOffset: sourceRow * stride,
                    to: windowBuffer,
                    destinationOffset: (slot * window + row) * stride,
                    size: stride)
            }
            windowDirty[layer][slot * window + row] = false
        }
        blit.endEncoding()
    }

    // MARK: - Commit and reset

    /// Moves this step's rows out of the window (decode) or staging (prefill)
    /// into the arena. Runs after the step's command buffers have completed.
    func commitStreamedRows(slot: Int, from position: Int, count: Int) {
        guard windowTokens != nil else { return }
        for layer in 0..<kinds.count where streamsLayer(layer) {
            let index = layer * slots + slot
            let stride = strides[layer]
            if prefillStagingActive[index] {
                let start = prefillStagingStart[index]
                let staged = prefillStagingCount[index]
                let available = min(count, max(0, start + staged - position))
                if available > 0,
                    let k = prefillStagingK[index],
                    let v = prefillStagingV[index],
                    let hostK = hostK[layer],
                    let hostV = hostV[layer]
                {
                    let sourceRow =
                        slot * prefillStagingCapacity[index]
                        + (position - start)
                    let destinationRow = slot * maxContext + position
                    memcpy(
                        hostK.advanced(by: destinationRow * stride),
                        k.contents().advanced(by: sourceRow * stride),
                        available * stride)
                    memcpy(
                        hostV.advanced(by: destinationRow * stride),
                        v.contents().advanced(by: sourceRow * stride),
                        available * stride)
                }
                prefillStagingActive[index] = false
                prefillStagingStart[index] = 0
                prefillStagingCount[index] = 0
            } else {
                for position in position..<(position + count) {
                    guard
                        let row = windowRow(
                            layer: layer, slot: slot, position: position)
                    else { continue }
                    flushWindowRow(layer: layer, slot: slot, row: row)
                }
            }
            windowPinned[layer][slot] = -1
        }
    }

    func resetStreamedWindow(slot: Int) {
        guard windowTokens != nil else { return }
        let window = windowTokens ?? 0
        for layer in 0..<kinds.count where streamsLayer(layer) {
            let mapBase = slot * maxContext
            let rowBase = slot * window
            // Clear only the rows this slot actually owns: a 1M-context
            // reset walks the window, not the logical map.
            for row in 0..<window {
                let occupant = windowOccupant[layer][rowBase + row]
                if occupant != UInt32.max {
                    windowRowMap[layer][mapBase + Int(occupant)] = UInt32.max
                }
                windowOccupant[layer][rowBase + row] = UInt32.max
                windowDirty[layer][rowBase + row] = false
            }
            windowCursor[layer][slot] = 0
            windowPinned[layer][slot] = -1
            let index = layer * slots + slot
            prefillStagingActive[index] = false
            prefillStagingStart[index] = 0
            prefillStagingCount[index] = 0
        }
    }

    func resetStreamedArenas() {
        guard windowTokens != nil else { return }
        let page = Int(getpagesize())
        for layer in 0..<kinds.count where streamsLayer(layer) {
            let length = hostArenaLengths[layer]
            guard length > 0 else { continue }
            let rounded = ((length + page - 1) / page) * page
            if let k = hostK[layer] {
                _ = posix_madvise(k, rounded, POSIX_MADV_DONTNEED)
            }
            if let v = hostV[layer] {
                _ = posix_madvise(v, rounded, POSIX_MADV_DONTNEED)
            }
            windowRowMap[layer] = [UInt32](
                repeating: UInt32.max, count: maxContext * slots)
            windowOccupant[layer] = [UInt32](
                repeating: UInt32.max, count: (windowTokens ?? 0) * slots)
            windowDirty[layer] = [Bool](
                repeating: false, count: (windowTokens ?? 0) * slots)
            windowCursor[layer] = [Int](repeating: 0, count: slots)
            windowPinned[layer] = [Int](repeating: -1, count: slots)
            for slot in 0..<slots {
                prefillStagingActive[layer * slots + slot] = false
                prefillStagingStart[layer * slots + slot] = 0
                prefillStagingCount[layer * slots + slot] = 0
            }
        }
        windowCopyBytes = 0
        windowEvictions = 0
        windowRowCopies = 0
    }

    /// One row of the authoritative arena, for the acceptance tests and
    /// diagnostics. Reading through `kSlot` would gather the row back into the
    /// window and is not a way to inspect what was committed.
    func hostRowBytes(
        layer: Int, slot: Int, position: Int, key: Bool
    ) -> [UInt8] {
        let stride = strides[layer]
        guard let base = key ? hostK[layer] : hostV[layer] else { return [] }
        let pointer = base.advanced(
            by: (slot * maxContext + position) * stride
        ).assumingMemoryBound(to: UInt8.self)
        return (0..<stride).map { pointer[$0] }
    }

    /// Snapshot bytes for one streamed layer, read from the arena rather than
    /// from the window (which may not hold the snapshot's prefix).
    func appendStreamedSnapshot(
        layer: Int, length: Int, to payload: inout Data
    ) {
        precondition(length <= hostArenaLengths[layer])
        if let k = hostK[layer] {
            payload.append(
                k.assumingMemoryBound(to: UInt8.self), count: length)
        }
        if let v = hostV[layer] {
            payload.append(
                v.assumingMemoryBound(to: UInt8.self), count: length)
        }
    }

    func restoreStreamedSnapshot(
        layer: Int, source: UnsafeRawPointer, length: Int
    ) throws {
        // Returning silently here would leave the arena stale while the
        // caller reports a successful restore — a fail-open backstop, so it
        // throws instead. The caller's segment-length check makes this
        // unreachable today; the throw is what keeps it that way.
        guard let k = hostK[layer], let v = hostV[layer],
            length <= hostArenaLengths[layer]
        else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        memcpy(k, source, length)
        memcpy(v, source.advanced(by: length), length)
        // The window's map no longer describes the arena: every row must be
        // treated as absent until a gather copies it back in.
        windowRowMap[layer] = [UInt32](
            repeating: UInt32.max, count: maxContext * slots)
        windowOccupant[layer] = [UInt32](
            repeating: UInt32.max, count: (windowTokens ?? 0) * slots)
        windowDirty[layer] = [Bool](
            repeating: false, count: (windowTokens ?? 0) * slots)
        windowCursor[layer] = [Int](repeating: 0, count: slots)
        windowPinned[layer] = [Int](repeating: -1, count: slots)
    }
}
