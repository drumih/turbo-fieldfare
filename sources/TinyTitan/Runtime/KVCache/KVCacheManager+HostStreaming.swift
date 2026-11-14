import Darwin
import Foundation
import Metal
import TinyTitanCore

/// The indirection a windowed attention call reads.
///
/// `indices` is the logical position of each selected key, `rows` the window
/// row that currently holds it. Both are ascending in the selection's order,
/// which is what keeps the K/V reads sequential.
struct KVWindowSelection {
    let indices: MTLBuffer
    let rows: MTLBuffer
    let count: Int
}

/// A prefill chunk's selection, remapped through the window.
///
/// `indices`/`rows` keep the `[token, indexWidth]` layout the tiled prefill
/// kernel already reads; `counts[token]` says how many entries that row uses.
/// The kernel's `useKeep == 3` path reads the logical key from `indices` (for
/// the visibility check) and the physical row from `rows`.
struct KVPrefillWindow {
    let indices: MTLBuffer
    let rows: MTLBuffer
    let counts: MTLBuffer
    let indexStride: Int
    let mask: MTLBuffer
}

extension KVCacheManager {

    // MARK: - Layout

    func initializeHostStreaming(precision: KVCachePrecision) {
        let count = config.numLayers
        windowRowMap = Array(repeating: [], count: count)
        windowOccupant = Array(repeating: [], count: count)
        windowCursor = Array(repeating: [], count: count)
        windowDirty = Array(repeating: [], count: count)
        windowPinned = Array(repeating: [], count: count)
        prefillStagingK = Array(repeating: nil, count: count * slots)
        prefillStagingV = Array(repeating: nil, count: count * slots)
        prefillStagingCapacity = Array(repeating: 0, count: count * slots)
        prefillStagingStart = Array(repeating: 0, count: count * slots)
        prefillStagingCount = Array(repeating: 0, count: count * slots)
        prefillStagingActive = Array(repeating: false, count: count * slots)
        guard let window = windowTokens else { return }
        windowNeeded = [UInt8](repeating: 0, count: maxContext)
        for layer in 0..<count where streamsLayer(layer) {
            let rows = window * slots
            windowRowMap[layer] = [UInt32](
                repeating: UInt32.max, count: maxContext * slots)
            windowOccupant[layer] = [UInt32](
                repeating: UInt32.max, count: rows)
            windowCursor[layer] = [Int](repeating: 0, count: slots)
            windowDirty[layer] = [Bool](repeating: false, count: rows)
            windowPinned[layer] = [Int](repeating: -1, count: slots)
        }
        guard
            let mask = device.makeBuffer(
                length: MemoryLayout<UInt32>.size,
                options: .storageModeShared)
        else { return }
        mask.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        mask.label = "kv.window.emptyMask"
        windowEmptyMask = mask
    }

    /// True when this layer's authoritative rows live on the host and its
    /// Metal buffer is only the window.
    public func streamsLayer(_ layer: Int) -> Bool {
        windowTokens != nil && kinds[layer] == .full
    }

    /// Bytes the window costs in Metal for the streamed layers, K and V.
    public var windowBytes: Int {
        var total = 0
        for layer in 0..<kinds.count where streamsLayer(layer) {
            total += kBuffers[layer].length + vBuffers[layer].length
        }
        return total
    }

    /// Bytes the authoritative host arenas occupy, K and V.
    public var hostArenaBytes: Int {
        hostArenaLengths.reduce(0, +) * 2
    }

    /// The prefill chunk size this tier can serve without the union of a
    /// chunk's selections overflowing the window.
    ///
    /// The worst case is every row selecting a disjoint `selectionWidth` set,
    /// so the bound is `windowTokens / selectionWidth`. Selection entries are
    /// a quarter of a row's budget per block in practice; the bound is the
    /// correctness limit, not the expected cost.
    public func streamingPrefillChunkTokens(configured: Int) -> Int {
        guard windowTokens != nil, config.sparseIndexer.enabled else {
            return configured
        }
        let width =
            config.sparseIndexer.budget + config.sparseIndexer.compressRatio - 1
        guard width > 0 else { return configured }
        return max(1, min(configured, (windowTokens ?? configured) / width))
    }

    // MARK: - Window rows

    func windowRow(layer: Int, slot: Int, position: Int) -> Int? {
        precondition(position >= 0 && position < maxContext)
        let value = windowRowMap[layer][slot * maxContext + position]
        guard value != UInt32.max else { return nil }
        let row = Int(value)
        let base = slot * (windowTokens ?? 0)
        guard windowOccupant[layer][base + row] == UInt32(position) else {
            return nil
        }
        return row
    }

    func hostRow(
        layer: Int, slot: Int, position: Int,
        key: Bool,
        windowRow: Int
    ) -> (source: UnsafeMutableRawPointer, destination: UnsafeMutableRawPointer) {
        let stride = strides[layer]
        let hostBase: UnsafeMutableRawPointer
        switch (key, hostK[layer], hostV[layer]) {
        case (true, let pointer?, _), (false, _, let pointer?):
            hostBase = pointer
        case _:
            preconditionFailure("streamed layer \(layer) has no host arena")
        }
        let host = hostBase.advanced(
            by: (slot * maxContext + position) * stride)
        let buffer = key ? kBuffers[layer] : vBuffers[layer]
        let window = buffer.contents().advanced(
            by: (slot * (windowTokens ?? 0) + windowRow) * stride)
        return (host, window)
    }

    @discardableResult
    func ensureWindowRow(
        layer: Int, slot: Int, position: Int, forWrite: Bool
    ) -> Int {
        if let row = windowRow(layer: layer, slot: slot, position: position) {
            if forWrite {
                windowPinned[layer][slot] = position
                windowDirty[layer][slot * (windowTokens ?? 0) + row] = true
            }
            return row
        }
        let rows = windowTokens ?? 0
        precondition(rows > 0)
        var chosen = -1
        for _ in 0..<rows {
            let row = windowCursor[layer][slot]
            windowCursor[layer][slot] = (row + 1) % rows
            let index = slot * rows + row
            let occupant = windowOccupant[layer][index]
            let protectedKey =
                layer * (slots * rows) + slot * rows + row
            if windowProtected.contains(protectedKey) { continue }
            let pinned = windowPinned[layer][slot]
            if occupant != UInt32.max, pinned >= 0,
                occupant == UInt32(pinned)
            {
                // The token being written this step must still be there when
                // attention runs; skip it instead of flushing it away.
                continue
            }
            if occupant != UInt32.max {
                flushWindowRow(layer: layer, slot: slot, row: row)
                windowRowMap[layer][slot * maxContext + Int(occupant)] = UInt32.max
                windowOccupant[layer][index] = UInt32.max
                windowDirty[layer][index] = false
                windowEvictions += 1
            }
            chosen = row
            break
        }
        precondition(
            chosen >= 0,
            "KV window has no free row for layer \(layer) slot \(slot); "
                + "the selection plus the in-flight token must fit \(rows) rows")
        let index = slot * rows + chosen
        windowOccupant[layer][index] = UInt32(position)
        windowRowMap[layer][slot * maxContext + position] = UInt32(chosen)
        windowDirty[layer][index] = forWrite
        if forWrite {
            windowPinned[layer][slot] = position
        }
        return chosen
    }

    /// Copies one dirty window row back to the arena. Called only when the GPU
    /// is between steps (eviction during a later step, or `advance`).
    func flushWindowRow(layer: Int, slot: Int, row: Int) {
        let index = slot * (windowTokens ?? 0) + row
        guard windowDirty[layer][index] else { return }
        let occupant = windowOccupant[layer][index]
        guard occupant != UInt32.max else {
            windowDirty[layer][index] = false
            return
        }
        let stride = strides[layer]
        for key in [true, false] {
            let pointers = hostRow(
                layer: layer, slot: slot, position: Int(occupant),
                key: key, windowRow: row)
            memcpy(pointers.source, pointers.destination, stride)
        }
        windowDirty[layer][index] = false
    }

    /// Reserves `position` for this step's write and marks it resident.
    func pinForDecode(layer: Int, slot: Int, position: Int) {
        _ = ensureWindowRow(
            layer: layer, slot: slot, position: position, forWrite: false)
        windowPinned[layer][slot] = position
    }

    // MARK: - Selection gather

    private func growSelectionScratch(entries: Int, count: Int) throws {
        let entryBytes = entries * MemoryLayout<UInt32>.size
        if let indices = windowIndicesBuf, indices.length >= entryBytes,
            let rows = windowRowsBuf, rows.length >= entryBytes
        {
            windowSelectionCapacity = max(windowSelectionCapacity, entries)
            _ = indices
            _ = rows
            return
        }
        let target = max(entries, 1)
        guard
            let indices = device.makeBuffer(
                length: target * MemoryLayout<UInt32>.size,
                options: .storageModeShared),
            let rows = device.makeBuffer(
                length: target * MemoryLayout<UInt32>.size,
                options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        indices.label = "kv.window.selectionIndices"
        rows.label = "kv.window.selectionRows"
        windowIndicesBuf = indices
        windowRowsBuf = rows
        windowSelectionCapacity = target
    }

    func growPrefillSelectionScratch(
        entries: Int, count: Int
    ) throws {
        let entryBytes = entries * MemoryLayout<UInt32>.size
        let countBytes = count * MemoryLayout<UInt32>.size
        if let indices = prefillIndicesBuf,
            let rows = prefillRowsBuf, let counts = prefillCountsBuf,
            indices.length >= entryBytes,
            counts.length >= countBytes
        {
            _ = rows
            return
        }
        let target = max(entries, 1)
        guard
            let indices = device.makeBuffer(
                length: target * MemoryLayout<UInt32>.size,
                options: .storageModeShared),
            let rows = device.makeBuffer(
                length: target * MemoryLayout<UInt32>.size,
                options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        if (prefillCountsBuf?.length ?? 0) < countBytes
            || prefillCountsBuf == nil
        {
            guard
                let counts = device.makeBuffer(
                    length: max(count, 1) * MemoryLayout<UInt32>.size,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            counts.label = "kv.window.prefillCounts"
            prefillCountsBuf = counts
        }
        indices.label = "kv.window.prefillIndices"
        rows.label = "kv.window.prefillRows"
        prefillIndicesBuf = indices
        prefillRowsBuf = rows
        prefillSelectionCapacity = target
    }

    /// Gathers the keys a decode step will read into the window and returns
    /// the indirection arrays the attention kernel binds.
    ///
    /// `indices` may be nil, which means the dense path: every visible key is
    /// selected. That is also the path inside QSA's exact window, where the
    /// selection is vacuous. The current token's row is pinned by the write
    /// that produced it, so the gather can evict every other row.
    func prepareDecodeWindow(
        layer: Int, slot: Int, position: Int,
        indices: MTLBuffer?, count: Int
    ) throws -> KVWindowSelection {
        precondition(streamsLayer(layer))
        let window = windowTokens ?? 0
        guard count > 0 else {
            throw ModelError.internalInconsistency(
                detail: "decode selection is empty at layer \(layer)")
        }
        guard count <= window else {
            throw ModelError.internalInconsistency(
                detail:
                    "decode selection \(count) does not fit the KV window "
                    + "\(window) at layer \(layer) position \(position); "
                    + "raise --host-kv-window or enable the sparse indexer")
        }
        pinForDecode(layer: layer, slot: slot, position: position)
        windowProtected.removeAll(keepingCapacity: true)
        defer { windowProtected.removeAll(keepingCapacity: true) }
        try growSelectionScratch(entries: count, count: count)
        guard let indexBuf = windowIndicesBuf, let rowBuf = windowRowsBuf else {
            throw ModelError.residentBufferWrapFailed
        }
        let indexPtr = indexBuf.contents().bindMemory(
            to: UInt32.self, capacity: count)
        let rowPtr = rowBuf.contents().bindMemory(
            to: UInt32.self, capacity: count)
        let selected = indices?.contents().bindMemory(
            to: UInt32.self, capacity: count)
        for i in 0..<count {
            let logical = selected.map { Int($0[i]) } ?? i
            precondition(
                logical >= 0 && logical <= position,
                "selection entry \(logical) is past the current position \(position)")
            let alreadyResident =
                windowRow(
                    layer: layer, slot: slot, position: logical) != nil
            let row = ensureWindowRow(
                layer: layer, slot: slot, position: logical, forWrite: false)
            windowProtected.insert(
                layer * (slots * window)
                    + slot * window + row)
            if !alreadyResident, logical != position {
                let stride = strides[layer]
                for key in [true, false] {
                    let pointers = hostRow(
                        layer: layer, slot: slot, position: logical,
                        key: key, windowRow: row)
                    memcpy(pointers.destination, pointers.source, stride)
                }
                windowCopyBytes += strides[layer] * 2
                windowRowCopies += 1
            }
            indexPtr[i] = UInt32(logical)
            rowPtr[i] = UInt32(row)
        }
        return KVWindowSelection(
            indices: indexBuf, rows: rowBuf, count: count)
    }
}
