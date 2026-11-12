import Foundation
import Metal

/// Why a wide-verify rollback window refused an operation. Every case is a
/// caller error that would otherwise leave the GDN state, the KV cursor or the
/// n-gram context silently out of step with scalar decode.
enum WideVerifyRollbackError: Error, Equatable, CustomStringConvertible {
    case windowNotOpen
    case rowsOutOfRange(rows: Int, capacity: Int)
    case unknownLayer(Int)
    case missingLayer(Int)
    case retentionOverflow(layer: Int, retained: Int, adding: Int, rows: Int)
    case retentionIncomplete(layer: Int, retained: Int, needed: Int)
    case acceptedRowsOutOfRange(accepted: Int, rows: Int)
    case positionMismatch(expected: Int, actual: Int)
    case windowMismatch(String)
    case ringTooSmall(layer: Int, capacity: Int, window: Int, rows: Int)
    case configMismatch

    var description: String {
        switch self {
        case .windowNotOpen:
            "no wide-verify rollback window is open"
        case .rowsOutOfRange(let rows, let capacity):
            "wide-verify window of \(rows) rows exceeds 1...\(capacity)"
        case .unknownLayer(let layer):
            "layer \(layer) is not a linear layer of this rollback window"
        case .missingLayer(let layer):
            "rollback did not supply linear layer \(layer)"
        case .retentionOverflow(let layer, let retained, let adding, let rows):
            "layer \(layer): retaining \(adding) rows after \(retained) exceeds the \(rows)-row window"
        case .retentionIncomplete(let layer, let retained, let needed):
            "layer \(layer): \(retained) rows retained, rollback needs \(needed)"
        case .acceptedRowsOutOfRange(let accepted, let rows):
            "accepted \(accepted) rows of a \(rows)-row pass"
        case .positionMismatch(let expected, let actual):
            "KV cursor is at \(actual), expected \(expected)"
        case .windowMismatch(let detail):
            "rollback does not match the open window: \(detail)"
        case .ringTooSmall(let layer, let capacity, let window, let rows):
            "layer \(layer): ring of \(capacity) rows cannot hold a \(window)-row window plus \(rows - 1) rejected rows"
        case .configMismatch:
            "GDN kernels were built for a different linear-attention shape"
        }
    }
}

/// GPU storage and encoders that undo a wide-verify pass's Gated-DeltaNet
/// advance down to its accepted rows. The design and its memory budget are in
/// `RealForwardRunner+WideVerifyRollback.swift`.
///
/// Per linear layer it holds one slot's state and conv tail as they were
/// before the pass, and the pass's raw `in_proj_qkv`, `a` and `b` rows. A
/// rollback restores the snapshot and replays the first k rows through the
/// prefill conv, conv-tail update, q/k norm and delta kernels -- the same
/// kernels on the same inputs, so the result is the state those k rows leave.
final class WideVerifyGDNRollback {
    /// A buffer and the byte offset a kernel binds it at.
    struct Region {
        let buffer: MTLBuffer
        let offset: Int

        init(_ buffer: MTLBuffer, offset: Int = 0) {
            self.buffer = buffer
            self.offset = offset
        }
    }

    /// One linear layer's live state for the window's slot.
    struct LayerState {
        let layer: Int
        let state: Region
        let tail: Region
    }

    /// A layer's live state plus the parameters its replay reads.
    struct LayerReplay {
        let live: LayerState
        let convWeight: Region
        let aLog: Region
        let dtBias: Region
    }

    let config: LinearAttentionConfig
    let capacityRows: Int
    /// The linear layers this window snapshots, in layer order.
    let layers: [Int]
    let stateBytes: Int
    let tailBytes: Int

    private let ordinal: [Int: Int]
    private let snapshotStates: MTLBuffer?
    private let snapshotTails: MTLBuffer?
    private let qkvRows: MTLBuffer?
    private let aRows: MTLBuffer?
    private let bRows: MTLBuffer?
    private let replayConvOut: MTLBuffer?
    private let replayY: MTLBuffer?

    private var retained: [Int]
    private(set) var isOpen = false
    private(set) var rows = 0
    private(set) var slot = 0
    private(set) var startPosition = 0
    /// The runner's n-gram token context before the pass.
    private(set) var savedPLEContext: [Int32] = []

    private static let half = MemoryLayout<Float16>.stride

    private var qkvRowBytes: Int { config.qkvDim * Self.half }
    private var headRowBytes: Int { config.numVHeads * Self.half }

    /// Total bytes this window allocates.
    let allocatedBytes: Int

    init(
        device: MTLDevice, config: LinearAttentionConfig,
        layers: [Int], capacityRows: Int
    ) throws {
        precondition(capacityRows > 0, "capacityRows must be positive")
        self.config = config
        self.capacityRows = capacityRows
        self.layers = layers
        let stateBytes =
            config.numVHeads * config.valueHeadDim * config.keyHeadDim
            * MemoryLayout<Float>.stride
        let tailBytes = max(0, config.convKernelSize - 1) * config.qkvDim * Self.half
        self.stateBytes = stateBytes
        self.tailBytes = tailBytes
        self.ordinal = Dictionary(
            uniqueKeysWithValues: layers.enumerated().map { ($1, $0) })
        self.retained = Array(repeating: 0, count: layers.count)
        let count = layers.count
        let qkvBytes = config.qkvDim * Self.half
        let headBytes = config.numVHeads * Self.half
        func make(_ length: Int, _ label: String) throws -> MTLBuffer? {
            guard length > 0 else { return nil }
            guard let buffer = device.makeBuffer(length: length, options: .storageModePrivate)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            buffer.label = label
            return buffer
        }
        let sizes = (
            states: count * stateBytes, tails: count * tailBytes,
            qkv: count * capacityRows * qkvBytes,
            heads: count * capacityRows * headBytes,
            convOut: count > 0 ? capacityRows * qkvBytes : 0,
            y: count > 0 ? capacityRows * config.valueDim * Self.half : 0
        )
        self.snapshotStates = try make(sizes.states, "wideverify.snapshot.state")
        self.snapshotTails = try make(sizes.tails, "wideverify.snapshot.tail")
        self.qkvRows = try make(sizes.qkv, "wideverify.retained.qkv")
        self.aRows = try make(sizes.heads, "wideverify.retained.a")
        self.bRows = try make(sizes.heads, "wideverify.retained.b")
        self.replayConvOut = try make(sizes.convOut, "wideverify.replay.convOut")
        self.replayY = try make(sizes.y, "wideverify.replay.y")
        self.allocatedBytes =
            sizes.states + sizes.tails + sizes.qkv + 2 * sizes.heads
            + sizes.convOut + sizes.y
    }

    /// Starts a window over a `rows`-row pass of `slot` at `startPosition`.
    func open(rows: Int, slot: Int, startPosition: Int, pleContext: [Int32]) throws {
        guard rows >= 1, rows <= capacityRows else {
            throw WideVerifyRollbackError.rowsOutOfRange(rows: rows, capacity: capacityRows)
        }
        self.rows = rows
        self.slot = slot
        self.startPosition = startPosition
        self.savedPLEContext = pleContext
        for index in retained.indices { retained[index] = 0 }
        isOpen = true
    }

    /// Ends the window; later retention hooks see it closed and do nothing.
    func close() {
        isOpen = false
        rows = 0
        for index in retained.indices { retained[index] = 0 }
    }

    /// Rows of `layer` retained so far in the open window.
    func retainedRows(layer: Int) -> Int {
        ordinal[layer].map { retained[$0] } ?? 0
    }

    /// True while the open window still wants rows of `layer`.
    func isRetaining(layer: Int) -> Bool {
        guard isOpen, let index = ordinal[layer] else { return false }
        return retained[index] < rows
    }

    /// Copies every layer's live state and conv tail into the snapshot.
    func encodeSnapshot(commandBuffer: MTLCommandBuffer, layers live: [LayerState]) throws {
        guard isOpen else { throw WideVerifyRollbackError.windowNotOpen }
        try requireCoverage(live.map(\.layer))
        guard !live.isEmpty, let states = snapshotStates, let tails = snapshotTails else {
            return
        }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        for entry in live {
            let index = try layerOrdinal(entry.layer)
            blit.copy(
                from: entry.state.buffer, sourceOffset: entry.state.offset,
                to: states, destinationOffset: index * stateBytes, size: stateBytes)
            if tailBytes > 0 {
                blit.copy(
                    from: entry.tail.buffer, sourceOffset: entry.tail.offset,
                    to: tails, destinationOffset: index * tailBytes, size: tailBytes)
            }
        }
        blit.endEncoding()
    }

    /// Appends `count` rows of `layer`'s raw qkv, `a` and `b` projections.
    ///
    /// The chunked path retains a layer's rows in one call; a per-row path
    /// calls once per row with `count == 1`, in row order.
    func encodeRetain(
        commandBuffer: MTLCommandBuffer, layer: Int,
        qkv: Region, a: Region, b: Region, rows count: Int
    ) throws {
        guard isOpen else { throw WideVerifyRollbackError.windowNotOpen }
        let index = try layerOrdinal(layer)
        let first = retained[index]
        guard count > 0, first + count <= rows else {
            throw WideVerifyRollbackError.retentionOverflow(
                layer: layer, retained: first, adding: count, rows: rows)
        }
        guard let qkvRows, let aRows, let bRows,
            let blit = commandBuffer.makeBlitCommandEncoder()
        else {
            throw MetalError.commandEncoderFailed
        }
        let row = index * capacityRows + first
        blit.copy(
            from: qkv.buffer, sourceOffset: qkv.offset,
            to: qkvRows, destinationOffset: row * qkvRowBytes,
            size: count * qkvRowBytes)
        blit.copy(
            from: a.buffer, sourceOffset: a.offset,
            to: aRows, destinationOffset: row * headRowBytes,
            size: count * headRowBytes)
        blit.copy(
            from: b.buffer, sourceOffset: b.offset,
            to: bRows, destinationOffset: row * headRowBytes,
            size: count * headRowBytes)
        blit.endEncoding()
        retained[index] = first + count
    }

    /// Restores the snapshot and replays the first `accepted` rows. Returns
    /// false, encoding nothing, when every row was accepted.
    @discardableResult
    func encodeRollback(
        commandBuffer: MTLCommandBuffer, gdn: GDN,
        acceptedRows accepted: Int, layers replay: [LayerReplay]
    ) throws -> Bool {
        guard isOpen else { throw WideVerifyRollbackError.windowNotOpen }
        guard accepted >= 0, accepted <= rows else {
            throw WideVerifyRollbackError.acceptedRowsOutOfRange(accepted: accepted, rows: rows)
        }
        guard accepted < rows else { return false }
        guard gdn.config.qkvDim == config.qkvDim,
            gdn.config.numVHeads == config.numVHeads,
            gdn.config.convKernelSize == config.convKernelSize
        else {
            throw WideVerifyRollbackError.configMismatch
        }
        try requireCoverage(replay.map(\.live.layer))
        for entry in replay {
            let have = retained[try layerOrdinal(entry.live.layer)]
            guard have >= accepted else {
                throw WideVerifyRollbackError.retentionIncomplete(
                    layer: entry.live.layer, retained: have, needed: accepted)
            }
        }
        guard !replay.isEmpty else { return true }
        try encodeRestore(commandBuffer: commandBuffer, layers: replay.map(\.live))
        if accepted > 0 {
            for entry in replay {
                try encodeReplay(
                    commandBuffer: commandBuffer, gdn: gdn, entry: entry, rows: accepted)
            }
        }
        return true
    }

    private func encodeRestore(commandBuffer: MTLCommandBuffer, layers live: [LayerState]) throws {
        guard let states = snapshotStates, let tails = snapshotTails,
            let blit = commandBuffer.makeBlitCommandEncoder()
        else {
            throw MetalError.commandEncoderFailed
        }
        for entry in live {
            let index = try layerOrdinal(entry.layer)
            blit.copy(
                from: states, sourceOffset: index * stateBytes,
                to: entry.state.buffer, destinationOffset: entry.state.offset,
                size: stateBytes)
            if tailBytes > 0 {
                blit.copy(
                    from: tails, sourceOffset: index * tailBytes,
                    to: entry.tail.buffer, destinationOffset: entry.tail.offset,
                    size: tailBytes)
            }
        }
        blit.endEncoding()
    }

    /// The chunked-prefill GDN chain over `rows` retained rows, minus the
    /// gated norm and out-projection: the replay wants the state, not `y`.
    private func encodeReplay(
        commandBuffer cb: MTLCommandBuffer, gdn: GDN,
        entry: LayerReplay, rows count: Int
    ) throws {
        guard let qkvRows, let aRows, let bRows, let replayConvOut, let replayY else {
            throw MetalError.commandEncoderFailed
        }
        let row = try layerOrdinal(entry.live.layer) * capacityRows
        let tail = entry.live.tail
        try gdn.encodeConvPrefill(
            commandBuffer: cb,
            tail: tail.buffer, tailOffset: tail.offset,
            qkvRows: qkvRows, qkvRowsOffset: row * qkvRowBytes,
            convWeight: entry.convWeight.buffer,
            convWeightOffset: entry.convWeight.offset,
            out: replayConvOut, rows: count)
        try gdn.encodeConvTailUpdate(
            commandBuffer: cb,
            tail: tail.buffer, tailOffset: tail.offset,
            qkvRows: qkvRows, qkvRowsOffset: row * qkvRowBytes,
            rows: count)
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: replayConvOut, rows: count)
        try gdn.encodeDeltaStepPrefill(
            commandBuffer: cb,
            convOut: replayConvOut,
            aProj: aRows, aProjOffset: row * headRowBytes,
            bProj: bRows, bProjOffset: row * headRowBytes,
            aLog: entry.aLog.buffer, aLogOffset: entry.aLog.offset,
            dtBias: entry.dtBias.buffer, dtBiasOffset: entry.dtBias.offset,
            state: entry.live.state.buffer, stateOffset: entry.live.state.offset,
            y: replayY, rows: count)
    }

    private func layerOrdinal(_ layer: Int) throws -> Int {
        guard let index = ordinal[layer] else {
            throw WideVerifyRollbackError.unknownLayer(layer)
        }
        return index
    }

    /// Snapshot and rollback must name every layer of the window: a layer left
    /// out would keep the rejected rows in its state.
    private func requireCoverage(_ named: [Int]) throws {
        for layer in named { _ = try layerOrdinal(layer) }
        let present = Set(named)
        if let missing = layers.first(where: { !present.contains($0) }) {
            throw WideVerifyRollbackError.missingLayer(missing)
        }
    }
}

/// The non-GPU parts of a wide-verify rollback, kept pure so they can be
/// tested without a model.
enum WideVerifyRollbackPlan {
    /// The n-gram context scalar decode would hold after consuming `accepted`
    /// on top of `saved` (most recent token first, capped at `ngramSize`).
    ///
    /// Rebuilt rather than trimmed: the pass already evicted the oldest
    /// tokens, so dropping its rejected rows cannot recover them once the
    /// pass is wider than the n-gram window.
    static func pleContext(
        after saved: [Int32], accepting accepted: ArraySlice<Int32>, ngramSize: Int
    ) -> [Int32] {
        var context = saved
        for token in accepted {
            context.insert(token, at: 0)
            if context.count > ngramSize {
                context.removeLast(context.count - ngramSize)
            }
        }
        return context
    }

    /// A ring-backed sliding-window layer wraps the pass's rows over its
    /// oldest ones. Rewinding is safe only when every overwritten row is
    /// already outside the window the next decode step reads.
    static func ringRewindIsSafe(ringCapacity: Int, slidingWindow: Int, rows: Int) -> Bool {
        ringCapacity == 0 || ringCapacity - slidingWindow >= rows - 1
    }

    /// Refuses a window whose rejected rows could clobber live ring rows.
    static func checkRings(_ kv: KVCacheManager, slidingWindow: Int, rows: Int) throws {
        for layer in 0..<kv.config.numLayers {
            let ring = kv.ringCapacity(layer: layer)
            guard ringRewindIsSafe(ringCapacity: ring, slidingWindow: slidingWindow, rows: rows)
            else {
                throw WideVerifyRollbackError.ringTooSmall(
                    layer: layer, capacity: ring, window: slidingWindow, rows: rows)
            }
        }
    }

    /// The KV cursor a pass of `rows` from `startPosition` must have left.
    static func requirePassCursor(
        _ kv: KVCacheManager, slot: Int, startPosition: Int, rows: Int
    ) throws {
        let actual = kv.position(slot: slot)
        guard actual == startPosition + rows else {
            throw WideVerifyRollbackError.positionMismatch(
                expected: startPosition + rows, actual: actual)
        }
    }

    /// Moves the cursor back to the accepted rows. The rejected rows stay in
    /// the store but sit past the valid length every attention kernel and the
    /// sparse indexer bound their reads by, and the next writes overwrite them.
    static func rewindKV(
        _ kv: KVCacheManager, slot: Int, startPosition: Int,
        rows: Int, acceptedRows: Int
    ) throws {
        guard acceptedRows >= 0, acceptedRows <= rows else {
            throw WideVerifyRollbackError.acceptedRowsOutOfRange(
                accepted: acceptedRows, rows: rows)
        }
        try requirePassCursor(kv, slot: slot, startPosition: startPosition, rows: rows)
        try kv.rewind(slot: slot, to: startPosition + acceptedRows)
    }
}
