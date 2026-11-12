import Foundation
import Metal

// Wide verify: rollback after a partly accepted draft
// (docs/plan-wide-verify.md, task 9).
//
// A W-row verify pass (W <= 9) from position P feeds x_0..x_{W-1} and advances
// every piece of per-sequence state by W rows. Acceptance keeps k of them
// (1 <= k <= W). The engine must then hold exactly the state scalar decode
// would after consuming x_0..x_{k-1}. Per piece:
//
// 1. GDN recurrent state and conv tail (each linear layer, the pass's slot).
//    Not invertible bit-exactly, so it is snapshotted before the pass and
//    rebuilt after it:
//    - before: one blit per layer copies the slot's FP32 state and FP16 tail
//      into a private snapshot (`beginWideVerifyRollbackWindow`);
//    - during: each linear layer's branch blits the pass's raw `in_proj_qkv`
//      rows and its `a` and `b` rows into the window (`wideVerifyRetain`);
//    - on a partial accept: the snapshot is blitted back and rows 0..k-1 are
//      replayed through the prefill conv, conv-tail update, q/k norm and delta
//      kernels with rows = k. The conv is deterministic, so it reproduces the
//      post-norm conv output from the restored tail and the same raw rows, and
//      the prefill delta recurrence is bit-identical to sequential decode
//      (GDNKernelTests.prefillChunkMatchesSequentialDecode). The replayed
//      state is therefore the state the pass's own path leaves after k rows.
//
//    What to retain, against memory. Qwen3.8-Flash-Next has 36 linear layers
//    of 48, Hv = 48, Dv = Dk = 128, qkvDim = 10240, conv kernel 4:
//    - snapshot: 36 x (3 MiB state + 60 KiB tail) = ~110 MiB; needed by
//      every option;
//    - raw qkv + a + b, W = 9: 36 x 9 x (20480 + 192) B = ~6.4 MiB (chosen);
//    - post-norm conv output as well (to skip the replay conv): twice the
//      above for a conv over k rows that costs microseconds; rejected;
//    - a state checkpoint after every row instead of a replay: 9 x 110 MiB
//      = ~1 GiB; rejected.
//    The replay touches k rows and the snapshot is two ~110 MiB blits (~1 ms
//    each at unified-memory bandwidth); a full accept (k == W) does nothing.
//
// 2. KV cache (full-attention layers). The pass wrote rows P..P+W-1; the
//    cursor moves back to P+k. Every attention read is bounded by the valid
//    length (decode `seqLen = position + 1`, prefill `kvValidCount`), and the
//    next writes land on the stale rows first. A ring-backed sliding-window
//    layer is the exception -- the pass may have wrapped over rows the window
//    still needs -- so the window is refused unless the ring holds the window
//    plus W - 1 spare rows (`WideVerifyRollbackPlan.checkRings`).
//
// 3. QSA indexer key cache. Raw keys are addressed by position and the pooled
//    block a key lands in is repooled from the raw keys present
//    (`encodeAppendKey` counts members up to the position; prefill pools with
//    `visibleKeys = end`), and scoring reads blocks up to `position /
//    compressRatio`. It has no cursor of its own, so moving the KV cursor is
//    the whole rollback, exactly as for the MTP pair rollback.
//
// 4. PLE n-gram context. `rewindPLE` trims the rejected tokens off the
//    context, which is wrong once W exceeds the n-gram size: the pass already
//    evicted the older tokens. The context is instead rebuilt from the one
//    saved at begin plus x_0..x_{k-1}. The PLE convolution window is rewound
//    with `PLEBlock.rewindWindow(acceptedRows: k, passRows: W)`, which is
//    exact because the pass ran one multi-row `encodeRows`.
//
// 5. Counters: the KV cursor is the only position counter; the transient
//    prefill/readahead cursors are cleared as after the MTP pair rollback.
//
// Calling contract (under `forwardStepGate`, like the verify pass itself):
// begin -> verify pass (whose GDN branch retains rows) -> exactly one of
// `rollbackWideVerify(acceptedRows:...)` or `endWideVerifyRollbackWindow()`.
// `runVerifyGreedyRows` does the begin itself and returns with the window
// open; the lifetime note on it says who closes it. A runner `reset()` and
// every `resetTransientState()` close an open window.

extension RealForwardRunner {
    /// The open window when it still wants layer `L`'s rows, else nil. The GDN
    /// branch of a verify pass calls `encodeRetain` on it after the layer's
    /// projections, on the command buffer that ran them.
    func wideVerifyRetain(layer L: Int) -> WideVerifyGDNRollback? {
        guard let window = fusedDecodeState.wideVerifyRollback,
            window.isRetaining(layer: L)
        else { return nil }
        return window
    }

    /// Opens a rollback window for a `rows`-row verify pass of `slot` starting
    /// at `startPosition`, and encodes the GDN snapshot ahead of the pass.
    func beginWideVerifyRollbackWindow(
        rows: Int, startPosition: Int, slot: Int = 0
    ) throws {
        guard let kv else {
            throw PrefillError.chunkedUnsupported("wide verify rollback needs a KV cache")
        }
        let actual = kv.position(slot: slot)
        guard actual == startPosition else {
            throw WideVerifyRollbackError.positionMismatch(
                expected: startPosition, actual: actual)
        }
        try WideVerifyRollbackPlan.checkRings(kv, slidingWindow: cfg.slidingWindow, rows: rows)
        let window = try wideVerifyRollbackWindow(rows: rows)
        try window.open(
            rows: rows, slot: slot, startPosition: startPosition, pleContext: pleContext)
        let live = wideVerifyLiveLayers(slot: slot)
        guard !live.isEmpty else { return }
        guard let cb = ctx.queue.makeCommandBuffer() else {
            window.close()
            throw ModelError.residentBufferWrapFailed
        }
        do {
            try window.encodeSnapshot(commandBuffer: cb, layers: live)
        } catch {
            window.close()
            throw error
        }
        // Queue order puts the snapshot ahead of the pass; nothing waits here.
        cb.commit()
    }

    /// Rolls the verify pass back to its first `k` rows. `tokens` are the
    /// pass's W input tokens; `k == W` only closes the window.
    func rollbackWideVerify(
        acceptedRows k: Int, tokens: [Int32], startPosition: Int, slot: Int = 0
    ) throws {
        guard let window = fusedDecodeState.wideVerifyRollback, window.isOpen else {
            throw WideVerifyRollbackError.windowNotOpen
        }
        defer { window.close() }
        guard window.slot == slot, window.startPosition == startPosition,
            tokens.count == window.rows
        else {
            throw WideVerifyRollbackError.windowMismatch(
                "slot \(slot), start \(startPosition), \(tokens.count) tokens against "
                    + "slot \(window.slot), start \(window.startPosition), \(window.rows) rows")
        }
        guard k >= 0, k <= window.rows else {
            throw WideVerifyRollbackError.acceptedRowsOutOfRange(accepted: k, rows: window.rows)
        }
        guard let kv else { throw WideVerifyRollbackError.windowNotOpen }
        try WideVerifyRollbackPlan.requirePassCursor(
            kv, slot: slot, startPosition: startPosition, rows: window.rows)
        guard k < window.rows else { return }
        do {
            try restoreWideVerifyGDN(window: window, acceptedRows: k)
            try WideVerifyRollbackPlan.rewindKV(
                kv, slot: slot, startPosition: startPosition,
                rows: window.rows, acceptedRows: k)
            if let pleBlock, let pleHash {
                pleBlock.rewindWindow(acceptedRows: k, passRows: window.rows)
                pleContext = WideVerifyRollbackPlan.pleContext(
                    after: window.savedPLEContext, accepting: tokens[..<k],
                    ngramSize: pleHash.ngramSize)
            }
            resetTransientState()
        } catch {
            // Part of the state may already be rolled back: no answer beats
            // a fluent wrong one.
            reset()
            throw error
        }
    }

    /// Closes the window without touching state, for a fully accepted pass.
    func endWideVerifyRollbackWindow() {
        fusedDecodeState.wideVerifyRollback?.close()
    }

    /// The window, reused while it is wide enough and reallocated otherwise.
    private func wideVerifyRollbackWindow(rows: Int) throws -> WideVerifyGDNRollback {
        if let window = fusedDecodeState.wideVerifyRollback, window.capacityRows >= rows {
            return window
        }
        fusedDecodeState.wideVerifyRollback = nil
        let layers = (0..<cfg.numLayers).filter { gdnState?.isLinear(layer: $0) ?? false }
        let window = try WideVerifyGDNRollback(
            device: ctx.device, config: cfg.linearAttention,
            layers: layers, capacityRows: rows)
        fusedDecodeState.wideVerifyRollback = window
        return window
    }

    private func wideVerifyLiveLayers(slot: Int) -> [WideVerifyGDNRollback.LayerState] {
        guard let gdnState else { return [] }
        return (0..<cfg.numLayers).compactMap { layer in
            guard gdnState.isLinear(layer: layer) else { return nil }
            let state = gdnState.stateSlot(layer: layer, slot: slot)
            let tail = gdnState.convTailSlot(layer: layer, slot: slot)
            return WideVerifyGDNRollback.LayerState(
                layer: layer,
                state: .init(state.buffer, offset: state.offset),
                tail: .init(tail.buffer, offset: tail.offset))
        }
    }

    private func restoreWideVerifyGDN(
        window: WideVerifyGDNRollback, acceptedRows k: Int
    ) throws {
        let live = wideVerifyLiveLayers(slot: window.slot)
        guard !live.isEmpty else { return }
        guard let gdn else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention state without GDN kernels")
        }
        let replay = try live.map { entry in
            let conv = try model.linearConv1d(layer: entry.layer)
            let aLog = try model.linearALog(layer: entry.layer)
            let dtBias = try model.linearDtBias(layer: entry.layer)
            return WideVerifyGDNRollback.LayerReplay(
                live: entry,
                convWeight: .init(conv.buffer, offset: Int(conv.offset)),
                aLog: .init(aLog.buffer, offset: Int(aLog.offset)),
                dtBias: .init(dtBias.buffer, offset: Int(dtBias.offset)))
        }
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try window.encodeRollback(
            commandBuffer: cb, gdn: gdn, acceptedRows: k, layers: replay)
        cb.commit()
        try waitForCompletion(cb)
    }
}
