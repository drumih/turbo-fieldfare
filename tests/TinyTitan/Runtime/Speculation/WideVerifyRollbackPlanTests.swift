import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The non-GPU half of a wide-verify rollback: the n-gram context rebuild, the
/// KV cursor rewind and the ring-safety bound.
@Suite struct WideVerifyRollbackPlanTests {
    /// What scalar decode's `gatherPLERows` leaves after `tokens`.
    private func decodeContext(_ start: [Int32], _ tokens: [Int32], ngramSize: Int) -> [Int32] {
        var context = start
        for token in tokens {
            context.insert(token, at: 0)
            if context.count > ngramSize { context.removeLast(context.count - ngramSize) }
        }
        return context
    }

    @Test(arguments: [0, 1, 2, 5, 6])
    func pleContextMatchesDecodeOverAcceptedTokens(accepted: Int) {
        let saved: [Int32] = [12, 11, 10]
        let pass: [Int32] = [20, 21, 22, 23, 24, 25]
        let rebuilt = WideVerifyRollbackPlan.pleContext(
            after: saved, accepting: pass[..<accepted], ngramSize: 3)
        #expect(rebuilt == decodeContext(saved, Array(pass[..<accepted]), ngramSize: 3))
    }

    /// Trimming the rejected tokens off the post-pass context -- what
    /// `rewindPLE` does for the two-row pair -- cannot recover tokens the pass
    /// evicted once it is wider than the n-gram window.
    @Test func trimmingIsWrongForWidePasses() {
        let saved: [Int32] = [12, 11, 10]
        let pass: [Int32] = [20, 21, 22, 23, 24, 25]
        var trimmed = decodeContext(saved, pass, ngramSize: 3)
        trimmed.removeFirst(min(pass.count - 1, trimmed.count))
        let rebuilt = WideVerifyRollbackPlan.pleContext(
            after: saved, accepting: pass[..<1], ngramSize: 3)
        #expect(rebuilt == [20, 12, 11])
        #expect(trimmed != rebuilt)
    }

    @Test func ringSafetyNeedsWindowPlusRejectedRows() {
        #expect(
            WideVerifyRollbackPlan.ringRewindIsSafe(ringCapacity: 0, slidingWindow: 512, rows: 9))
        #expect(
            WideVerifyRollbackPlan.ringRewindIsSafe(ringCapacity: 520, slidingWindow: 512, rows: 9))
        #expect(
            !WideVerifyRollbackPlan.ringRewindIsSafe(ringCapacity: 519, slidingWindow: 512, rows: 9)
        )
        // The production ring is window + prefill chunk, so a pass up to the
        // chunk width plus one always fits.
        #expect(
            WideVerifyRollbackPlan.ringRewindIsSafe(
                ringCapacity: 512 + 32, slidingWindow: 512, rows: 9))
    }

    @Test func kvCursorReturnsToAcceptedRowsOnItsSlotOnly() throws {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(
            device: ctx.device, config: .qwen36_35B_A3B, maxContext: 64, slots: 2)
        try WideVerifyRollbackPlan.checkRings(kv, slidingWindow: 0, rows: 9)
        kv.advance(slot: 0, by: 5)
        kv.advance(slot: 1, by: 10)
        kv.advance(slot: 1, by: 6)
        try WideVerifyRollbackPlan.rewindKV(
            kv, slot: 1, startPosition: 10, rows: 6, acceptedRows: 2)
        #expect(kv.position(slot: 1) == 12)
        #expect(kv.position(slot: 0) == 5)
        // A second rewind of the same pass sees the cursor already moved.
        #expect(throws: WideVerifyRollbackError.positionMismatch(expected: 16, actual: 12)) {
            try WideVerifyRollbackPlan.rewindKV(
                kv, slot: 1, startPosition: 10, rows: 6, acceptedRows: 2)
        }
        #expect(throws: WideVerifyRollbackError.acceptedRowsOutOfRange(accepted: 7, rows: 6)) {
            try WideVerifyRollbackPlan.rewindKV(
                kv, slot: 1, startPosition: 6, rows: 6, acceptedRows: 7)
        }
        // Views bound reads by the cursor, so the rejected rows are invisible.
        let layer = try #require(
            (0..<kv.config.numLayers).first { kv.layerKind($0) != .linear })
        let view = kv.keyView(layer: layer, slot: 1, validTokenCount: kv.position(slot: 1))
        #expect(view.validTokenCount == 12)
    }
}
