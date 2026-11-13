import Foundation
import Metal

/// The W-row verify pass for sampled decoding (docs/plan-wide-verify.md,
/// task 10): the same pass as `verifyGreedyRows`, but each row's full FP16
/// logits are kept so `WideVerifySampler` can run scalar decode's own sampler
/// on them, penalties and seeded draw included.
///
/// The W-row head (`encodeVerifyHeadRows`) writes every row's logits straight
/// into the caller's `[rows, vocab]` buffer; the greedy pass writes the same
/// head's rows into its own buffer and argmaxes them on the host.
extension RealForwardRunner {
    /// Bytes of one row of verify logits: FP16 `[vocab]`, the layout the
    /// scalar sampler reads from `RawCompletionScratch.logits`.
    package var wideVerifyRowLogitsStride: Int {
        cfg.vocabSize * MemoryLayout<Float16>.stride
    }

    /// A host-readable `[rows, vocab]` FP16 buffer for `verifyRowLogits`.
    /// Allocate it once per generation at the widest W and reuse it.
    package func makeWideVerifyRowLogitsBuffer(rows: Int) throws -> MTLBuffer {
        guard rows > 0,
            let buffer = ctx.device.makeBuffer(
                length: rows * wideVerifyRowLogitsStride, options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        return buffer
    }

    /// Runs one W-row verify pass over `tokens` at `startPosition..` and
    /// writes row i's logits (the distribution of the token after `tokens[i]`)
    /// to `rowLogits` at byte offset `i * wideVerifyRowLogitsStride`.
    ///
    /// Like `verifyGreedyRows` it leaves the KV cache and GDN state advanced
    /// by W rows; the caller restores or commits. The returned
    /// `predictions` is empty: the caller samples the rows.
    package func verifyRowLogits(
        _ tokens: [Int32],
        startPosition: Int,
        into rowLogits: MTLBuffer
    ) async throws -> WideVerification {
        let rows = tokens.count
        guard rows >= 2, rows <= wideVerifyMaximumRows else {
            throw PrefillError.chunkedUnsupported(
                "wide verify needs 2...\(wideVerifyMaximumRows) rows, got \(rows)")
        }
        guard rowLogits.storageMode == .shared,
            rowLogits.length >= rows * wideVerifyRowLogitsStride
        else {
            throw PrefillError.chunkedUnsupported(
                "wide verify row logits need a shared buffer of \(rows) x \(cfg.vocabSize) FP16")
        }
        try await forwardStepGate.acquire()
        do {
            let result = try await runVerifyGreedyRows(
                tokens, startPosition: startPosition, rowLogits: rowLogits)
            await forwardStepGate.release()
            return result
        } catch {
            await forwardStepGate.release()
            throw error
        }
    }
}
