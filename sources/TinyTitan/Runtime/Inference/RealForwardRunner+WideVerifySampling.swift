import Foundation
import Metal

/// The W-row verify pass for sampled decoding (docs/plan-wide-verify.md,
/// task 10): the same pass as `verifyGreedyRows`, but each row's full FP16
/// logits are kept so `WideVerifySampler` can run scalar decode's own sampler
/// on them, penalties and seeded draw included.
///
/// The head writes the rows into `verificationLogits` two at a time -- the very
/// buffer `verifyGreedyRows` argmaxes on the host -- and a blit in the same
/// command buffer copies each pair out to the caller's `[rows, vocab]` buffer.
/// The greedy pass encodes no blit and is unchanged.
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

    /// Copies the pair head's `rows` rows out of `verificationLogits` into
    /// `rowLogits` starting at row `firstRow`, on `cb` after the head.
    func encodeRowLogitsCapture(
        commandBuffer cb: MTLCommandBuffer,
        into rowLogits: MTLBuffer,
        firstRow: Int,
        rows: Int
    ) throws {
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(
            from: verificationLogits, sourceOffset: 0,
            to: rowLogits, destinationOffset: firstRow * wideVerifyRowLogitsStride,
            size: rows * wideVerifyRowLogitsStride)
        blit.endEncoding()
    }
}
