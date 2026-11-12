import Foundation
import Metal

/// Accepts or rejects a W-row verify pass's drafts under any sampling
/// configuration, emitting exactly what scalar decode would have emitted
/// (docs/plan-wide-verify.md, task 10).
///
/// Scalar decode (`runRawCompletion`) samples each token with `sampleOnce`:
/// the host applies the repetition and presence penalties to the shared FP16
/// logits buffer, then the GPU runs the softcap+softmax front end and the
/// seeded Top-K / Top-P draw. This type does not reimplement any of that. For
/// row i it copies the row's logits into the same `RawCompletionScratch.logits`
/// buffer and calls the same `sampleOnce` with the same scratch sampler, so the
/// math, kernels and the incremental penalty table are scalar decode's own.
/// What it supplies are the two inputs scalar decode would have had at that
/// token:
///
/// - the penalty history: the committed history plus the drafts accepted
///   before row i (`drafts[0..<i]`), and nothing after them. The history only
///   ever grows by appending, which is what the sampler's incremental table
///   folds; a rejected draft is never appended, because acceptance stops at
///   the row that rejects it.
/// - the seed position: scalar decode passes the index of the token being
///   generated (`generated`, counting from 0 at the first decoded token), not
///   the KV position. Row i's token is generated token `firstSampleIndex + i`.
///
/// Byte identity with scalar decode then rests on the rows' logits matching
/// scalar decode's (the verify pass's job) and on the caller following the
/// contract below.
///
/// Caller contract:
/// - `scratch` is the generation's own scratch, whose sampler has seen the
///   history up to now (it is reset once per generation, as scalar decode
///   does); scalar steps and wide passes can then be interleaved freely.
/// - after `resolve`, append every emitted token to the history -- scalar
///   decode appends each token it emits -- and pass `firstSampleIndex +
///   emitted.count` next time.
/// - stop handling stays with the caller: scalar decode stops at the first
///   stop token, stop string or `maxNewTokens`, so the caller truncates
///   `emitted` there and ends the generation. Nothing sampled past that point
///   is ever shown, and the sampler is reset before the next generation.
package struct WideVerifySampler {
    private let scratch: RawCompletionScratch
    private let context: MetalContext
    private let config: GenerationConfig

    /// Refuses a constrained request: the grammar is advanced by the decode
    /// loop between tokens, so a row's mask would depend on tokens this pass
    /// has not committed yet. Such a request takes the scalar path, as it does
    /// for MTP.
    package init(
        scratch: RawCompletionScratch, context: MetalContext, config: GenerationConfig
    ) throws {
        guard config.constraint == nil else {
            throw GeneratorError.invalidGenerationConfig(
                "constrained decoding does not support the wide verify path")
        }
        self.scratch = scratch
        self.context = context
        self.config = config
    }

    /// Bytes of one logits row: FP16 `[vocab]`.
    package var rowStride: Int { scratch.sampler.vocab * MemoryLayout<Float16>.stride }

    /// - Parameters:
    ///   - rowLogits: `[drafts.count + 1, vocab]` FP16 in a shared buffer; row
    ///     i is the distribution of the token after the pass's input row i
    ///     (`RealForwardRunner.verifyRowLogits`).
    ///   - drafts: the `W - 1` drafted tokens the pass verified (input rows
    ///     1...W-1).
    ///   - history: the committed history exactly as scalar decode would pass
    ///     it for row 0's token: prompt plus every emitted token, ending with
    ///     the pass's boundary token.
    ///   - firstSampleIndex: how many tokens this generation has emitted so
    ///     far, i.e. scalar decode's `position` argument for row 0's token.
    ///   - timing: the runner whose kernel timeline the sample spans join.
    package func resolve(
        rowLogits: MTLBuffer,
        drafts: ArraySlice<Int32>,
        history: [Int32],
        firstSampleIndex: Int,
        timing: RealForwardRunner? = nil
    ) throws -> DraftAcceptance.Outcome {
        let stride = rowStride
        guard rowLogits.storageMode == .shared,
            rowLogits.length >= (drafts.count + 1) * stride
        else {
            throw GeneratorError.invalidGenerationConfig(
                "wide verify row logits need a shared buffer of \(drafts.count + 1) rows")
        }
        var rowHistory = history
        rowHistory.reserveCapacity(history.count + drafts.count)
        return try DraftAcceptance.resolve(drafts: drafts) { row, accepted in
            // Append-only: `accepted` is `drafts[0..<row]`, one longer per row.
            let appended = rowHistory.count - history.count
            rowHistory.append(contentsOf: accepted.dropFirst(appended))
            scratch.logits.contents().copyMemory(
                from: rowLogits.contents().advanced(by: row * stride), byteCount: stride)
            return try sampleOnce(
                scratch: scratch, context: context, history: rowHistory,
                config: config, position: firstSampleIndex + row, timing: timing)
        }
    }
}
