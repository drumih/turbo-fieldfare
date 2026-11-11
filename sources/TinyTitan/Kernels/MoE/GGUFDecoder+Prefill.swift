import Foundation

/// Prompt prefill for the as-is GGUF decoder.
///
/// The decoder runs one token at a time; prefill is the loop that walks a
/// prompt through it, handing each token the n-gram window the PLE hash reads.
/// This is the GPU reference the ANE sidecar path has to beat: the sidecar
/// replaces only the full-attention layers' block, so the rest of this loop -
/// GDN, the MoE, the output tail - stays as it is either way.
extension GGUFDecoder {
    package struct PrefillMeasurement: Sendable {
        package let logits: [Float]
        package let seconds: Double
        package let tokenCount: Int

        package var tokensPerSecond: Double {
            seconds > 0 ? Double(tokenCount) / seconds : 0
        }
    }

    /// The PLE hash context for the token at `position` of `sequence` (the
    /// token, then its predecessors newest first, as far back as the
    /// checkpoint's n-gram reaches).
    package func pleHistory(of sequence: [Int32], at position: Int) -> [Int32] {
        ple.history(of: sequence, at: position)
    }

    /// Runs the token at `position` of `sequence` - the whole sequence so far,
    /// oldest first - and returns its logits. The PLE window is derived here,
    /// so a caller cannot hand the hash a shorter context than the checkpoint
    /// declares.
    package func decode(sequence: [Int32], position: Int) throws -> [Float] {
        try decode(
            token: sequence[position], position: position,
            history: pleHistory(of: sequence, at: position))
    }

    /// Run `tokens` as a prompt and return the last token's logits with the
    /// wall time.
    package func prefill(tokens: [Int32]) throws -> PrefillMeasurement {
        precondition(!tokens.isEmpty, "prefill needs at least one token")
        var logits: [Float] = []
        let start = DispatchTime.now().uptimeNanoseconds
        for position in tokens.indices {
            logits = try decode(sequence: tokens, position: position)
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        return PrefillMeasurement(
            logits: logits, seconds: seconds, tokenCount: tokens.count)
    }
}
