import Foundation

/// Prompt prefill for the as-is GGUF decoder.
///
/// The decoder runs one token at a time; prefill is the loop that walks a
/// prompt through it, carrying the two-token history the PLE n-gram hash needs.
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

    /// Run `tokens` as a prompt and return the last token's logits with the
    /// wall time. `history` carries the token plus its predecessor, which is
    /// what `GGUFPLEStage.gather` hashes for the 3-gram table.
    package func prefill(tokens: [Int32]) throws -> PrefillMeasurement {
        precondition(!tokens.isEmpty, "prefill needs at least one token")
        var logits: [Float] = []
        let start = DispatchTime.now().uptimeNanoseconds
        for (position, token) in tokens.enumerated() {
            var history: [Int32] = [token]
            if position > 0 { history.append(tokens[position - 1]) }
            logits = try decode(token: token, position: position, history: history)
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        return PrefillMeasurement(
            logits: logits, seconds: seconds, tokenCount: tokens.count)
    }
}
