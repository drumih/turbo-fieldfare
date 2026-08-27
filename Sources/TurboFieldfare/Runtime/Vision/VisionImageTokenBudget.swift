import Foundation

/// What bounds an image request is context, not a fixed image count.
///
/// This lives in the runtime rather than in one surface because the server, the
/// CLI and the Mac app all have to answer "can these images be sent" the same
/// way. They did not: the server derived a budget from the context while the
/// app kept a hard limit of four, so the same conversation was accepted by one
/// and refused by the other.
public enum VisionImageTokenBudget {
    /// A rendered image placeholder becomes one BOI marker, the projected image
    /// tokens, and one EOI marker.
    public static let markerTokensPerImage = 2

    /// The most tokens a single image can cost, whatever its dimensions. A
    /// caller that has measured the real geometry should use it; one that has
    /// not — the app, deciding whether to accept a drop before anything is
    /// preprocessed — can bound the answer with this.
    public static var maximumTokensPerImage: Int {
        VisionConfig().maximumPooledTokens + markerTokensPerImage
    }

    public static func imageTokens(softTokenCounts: [Int]) -> Int {
        softTokenCounts.reduce(0) { $0 + $1 + markerTokensPerImage }
    }

    /// Only the prompt is required to fit. The requested reply is deliberately
    /// not counted: the text path clamps the reply length to the remaining
    /// context rather than refusing, and clients routinely ask for far more
    /// than the context holds — pinned Pi 0.82.1 sends `max_tokens: 16384`
    /// against an 8,192-token context. Counting it rejected requests the text
    /// path would have served.
    public static func requiredTokens(
        softTokenCounts: [Int],
        textTokens: Int
    ) -> Int {
        textTokens + imageTokens(softTokenCounts: softTokenCounts)
    }

    public static func fits(
        softTokenCounts: [Int],
        textTokens: Int,
        maxContext: Int
    ) -> Bool {
        requiredTokens(
            softTokenCounts: softTokenCounts,
            textTokens: textTokens) < maxContext
    }

    /// How many images of unknown size can still be accepted, assuming each
    /// costs the maximum. `reservedTextTokens` is the room left for the prompt
    /// itself; the runtime still rejects a combination that does not fit, so
    /// this only has to be a defensible ceiling rather than an exact one.
    public static func capacity(
        maxContext: Int,
        reservedTextTokens: Int
    ) -> Int {
        guard maxContext > 0,
              reservedTextTokens >= 0,
              reservedTextTokens < maxContext else {
            return 0
        }
        let available = maxContext - reservedTextTokens
        return available / maximumTokensPerImage
    }
}
