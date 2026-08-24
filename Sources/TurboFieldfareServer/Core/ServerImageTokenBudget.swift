import Foundation
import TurboFieldfare

/// The server's face of `VisionImageTokenBudget`: the shared arithmetic plus
/// the OpenAI-shaped rejection. The rule itself lives in the runtime so the
/// app and CLI cannot drift from it.
enum ServerImageTokenBudget {
    /// A rendered image placeholder becomes one BOI marker, the projected image
    /// tokens, and one EOI marker.
    static func imageTokens(softTokenCounts: [Int]) -> Int {
        VisionImageTokenBudget.imageTokens(softTokenCounts: softTokenCounts)
    }

    /// Only the prompt is required to fit. The requested reply is deliberately
    /// not counted: the text path clamps `max_tokens` to the remaining context
    /// rather than refusing, and clients routinely ask for the whole context —
    /// pinned Pi 0.82.1 sends `max_tokens: 16384`, the entire default
    /// 16,384-token context, on every request. Counting it here rejected
    /// requests the text path would have served.
    static func fits(
        softTokenCounts: [Int],
        textTokens: Int,
        maxContext: Int
    ) -> Bool {
        VisionImageTokenBudget.fits(
            softTokenCounts: softTokenCounts,
            textTokens: textTokens,
            maxContext: maxContext)
    }

    static func rejection(
        imageCount: Int,
        softTokenCounts: [Int],
        textTokens: Int,
        maxContext: Int
    ) -> ServerRequestError {
        let images = imageTokens(softTokenCounts: softTokenCounts)
        return ServerRequestError.invalid(
            message: "\(imageCount) image(s) need \(images) tokens; with "
                + "\(textTokens) text tokens that exceeds the "
                + "\(maxContext)-token context",
            param: "messages",
            code: "context_length_exceeded")
    }
}
