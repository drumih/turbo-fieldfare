import Foundation

/// Tracks the `<tool_call>` / `</tool_call>` barrier tokens as they are emitted,
/// so the decode loop can tell a tool turn from a plain one.
///
/// `StopReason.toolCalls` used to be decided by testing the *stop* token against
/// `tokenizer.toolResponseID`, which can never match: `stopTokenIDs` is
/// `[<|im_end|>, <|endoftext|>]`, and `<tool_response>` is the marker a *client*
/// sends back, not something the model emits to end a turn. The reason was
/// therefore unreachable, which left three callers dead — the tool-round prompt
/// cache reuse in `ServerPromptCache.matchToolContinuation`, the
/// `orphanToolResponse` diagnostic, and the `rawStop` diagnostic string.
///
/// The marker tokens are the right signal because they cannot be faked: the
/// streaming decoder requires each to be a literal ByteLevel barrier whose
/// content is exactly the marker (`GFTokenizer.validateStreamingDecoder`), so
/// `<tool_call>` reaches the output only as that one token and never as
/// ordinary text that a model could spell out.
struct ToolCallMarkerCounter {
    private var open = 0
    private var sawStart = false

    /// True only for a turn that opened a tool call *and* closed every call it
    /// opened.
    ///
    /// An unterminated call is a truncated turn, and a bare `</tool_call>` is
    /// not a turn that called a tool; both stay `.endOfTurn` on purpose, so the
    /// orphan diagnostic keeps its meaning: "the parse found no call in a turn
    /// that produced a well-formed one".
    var isCompleteToolTurn: Bool { sawStart && open == 0 }

    /// Counts one emitted token. Must see *every* token of the turn, including
    /// the stop token, so the caller observes before it classifies.
    mutating func observe(_ tokenID: Int32, start: Int32, end: Int32) {
        if tokenID == start {
            open += 1
            sawStart = true
        } else if tokenID == end, open > 0 {
            // The clamp keeps a stray closing marker from underflowing into a
            // negative depth, which would read as "balanced" forever after.
            open -= 1
        }
    }
}
