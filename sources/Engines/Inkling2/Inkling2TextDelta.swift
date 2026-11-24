import Foundation

/// Turns the decoded text of a growing answer into the deltas a streaming
/// client receives (TT-128).
///
/// A token can end inside a multi-byte character, and the tokenizer then decodes
/// the incomplete bytes as U+FFFD. That replacement is held back until a later
/// token completes the character, so a client never sees a character that the
/// next delta would have to take back. The deltas concatenate to the final text.
public struct Inkling2TextDelta: Sendable {
    private var emitted = ""

    public init() {}

    /// The new text since the last call, without a trailing replacement
    /// character that may still be completed.
    public mutating func advance(to text: String) -> String {
        var stable = Substring(text)
        while stable.last == "\u{FFFD}" { stable = stable.dropLast() }
        guard stable.hasPrefix(emitted), stable.count > emitted.count else { return "" }
        let delta = String(stable.dropFirst(emitted.count))
        emitted = String(stable)
        return delta
    }

    /// Whatever the final text holds beyond what was emitted, held-back
    /// replacements included: at the end of the answer nothing can complete them.
    public mutating func finish(_ text: String) -> String {
        guard text.hasPrefix(emitted), text.count > emitted.count else { return "" }
        let delta = String(text.dropFirst(emitted.count))
        emitted = text
        return delta
    }
}
