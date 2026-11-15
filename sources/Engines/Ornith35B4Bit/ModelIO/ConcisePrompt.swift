import Foundation

/// Built-in system prompts for TinyTitan "concise mode": answer directly and
/// lean, without preamble, filler, or closing codas.
///
/// The prompts are derived from the Nail-Qwen3.6-35B-A3B chat template
/// (peculiar-ragdoll), which force-appends a terseness prompt to every
/// request. Measured on M3 24 GB, temp 0, 8 chat questions:
///
///   quant | baseline | concise (standard prompt)
///   ------+----------+---------------------------
///   4-bit |  3,788+  | 1,480 (−61%)
///   6-bit |  3,714+  | 1,680 (−55%)
///   8-bit |  3,635+  | 1,570 (−57%)
///
/// (`+` = baseline hit the 512-token cap). The standard prompt ships for
/// every quantization so behavior stays consistent across the family. An
/// experimental "strengthened" prompt measured 759 tokens (−79%) on 8-bit but
/// was judged too aggressive — it breaks cross-quant consistency and risks
/// dropping nuance on complex answers — so it is not shipped. See the wiki
/// FAQ and Optimization Journey for the full tables.
public enum ConcisePrompt {
    /// Shipped prompt for every quantization. Measured: 1,480 / 1,680 / 1,570
    /// tokens for 8 answers vs 3,788+ / 3,714+ / 3,635+ baseline.
    public static let standard = """
        You are in concise mode. Think before answering, then answer directly.
        Lead with the answer, then include only what the answer needs to be correct and usable.
        Never: open with preamble or pleasantries; restate the question; add filler transitions; hedge with niceties; repeat a point already made; or add a closing summary, follow-up offer, or 'let me know if you have questions' coda.
        Always: keep essential steps, caveats, uncertainties, and specifics — never drop correctness or a needed warning for brevity. Keep the final answer lean: use the least structure that conveys it (plain prose when short; lists or code only when they earn their place). If genuinely uncertain, say so and explain why — never omit uncertainty for brevity's sake.
        If a user request is genuinely ambiguous, ask one sharp question instead of guessing.
        When the answer is complete, stop — end with the answer itself.
        """

    /// Experimental prompt that measured 759 tokens (−79%) on 8-bit — more
    /// aggressive than `standard` but not shipped, because it is inconsistent
    /// with 4/6-bit behavior and risks dropping nuance on complex answers.
    public static let strengthened = """
        You are in concise mode. Answer with the answer only — lead with it, then add exactly what is needed to be correct and usable, nothing more.
        Never: open with preamble, pleasantries, or 'here is...' introductions; restate the question; add filler transitions; hedge with niceties; repeat a point; explain or justify the answer's structure; or add a closing summary, wrap-up sentence, or follow-up offer. When the answer is complete, stop — end with the answer itself.
        Always: keep essential steps, caveats, uncertainties, and specifics — brevity never drops correctness. Use the least structure that conveys the answer (plain prose when short; lists or code only when they earn their place). If genuinely uncertain, say so and explain why. If the request is genuinely ambiguous, ask one sharp question instead of guessing.
        """

    // There is no per-quantization variant, and there used to be a function
    // claiming there was: `prompt(forRoutedExpertBits:)` returned `standard` for
    // every width (`strengthened` has never shipped), and its callers read the
    // manifest -- pinned to one family, so it silently fell back to 4 for every
    // other one -- to compute a value that was then discarded. `standard` is the
    // only prompt, so it is named directly at each call site rather than behind a
    // selection that selects nothing.

    /// Apply the concise system prompt to a message list.
    ///
    /// TinyTitan's chat template requires exactly one leading system message, so
    /// the concise prompt is folded into the existing first system/developer
    /// message (appended at the end, so our instructions come last within the
    /// system block — the Nail template's append-after-user-guidance idea) or
    /// becomes the opening system message when none exists.
    public static func appendingSystemPrompt(
        _ prompt: String,
        to messages: [GFTokenizer.Message]
    ) -> [GFTokenizer.Message] {
        guard
            let index = messages.firstIndex(where: {
                $0.role == .system || $0.role == .developer
            })
        else {
            return [GFTokenizer.Message(role: .system, content: prompt)] + messages
        }
        var result = messages
        let existing = result[index]
        result[index] = GFTokenizer.Message(
            role: .system,
            content: (existing.content ?? "") + "\n\n" + prompt)
        return result
    }
}
