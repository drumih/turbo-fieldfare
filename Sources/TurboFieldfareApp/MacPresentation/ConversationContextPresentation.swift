import Foundation

/// What the HUD says about the conversation's place in the context window, and
/// about how much of the last turn was served from the cache.
///
/// A cache nobody can see is one that can regress to nothing without a single
/// bug report — LM Studio and Ollama have both shipped that state, where reuse
/// silently stopped and users only noticed as slowness. The figure is reported
/// per turn for the same reason the OpenAI usage payload reports
/// `cached_tokens`.
public enum ConversationContextPresentation {
    /// Compact gauge for the HUD: tokens held against the window.
    public static func gauge(kvTokens: Int, maxContext: Int) -> String {
        guard maxContext > 0 else { return "\u{2014}" }
        return "\(compact(kvTokens))/\(compact(maxContext))"
    }

    /// What the KV holds *right now*, during a turn.
    ///
    /// The committed count only moves when a turn finishes, so a gauge reading
    /// it sits still through prefill and decode — exactly while the number it
    /// reports is changing — and then jumps. These three are the live figures
    /// the run already reports.
    ///
    /// `prefillDone` is absolute: `runRawCompletion` reports progress as
    /// `cachedPromptTokens + done`, so during prefill it *is* the KV position.
    /// Once a token has been sampled the prompt is fully in. The newest sampled
    /// token is still the boundary for the next producer call, so only the
    /// earlier generated tokens are committed.
    public static func liveTokens(prefillDone: Int,
                                  prefillTotal: Int,
                                  generated: Int,
                                  committed: Int) -> Int {
        if generated > 0 { return prefillTotal + generated - 1 }
        if prefillDone > 0 { return prefillDone }
        // Nothing reported yet — an image is still encoding, or the first chunk
        // has not landed. The committed figure is the last thing that was true.
        return committed
    }

    /// Percentage of the window in use, clamped, for a progress affordance.
    public static func fraction(kvTokens: Int, maxContext: Int) -> Double {
        guard maxContext > 0 else { return 0 }
        return min(1, max(0, Double(kvTokens) / Double(maxContext)))
    }

    /// True once the conversation is close enough to the wall that the user
    /// should be told before a turn is refused rather than after.
    public static func isNearingLimit(kvTokens: Int, maxContext: Int) -> Bool {
        fraction(kvTokens: kvTokens, maxContext: maxContext) >= 0.8
    }

    /// The words behind the gauge. Names the cached figure explicitly, because
    /// "it feels fast" is not evidence that anything was reused.
    public static func explanation(
        kvTokens: Int,
        maxContext: Int,
        cachedTokens: Int?
    ) -> String {
        var parts: [String] = []
        parts.append(
            "This chat is holding \(number(kvTokens)) of \(number(maxContext)) context tokens.")
        if let cachedTokens, cachedTokens > 0 {
            parts.append(
                "The last turn reused \(number(cachedTokens)) of them instead of "
                    + "processing the conversation again.")
        } else if cachedTokens != nil {
            parts.append("The last turn had nothing to reuse; it started a conversation.")
        }
        if isNearingLimit(kvTokens: kvTokens, maxContext: maxContext) {
            parts.append(
                "When the window fills, a turn is refused rather than silently "
                    + "dropping earlier ones. Start a new chat to reset it.")
        }
        return parts.joined(separator: " ")
    }

    private static func compact(_ value: Int) -> String {
        if value >= 1_000 {
            let thousands = Double(value) / 1_000
            return thousands >= 10
                ? "\(Int(thousands.rounded()))K"
                : String(format: "%.1fK", thousands)
        }
        return "\(value)"
    }

    private static func number(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }
}
