import Foundation

enum ServerLog {
    static func accepted(id: String, streaming: Bool) {
        write("request \(id) accepted streaming=\(streaming)")
    }

    static func prepared(id: String, promptTokens: Int?) {
        let count = promptTokens.map(String.init) ?? "backend-managed"
        write("request \(id) prepared prompt=\(count)")
    }

    static func queued(id: String) {
        write("request \(id) queued")
    }

    static func generating(id: String) {
        write("request \(id) generating")
    }

    static func completed(id: String,
                          duration: Duration,
                          completion: ServerCompletion) {
        write(completedMessage(id: id, duration: duration, completion: completion))
    }

    /// Split out so the throughput arithmetic can be checked without a log sink
    /// and without a decode. `pp_tok_s` divides by the tokens actually
    /// prefilled: a prompt-cache hit contributes no prefill work, so counting
    /// cached tokens as prefilled would report a rate the runtime never
    /// achieved. Carries counts and durations only, never content.
    static func completedMessage(id: String,
                                 duration: Duration,
                                 completion: ServerCompletion) -> String {
        let usage = completion.usage
        // Clamped because both counts come from the runtime: a cached count
        // above the prompt would otherwise log a negative token count and a
        // negative rate rather than the zero work that was done.
        let computedPrefillTokens = max(
            usage.promptTokens - usage.promptTokensDetails.cachedTokens, 0)
        let prefillTokensPerSecond = completion.prefillSeconds > 0
            ? Double(computedPrefillTokens) / completion.prefillSeconds
            : 0
        let decodeTokensPerSecond = completion.decodeSeconds > 0
            ? Double(usage.completionTokens) / completion.decodeSeconds
            : 0
        return "request \(id) completed in \(format(duration)) "
            + "prompt=\(usage.promptTokens) "
            + "cached=\(usage.promptTokensDetails.cachedTokens) "
            + "completion=\(usage.completionTokens) "
            + "pp=\(formatSeconds(completion.prefillSeconds)) "
            + "pp_tok_s=\(formatRate(prefillTokensPerSecond)) "
            + "tg=\(formatSeconds(completion.decodeSeconds)) "
            + "tg_tok_s=\(formatRate(decodeTokensPerSecond)) "
            + "finish=\(completion.finishReason)"
    }

    /// What retention costs at worst, said once at load.
    ///
    /// Reported because nothing else can say it: per-slot KV is not in the
    /// model manifest and not in any existing line, so sizing a server meant
    /// deriving it from the architecture by hand.
    ///
    /// Named a ceiling because that is what it is. The figure is what a slot
    /// allocates, and a slot goes resident only as far as the conversation in
    /// it reaches: the sliding-window ring is fully touched within the first
    /// ~1,300 tokens, while the full-attention layers fault in proportionally.
    /// `reset()` madvises the pages back when a lineage is dropped. Reporting
    /// it bare invited an operator to check Activity Monitor, find a smaller
    /// number, and conclude the line was lying.
    ///
    /// The gap is narrower than it sounds, because the ring is the larger half
    /// and is reached early: a quarter of the context is 57% of the ceiling,
    /// not a quarter of it. Size for the ceiling on a machine that must not
    /// swap.
    static func promptCacheSlots(count: Int,
                                 bytesPerSlot: Int,
                                 mode: ServerPromptCacheMode) {
        let mib = Double(bytesPerSlot) / (1_024 * 1_024)
        write("prompt cache mode=\(mode.rawValue) slots=\(count) "
            + "kv_per_slot_max=\(String(format: "%.1f", mib))MiB "
            + "kv_total_max=\(String(format: "%.1f", mib * Double(count)))MiB "
            + "(ceiling; a slot goes resident as far as its conversation reaches)")
    }

    /// Which lineage a request resolved to, and why it could not continue one
    /// when it could not.
    ///
    /// The session is an actor, so generation is serialized: this line always
    /// belongs to the request between the preceding `generating` and the
    /// following `completed`. Carries a reason code only, never prompt content,
    /// and never the caller's `prompt_cache_key` — the slot index says which
    /// lineage without repeating the name the caller gave it.
    static func promptCacheResolved(slot: Int,
                                    slotCount: Int,
                                    occupied: Int,
                                    reason: ServerPromptCacheMissReason?) {
        write("prompt cache slot=\(slot) of \(slotCount) occupied=\(occupied) "
            + "outcome=\(reason?.rawValue ?? "hit")")
    }

    /// A request that asked not to participate in the cache.
    ///
    /// Reported alongside the occupancy it did not disturb, because that is the
    /// claim being made: opting out costs a prefill and costs nobody else their
    /// prefix. Without the line the two cases are indistinguishable in a log —
    /// a request that opted out and one that missed both show `cached=0`.
    static func promptCacheOptedOut(lineage: Int,
                                    slotCount: Int,
                                    occupied: Int) {
        write("prompt cache opted-out lineage=\(lineage) "
            + "slots=\(slotCount) occupied=\(occupied) outcome=not-cached")
    }

    static func visionPackInvalid(at url: URL, error: Error) {
        write("vision pack at \(url.path) is invalid: \(String(reflecting: error))")
    }

    static func visionRuntimeUnsupported(at url: URL, error: Error) {
        write("vision runtime for pack at \(url.path) is unsupported: "
            + String(reflecting: error))
    }

    static func visionRuntimeUnsupported() {
        write("vision runtime is unsupported: the image tower requires an M2 or newer Mac")
    }

    /// A prefix that could not be continued because its bridge failed to
    /// render. Carries the underlying error, because this miss means the
    /// template and the cached turn disagree, which no other miss does.
    static func promptCacheBridgeFailed(error: Error) {
        write("prompt cache miss reason=bridge-render-failed "
            + "error=\(String(reflecting: error))")
    }

    /// A request whose client stopped listening. Kept distinct from `failed`
    /// because a cancel is not a server fault and should not read as one, and
    /// added because excluding it from logging altogether left the request's last
    /// line as `generating` forever: an operator could not tell a running request
    /// from an abandoned one, or from a crashed one. Carries no content.
    static func cancelled(id: String, phase: String, duration: Duration) {
        write(cancelledMessage(id: id, phase: phase, duration: duration))
    }

    /// Split out so the wording can be checked without a log sink: this line
    /// must stay distinguishable from `failed`, must name the phase, and must
    /// carry no prompt or generated content.
    static func cancelledMessage(id: String, phase: String, duration: Duration) -> String {
        "request \(id) cancelled by client in \(format(duration)) phase=\(phase)"
    }

    static func failed(id: String,
                       phase: String,
                       status: UInt,
                       error: Error) {
        write("request \(id) failed phase=\(phase) status=\(status) "
            + "error=\(String(reflecting: error))")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return formatSeconds(seconds)
    }

    private static func formatSeconds(_ seconds: Double) -> String {
        String(format: "%.3fs", seconds)
    }

    private static func formatRate(_ rate: Double) -> String {
        String(format: "%.3f", rate)
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}
