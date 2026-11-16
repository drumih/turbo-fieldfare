import TinyTitanCore
import Foundation

package enum ServerLog {
    /// S32: requests log from concurrent tasks; serialize stderr writes so
    /// lines never interleave.
    private static let writeLock = NSLock()

    /// Where every line goes. `nil` is the default: stderr, which is what a
    /// server wants and what every line did before this existed.
    ///
    /// Process-wide on purpose. The orchestrator logs statically from deep
    /// inside the engine, and threading a logger through every call site would
    /// put a parameter on functions that have nothing to do with logging.
    /// `Engine.init` installs what `EngineConfiguration.logSink` asks for, so
    /// the most recently created engine owns the destination — stated on the
    /// configuration rather than implied, because a second engine silently
    /// inheriting the first one's sink would be a surprise.
    nonisolated(unsafe) private static var sink: (@Sendable (String) -> Void)?

    /// Install the destination for this library's log lines; `nil` restores
    /// stderr.
    ///
    /// The sink is called with the log lock held, so lines from concurrent
    /// requests cannot interleave. It must therefore not log through this
    /// library — that would deadlock, not recurse.
    package static func useSink(_ target: (@Sendable (String) -> Void)?) {
        writeLock.withLock { sink = target }
    }

    /// Hand one finished line (newline included) to the sink, or to stderr.
    private static func emit(_ text: String) {
        writeLock.withLock {
            if let sink {
                sink(text)
            } else {
                FileHandle.standardError.write(Data(text.utf8))
            }
        }
    }

    package static func accepted(id: String, streaming: Bool) {
        write("request \(id) accepted streaming=\(streaming)")
    }

    package static func queued(id: String) {
        write("request \(id) queued")
    }

    /// A request asked for a reasoning level the served model cannot render,
    /// and was answered with the nearest it can.
    ///
    /// Logged rather than returned: the whole point is that the agent keeps
    /// working, so this must not be an error, but an operator staring at a
    /// model that "ignores" `xhigh` deserves to see why. One line per
    /// request, and only when something actually changed.
    package static func reasoningFallback(id: String, notes: [String]) {
        for note in notes {
            write("request \(id) reasoning: \(note)")
        }
    }

    package static func generating(id: String) {
        write("request \(id) generating")
    }

    package static func completed(
        id: String,
        duration: Duration,
        completion: ServerCompletion
    ) {
        let usage = completion.usage
        write(
            "request \(id) completed in \(format(duration)) "
                + "prompt=\(usage.promptTokens) "
                + "cached=\(usage.promptTokensDetails.cachedTokens) "
                + "completion=\(usage.completionTokens) "
                + "finish=\(completion.finishReason)")
        for trip in completion.watchdogTrips {
            watchdog(id: id, trip: trip)
        }
        // The model thought although the request rendered with thinking off.
        // Worth a line because the consequence is not obvious: those tokens are
        // billed, and a client that caps `max_tokens` sees an empty answer
        // rather than a short one. Counted, never quoted -- generated text does
        // not belong in a log line.
        if completion.unrequestedReasoning > 0 {
            write(
                "request \(id) thinking off, but the model wrote "
                    + "\(completion.unrequestedReasoning) characters of reasoning; "
                    + "they are in reasoning_content, not content")
        }
    }

    package static func failed(
        id: String,
        phase: String,
        status: UInt,
        error: Error
    ) {
        write(
            "request \(id) failed phase=\(phase) status=\(status) "
                + "error=\(String(reflecting: error))")
    }

    /// Strip report for one request (TINYTITAN_STRIP_CLI_PROMPT on). The reminder
    /// and token counters are the early-warning signal: if a CLI changes its
    /// bloat template, "reminders=" drops to 0 or "prompt=" jumps back to the
    /// thousands, visible here without a model run.
    static func strip(stats: CLIStrip.Stats, promptTokens: Int) {
        write(
            "strip v\(CLIStrip.version) "
                + "system=\(stats.systemDropped) "
                + "developer=\(stats.developerDropped) "
                + "toolRole=\(stats.toolRoleDropped) "
                + "tools=\(stats.toolsDropped) "
                + "toolCalls=\(stats.toolCallsDropped) "
                + "reminders=\(stats.reminderCharsRemoved)chars "
                + "messageFallback=\(stats.emptyMessageFallbacks) "
                + "requestFallback=\(stats.emptyRequestFallback) "
                + "prompt=\(promptTokens)")
    }

    /// Model residency transitions under --lazy-load / --idle-unload-seconds.
    /// Always logged rather than hidden behind a debug env var: a server that
    /// silently dropped several GB is exactly what an operator needs to see in
    /// the log when a later request is unexpectedly slow.
    package static func residency(_ transition: String) {
        write("model \(transition)")
    }

    /// A `/v1/responses/compact` finished: which path produced the note and
    /// whether it fit.
    ///
    /// Worth a line because a compaction that quietly misses its budget, or one
    /// that fell back to trimming, is invisible in the caller's result — the
    /// window looks the same either way.
    package static func compacted(id: String, mode: String, noteTokens: Int, budget: Int) {
        write("request \(id) compacted mode=\(mode) note_tokens=\(noteTokens) budget=\(budget)")
    }

    /// `" prompt_cache=<mode>"` for a backend that can describe its cache, and
    /// nothing at all for one that cannot.
    ///
    /// The residency lines run when a model loads, so the mode they name has to
    /// come from the backend that just loaded: the server's own flag would
    /// report the previous model's cache after a switch between a GPU install
    /// and a CPU one, which has no cache to report.
    package static func promptCacheField(for backend: any ServerInferenceBackend) -> String {
        guard let described = backend as? any PromptCacheDescribing else { return "" }
        return " prompt_cache=\(described.promptCacheMode.rawValue)"
    }

    /// Memory subsystem events. Operational only: never a memory's contents,
    /// which can be anything the model chose to write.
    package static func memory(_ detail: String) {
        write("memory \(detail)")
    }

    /// The watchdog configuration, once, at startup. On stderr beside the
    /// memory line rather than in the launcher's banner: the launcher prints
    /// its own box and an operator reading a server log needs to know what
    /// was watching without reconstructing the environment.
    static func watchdogStartup(_ summary: String) {
        write(summary)
    }

    /// A watchdog trip. Operational only: what was repeated, or what was
    /// too short, is in the reply the user already has, and generated text
    /// does not belong in a log line any more than a memory's contents do.
    static func watchdog(id: String, trip: WatchdogSet.Trip) {
        write(
            "request \(id) watchdog \(trip.kind.rawValue) "
                + "\(trip.acted ? "stopped" : "observed") \(trip.message)")
    }

    /// One engine diagnostic line, without the log's timestamp prefix.
    ///
    /// The kit's operational telemetry (`TinyTitan generation …`, the RAM
    /// profile, the prompt-cache lines, the kernel/runner stage splits) is
    /// parsed out of a *merged* server log by `benchmark/*.py`, so its text is
    /// a contract. What is not a contract is the stream: a library must not
    /// write to stdout, which belongs to the program that embedded it, so every
    /// one of these goes to stderr byte-for-byte as it was printed before —
    /// unless the embedder installed a sink, which is the one thing that may
    /// redirect them.
    package static func diagnostic(_ message: String) {
        emit(message + "\n")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds =
            Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.3fs", seconds)
    }

    private static func write(_ message: String) {
        emit("[\(Date().formatted(.iso8601))] \(message)\n")
    }
}
