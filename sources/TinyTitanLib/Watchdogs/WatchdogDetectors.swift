import Foundation
import TinyTitan
import TinyTitanCore

/// A phrase repeating in the output.
///
/// The measured failure: Qwen 0.8B emitted "Wait, I need to check the facts
/// again" over and over until the 600-token budget ended the request, and
/// the user paid for every token of it.
///
/// The implementation is a rolling hash over a fixed window, so the cost per
/// byte is constant and does not grow with the output (B5). A naive
/// substring search over the accumulated text would be quadratic, and would
/// therefore be slowest on exactly the long generations most in need of
/// watching.
///
/// Three rules keep it honest on ordinary text:
///
///   * repeats must be **far enough apart**. Two windows eight bytes apart
///     are one run of something, not two occurrences of it.
///   * the window must be **varied enough**. A table rule `|---|---|---|`,
///     a line of dashes and a block of indentation all repeat perfectly and
///     none of them is a loop, so a window is only considered once it holds
///     at least twelve distinct bytes -- ordinary English holds around
///     twenty in forty, and so does code.
///   * repeats must fall inside a **recent history**, so a phrase that
///     recurs naturally across a long document (a heading, a variable name)
///     ages out rather than accumulating.
///
/// The thresholds are calibrated, not guessed: `benchmark/watchdog_calibrate.py`
/// runs this over every recorded reply and the count of false positives is
/// asserted to be zero.
package struct LoopWatchdog: Watchdog {
    package static let kind = WatchdogKind.loop

    private struct Occurrence {
        var first: Int
        var last: Int
        var count: Int
    }

    /// Closer than this and the two windows are one run, not a repeat.
    static let minimumPeriodBytes = 8
    /// Distinct bytes a window must hold to be worth counting. Twelve of
    /// forty admits prose and code and rejects rules, borders and runs.
    static let minimumDistinctBytes = 12

    private let window: Int
    private let history: Int
    private let repeats: Int
    /// base^window, for removing the byte that leaves the window.
    private let leavingFactor: UInt64
    private var ring: [UInt8]
    private var hash: UInt64 = 0
    private var index = 0
    private var seen: [UInt64: Occurrence] = [:]
    /// Occupancy of each byte value inside the window, kept incrementally so
    /// the variety test costs the same whatever the window length (B5).
    private var byteCounts = [Int](repeating: 0, count: 256)
    private var distinct = 0
    private var tripped = false

    private static let base: UInt64 = 1_000_003

    package init(configuration: WatchdogConfiguration = .off) {
        window = configuration.loopWindowBytes
        history = configuration.loopHistoryBytes
        repeats = configuration.loopRepeats
        ring = [UInt8](repeating: 0, count: window)
        var factor: UInt64 = 1
        for _ in 0..<window { factor = factor &* Self.base }
        leavingFactor = factor
    }

    package mutating func observe(
        _ chunk: String,
        at instant: ContinuousClock.Instant
    ) -> WatchdogVerdict {
        guard !tripped else { return .fine }
        for byte in chunk.utf8 {
            if let verdict = push(byte) { return verdict }
        }
        return .fine
    }

    private mutating func push(_ byte: UInt8) -> WatchdogVerdict? {
        let slot = index % window
        // A byte only leaves once the ring has been round once. Reading the
        // slot before then would take an initial zero for a real byte and
        // corrupt both the hash and the occupancy count.
        let displacesAByte = index >= window
        let leaving = ring[slot]
        ring[slot] = byte
        if byteCounts[Int(byte)] == 0 { distinct += 1 }
        byteCounts[Int(byte)] += 1
        hash = hash &* Self.base &+ UInt64(byte)
        if displacesAByte {
            hash = hash &- UInt64(leaving) &* leavingFactor
            byteCounts[Int(leaving)] -= 1
            if byteCounts[Int(leaving)] == 0 { distinct -= 1 }
        }
        index += 1
        // `index` is now one past the window's last byte, which is the
        // position the non-overlap and history arithmetic is written in.
        guard index >= window else { return nil }
        guard distinct >= Self.minimumDistinctBytes else { return nil }
        guard var entry = seen[hash] else {
            seen[hash] = Occurrence(first: index, last: index, count: 1)
            pruneIfCrowded()
            return nil
        }
        if index - entry.first > history {
            // Too old to be part of the same loop; start counting again.
            seen[hash] = Occurrence(first: index, last: index, count: 1)
            return nil
        }
        guard index - entry.last >= Self.minimumPeriodBytes else { return nil }
        entry.last = index
        entry.count += 1
        seen[hash] = entry
        guard entry.count >= repeats else { return nil }
        tripped = true
        let period = (entry.last - entry.first) / max(1, entry.count - 1)
        // Deliberately no excerpt: server logs carry operational facts, never
        // generated or remembered content. What was repeated is in the reply
        // the user already has.
        return .concern(
            "a \(window)-byte window repeated \(entry.count) times, "
                + "period \(period) bytes")
    }

    /// The dictionary can only hold as many live windows as the history, so
    /// dropping the stale ones on the rare crowded push keeps the memory
    /// bounded and the amortized cost constant.
    private mutating func pruneIfCrowded() {
        guard seen.count > history * 2 else { return }
        let cutoff = index - history
        seen = seen.filter { $0.value.last >= cutoff }
    }
}

/// A generation that has stopped producing.
///
/// B1: the clock starts at the **first emitted token**, never at the
/// request. Prefill emits nothing, and a measured 10k-token prompt on this
/// project spent 652 s in it; a watchdog started at the request would stop
/// every long prompt, which is worse than no watchdog at all.
///
/// Reported through `check`, not `observe`, because the evidence for a stall
/// is the absence of a call.
package struct StallWatchdog: Watchdog {
    package static let kind = WatchdogKind.stall

    private let threshold: Duration
    private let seconds: Double
    private var lastToken: ContinuousClock.Instant?
    private var tripped = false

    package init(configuration: WatchdogConfiguration = .off) {
        seconds = configuration.stallSeconds
        threshold = .seconds(configuration.stallSeconds)
    }

    package mutating func observe(
        _ chunk: String,
        at instant: ContinuousClock.Instant
    ) -> WatchdogVerdict {
        if !chunk.isEmpty { lastToken = instant }
        return .fine
    }

    package mutating func check(at instant: ContinuousClock.Instant) -> WatchdogVerdict {
        guard !tripped, let lastToken else { return .fine }
        guard instant - lastToken >= threshold else { return .fine }
        tripped = true
        return .concern("no visible token for \(Int(seconds))s")
    }
}

/// A reply that finished normally with essentially nothing in it.
///
/// Two of these are on record: Ornith returning forty tokens of empty
/// `<tool_call>` markup where a C99 program belonged, and a book session
/// returning 151 tokens where ten chapters did.
///
/// Only the first is caught here, deliberately. A short answer to a long
/// request is wrong but not malformed, and telling the two apart needs to
/// know what was asked -- a judgement, and the shadow's job, not this one's.
/// Reaching further from here would be the first source of false positives.
///
/// The rule is written in bytes rather than the tokens the plan proposed.
/// The visible content is a string at this point and its token count is not
/// known without re-encoding it, while a reasoning-heavy model can burn
/// hundreds of generated tokens and emit nothing -- so a token count would
/// measure the wrong thing anyway. Ninety-six bytes is about twenty-four
/// tokens of English.
package struct StubWatchdog: Watchdog {
    package static let kind = WatchdogKind.stub

    private let threshold: Int
    private let asked: Int

    package init(configuration: WatchdogConfiguration = .off) {
        threshold = configuration.stubVisibleBytes
        asked = configuration.stubAskedBytes
    }

    package mutating func finish(
        visibleBytes: Int,
        requestBytes: Int,
        finishReason: String
    ) -> WatchdogVerdict {
        // `tool_calls` is a real answer in a tool loop and `length` already
        // tells the client what happened. Only a normal stop can be a stub.
        guard finishReason == "stop", visibleBytes < threshold else { return .fine }
        // And something has to have been asked for. Judged without the
        // request, this rule calls "OK." a failure.
        guard requestBytes >= asked else { return .fine }
        return .concern(
            "finished normally with \(visibleBytes) visible bytes "
                + "for a \(requestBytes)-byte request")
    }
}

/// The same tool called with the same arguments, over and over.
///
/// B2: this one does not watch the output stream. TinyTitan returns a tool call
/// to the client and the repeat appears in the *next* request's history, so
/// the evidence is in the incoming messages and the check belongs beside the
/// existing unresolved-tool-call validation.
///
/// It is a pure function rather than a `Watchdog`: there is no stream and no
/// state to carry, and a request is inspected once.
package enum PingPongWatchdog {
    package static let kind = WatchdogKind.pingpong

    /// The rule is a **consecutive** run, not a tally.
    ///
    /// A long agent session legitimately reads the same file three times an
    /// hour apart, and counting every identical call in the history would
    /// call that a loop. What is never right is the same tool called with
    /// the same arguments three times in a row with nothing else in
    /// between: the model is not getting what it needs and is asking again
    /// identically, which is the failure this watches for.
    package static func inspect(
        _ messages: [GFTokenizer.Message],
        configuration: WatchdogConfiguration
    ) -> WatchdogVerdict {
        guard configuration.isEnabled else { return .fine }
        var previous: String?
        var run = 0
        var worst: (name: String, count: Int)?
        for message in messages {
            if message.toolCalls.isEmpty {
                // Anything that is not a tool call breaks the run, except the
                // tool results that answer one: a user turn, or the
                // assistant writing prose, means the conversation moved on.
                // Counting only within the tool-call subsequence would treat
                // three reads an hour apart as consecutive, which is the
                // false positive this rule exists to avoid.
                if message.role != .tool {
                    previous = nil
                    run = 0
                }
                continue
            }
            for call in message.toolCalls {
                let signature = "\(call.name)\u{1}\(canonical(call.arguments))"
                run = signature == previous ? run + 1 : 1
                previous = signature
                if run > (worst?.count ?? 0) { worst = (call.name, run) }
            }
        }
        guard let worst, worst.count >= configuration.pingPongRepeats else { return .fine }
        return .concern(
            "tool \(worst.name) called \(worst.count) times in a row "
                + "with identical arguments")
    }

    /// Sorted keys, so two calls that differ only in the order the client
    /// serialized them are recognised as the same call.
    private static func canonical(_ arguments: JSONValue) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(arguments) else { return "?" }
        return data.lossyUTF8String
    }
}
