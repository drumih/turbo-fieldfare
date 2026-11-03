// The server's inference API: events, completion, and the backend protocols.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import TinyTitan

/// A model the server was asked to load but this build cannot run.
enum ServerInferenceError: Error, CustomStringConvertible {
    case unsupportedModel(String)

    var description: String {
        switch self {
        case .unsupportedModel(let detail): return detail
        }
    }
}

/// Prefill chunk size when the caller has not asked for one. Both families
/// with a long-chunk default want it for the same reason -- routed experts are
/// what prefill spends itself on, and a longer chunk amortizes them.
package func defaultPrefillChunkTokens(family: ModelFamily, fallback: Int) -> Int {
    switch family {
    case .qwen36: return RuntimeConfiguration.qwenLongPrefillChunkTokens
    // 4,096 for Qwen3.8 too. Prefill's expert cache is inert -- a chunk routes
    // essentially every expert in a layer against 96 slots, so the hit rate is
    // 0.6% and each chunk re-streams what the last evicted. The cost is
    // therefore proportional to the chunk *count*: an 8k prompt is 5 chunks at
    // 2,048 and 3 at 4,096, measured at 167.5 -> 111.0 GiB of expert reads and
    // 506.4 -> 450.5 s of prefill (-11%), with identical output on a
    // multi-chunk prompt. The 2,048 here predates that measurement.
    //
    // It is a trade, not free. A/B/A on one machine state, 0.4% drift
    // between the repeated arms: decode 6.97 / 7.19 / 7.00 tok/s at
    // 4096 / 2048 / 4096, so 2,048 decodes ~3% faster -- the KV ring is
    // sized from the chunk and this machine feels the reservation.
    // 4,096 still wins for the long-prompt case it is chosen for: 56 s
    // of prefill on a 10k prompt against ~2 s of a 512-token generation.
    // A short-prompt, long-generation workload would want 2,048 back.
    case .qwen38flash, .qwen4flash:
        return RuntimeConfiguration.qwenLongPrefillChunkTokens
    // The dense Qwen 3.5 family wants the long chunk as well, and for a second
    // reason: the ANE prefill sidecar is a fixed 4,096-token program, and
    // `eligibleChunk` routes a chunk to it only when the configured chunk is
    // exactly that size — so a family left on the 128 default can never reach
    // the ANE at all, which is how the dense installs saw no ANE prefill
    // despite shipping a default-on switch. Measured on the dense 2B: chunk size
    // does not change the GPU path's output (byte-identical greedy text at 128
    // and at 4,096) and prefill time is flat, so this is a scheduling choice
    // that makes the ANE reachable, not a numerics change.
    //
    // The CLI's own switch did this before the library took the path over; when
    // the switch moved here the case did not come with it, and dense Qwen 3.5
    // quietly dropped to 128. That was a regression, not a preference.
    case .qwen35Dense: return RuntimeConfiguration.qwenLongPrefillChunkTokens
    default: return fallback
    }
}

package enum ServerInferenceEvent: Equatable, Sendable {
    /// Emitted once per generation, when the prompt has been read: `tokens` is
    /// its length and `cachedTokens` how much of it came from the prompt cache
    /// instead of being prefilled. A caller uses it to tell "still prefilling"
    /// from "content is coming". The HTTP surfaces ignore it — their clients
    /// count prompt tokens from the usage block instead.
    case promptProcessed(tokens: Int, cachedTokens: Int)
    /// Opt-in reading progress: `done` of `total` prompt tokens are prefilled.
    /// Emitted only for a request that set `return_progress`, once per chunk,
    /// before the first content event.
    case promptProgress(done: Int, total: Int)
    case content(String)
    /// Thought text from inside the model's `<think>` block. Kept apart from
    /// `content` so each surface can put it where its clients look for
    /// reasoning, and so nothing that judges the answer ever reads it.
    case reasoning(String)
    case toolCall(ParsedToolCall)
}

package struct ServerCompletion: Equatable, Sendable {
    package let content: String
    /// Everything the model thought, in order; empty with thinking off.
    /// `usage.completionTokens` already counts these tokens, as it always
    /// has -- only where the text goes has changed.
    package let reasoning: String
    /// Characters of `reasoning` the model wrote although this request's
    /// render had thinking off.
    ///
    /// Not a fault of the runtime -- some installs think with the switch off,
    /// measured on Qwen AgentWorld 35B-A3B 8-bit -- but not something to leave
    /// silent either: the operator asked for no thought, paid tokens for one,
    /// and a client that caps tokens gets an empty answer rather than a short
    /// one. Carried as a count, like the watchdog trips, so the HTTP layer can
    /// log it where generated text does not belong.
    package let unrequestedReasoning: Int
    package let toolCalls: [ParsedToolCall]
    package let finishReason: String
    package let usage: OpenAIUsage
    /// What the watchdogs saw, empty when they are off. Carried on the
    /// completion so the HTTP layer, which owns the request id, can log them
    /// on the one line that already reports how the request ended.
    package let watchdogTrips: [WatchdogSet.Trip]
    /// The client stop string that ended generation, when one did. OpenAI
    /// folds this into finish_reason "stop"; the Anthropic Messages API
    /// distinguishes it as stop_reason "stop_sequence" and names the string.
    package let stopSequence: String?
    /// Wall time the engine spent prefilling, and decoding. Zero when the
    /// backend does not report them.
    package let prefillSeconds: Double
    package let decodeSeconds: Double
    /// The decode loop's own stop reason, before the wire's finish_reason
    /// mapping flattens it.
    package let engineStopReason: StopReason?

    package init(
        content: String,
        toolCalls: [ParsedToolCall],
        finishReason: String,
        usage: OpenAIUsage,
        watchdogTrips: [WatchdogSet.Trip] = [],
        stopSequence: String? = nil,
        reasoning: String = "",
        unrequestedReasoning: Int = 0,
        prefillSeconds: Double = 0,
        decodeSeconds: Double = 0,
        engineStopReason: StopReason? = nil
    ) {
        self.content = content
        self.reasoning = reasoning
        self.unrequestedReasoning = unrequestedReasoning
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
        self.watchdogTrips = watchdogTrips
        self.stopSequence = stopSequence
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.engineStopReason = engineStopReason
    }
}

/// A backend that can count the prompt tokens a request would occupy
/// without generating. Kept apart from `ServerInferenceBackend` so wrappers
/// and test doubles that cannot count are not forced to pretend; the
/// Anthropic `count_tokens` endpoint answers 501 when the backend lacks it.
package protocol PromptTokenCounting: Sendable {
    func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int
}

/// A backend that can say which prompt-cache mode it is really running.
///
/// Kept apart from `ServerInferenceBackend` for the same reason
/// `PromptTokenCounting` is: an engine with no prompt cache (the CPU backend)
/// and the test doubles should not have to answer for a cache they do not have.
/// One resident slot serves a whole catalog, so a residency line that named the
/// mode from the server's flags rather than from the backend that just loaded
/// would report the previous model's cache after a switch.
package protocol PromptCacheDescribing: Sendable {
    /// The mode in force for this backend, never the one requested.
    var promptCacheMode: ServerPromptCacheMode { get }
}

package protocol ServerInferenceBackend: Sendable {
    /// The backend's configured context window, used to validate
    /// max_tokens/max_completion_tokens against the session's maxContext (S11).
    var maximumContext: Int { get }
    /// Sampling values used for whatever the request omits. A family whose
    /// model card differs from the house settings reports its own here, so a
    /// client that sends no temperature gets what the model was tuned for
    /// rather than what the last family to need tuning wanted.
    var samplingDefaults: GenerationDefaults.Sampling { get }
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion
}

extension ServerInferenceBackend {
    package var maximumContext: Int {
        RuntimeConfiguration.nativeMaximumContextTokens
    }
    package var samplingDefaults: GenerationDefaults.Sampling { GenerationDefaults.house }
}

/// A backend that owns the model's residency and can release it on demand.
///
/// Kept separate from `ServerInferenceBackend` rather than added to it with a
/// `false`-returning default: exactly one backend manages residency, and the
/// wrapper design exists so the HTTP layer stays unaware of loading at all.
/// Folding it into the inference protocol would make every conforming type —
/// including the plain session and every test stub — carry a member that only
/// answers "not me".
package protocol ResidencyManaging: Sendable {
    /// Releases the model's memory, waiting for in-flight requests to drain
    /// first. Returns true when a resident model was actually released.
    func unload() async -> Bool
}

/// Whether a client generation is running, readable without awaiting the
/// coordinator.
///
/// The CPU side-engine needs this before every token it produces, from
/// whatever thread it happens to be on, and `await`ing an actor to decide
/// how wide to run a GEMV would cost more than the decision is worth.
///
/// unchecked-invariant: `depth` is only ever read or written under `lock`.
package final class GenerationSignal: @unchecked Sendable {
    let lock = NSLock()
    var depth = 0

    package init() {}

    /// True while at least one client generation is in flight.
    package var isBusy: Bool { lock.withLock { depth > 0 } }

    package func enter() { lock.withLock { depth += 1 } }
    package func leave() { lock.withLock { depth = max(0, depth - 1) } }
}
