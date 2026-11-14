import Foundation
import TinyTitan
import TinyTitanCore

// Stored responses and the object a response echoes back.
//
// Split out of `ResponsesAPIModels.swift` (2026-09-28) under the 500-line-per-
// file rule (Task 8 of the cleanup runbook) as pure code motion; the store's
// `private` state moved with the class, so no access widened.
// MARK: - Stored responses

/// Finished responses kept for `previous_response_id`, `GET`, `DELETE` and
/// `input_items`. Bounded and in memory: this is the API's storage contract
/// for a single-user local server, not a database. The oldest entry goes
/// when the cap is reached.
/// unchecked-invariant: every field is guarded by `lock`.
public final class ResponseStore: @unchecked Sendable {
    public struct Entry: Sendable {
        /// The response object as returned to the client, JSON-encoded.
        public let responseJSON: Data
        /// The conversation the response was generated from, as input items
        /// (a prior chain already flattened in).
        public let inputItems: [ResponsesAPIRequest.Item]
        /// The response's output, as the input items a follow-up carries.
        public let outputItems: [ResponsesAPIRequest.Item]
        public let created: Date

        /// What a request naming this response as `previous_response_id`
        /// continues from.
        public var conversation: [ResponsesAPIRequest.Item] { inputItems + outputItems }
    }

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    private var order: [String] = []
    public let capacity: Int

    public init(capacity: Int = 256) {
        self.capacity = capacity
    }

    public func put(id: String, entry: Entry) {
        lock.withLock {
            // A re-put replaces the entry *and* refreshes its place in the
            // eviction order. Leaving a replaced response at its first insertion
            // point evicts the response the caller just wrote -- the newest thing
            // in the store -- while keeping whatever it replaced, which is the
            // opposite of "the oldest entry goes".
            if let existing = order.firstIndex(of: id) { order.remove(at: existing) }
            order.append(id)
            entries[id] = entry
            while order.count > capacity, let oldest = order.first {
                order.removeFirst()
                entries.removeValue(forKey: oldest)
            }
        }
    }

    public func get(_ id: String) -> Entry? {
        lock.withLock { entries[id] }
    }

    @discardableResult
    public func delete(_ id: String) -> Bool {
        lock.withLock {
            guard entries.removeValue(forKey: id) != nil else { return false }
            order.removeAll { $0 == id }
            return true
        }
    }

    public var count: Int { lock.withLock { entries.count } }

    /// An item of any stored conversation by its id, for `item_reference`.
    public func item(withID id: String) -> ResponsesAPIRequest.Item? {
        lock.withLock {
            for entry in entries.values {
                if let item = (entry.outputItems + entry.inputItems).first(where: { $0.id == id }) {
                    return item
                }
            }
            return nil
        }
    }
}

// MARK: - Response object builders

/// Everything a response object echoes back: the request's own fields, plus the
/// sampling the validator resolved from them and the served model's profile.
/// Built once per request so the in_progress, completed and stored objects agree.
public struct ResponsesAPIEcho: Sendable {
    public let instructions: String?
    public let maxOutputTokens: Int?
    public let maxToolCalls: Int?
    public let previousResponseID: String?
    public let metadata: JSONValue
    public let user: String?
    public let safetyIdentifier: String?
    public let promptCacheKey: String?
    public let serviceTier: String
    public let truncation: String
    /// The sampling the server actually ran, resolved by the validator from the
    /// request where the client named a value and from the served model's own
    /// profile where it did not. These are numbers, never nil: the Response
    /// schema requires them, and a client cannot otherwise see what was applied.
    /// The *request-side* fields stay nil when the client named none (C11), so
    /// the profile still supplies the value here rather than the echo doing it.
    public let temperature: Float
    public let topP: Float
    public let presencePenalty: Float
    public let frequencyPenalty: Float
    public let store: Bool
    public let tools: [ResponsesAPIRequest.Tool]
    public let namespaces: [String: String]
    public let toolChoice: JSONValue
    public let textVerbosity: String
    public let reasoningEffort: String?
    public let reasoningSummary: String?
    public let parallelToolCalls: Bool

    public init(
        request: ResponsesAPIRequest,
        effectiveEffort: ModelReasoningEffort?,
        applied: GenerationConfig
    ) {
        instructions = request.instructions
        maxOutputTokens = request.maxOutputTokens
        maxToolCalls = request.maxToolCalls
        previousResponseID = request.previousResponseID
        metadata = request.metadata ?? .object([:])
        user = request.user
        safetyIdentifier = request.safetyIdentifier
        promptCacheKey = request.promptCacheKey
        serviceTier = "default"
        truncation = request.truncation ?? "disabled"
        // Echo what the sampler was handed, not the raw request. The mapper left
        // an omitted field nil so validation could resolve it; by this point the
        // validator has, exactly as the generation config below shows.
        temperature = applied.temperature
        topP = applied.topP ?? GenerationDefaults.topP
        presencePenalty = applied.presencePenalty
        // frequency_penalty has no runtime knob: the validator admits only the
        // neutral value, so the number this server applied is always zero. The
        // schema requires the field, so it is echoed as that zero.
        frequencyPenalty = 0
        store = request.stores
        tools = request.tools ?? []
        namespaces = ResponsesAPIMapper.functionTools(request.tools).namespaces
        toolChoice = request.toolChoice ?? .string("auto")
        textVerbosity = request.text?.verbosity ?? "medium"
        reasoningEffort = request.reasoning?.effort ?? effectiveEffort?.rawValue
        reasoningSummary = request.reasoning?.summary
        parallelToolCalls = request.parallelToolCalls ?? true
    }
}
