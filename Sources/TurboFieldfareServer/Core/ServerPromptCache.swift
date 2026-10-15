import Foundation
import TurboFieldfare

public enum ServerPromptCacheMode: String, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
}

struct ServerPromptCacheDomain: Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Sendable, Equatable {
    let message: GFTokenizer.Message
    let rawStopReason: StopReason
}

struct ServerPromptCacheEntry: Sendable, Equatable {
    let domain: ServerPromptCacheDomain
    let inputMessages: [GFTokenizer.Message]
    /// Per-message image content hashes for `inputMessages`. Plain message
    /// equality cannot see images - image parts contribute nothing to a
    /// message's flattened text - so two prompts differing only in their image
    /// would compare equal without this.
    let inputImageIdentities: [[String]]
    let tools: [GFTokenizer.FunctionDefinition]
    let assistantTurn: CachedAssistantTurn
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int
}

/// Why a prefix could not be reused. Carried so a miss is diagnosable rather
/// than indistinguishable from every other miss: a bridge that fails to render
/// is a defect worth seeing, and a history that simply diverged is not.
enum ServerPromptCacheMissReason: String, Sendable, Equatable {
    case noEntry = "no-entry"
    case unusableEntry = "unusable-entry"
    case historyDiverged = "history-diverged"
    case unsupportedContinuation = "unsupported-continuation"
    case boundaryMismatch = "boundary-mismatch"
    case bridgeRenderFailed = "bridge-render-failed"
    case missingImageIdentity = "missing-image-identity"
    case imagesDiverged = "images-diverged"
}

/// Which retained lineage a request resolved to, and what it can do with it.
///
/// A slot is always named, hit or miss: a miss still has to prefill somewhere,
/// and where it prefills is what it overwrites.
struct ServerPromptCacheResolution: Sendable, Equatable {
    let slot: Int
    let match: ServerPromptCacheMatch

    var missReason: ServerPromptCacheMissReason? { match.missReason }
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss(ServerPromptCacheMissReason)
    case hit(effectivePromptIDs: [Int32], cachedPromptTokens: Int)
    /// The history and its images match, but the new turn carries its own
    /// image, so the effective prompt can only be produced by rendering. The
    /// caller renders just that turn as a continuation on the cached tokens;
    /// history is never re-rendered, because the KV holds tokens the model
    /// generated and a fresh render re-tokenises that assistant text.
    case renderThenResume(cachedPromptTokens: Int)

    static let miss: ServerPromptCacheMatch = .miss(.noEntry)

    var missReason: ServerPromptCacheMissReason? {
        if case .miss(let reason) = self { return reason }
        return nil
    }
}

struct ServerPromptCache: Sendable {
    /// One retained lineage.
    ///
    /// `key` is the client-supplied `prompt_cache_key` this slot answers for.
    /// It governs eviction only: a keyed request looks at its own slot and
    /// nowhere else, and unkeyed traffic looks only at slots no key has
    /// claimed. So a caller that names its conversation cannot have it thrown
    /// away by a caller that does not — which is the whole difference between
    /// two clients sharing a server and two clients fighting over it.
    struct Slot: Sendable {
        var entry: ServerPromptCacheEntry?
        var key: String?
        /// Tick of the last request to use this slot; the LRU order.
        var lastUsed: UInt64 = 0
    }

    private(set) var slots: [Slot]
    private var clock: UInt64 = 0

    init(slotCount: Int = 1) {
        precondition(slotCount > 0, "slotCount must be positive")
        self.slots = Array(repeating: Slot(), count: slotCount)
    }

    var slotCount: Int { slots.count }

    /// The historical single-prefix view: slot 0. Kept because a one-slot cache
    /// is exactly what this was before slots existed, and the suite that proves
    /// so reads through here.
    var entry: ServerPromptCacheEntry? { slots[0].entry }

    func kvBackedTokenIDs(slot: Int) -> [Int32]? { slots[slot].entry?.kvBackedTokenIDs }
    func inputMessageCount(slot: Int) -> Int? { slots[slot].entry?.inputMessages.count }
    func key(slot: Int) -> String? { slots[slot].key }

    /// How many slots hold a reusable prefix. Reported at completion so an
    /// operator can see retention working without reading token counts.
    var occupiedSlotCount: Int { slots.count(where: { $0.entry != nil }) }

    mutating func invalidate(slot: Int) {
        slots[slot].entry = nil
    }

    /// The slot a request runs in, and whether that slot can continue it.
    ///
    /// Selection never decides correctness. Every hit returned here has been
    /// verified by `match` against that entry's own domain, tools, history and
    /// images, so choosing the wrong slot can cost a prefill and can never
    /// serve a prefix the caller did not send. That is what makes guessing by
    /// recency safe.
    mutating func resolve(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32]?,
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheResolution {
        clock &+= 1
        let candidates = candidateSlots(for: request.promptCacheKey)
        var reported: ServerPromptCacheMissReason?
        for slot in candidates where slots[slot].entry != nil {
            let match = match(
                slot: slot,
                domain: domain,
                request: request,
                renderedPromptIDs: renderedPromptIDs,
                tokenizer: tokenizer)
            guard let reason = match.missReason else {
                slots[slot].lastUsed = clock
                return ServerPromptCacheResolution(slot: slot, match: match)
            }
            // Searching several slots produces several refusals, and reporting
            // whichever came last would bury the informative one: every slot
            // holding an unrelated conversation says `historyDiverged`, so a
            // bridge that failed to render on the slot the request actually
            // belongs to would never be seen. Keep the most specific.
            if let current = reported {
                if Self.diagnosticRank(reason) > Self.diagnosticRank(current) {
                    reported = reason
                }
            } else {
                reported = reason
            }
        }
        let victim = victimSlot(for: request.promptCacheKey)
        slots[victim].lastUsed = clock
        return ServerPromptCacheResolution(
            slot: victim,
            match: .miss(reported ?? .noEntry))
    }

    /// How much a miss reason says. `noEntry` and `historyDiverged` are what
    /// every slot holding somebody else's conversation answers; the rest name
    /// something specific about this request or this prefix, and are worth
    /// more than a truthful shrug from the slot next door.
    private static func diagnosticRank(_ reason: ServerPromptCacheMissReason) -> Int {
        switch reason {
        case .noEntry: 0
        case .historyDiverged: 1
        case .unusableEntry: 2
        default: 3
        }
    }

    /// Which slots a request may reuse, most recently used first, ties broken
    /// by index so the search order never depends on how a sort happened to
    /// arrange equal keys.
    ///
    /// A keyed request sees the slot bearing its key and nothing else: the key
    /// names a lineage, so there is no question of which one it meant. An
    /// unkeyed request sees every slot no key has claimed.
    private func candidateSlots(for key: String?) -> [Int] {
        let eligible: [Int]
        if let key {
            eligible = slots.indices.filter { slots[$0].key == key }
        } else {
            eligible = slots.indices.filter { slots[$0].key == nil }
        }
        return eligible.sorted {
            slots[$0].lastUsed == slots[$1].lastUsed
                ? $0 < $1
                : slots[$0].lastUsed > slots[$1].lastUsed
        }
    }

    /// The slot a miss overwrites.
    ///
    /// A key claims its own slot and keeps it. Where it has none yet, and for
    /// unkeyed traffic, an empty slot is taken before an occupied one and the
    /// least recently used before the rest. A key prefers to take from the
    /// unkeyed pool before it takes from another key, so two named
    /// conversations do not evict each other while unnamed traffic sits on a
    /// slot.
    private func victimSlot(for key: String?) -> Int {
        if let key, let claimed = slots.indices.first(where: { slots[$0].key == key }) {
            return claimed
        }
        let unkeyed = slots.indices.filter { slots[$0].key == nil }
        // An unkeyed request never takes a named slot while any unnamed one is
        // free to take; with every slot named it falls back to the whole pool,
        // because a request still has to prefill somewhere.
        let preferred = unkeyed.isEmpty ? Array(slots.indices) : unkeyed
        if let empty = preferred.first(where: { slots[$0].entry == nil }) {
            return empty
        }
        return preferred.min { slots[$0].lastUsed < slots[$1].lastUsed } ?? 0
    }

    /// Per-message image identity, or nil when a multimodal request did not
    /// carry one entry per message. Nil is fail-closed: without identity an
    /// image-bearing prompt would be indistinguishable from one carrying a
    /// different image, so it must neither publish nor match.
    static func identities(for request: ValidatedChatRequest) -> [[String]]? {
        let count = request.messages.count
        if request.multimodalMessages == nil, request.imageIdentities.isEmpty {
            return Array(repeating: [], count: count)
        }
        guard request.imageIdentities.count == count else { return nil }
        return request.imageIdentities
    }

    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool = false
    ) {
        publish(slot: 0,
                domain: domain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopStringFiltered)
    }

    /// Records what the completion in `slot` left behind, and binds the slot to
    /// the request's key.
    ///
    /// The key is taken even when the turn itself is unpublishable: the
    /// completion has already overwritten that lineage's KV, so the slot is
    /// spent either way, and leaving it unclaimed would let the next unkeyed
    /// request take the slot this caller is in the middle of using.
    mutating func publish(
        slot: Int,
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool = false
    ) {
        slots[slot].key = request.promptCacheKey
        publishEntry(
            slot: slot,
            domain: domain,
            request: request,
            content: content,
            calls: calls,
            result: result,
            stopStringFiltered: stopStringFiltered)
    }

    private mutating func publishEntry(
        slot: Int,
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool
    ) {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            slots[slot].entry = nil
            return
        }
        let historicalCalls = calls.map {
            GFTokenizer.HistoricalToolCall(
                id: $0.id,
                name: $0.name,
                arguments: $0.arguments)
        }
        let assistant = GFTokenizer.Message(
            role: .assistant,
            content: calls.isEmpty ? content : nil,
            toolCalls: historicalCalls)
        guard let identities = Self.identities(for: request) else {
            slots[slot].entry = nil
            return
        }
        slots[slot].entry = ServerPromptCacheEntry(
            domain: domain,
            inputMessages: request.messages,
            inputImageIdentities: identities,
            tools: request.tools,
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition)
    }

    /// `renderedPromptIDs` is nil for multimodal requests, where the rendered
    /// ids cannot establish identity: image placeholder tokens are the same
    /// token repeated whichever image they stand for.
    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32]?,
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        match(slot: 0,
              domain: domain,
              request: request,
              renderedPromptIDs: renderedPromptIDs,
              tokenizer: tokenizer)
    }

    /// Whether the prefix in one slot can serve this request. Which slots are
    /// asked, and in what order, is `resolve`'s business; this answers only for
    /// the one named, and answers on content alone.
    func match(
        slot: Int,
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32]?,
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        guard let entry = slots[slot].entry,
              entry.domain == domain,
              entry.tools == request.tools,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return .miss(.unusableEntry)
        }

        guard let requestIdentities = Self.identities(for: request) else {
            return .miss(.missingImageIdentity)
        }

        // Rendered ids cannot identify an image, so the cheap prefix path is
        // refused whenever either side carries one. Keeping that decision here
        // rather than in the caller means the invariant holds for every caller.
        if let renderedPromptIDs,
           requestIdentities.allSatisfy(\.isEmpty),
           entry.inputImageIdentities.allSatisfy(\.isEmpty),
           renderedPromptIDs.count > entry.kvPosition,
           renderedPromptIDs.prefix(entry.kvPosition)
            .elementsEqual(entry.kvBackedTokenIDs) {
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: entry.kvPosition)
        }

        let inputCount = entry.inputMessages.count
        guard request.messages.count > inputCount + 1,
              request.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
              assistantMatches(
                request.messages[inputCount],
                entry.assistantTurn.message) else {
            return .miss(.historyDiverged)
        }
        guard requestIdentities.prefix(inputCount)
            .elementsEqual(entry.inputImageIdentities) else {
            return .miss(.imagesDiverged)
        }
        // The text bridge encoders cannot render an image, so a continuation
        // carrying one resumes by rendering just that turn - but only after a
        // clean end of turn. A `.maxTokens` entry holds a generated token
        // outside the KV that the bridge does not replay, and a `.toolCalls`
        // entry's boundary is a tool-response marker it does not emit; both
        // would resume onto a prompt the client never sent.
        if !requestIdentities.dropFirst(inputCount).allSatisfy(\.isEmpty) {
            guard entry.assistantTurn.rawStopReason == .endOfTurn else {
                return .miss(.unsupportedContinuation)
            }
            return .renderThenResume(cachedPromptTokens: entry.kvPosition)
        }
        let continuation = Array(request.messages.dropFirst(inputCount + 1))

        if entry.assistantTurn.message.toolCalls.isEmpty {
            return matchTextContinuation(
                entry: entry,
                continuation: continuation,
                tokenizer: tokenizer)
        }
        return matchToolContinuation(
            entry: entry,
            request: request,
            continuation: continuation,
            tokenizer: tokenizer)
    }

    private func assistantMatches(
        _ incoming: GFTokenizer.Message,
        _ cached: GFTokenizer.Message
    ) -> Bool {
        guard incoming.role == .assistant,
              cached.role == .assistant,
              incoming.toolCalls == cached.toolCalls,
              incoming.toolCallID == cached.toolCallID,
              incoming.name == cached.name else {
            return false
        }
        if !cached.toolCalls.isEmpty {
            return (incoming.content ?? "").isEmpty
                && (cached.content ?? "").isEmpty
        }
        return incoming.content == cached.content
    }

    private func matchTextContinuation(
        entry: ServerPromptCacheEntry,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        guard continuation.count == 1,
              continuation[0].role == .user,
              let content = continuation[0].content,
              continuation[0].toolCalls.isEmpty,
              continuation[0].toolCallID == nil,
              entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens else {
            return .miss(.unsupportedContinuation)
        }
        var bridge = tokenizer.encodeTextContinuation(userContent: content)
        if entry.assistantTurn.rawStopReason == .maxTokens {
            bridge = entry.uncommittedBoundaryTokenIDs + bridge
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return .miss(.boundaryMismatch)
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }

    private func matchToolContinuation(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        let calls = entry.assistantTurn.message.toolCalls
        guard entry.assistantTurn.rawStopReason == .toolCalls,
              continuation.count == calls.count,
              zip(continuation, calls).allSatisfy({ message, call in
                  message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
              }) else {
            return .miss(.unsupportedContinuation)
        }
        // Not `try?` discarded: a bridge that fails to render is a distinct and
        // diagnosable condition, not the same as a history that simply
        // diverged, and both would otherwise arrive as one anonymous miss.
        let bridge: [Int32]
        do {
            bridge = try tokenizer.encodeToolResultContinuation(
                cachedMessages: entry.inputMessages,
                assistant: entry.assistantTurn.message,
                incomingMessages: request.messages,
                tools: request.tools)
        } catch {
            ServerLog.promptCacheBridgeFailed(error: error)
            return .miss(.bridgeRenderFailed)
        }
        guard bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss(.boundaryMismatch)
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }
}
