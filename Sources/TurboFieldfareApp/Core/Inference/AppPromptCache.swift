import Foundation
import TurboFieldfare
import TurboFieldfareDecodeProtocol

extension GFTokenizer.Message {
    init(_ message: DecodeChatMessage) {
        self.init(role: GFTokenizer.Role(rawValue: message.role.rawValue) ?? .user,
                  content: message.content)
    }
}

/// App-side prefix cache. Deliberately simpler than `ServerPromptCache`: the
/// app path is text-only (no tools, no system prompt), so the domain carries a
/// template identity constant instead of a jinja-file digest and the match
/// logic only implements the plain-prefix and text-continuation paths.
struct AppPromptCacheDomain: Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Sendable, Equatable {
    let content: String
    let rawStopReason: StopReason
}

struct AppPromptCacheEntry: Sendable, Equatable {
    let domain: AppPromptCacheDomain
    let inputMessages: [DecodeChatMessage]
    let assistantTurn: CachedAssistantTurn
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int
}

enum AppPromptCacheMatch: Sendable, Equatable {
    case miss
    case hit(effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

struct AppPromptCache: Sendable {
    private(set) var entry: AppPromptCacheEntry?

    mutating func invalidate() {
        entry = nil
    }

    mutating func publish(
        domain: AppPromptCacheDomain,
        messages: [DecodeChatMessage],
        content: String,
        result: RawDecodeResult
    ) {
        // A reusable prefix must be exactly the committed KV state: the stop
        // token is sampled but never written to KV, so a clean endOfTurn /
        // maxTokens finish leaves exactly one uncommitted boundary token.
        // Any other exit poisons the cache; the caller invalidates on failure.
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              result.reason == .endOfTurn
                || result.reason == .maxTokens else {
            entry = nil
            return
        }
        entry = AppPromptCacheEntry(
            domain: domain,
            inputMessages: messages,
            assistantTurn: CachedAssistantTurn(
                content: content,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition)
    }

    func match(
        domain: AppPromptCacheDomain,
        messages: [DecodeChatMessage],
        renderedPromptIDs: [Int32],
        tokenizer: GFTokenizer
    ) -> AppPromptCacheMatch {
        guard let entry,
              entry.domain == domain,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return .miss
        }

        // Plain token-prefix hit. Rare in multi-turn: the KV holds sampled
        // token IDs while a re-render re-encodes assistant TEXT, which does not
        // reproduce them. Kept as the free fast path for round-trip cases.
        if renderedPromptIDs.count > entry.kvPosition,
           renderedPromptIDs.prefix(entry.kvPosition)
            .elementsEqual(entry.kvBackedTokenIDs) {
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: entry.kvPosition)
        }

        // Text-continuation bridge: the cache holds sampled IDs; the next
        // request re-encodes the assistant turn from text. Only the new user
        // turn is encoded (`encodeTextContinuation`) and appended to the KV
        // prefix, so a follow-up turn prefills just the delta.
        // WARNING: the assistant-content comparison below is byte-exact. Any
        // normalization of assistant text (trimming, whitespace collapsing)
        // silently degrades this to a full re-prefill — a performance cliff,
        // not a correctness bug.
        let inputCount = entry.inputMessages.count
        guard messages.count > inputCount + 1,
              messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
              messages[inputCount].role == .assistant,
              messages[inputCount].content == entry.assistantTurn.content else {
            return .miss
        }
        let continuation = Array(messages.dropFirst(inputCount + 1))
        guard continuation.count == 1,
              continuation[0].role == .user,
              !continuation[0].content.isEmpty,
              entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens else {
            return .miss
        }
        var bridge = tokenizer.encodeTextContinuation(userContent: continuation[0].content)
        if entry.assistantTurn.rawStopReason == .maxTokens {
            // The last sampled token never committed to KV; prepend it so the
            // bridge starts at the exact KV boundary.
            bridge = entry.uncommittedBoundaryTokenIDs + bridge
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }
}
