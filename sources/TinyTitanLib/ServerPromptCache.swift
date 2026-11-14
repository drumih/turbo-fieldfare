import Foundation
import TinyTitan
import TinyTitanCore

package enum ServerPromptCacheMode: String, Codable, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
    case multiPrefix = "multi-prefix"
}

struct ServerPromptCacheDomain: Codable, Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Codable, Sendable, Equatable {
    let message: GFTokenizer.Message
    let rawStopReason: StopReason
}

struct ServerPromptCacheEntry: Codable, Sendable, Equatable {
    let id: UUID
    let domain: ServerPromptCacheDomain
    let inputMessages: [GFTokenizer.Message]
    let tools: [GFTokenizer.FunctionDefinition]
    let assistantTurn: CachedAssistantTurn
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss
    case hit(entryID: UUID, effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

struct ServerPromptCachePublication: Sendable, Equatable {
    let entry: ServerPromptCacheEntry
    let evictedEntryIDs: [UUID]
}

struct ServerPromptCache: Sendable {
    private let maximumEntries: Int
    private(set) var entries: [ServerPromptCacheEntry]

    init(
        maximumEntries: Int = 1,
        entries: [ServerPromptCacheEntry] = []
    ) {
        precondition(maximumEntries > 0, "maximumEntries must be positive")
        self.maximumEntries = maximumEntries
        self.entries = Array(entries.suffix(maximumEntries))
    }

    mutating func invalidate() {
        entries.removeAll(keepingCapacity: true)
    }

    mutating func remove(entryIDs: some Sequence<UUID>) {
        let removed = Set(entryIDs)
        guard !removed.isEmpty else { return }
        entries.removeAll { removed.contains($0.id) }
    }

    @discardableResult
    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool = false
    ) -> ServerPromptCachePublication? {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
            !result.kvBackedTokenIDs.isEmpty,
            result.uncommittedBoundaryTokenIDs.count == 1,
            !stopStringFiltered,
            result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens
        else {
            return nil
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
        let entry = ServerPromptCacheEntry(
            id: UUID(),
            domain: domain,
            inputMessages: request.messages,
            tools: request.tools,
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition)
        var evicted = entries.filter {
            $0.domain == entry.domain
                && $0.kvBackedTokenIDs == entry.kvBackedTokenIDs
                && $0.tools == entry.tools
        }.map(\.id)
        entries.removeAll { evicted.contains($0.id) }
        entries.append(entry)
        while entries.count > maximumEntries {
            evicted.append(entries.removeFirst().id)
        }
        return ServerPromptCachePublication(
            entry: entry,
            evictedEntryIDs: evicted)
    }

    mutating func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        var best: (index: Int, effective: [Int32], cached: Int)?
        for (index, entry) in entries.enumerated() {
            guard entry.domain == domain,
                entry.tools == request.tools,
                entry.kvPosition == entry.kvBackedTokenIDs.count,
                entry.kvPosition > 0,
                entry.uncommittedBoundaryTokenIDs.count == 1,
                let candidate = match(
                    entry: entry,
                    request: request,
                    renderedPromptIDs: renderedPromptIDs,
                    tokenizer: tokenizer)
            else { continue }
            if candidate.cached > (best?.cached ?? -1) {
                best = (index, candidate.effective, candidate.cached)
            }
        }
        guard let best else { return .miss }
        let matched = entries.remove(at: best.index)
        entries.append(matched)
        return .hit(
            entryID: matched.id,
            effectivePromptIDs: best.effective,
            cachedPromptTokens: best.cached)
    }

    private func match(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: GFTokenizer
    ) -> (effective: [Int32], cached: Int)? {
        guard entry.kvPosition == entry.kvBackedTokenIDs.count,
            entry.kvPosition > 0,
            entry.uncommittedBoundaryTokenIDs.count == 1
        else {
            return nil
        }

        // S12: direct prefix hit. `>=` (not `>`) so an identical-prompt
        // replay whose render is exactly the entry's KV-backed prefix also
        // hits; the caller restores the entry state without extending it.
        if renderedPromptIDs.count >= entry.kvPosition,
            renderedPromptIDs.prefix(entry.kvPosition)
                .elementsEqual(entry.kvBackedTokenIDs)
        {
            return (renderedPromptIDs, entry.kvPosition)
        }

        let inputCount = entry.inputMessages.count
        guard request.messages.count > inputCount + 1,
            request.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
            assistantMatches(
                request.messages[inputCount],
                entry.assistantTurn.message)
        else {
            return nil
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
            incoming.name == cached.name
        else {
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
    ) -> (effective: [Int32], cached: Int)? {
        // S13: support multi-turn continuations by matching on the trailing
        // user message. The whole tail after the cached assistant turn is
        // re-encoded with the same text-only ChatML template used to render
        // the original prompt, so the prefill reproduces the request's render
        // byte-for-byte. Kept conservative: the tail must be plain text-only
        // turns (no tool calls/results or tool ids, which the text template
        // cannot represent) and must end in a user message so the generation
        // suffix applies.
        guard let last = continuation.last,
            last.role == .user,
            continuation.allSatisfy({
                $0.role != .tool && $0.toolCallID == nil && $0.toolCalls.isEmpty
            }),
            entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens,
            let renderedTail = try? tokenizer.applyChatTemplate(continuation)
        else {
            return nil
        }
        // The bridge begins with the cached turn's closing <|im_end|>, then
        // the rendered tail (which includes the generation suffix).
        var bridge =
            [tokenizer.endOfTurnID]
            + tokenizer.encode("\n" + renderedTail, addBOS: false)
        if entry.assistantTurn.rawStopReason == .maxTokens {
            // S14: the uncommitted boundary token (the last generated token,
            // never committed to KV) must be replayed first — but apply the
            // same first-token dedup as the endOfTurn branch so a bridge that
            // already begins with the boundary token is not doubled.
            if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
                bridge = entry.uncommittedBoundaryTokenIDs + bridge
            }
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return nil
        }
        return (entry.kvBackedTokenIDs + bridge, entry.kvPosition)
    }

    private func matchToolContinuation(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> (effective: [Int32], cached: Int)? {
        let calls = entry.assistantTurn.message.toolCalls
        guard entry.assistantTurn.rawStopReason == .toolCalls,
            continuation.count == calls.count,
            zip(continuation, calls).allSatisfy({ message, call in
                message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
            })
        else {
            return nil
        }
        guard
            let bridge = try? tokenizer.encodeToolResultContinuation(
                cachedMessages: entry.inputMessages,
                assistant: entry.assistantTurn.message,
                incomingMessages: request.messages,
                tools: request.tools),
            bridge.first == entry.uncommittedBoundaryTokenIDs.first
        else {
            return nil
        }
        return (entry.kvBackedTokenIDs + bridge, entry.kvPosition)
    }
}
