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
    private(set) var entry: ServerPromptCacheEntry?

    var kvBackedTokenIDs: [Int32]? { entry?.kvBackedTokenIDs }
    var inputMessageCount: Int? { entry?.inputMessages.count }

    mutating func invalidate() {
        entry = nil
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
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            entry = nil
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
            entry = nil
            return
        }
        entry = ServerPromptCacheEntry(
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
        guard let entry,
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
