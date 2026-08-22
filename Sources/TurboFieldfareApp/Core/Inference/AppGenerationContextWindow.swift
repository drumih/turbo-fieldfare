import Foundation
import TurboFieldfare

public struct AppGenerationContextFit: Equatable, Sendable {
    public let messages: [AppGenerationMessage]
    public let promptTokenCount: Int
    public let removedMessageCount: Int

    public init(messages: [AppGenerationMessage],
                promptTokenCount: Int,
                removedMessageCount: Int) {
        self.messages = messages
        self.promptTokenCount = promptTokenCount
        self.removedMessageCount = removedMessageCount
    }
}

public struct AppPreparedGenerationRequest: Equatable, Sendable {
    public let request: AppGenerationRequest
    public let removedMessages: [AppGenerationMessage]

    public init(request: AppGenerationRequest,
                removedMessages: [AppGenerationMessage]) {
        self.request = request
        self.removedMessages = removedMessages
    }
}

public enum AppGenerationContextWindow {
    public static func fit(
        messages: [AppGenerationMessage],
        maximumPromptTokens: Int,
        maxNewTokens: Int,
        tokenCount: ([AppGenerationMessage]) throws -> Int
    ) throws -> AppGenerationContextFit {
        guard maximumPromptTokens > 0 else {
            throw AppInferenceError.invalidRequest(
                "Max context must be greater than zero.")
        }

        guard !messages.isEmpty else {
            throw AppInferenceError.invalidRequest("Prompt cannot be empty.")
        }
        guard messages.last?.role == .user else {
            throw AppInferenceError.invalidRequest(
                "Conversation must end with a user message.")
        }

        let suffixStarts = messages.indices.filter { index in
            if index == messages.startIndex {
                return messages[index].role != .assistant
            }
            return messages[index].role == .user
        }
        guard !suffixStarts.isEmpty else {
            throw AppInferenceError.invalidRequest(
                "Conversation has no usable user turn.")
        }

        var lowerBound = suffixStarts.startIndex
        var upperBound = suffixStarts.index(before: suffixStarts.endIndex)
        var bestFit: AppGenerationContextFit?
        while lowerBound <= upperBound {
            let distance = suffixStarts.distance(
                from: lowerBound,
                to: upperBound)
            let midpoint = suffixStarts.index(
                lowerBound,
                offsetBy: distance / 2)
            let start = suffixStarts[midpoint]
            let candidate = Array(messages[start...])
            let count = try tokenCount(candidate)
            if count < maximumPromptTokens {
                bestFit = AppGenerationContextFit(
                    messages: candidate,
                    promptTokenCount: count,
                    removedMessageCount: start)
                if midpoint == suffixStarts.startIndex { break }
                upperBound = suffixStarts.index(before: midpoint)
            } else {
                if midpoint == suffixStarts.index(before: suffixStarts.endIndex) {
                    break
                }
                lowerBound = suffixStarts.index(after: midpoint)
            }
        }

        guard let bestFit else {
            let currentTurn = [messages[suffixStarts.last!]]
            let count = try tokenCount(currentTurn)
            throw AppInferenceError.contextOverflow(
                prompt: count,
                maxNew: maxNewTokens,
                maxContext: maximumPromptTokens)
        }
        return bestFit
    }

    public static func prepare(
        _ request: AppGenerationRequest,
        tokenizer: GFTokenizer
    ) throws -> AppGenerationRequest {
        try prepareWithReport(request, tokenizer: tokenizer).request
    }

    public static func prepareWithReport(
        _ request: AppGenerationRequest,
        tokenizer: GFTokenizer
    ) throws -> AppPreparedGenerationRequest {
        let fit = try fit(
            messages: request.messages,
            maximumPromptTokens: request.maxContextTokens,
            maxNewTokens: request.maxNewTokens
        ) { messages in
            let rendered = try tokenizer.applyChatTemplate(
                messages.map(tokenizerMessage))
            return tokenizer.encode(rendered, addBOS: false).count
        }
        var prepared = request
        prepared.messages = fit.messages
        return AppPreparedGenerationRequest(
            request: prepared,
            removedMessages: Array(
                request.messages.prefix(fit.removedMessageCount)))
    }

    public static func prepareUsingModelTokenizer(
        _ request: AppGenerationRequest
    ) async throws -> AppGenerationRequest {
        try await prepareUsingModelTokenizerWithReport(request).request
    }

    public static func prepareUsingModelTokenizerWithReport(
        _ request: AppGenerationRequest
    ) async throws -> AppPreparedGenerationRequest {
        do {
            let tokenizer = try await GFTokenizer.load(
                forModelDirectory: request.modelDirectory)
            try Task.checkCancellation()
            return try prepareWithReport(request, tokenizer: tokenizer)
        } catch is CancellationError {
            throw CancellationError()
        } catch let appError as AppInferenceError {
            throw appError
        } catch {
            throw AppInferenceError.tokenizerUnavailable("\(error)")
        }
    }

    private static func tokenizerMessage(
        _ message: AppGenerationMessage
    ) -> GFTokenizer.Message {
        let role: GFTokenizer.Role = switch message.role {
        case .system: .system
        case .user: .user
        case .assistant: .assistant
        }
        return GFTokenizer.Message(role: role, content: message.content)
    }
}
