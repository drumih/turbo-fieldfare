import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppGenerationContextWindowTests {
    @Test func keepsSystemMemoryWhenTheWholeConversationFits() throws {
        let messages: [AppGenerationMessage] = [
            .init(role: .system, content: "memory"),
            .init(role: .user, content: "older"),
            .init(role: .assistant, content: "answer"),
            .init(role: .user, content: "new"),
        ]

        let fit = try AppGenerationContextWindow.fit(
            messages: messages,
            maximumPromptTokens: 100,
            maxNewTokens: 20,
            tokenCount: mockTokenCount)

        #expect(fit.messages == messages)
        #expect(fit.removedMessageCount == 0)
    }

    @Test func removesOldestCompleteTurnsUntilPromptFits() throws {
        let messages: [AppGenerationMessage] = [
            .init(role: .user, content: "older"),
            .init(role: .assistant, content: "answer"),
            .init(role: .user, content: "new"),
        ]

        let fit = try AppGenerationContextWindow.fit(
            messages: messages,
            maximumPromptTokens: 12,
            maxNewTokens: 20,
            tokenCount: mockTokenCount)

        #expect(fit.messages == [
            AppGenerationMessage(role: .user, content: "new"),
        ])
        #expect(fit.removedMessageCount == 2)
        #expect(fit.promptTokenCount == 5)
    }

    @Test func preservesAContiguousRecentConversation() throws {
        let messages: [AppGenerationMessage] = [
            .init(role: .user, content: "old question"),
            .init(role: .assistant, content: "old answer"),
            .init(role: .user, content: "recent"),
            .init(role: .assistant, content: "reply"),
            .init(role: .user, content: "follow-up"),
        ]

        let fit = try AppGenerationContextWindow.fit(
            messages: messages,
            maximumPromptTokens: 26,
            maxNewTokens: 20,
            tokenCount: mockTokenCount)

        #expect(fit.messages.map(\.content) == ["recent", "reply", "follow-up"])
        #expect(fit.messages.first?.role == .user)
        #expect(fit.messages.last?.role == .user)
    }

    @Test func rejectsOnlyWhenTheCurrentUserTurnCannotFit() {
        let messages = [
            AppGenerationMessage(role: .user, content: "far too long"),
        ]

        #expect(throws: AppInferenceError.self) {
            _ = try AppGenerationContextWindow.fit(
                messages: messages,
                maximumPromptTokens: 5,
                maxNewTokens: 20,
                tokenCount: mockTokenCount)
        }
    }

    @Test func exactContextLimitIsRejectedToLeaveDecodeCapacity() {
        let messages = [
            AppGenerationMessage(role: .user, content: "abc"),
        ]

        #expect(throws: AppInferenceError.self) {
            _ = try AppGenerationContextWindow.fit(
                messages: messages,
                maximumPromptTokens: 5,
                maxNewTokens: 1,
                tokenCount: mockTokenCount)
        }
    }

    @Test func invalidConversationShapesAreRejectedBeforeCountingTokens() {
        var tokenCountWasCalled = false
        let counter: ([AppGenerationMessage]) throws -> Int = { _ in
            tokenCountWasCalled = true
            return 1
        }

        for messages in [
            [AppGenerationMessage](),
            [AppGenerationMessage(role: .assistant, content: "answer")],
            [
                AppGenerationMessage(role: .user, content: "question"),
                AppGenerationMessage(role: .assistant, content: "answer"),
            ],
        ] {
            #expect(throws: AppInferenceError.self) {
                _ = try AppGenerationContextWindow.fit(
                    messages: messages,
                    maximumPromptTokens: 10,
                    maxNewTokens: 1,
                    tokenCount: counter)
            }
        }
        #expect(!tokenCountWasCalled)
    }

    @Test func nonPositiveContextIsRejectedBeforeCountingTokens() {
        for limit in [0, -1] {
            var tokenCountWasCalled = false
            #expect(throws: AppInferenceError.self) {
                _ = try AppGenerationContextWindow.fit(
                    messages: [
                        AppGenerationMessage(role: .user, content: "question"),
                    ],
                    maximumPromptTokens: limit,
                    maxNewTokens: 1
                ) { _ in
                    tokenCountWasCalled = true
                    return 1
                }
            }
            #expect(!tokenCountWasCalled)
        }
    }

    @Test func tokenizationErrorsPropagateWithoutBeingReclassified() {
        struct TokenizationProbeError: Error {}

        #expect(throws: TokenizationProbeError.self) {
            _ = try AppGenerationContextWindow.fit(
                messages: [
                    AppGenerationMessage(role: .user, content: "question"),
                ],
                maximumPromptTokens: 10,
                maxNewTokens: 1
            ) { _ in
                throw TokenizationProbeError()
            }
        }
    }

    @Test func preparedRequestKeepsGenerationOptionsAndReportsExactPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages: [AppGenerationMessage] = [
            .init(
                role: .user,
                content: String(repeating: "old ", count: 300)),
            .init(role: .assistant, content: "old answer"),
            .init(role: .user, content: "current"),
        ]
        let request = AppGenerationRequest(
            modelDirectory: FileManager.default.temporaryDirectory,
            messages: messages,
            maxNewTokens: 17,
            maxContextTokens: 64,
            temperature: 0.7,
            topK: 32,
            topP: 0.8,
            repetitionPenalty: 1.1,
            runtimeOptions: AppRuntimeOptions(
                expertCacheSlots: 32,
                prefillEnabled: false))

        let prepared = try AppGenerationContextWindow.prepareWithReport(
            request,
            tokenizer: tokenizer)

        #expect(prepared.removedMessages == Array(messages.prefix(2)))
        #expect(prepared.request.messages == [messages.last!])
        #expect(prepared.request.maxNewTokens == 17)
        #expect(prepared.request.temperature == 0.7)
        #expect(prepared.request.topK == 32)
        #expect(prepared.request.topP == 0.8)
        #expect(prepared.request.repetitionPenalty == 1.1)
        #expect(prepared.request.runtimeOptions.expertCacheSlots == 32)
        #expect(!prepared.request.runtimeOptions.prefillEnabled)
    }

    @Test func realTokenizerFitsMultilingualHistoryExactly() async throws {
        let tokenizer = try await GFTokenizer.load()
        let request = AppGenerationRequest(
            modelDirectory: FileManager.default.temporaryDirectory,
            messages: [
                .init(
                    role: .user,
                    content: String(repeating: "старый контекст 🦝 ", count: 200)),
                .init(role: .assistant, content: "старый ответ"),
                .init(role: .user, content: "Короткий текущий вопрос"),
            ],
            maxNewTokens: 64,
            maxContextTokens: 128)

        let preparation = try AppGenerationContextWindow.prepareWithReport(
            request,
            tokenizer: tokenizer)
        let prepared = preparation.request
        let rendered = try tokenizer.applyChatTemplate(
            prepared.messages.map { message in
                let role: GFTokenizer.Role = switch message.role {
                case .system: .system
                case .user: .user
                case .assistant: .assistant
                }
                return GFTokenizer.Message(role: role, content: message.content)
            })

        #expect(tokenizer.encode(rendered, addBOS: false).count < 128)
        #expect(prepared.messages.map(\.content) == ["Короткий текущий вопрос"])
        #expect(preparation.removedMessages.count == 2)
    }

    @Test func realTokenizerReportsWhenRollingMemoryMustBeShortened() async throws {
        let tokenizer = try await GFTokenizer.load()
        let oversizedMemory = AppGenerationMessage(
            role: .system,
            content: String(repeating: "compressed memory ", count: 200))
        let currentTurn = AppGenerationMessage(
            role: .user,
            content: "Continue with the current request.")
        let request = AppGenerationRequest(
            modelDirectory: FileManager.default.temporaryDirectory,
            messages: [oversizedMemory, currentTurn],
            maxNewTokens: 32,
            maxContextTokens: 96)

        let preparation = try AppGenerationContextWindow.prepareWithReport(
            request,
            tokenizer: tokenizer)

        #expect(preparation.request.messages == [currentTurn])
        #expect(preparation.removedMessages == [oversizedMemory])
    }

    private func mockTokenCount(_ messages: [AppGenerationMessage]) -> Int {
        2 + messages.reduce(0) { $0 + $1.content.count }
    }
}
