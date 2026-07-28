import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppGenerationRequestTests {
    private let existingDirectory = FileManager.default.temporaryDirectory

    @Test func defaultRequestUsesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "hello")
        #expect(request.maxNewTokens == 4_096)
        #expect(request.temperature == 0.2)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
    }

    @Test func temperatureZeroRemainsPureGreedyWithTruncationDefaults() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello",
                                           temperature: 0)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.isPureGreedy)
    }

    @Test func emptyPromptRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "   ")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidMaxTokensRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", maxNewTokens: 0)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidSlotCountRejected() {
        var options = AppRuntimeOptions()
        options.expertCacheSlots = 7
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", runtimeOptions: options)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func repetitionPenaltyBelowOneRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", repetitionPenalty: 0.9)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidTopKRejected() {
        for topK in [0, 257] {
            let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                               prompt: "hello", topK: topK)
            #expect(throws: AppInferenceError.self) {
                try request.validate()
            }
        }
    }

    @Test func invalidTopPRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topP: 1.1)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func stochasticTopPRequiresTopK() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topK: nil, topP: 0.95)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func missingModelDirectoryRejected() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            prompt: "hello")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func conversationCannotStartWithAssistant() {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            messages: [
                AppGenerationMessage(role: .assistant, content: "orphan answer"),
                AppGenerationMessage(role: .user, content: "question"),
            ])
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func everySystemMessageAfterTheFirstPositionIsRejected() {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            messages: [
                AppGenerationMessage(role: .system, content: "first"),
                AppGenerationMessage(role: .user, content: "question"),
                AppGenerationMessage(role: .system, content: "second"),
                AppGenerationMessage(role: .user, content: "follow-up"),
            ])
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func completeMultiTurnConversationWithLeadingSystemMessageIsValid() throws {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            messages: [
                AppGenerationMessage(role: .system, content: "memory"),
                AppGenerationMessage(role: .user, content: "question"),
                AppGenerationMessage(role: .assistant, content: "answer"),
                AppGenerationMessage(role: .user, content: "follow-up"),
            ])

        try request.validate()
        #expect(request.prompt == "follow-up")
    }

    @Test func emptyConversationEmptyMessageAndTrailingAssistantAreRejected() {
        let invalidMessageSets: [[AppGenerationMessage]] = [
            [],
            [.init(role: .user, content: "\n\t ")],
            [
                .init(role: .user, content: "question"),
                .init(role: .assistant, content: "answer"),
            ],
            [
                .init(role: .user, content: "question"),
                .init(role: .assistant, content: " "),
                .init(role: .user, content: "follow-up"),
            ],
        ]

        for messages in invalidMessageSets {
            let request = AppGenerationRequest(
                modelDirectory: existingDirectory,
                messages: messages)
            #expect(throws: AppInferenceError.self) {
                try request.validate()
            }
        }
    }

    @Test func compatibilityPromptSetterReplacesPriorConversation() {
        var request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            messages: [
                .init(role: .system, content: "memory"),
                .init(role: .user, content: "old"),
                .init(role: .assistant, content: "answer"),
                .init(role: .user, content: "follow-up"),
            ])

        request.prompt = "replacement"

        #expect(request.messages == [
            AppGenerationMessage(role: .user, content: "replacement"),
        ])
        #expect(request.prompt == "replacement")
    }

    @Test func generationMessageCodableRoundTripKeepsEveryRole() throws {
        let messages: [AppGenerationMessage] = [
            .init(role: .system, content: "memory"),
            .init(role: .user, content: "question"),
            .init(role: .assistant, content: "answer"),
        ]

        let decoded = try JSONDecoder().decode(
            [AppGenerationMessage].self,
            from: JSONEncoder().encode(messages))

        #expect(decoded == messages)
    }
}
