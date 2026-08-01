import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppGenerationRequestTests {
    private let existingDirectory = FileManager.default.temporaryDirectory

    @Test func defaultRequestUsesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, messages: [.init(role: .user, content: "hello")])
        #expect(request.maxNewTokens == 4_096)
        #expect(request.temperature == 0.2)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
    }

    @Test func temperatureZeroRemainsPureGreedyWithTruncationDefaults() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           messages: [.init(role: .user, content: "hello")],
                                           temperature: 0)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.isPureGreedy)
    }

    @Test func emptyPromptRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, messages: [.init(role: .user, content: "   ")])
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidMaxTokensRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           messages: [.init(role: .user, content: "hello")], maxNewTokens: 0)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidSlotCountRejected() {
        var options = AppRuntimeOptions()
        options.expertCacheSlots = 7
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           messages: [.init(role: .user, content: "hello")], runtimeOptions: options)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func repetitionPenaltyBelowOneRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           messages: [.init(role: .user, content: "hello")], repetitionPenalty: 0.9)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidTopKRejected() {
        for topK in [0, 257] {
            let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                               messages: [.init(role: .user, content: "hello")], topK: topK)
            #expect(throws: AppInferenceError.self) {
                try request.validate()
            }
        }
    }

    @Test func invalidTopPRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           messages: [.init(role: .user, content: "hello")], topP: 1.1)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func stochasticTopPRequiresTopK() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           messages: [.init(role: .user, content: "hello")], topK: nil, topP: 0.95)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func missingModelDirectoryRejected() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            messages: [.init(role: .user, content: "hello")])
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }
}
