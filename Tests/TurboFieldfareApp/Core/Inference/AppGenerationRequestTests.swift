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
        #expect(request.messages == [AppChatMessage(role: .user, content: "hello")])
    }

    @Test func requestPreservesConversationHistory() throws {
        let messages = [
            AppChatMessage(role: .user, content: "First question"),
            AppChatMessage(role: .assistant, content: "First answer"),
            AppChatMessage(role: .user, content: "Follow-up question"),
        ]
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "Follow-up question",
            messages: messages)

        try request.validate()
        #expect(request.messages == messages)
    }

    @Test func requestPreservesManagedImageReference() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "AppGenerationImageRequest-\(UUID().uuidString)",
            isDirectory: true)
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: model,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("image.png")
        try Self.onePixelPNG.write(to: source)
        let image = try AppChatAttachmentStore.importImage(
            from: source,
            chatID: UUID(),
            forModelDirectory: model)
        let message = AppChatMessage(
            role: .user,
            content: "What is shown?",
            images: [image])
        let request = AppGenerationRequest(
            modelDirectory: model,
            prompt: message.content,
            messages: [message])

        try request.validate()
        #expect(request.messages.last?.images == [image])
    }

    @Test func imageIsRejectedOnNonuserMessage() {
        let image = syntheticImage()
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "Question",
            messages: [
                AppChatMessage(
                    role: .assistant,
                    content: "Answer",
                    images: [image]),
                AppChatMessage(role: .user, content: "Question"),
            ])

        #expect(throws: AppInferenceError.self) {
            try request.validate(requireModelDirectory: false)
        }
    }

    @Test func requestRejectsMoreThanTheBoundedImageLimit() {
        let messages = (0...AppGenerationRequest.maximumImageAttachments).map { index in
            AppChatMessage(
                role: .user,
                content: "Question \(index)",
                images: [syntheticImage()])
        }
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: messages.last!.content,
            messages: messages)

        #expect(throws: AppInferenceError.self) {
            try request.validate(requireModelDirectory: false)
        }
    }

    @Test func systemInstructionsMustPrecedeTheConversation() throws {
        let valid = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "Follow-up question",
            messages: [
                AppChatMessage(role: .system, content: "Be concise."),
                AppChatMessage(role: .user, content: "Follow-up question"),
            ])
        try valid.validate()

        let invalid = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "Follow-up question",
            messages: [
                AppChatMessage(role: .user, content: "First question"),
                AppChatMessage(role: .system, content: "Be concise."),
                AppChatMessage(role: .user, content: "Follow-up question"),
            ])
        #expect(throws: AppInferenceError.self) {
            try invalid.validate()
        }
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

    private func syntheticImage() -> AppImageAttachment {
        AppImageAttachment(
            relativePath: "chat/image.png",
            originalFilename: "image.png",
            mediaTypeIdentifier: "public.png",
            pixelWidth: 1,
            pixelHeight: 1,
            byteCount: 1,
            sha256: String(repeating: "a", count: 64))
    }

    private static let onePixelPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")!
}
