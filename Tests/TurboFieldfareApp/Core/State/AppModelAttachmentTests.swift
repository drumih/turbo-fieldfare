import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite @MainActor struct AppModelAttachmentTests {
    private func attachment(filename: String = "notes.txt",
                            text: String = "body text",
                            size: UInt64 = 100) -> DocumentAttachment {
        DocumentAttachment(filename: filename,
                           type: .txt,
                           fileSize: size,
                           extractedText: text)
    }

    @Test
    func attachDocumentsExtractsText() async throws {
        let model = AppModel()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("notes.txt")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data("extracted via attach".utf8).write(to: url)

        model.attachDocuments(from: [url])
        await waitForExtraction(model)

        #expect(!model.isExtractingAttachment)
        #expect(model.attachments.count == 1)
        #expect(model.attachments[0].filename == "notes.txt")
        #expect(model.attachments[0].extractedText == "extracted via attach")
        #expect(model.attachmentErrors.isEmpty)
    }

    
    @Test func attachDocumentsSurfacesErrors() async throws {
        let model = AppModel()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("notes.xyz")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data("ignored".utf8).write(to: url)

        model.attachDocuments(from: [url])
        await waitForExtraction(model)

        #expect(model.attachments.isEmpty)
        #expect(model.attachmentErrors.count == 1)
        #expect(model.attachmentErrors[0].contains("notes.xyz"))
    }

    
    @Test func hasAttachmentsReflectsCollection() {
        let model = AppModel()
        #expect(!model.hasAttachments)

        model.attachments = [attachment()]
        #expect(model.hasAttachments)
    }

    
    @Test func promptWithAttachmentsAppendsContext() {
        let model = AppModel()
        model.promptText = "Summarize this file"
        model.attachments = [attachment()]

        let composed = model.promptWithAttachments

        #expect(composed.hasPrefix("Summarize this file"))
        #expect(composed.contains("[Document: notes.txt]"))
        #expect(composed.contains("body text"))
    }

    
    @Test func promptWithAttachmentsFallsBackToPrompt() {
        let model = AppModel()
        model.promptText = "No attachments"
        #expect(model.promptWithAttachments == "No attachments")
    }

    
    @Test func totalAttachmentBytesSumsSizes() {
        let model = AppModel()
        model.attachments = [
            attachment(size: 100),
            attachment(filename: "second.txt", size: 250),
        ]
        #expect(model.totalAttachmentBytes == 350)
    }

    
    @Test func makeRequestInjectsAttachmentContext() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "Analyze"
        model.attachments = [attachment()]

        let request = try model.makeRequest()

        #expect(request.prompt.contains("[Document: notes.txt]"))
        #expect(request.prompt.hasPrefix("Analyze"))
    }

    
    @Test func removeAttachmentRemovesOnlyMatching() {
        let model = AppModel()
        let first = attachment()
        let second = attachment(filename: "second.txt")
        model.attachments = [first, second]

        model.removeAttachment(first)

        #expect(model.attachments == [second])
    }

    
    @Test func clearAttachmentsClearsErrorsToo() {
        let model = AppModel()
        model.attachments = [attachment()]
        model.attachmentErrors = ["boom"]

        model.clearAttachments()

        #expect(model.attachments.isEmpty)
        #expect(model.attachmentErrors.isEmpty)
    }

    /// Polls until the detached extraction task publishes its result.
    
    private func waitForExtraction(_ model: AppModel, timeoutMilliseconds: Int = 5_000) async {
        let stepMilliseconds = 20
        var waited = 0
        while model.isExtractingAttachment && waited < timeoutMilliseconds {
            try? await Task.sleep(for: .milliseconds(stepMilliseconds))
            waited += stepMilliseconds
        }
    }
}
