import AppKit
import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct DocumentAttachmentTests {
    private let extractor = DocumentExtractor()

    /// Writes a file to a unique temporary location.
    private func temporaryFile(named name: String, data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try data.write(to: url)
        return url
    }

    private func temporaryFile(named name: String, text: String) throws -> URL {
        try temporaryFile(named: name, data: Data(text.utf8))
    }

    // MARK: - Extraction

    @Test func extractsPlainText() throws {
        let url = try temporaryFile(named: "notes.txt", text: "Hello TurboFieldfare\nSecond line.")
        let attachment = try extractor.extract(from: url)

        #expect(attachment.type == .txt)
        #expect(attachment.extractedText == "Hello TurboFieldfare\nSecond line.")
        #expect(attachment.pageCount == nil)
        #expect(!attachment.truncated)
        #expect(attachment.originalLength == attachment.extractedText.count)
        #expect(attachment.filename == "notes.txt")
    }

    @Test func extractsMarkdown() throws {
        let url = try temporaryFile(named: "README.md", text: "# Title\n\nSome *markdown*.")
        let attachment = try extractor.extract(from: url)

        #expect(attachment.type == .md)
        #expect(attachment.extractedText.contains("# Title"))
        #expect(attachment.extractedText.contains("markdown"))
    }

    @Test func extractsRTF() throws {
        let attributed = NSAttributedString(string: "RTF content here")
        let data = try attributed.data(from: NSRange(location: 0, length: attributed.length),
                                       documentAttributes: [.documentType: NSAttributedString.DocumentType.rtf])
        let url = try temporaryFile(named: "notes.rtf", data: data)

        let attachment = try extractor.extract(from: url)

        #expect(attachment.type == .rtf)
        #expect(attachment.extractedText.contains("RTF content here"))
    }

    @Test func extractsDOCX() throws {
        let attributed = NSAttributedString(string: "Word document body")
        let data = try attributed.data(from: NSRange(location: 0, length: attributed.length),
                                       documentAttributes: [.documentType: NSAttributedString.DocumentType.officeOpenXML])
        let url = try temporaryFile(named: "notes.docx", data: data)

        let attachment = try extractor.extract(from: url)

        #expect(attachment.type == .docx)
        #expect(attachment.extractedText.contains("Word document body"))
    }

    @MainActor
    @Test func extractsPDF() throws {
        let text = "PDF extraction works"
        let textView = NSTextView(frame: NSRect(x: 0, y: 0, width: 300, height: 120))
        textView.string = text
        textView.layoutManager?.ensureLayout(for: textView.textContainer!)
        let data = textView.dataWithPDF(inside: textView.bounds)
        let url = try temporaryFile(named: "notes.pdf", data: data)

        let attachment = try extractor.extract(from: url)

        #expect(attachment.type == .pdf)
        #expect(attachment.pageCount != nil)
        #expect(attachment.extractedText.contains("PDF extraction works"))
    }

    // MARK: - Errors

    @Test func rejectsUnsupportedFormat() throws {
        let url = try temporaryFile(named: "notes.xyz", text: "ignored")

        #expect(throws: DocumentError.self) {
            try extractor.extract(from: url)
        }
    }

    @Test func rejectsOversizedFile() throws {
        // Sparse file: logical size exceeds the limit without consuming disk space.
        let url = try temporaryFile(named: "big.txt", data: Data())
        let handle = try FileHandle(forWritingTo: url)
        try handle.seek(toOffset: DocumentExtractor.maximumFileSize + 1)
        try handle.write(contentsOf: Data([0]))
        try handle.close()

        do {
            _ = try extractor.extract(from: url)
            Issue.record("Expected fileTooLarge error")
        } catch let error as DocumentError {
            guard case .fileTooLarge = error else {
                Issue.record("Unexpected error: \(error)")
                return
            }
        }
    }

    @Test func rejectsEmptyDocument() throws {
        let url = try temporaryFile(named: "blank.txt", text: "   \n\t ")

        do {
            _ = try extractor.extract(from: url)
            Issue.record("Expected emptyDocument error")
        } catch let error as DocumentError {
            guard case .emptyDocument = error else {
                Issue.record("Unexpected error: \(error)")
                return
            }
        }
    }

    // MARK: - Truncation

    @Test func truncatesVeryLongText() throws {
        let longText = String(repeating: "a", count: DocumentExtractor.maximumTextLength + 1)
        let url = try temporaryFile(named: "long.txt", text: longText)

        let attachment = try extractor.extract(from: url)

        #expect(attachment.truncated)
        #expect(attachment.extractedText.count == DocumentExtractor.maximumTextLength)
        #expect(attachment.originalLength == DocumentExtractor.maximumTextLength + 1)
    }

    // MARK: - Attachment presentation

    @Test func promptContextWrapsDocument() {
        let attachment = DocumentAttachment(filename: "notes.txt",
                                            type: .txt,
                                            fileSize: 100,
                                            extractedText: "body text")

        let context = attachment.promptContext()

        #expect(context.contains("[Document: notes.txt]"))
        #expect(context.contains("body text"))
        #expect(context.contains("[End of document]"))
    }

    @Test func promptContextMarksTruncation() {
        let attachment = DocumentAttachment(filename: "long.txt",
                                            type: .txt,
                                            fileSize: 100,
                                            extractedText: "body",
                                            truncated: true,
                                            originalLength: 500_001)

        #expect(attachment.promptContext().contains("document truncated"))
    }

    @Test func previewTruncatesLongText() {
        let long = String(repeating: "x", count: 500)
        let attachment = DocumentAttachment(filename: "f.txt",
                                            type: .txt,
                                            fileSize: 1,
                                            extractedText: long)

        let preview = attachment.preview(maxLength: 200)
        #expect(preview.count == 201)
        #expect(preview.hasSuffix("…"))
    }

    @Test func previewKeepsShortText() {
        let attachment = DocumentAttachment(filename: "f.txt",
                                            type: .txt,
                                            fileSize: 1,
                                            extractedText: "short")
        #expect(attachment.preview(maxLength: 200) == "short")
    }

    @Test func formattedSizeUsesFileStyle() {
        let attachment = DocumentAttachment(filename: "f.txt",
                                            type: .txt,
                                            fileSize: 1_048_576,
                                            extractedText: "")
        #expect(attachment.formattedSize == "1 MB")
    }

    @Test func supportedTypesCoverAllDocumentCases() {
        #expect(DocumentExtractor.supportedTypes.count == DocumentType.allCases.count)
        for type in DocumentType.allCases {
            #expect(DocumentExtractor.supportedTypes.contains(type.utType))
        }
    }
}
