import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct DocumentTextExtractorTests {
    @Test func advertisesExactlyTheFourSupportedDocumentFamilies() {
        let extensions = Set(
            DocumentTextExtractor.supportedContentTypes.compactMap(
                \.preferredFilenameExtension))

        #expect(extensions.contains("pdf"))
        #expect(extensions.contains("docx"))
        #expect(extensions.contains("pptx"))
        #expect(extensions.contains("xlsx"))
        #expect(DocumentTextExtractor.supportedContentTypes.count == 4)
    }

    @MainActor
    @Test func extractsSelectablePDFText() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("sample.pdf")
        let view = NSTextView(frame: NSRect(x: 0, y: 0, width: 500, height: 300))
        view.string = "PDF revenue summary"
        try view.dataWithPDF(inside: view.bounds).write(to: url)

        let document = try DocumentTextExtractor.extract(from: url)

        #expect(document.formatLabel == "PDF")
        #expect(document.text.contains("PDF revenue summary"))
    }

    @Test func extractsDOCXParagraphs() throws {
        let fixture = try makeOfficeArchive(
            extensionName: "docx",
            entries: [
                "word/document.xml": """
                <?xml version="1.0" encoding="UTF-8"?>
                <w:document xmlns:w="urn:word">
                  <w:body>
                    <w:p><w:r><w:t>Quarterly report</w:t></w:r></w:p>
                    <w:p><w:r><w:t>Revenue increased.</w:t></w:r></w:p>
                  </w:body>
                </w:document>
                """,
            ])
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let document = try DocumentTextExtractor.extract(from: fixture.archive)

        #expect(document.formatLabel == "Word")
        #expect(document.text.contains("Quarterly report"))
        #expect(document.text.contains("Revenue increased."))
    }

    @Test func docxIncludesSupplementarySectionsInStableOrder() throws {
        let fixture = try makeOfficeArchive(
            extensionName: "DOCX",
            entries: [
                "word/comments.xml": """
                <w:comments xmlns:w="urn:word">
                  <w:p><w:r><w:t>Comment</w:t></w:r></w:p>
                </w:comments>
                """,
                "word/footer1.xml": """
                <w:ftr xmlns:w="urn:word">
                  <w:p><w:r><w:t>Footer</w:t></w:r></w:p>
                </w:ftr>
                """,
                "word/header1.xml": """
                <w:hdr xmlns:w="urn:word">
                  <w:p><w:r><w:t>Header</w:t></w:r></w:p>
                </w:hdr>
                """,
                "word/document.xml": """
                <w:document xmlns:w="urn:word">
                  <w:body><w:p><w:r><w:t>Body</w:t></w:r>
                  <w:tab/><w:r><w:t>Tabbed</w:t></w:r><w:br/>
                  <w:r><w:t>Next</w:t></w:r></w:p></w:body>
                </w:document>
                """,
            ])
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let document = try DocumentTextExtractor.extract(from: fixture.archive)

        #expect(document.formatLabel == "Word")
        #expect(document.text.contains("Body\tTabbed\nNext"))
        let body = try #require(document.text.range(of: "Body"))
        let header = try #require(document.text.range(of: "[Header1]"))
        let footer = try #require(document.text.range(of: "[Footer1]"))
        let comment = try #require(document.text.range(of: "[Comments]"))
        #expect(body.lowerBound < header.lowerBound)
        #expect(header.lowerBound < footer.lowerBound)
        #expect(footer.lowerBound < comment.lowerBound)
    }

    @Test func extractsPPTXSlidesInNumericOrder() throws {
        let fixture = try makeOfficeArchive(
            extensionName: "pptx",
            entries: [
                "ppt/slides/slide2.xml": """
                <p:sld xmlns:p="urn:p" xmlns:a="urn:a">
                  <a:p><a:r><a:t>Second slide</a:t></a:r></a:p>
                </p:sld>
                """,
                "ppt/slides/slide1.xml": """
                <p:sld xmlns:p="urn:p" xmlns:a="urn:a">
                  <a:p><a:r><a:t>First slide</a:t></a:r></a:p>
                </p:sld>
                """,
            ])
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let document = try DocumentTextExtractor.extract(from: fixture.archive)

        #expect(document.formatLabel == "PowerPoint")
        #expect(document.text.contains("[Slide 1]\nFirst slide"))
        #expect(document.text.contains("[Slide 2]\nSecond slide"))
        #expect(document.text.range(of: "First slide")!.lowerBound
                < document.text.range(of: "Second slide")!.lowerBound)
    }

    @Test func extractsXLSXSharedStringsNumbersAndFormulas() throws {
        let fixture = try makeOfficeArchive(
            extensionName: "xlsx",
            entries: [
                "xl/workbook.xml": """
                <workbook xmlns:r="urn:relationships">
                  <sheets><sheet name="Budget" r:id="rId1"/></sheets>
                </workbook>
                """,
                "xl/_rels/workbook.xml.rels": """
                <Relationships>
                  <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
                </Relationships>
                """,
                "xl/sharedStrings.xml": """
                <sst><si><t>Revenue</t></si></sst>
                """,
                "xl/worksheets/sheet1.xml": """
                <worksheet>
                  <sheetData>
                    <row r="1">
                      <c r="A1" t="s"><v>0</v></c>
                      <c r="B1"><v>42</v></c>
                    </row>
                    <row r="2">
                      <c r="B2"><f>SUM(B1:B1)</f><v>42</v></c>
                    </row>
                  </sheetData>
                </worksheet>
                """,
            ])
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let document = try DocumentTextExtractor.extract(from: fixture.archive)

        #expect(document.formatLabel == "Excel")
        #expect(document.text.contains("[Sheet: Budget]"))
        #expect(document.text.contains("A1: Revenue"))
        #expect(document.text.contains("B1: 42"))
        #expect(document.text.contains("B2: =SUM(B1:B1) → 42"))
    }

    @Test func extractsXLSXInlineStringsBooleansAndFormulaWithoutCachedValue() throws {
        let fixture = try makeOfficeArchive(
            extensionName: "xlsx",
            entries: [
                "xl/worksheets/sheet2.xml": """
                <worksheet><sheetData><row>
                  <c r="A1" t="inlineStr"><is><t>Inline</t></is></c>
                  <c r="B1" t="b"><v>1</v></c>
                  <c r="C1"><f>A1</f></c>
                </row></sheetData></worksheet>
                """,
                "xl/worksheets/sheet1.xml": """
                <worksheet><sheetData><row>
                  <c r="A1" t="b"><v>0</v></c>
                </row></sheetData></worksheet>
                """,
            ])
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let document = try DocumentTextExtractor.extract(from: fixture.archive)

        #expect(document.text.contains("[Sheet: Sheet 1]\nA1: FALSE"))
        #expect(document.text.contains(
            "[Sheet: Sheet 2]\nA1: Inline\tB1: TRUE\tC1: =A1"))
        let first = try #require(document.text.range(of: "[Sheet: Sheet 1]"))
        let second = try #require(document.text.range(of: "[Sheet: Sheet 2]"))
        #expect(first.lowerBound < second.lowerBound)
    }

    @Test func extractionNormalizesWhitespaceAndReportsCharacterTruncation() throws {
        let oversizedText = String(
            repeating: "x",
            count: DocumentTextExtractor.maximumExtractedCharacters + 100)
        let fixture = try makeOfficeArchive(
            extensionName: "docx",
            entries: [
                "word/document.xml": """
                <w:document xmlns:w="urn:word"><w:body>
                  <w:p><w:r><w:t>  first  </w:t></w:r></w:p>
                  <w:p><w:r><w:t>   </w:t></w:r></w:p>
                  <w:p><w:r><w:t>   </w:t></w:r></w:p>
                  <w:p><w:r><w:t>\(oversizedText)</w:t></w:r></w:p>
                </w:body></w:document>
                """,
            ])
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let document = try DocumentTextExtractor.extract(from: fixture.archive)

        #expect(document.wasTruncated)
        #expect(document.text.count
            == DocumentTextExtractor.maximumExtractedCharacters)
        #expect(document.text.hasPrefix("first\n\n"))
        #expect(!document.text.contains("\n\n\n"))
    }

    @MainActor
    @Test func unsupportedEmptyAndMalformedFilesReturnSpecificErrors() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let unsupported = root.appendingPathComponent("notes.txt")
        try Data("plain text".utf8).write(to: unsupported)
        #expect(throws: DocumentTextExtractionError.unsupportedFormat("notes.txt")) {
            _ = try DocumentTextExtractor.extract(from: unsupported)
        }

        let emptyPDF = root.appendingPathComponent("empty.pdf")
        let view = NSTextView(frame: NSRect(x: 0, y: 0, width: 100, height: 100))
        try view.dataWithPDF(inside: view.bounds).write(to: emptyPDF)
        #expect(throws: DocumentTextExtractionError.noExtractableText("empty.pdf")) {
            _ = try DocumentTextExtractor.extract(from: emptyPDF)
        }

        let malformed = root.appendingPathComponent("broken.pptx")
        try Data("not a zip archive".utf8).write(to: malformed)
        #expect(throws: DocumentTextExtractionError.invalidArchive("broken.pptx")) {
            _ = try DocumentTextExtractor.extract(from: malformed)
        }
    }

    @Test func officeArchivesMissingTheirRequiredPayloadAreRejected() throws {
        let docx = try makeOfficeArchive(
            extensionName: "docx",
            entries: [
                "word/header1.xml": "<w:hdr xmlns:w=\"urn:word\"/>",
            ])
        defer { try? FileManager.default.removeItem(at: docx.root) }
        #expect(throws: DocumentTextExtractionError.invalidArchive(
            docx.archive.lastPathComponent)) {
            _ = try DocumentTextExtractor.extract(from: docx.archive)
        }

        let pptx = try makeOfficeArchive(
            extensionName: "pptx",
            entries: [
                "ppt/notesSlides/notesSlide1.xml": "<p:notes/>",
            ])
        defer { try? FileManager.default.removeItem(at: pptx.root) }
        #expect(throws: DocumentTextExtractionError.invalidArchive(
            pptx.archive.lastPathComponent)) {
            _ = try DocumentTextExtractor.extract(from: pptx.archive)
        }
    }

    @Test func extractionErrorsHaveActionableUserMessages() {
        let cases: [(DocumentTextExtractionError, String)] = [
            (.unsupportedFormat("a.txt"), "not a supported"),
            (.unreadableFile("a.pdf"), "could not be read"),
            (.invalidArchive("a.docx"), "not a valid Office document"),
            (.documentTooLarge("a.xlsx"), "too large"),
            (.noExtractableText("scan.pdf"), "require OCR"),
        ]

        for (error, fragment) in cases {
            #expect(error.localizedDescription.contains(fragment))
        }
    }

    private struct OfficeFixture {
        let root: URL
        let archive: URL
    }

    private func makeOfficeArchive(
        extensionName: String,
        entries: [String: String]
    ) throws -> OfficeFixture {
        let root = try makeTemporaryRoot()
        let contents = root.appendingPathComponent("contents", isDirectory: true)
        try FileManager.default.createDirectory(
            at: contents,
            withIntermediateDirectories: true)

        for (path, text) in entries {
            let url = contents.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try Data(text.utf8).write(to: url)
        }

        let archive = root.appendingPathComponent("fixture.\(extensionName)")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        process.arguments = ["-q", "-r", archive.path, "."]
        process.currentDirectoryURL = contents
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw FixtureError.zipFailed
        }
        return OfficeFixture(root: root, archive: archive)
    }

    private func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "DocumentTextExtractorTests-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        return root
    }

    private enum FixtureError: Error {
        case zipFailed
    }
}
