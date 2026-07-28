import Foundation
import PDFKit
import UniformTypeIdentifiers

public struct ExtractedPromptDocument: Equatable, Sendable {
    public let fileName: String
    public let formatLabel: String
    public let text: String
    public let wasTruncated: Bool

    public init(fileName: String,
                formatLabel: String,
                text: String,
                wasTruncated: Bool) {
        self.fileName = fileName
        self.formatLabel = formatLabel
        self.text = text
        self.wasTruncated = wasTruncated
    }
}

public enum DocumentTextExtractionError: LocalizedError, Equatable, Sendable {
    case unsupportedFormat(String)
    case unreadableFile(String)
    case invalidArchive(String)
    case documentTooLarge(String)
    case noExtractableText(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedFormat(let file):
            return "\(file) is not a supported PDF, Word, PowerPoint, or Excel file."
        case .unreadableFile(let file):
            return "\(file) could not be read."
        case .invalidArchive(let file):
            return "\(file) is not a valid Office document."
        case .documentTooLarge(let file):
            return "\(file) is too large to extract safely."
        case .noExtractableText(let file):
            return "No selectable text was found in \(file). Scanned PDFs require OCR."
        }
    }
}

public enum DocumentTextExtractor {
    public static let maximumExtractedCharacters = 240_000

    public static var supportedContentTypes: [UTType] {
        [UTType.pdf] + ["docx", "pptx", "xlsx"].compactMap {
            UTType(filenameExtension: $0)
        }
    }

    public static func extract(from url: URL) throws -> ExtractedPromptDocument {
        let didAccessSecurityScope = url.startAccessingSecurityScopedResource()
        defer {
            if didAccessSecurityScope {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let fileName = url.lastPathComponent
        let extensionName = url.pathExtension.lowercased()
        let extracted: (label: String, text: String)

        switch extensionName {
        case "pdf":
            extracted = ("PDF", try extractPDF(at: url))
        case "docx":
            extracted = ("Word", try extractDOCX(at: url))
        case "pptx":
            extracted = ("PowerPoint", try extractPPTX(at: url))
        case "xlsx":
            extracted = ("Excel", try extractXLSX(at: url))
        default:
            throw DocumentTextExtractionError.unsupportedFormat(fileName)
        }

        let normalized = normalize(extracted.text)
        guard !normalized.isEmpty else {
            throw DocumentTextExtractionError.noExtractableText(fileName)
        }
        let text = String(normalized.prefix(maximumExtractedCharacters))
        return ExtractedPromptDocument(
            fileName: fileName,
            formatLabel: extracted.label,
            text: text,
            wasTruncated: text.count < normalized.count)
    }

    private static func extractPDF(at url: URL) throws -> String {
        guard let document = PDFDocument(url: url) else {
            throw DocumentTextExtractionError.unreadableFile(url.lastPathComponent)
        }

        var pages: [String] = []
        var extractedCharacterCount = 0
        pages.reserveCapacity(document.pageCount)
        for index in 0..<document.pageCount {
            guard let text = document.page(at: index)?.string,
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }
            pages.append("[Page \(index + 1)]\n\(text)")
            extractedCharacterCount += text.count
            if extractedCharacterCount > maximumExtractedCharacters * 2 {
                break
            }
        }
        return pages.joined(separator: "\n\n")
    }

    private static func extractDOCX(at url: URL) throws -> String {
        let archive = try OfficeArchive(url: url)
        var entries = archive.entries.keys.filter { entry in
            entry == "word/document.xml"
                || (entry.hasPrefix("word/header") && entry.hasSuffix(".xml"))
                || (entry.hasPrefix("word/footer") && entry.hasSuffix(".xml"))
                || ["word/footnotes.xml", "word/endnotes.xml", "word/comments.xml"]
                    .contains(entry)
        }
        entries.sort { left, right in
            docxPriority(left) < docxPriority(right)
        }
        guard entries.contains("word/document.xml") else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }

        return try entries.map { entry in
            let parser = FlowingTextXMLParser()
            let text = try parser.parse(archive.data(for: entry))
            guard entry != "word/document.xml" else { return text }
            let section = URL(fileURLWithPath: entry).deletingPathExtension().lastPathComponent
            return "[\(section.capitalized)]\n\(text)"
        }.joined(separator: "\n\n")
    }

    private static func extractPPTX(at url: URL) throws -> String {
        let archive = try OfficeArchive(url: url)
        let slides = archive.entries.keys.compactMap { entry -> (Int, String)? in
            guard entry.hasPrefix("ppt/slides/slide"),
                  entry.hasSuffix(".xml"),
                  !entry.contains("/_rels/") else {
                return nil
            }
            let name = URL(fileURLWithPath: entry).deletingPathExtension().lastPathComponent
            guard let number = Int(name.dropFirst("slide".count)) else { return nil }
            return (number, entry)
        }.sorted { $0.0 < $1.0 }
        guard !slides.isEmpty else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }

        return try slides.map { number, entry in
            let text = try FlowingTextXMLParser().parse(archive.data(for: entry))
            return "[Slide \(number)]\n\(text)"
        }.joined(separator: "\n\n")
    }

    private static func extractXLSX(at url: URL) throws -> String {
        let archive = try OfficeArchive(url: url)
        let sharedStrings: [String]
        if archive.entries["xl/sharedStrings.xml"] != nil {
            sharedStrings = try SharedStringsXMLParser().parse(
                archive.data(for: "xl/sharedStrings.xml"))
        } else {
            sharedStrings = []
        }

        let sheets = try workbookSheets(in: archive)
        guard !sheets.isEmpty else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }

        return try sheets.map { sheet in
            let rows = try WorksheetXMLParser(sharedStrings: sharedStrings).parse(
                archive.data(for: sheet.entry))
            return "[Sheet: \(sheet.name)]\n\(rows)"
        }.joined(separator: "\n\n")
    }

    private static func workbookSheets(in archive: OfficeArchive) throws -> [WorkbookSheet] {
        if archive.entries["xl/workbook.xml"] != nil,
           archive.entries["xl/_rels/workbook.xml.rels"] != nil {
            let sheetReferences = try WorkbookXMLParser().parse(
                archive.data(for: "xl/workbook.xml"))
            let relationships = try RelationshipsXMLParser().parse(
                archive.data(for: "xl/_rels/workbook.xml.rels"))
            let resolved = sheetReferences.compactMap { sheet -> WorkbookSheet? in
                guard let target = relationships[sheet.relationshipID] else { return nil }
                let entry = normalizedWorkbookTarget(target)
                guard archive.entries[entry] != nil else { return nil }
                return WorkbookSheet(name: sheet.name, entry: entry)
            }
            if !resolved.isEmpty {
                return resolved
            }
        }

        return archive.entries.keys.compactMap { entry -> (Int, String)? in
            guard entry.hasPrefix("xl/worksheets/sheet"),
                  entry.hasSuffix(".xml"),
                  !entry.contains("/_rels/") else {
                return nil
            }
            let name = URL(fileURLWithPath: entry).deletingPathExtension().lastPathComponent
            guard let number = Int(name.dropFirst("sheet".count)) else { return nil }
            return (number, entry)
        }
        .sorted { $0.0 < $1.0 }
        .map { WorkbookSheet(name: "Sheet \($0.0)", entry: $0.1) }
    }

    private static func normalizedWorkbookTarget(_ target: String) -> String {
        let withoutLeadingSlash = target.hasPrefix("/") ? String(target.dropFirst()) : target
        if withoutLeadingSlash.hasPrefix("xl/") {
            return withoutLeadingSlash
        }
        let path = "xl/" + withoutLeadingSlash
        return (path as NSString).standardizingPath
    }

    private static func docxPriority(_ entry: String) -> (Int, String) {
        if entry == "word/document.xml" { return (0, entry) }
        if entry.hasPrefix("word/header") { return (1, entry) }
        if entry.hasPrefix("word/footer") { return (2, entry) }
        if entry == "word/footnotes.xml" { return (3, entry) }
        if entry == "word/endnotes.xml" { return (4, entry) }
        return (5, entry)
    }

    private static func normalize(_ source: String) -> String {
        let source = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .replacingOccurrences(of: "\u{0}", with: "")
        var lines: [String] = []
        lines.reserveCapacity(source.count / 40)
        var previousLineWasEmpty = false

        for rawLine in source.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                if !previousLineWasEmpty {
                    lines.append("")
                }
                previousLineWasEmpty = true
            } else {
                lines.append(line)
                previousLineWasEmpty = false
            }
        }
        return lines.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private struct OfficeArchive {
    static let maximumEntryBytes = 32 * 1_024 * 1_024
    static let maximumSelectedBytes = 64 * 1_024 * 1_024

    let url: URL
    let entries: [String: Int]

    init(url: URL) throws {
        self.url = url
        let result = try Self.runUnzip(arguments: ["-l", url.path])
        guard result.status == 0,
              let listing = String(data: result.output, encoding: .utf8) else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }

        var parsedEntries: [String: Int] = [:]
        for line in listing.split(separator: "\n") {
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count >= 4, let length = Int(fields[0]) else { continue }
            let name = fields.dropFirst(3).joined(separator: " ")
            parsedEntries[name] = length
        }
        guard !parsedEntries.isEmpty else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }
        let selectedBytes = parsedEntries.reduce(0) { partial, entry in
            entry.key.hasSuffix(".xml") ? partial + entry.value : partial
        }
        guard selectedBytes <= Self.maximumSelectedBytes else {
            throw DocumentTextExtractionError.documentTooLarge(url.lastPathComponent)
        }
        entries = parsedEntries
    }

    func data(for entry: String) throws -> Data {
        guard let byteCount = entries[entry] else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }
        guard byteCount <= Self.maximumEntryBytes else {
            throw DocumentTextExtractionError.documentTooLarge(url.lastPathComponent)
        }
        let result = try Self.runUnzip(arguments: ["-p", url.path, entry])
        guard result.status == 0, result.output.count <= Self.maximumEntryBytes else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }
        return result.output
    }

    private static func runUnzip(arguments: [String]) throws -> (status: Int32, output: Data) {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            throw DocumentTextExtractionError.unreadableFile(
                arguments.last.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "document")
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, data)
    }
}

private final class FlowingTextXMLParser: NSObject, XMLParserDelegate {
    private var output = ""
    private var capturedText: String?

    func parse(_ data: Data) throws -> String {
        output = ""
        capturedText = nil
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Office document")
        }
        return output
    }

    func parser(_ parser: XMLParser,
                didStartElement elementName: String,
                namespaceURI: String?,
                qualifiedName qName: String?,
                attributes attributeDict: [String: String] = [:]) {
        switch localName(elementName) {
        case "t":
            capturedText = ""
        case "tab":
            output += "\t"
        case "br":
            output += "\n"
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard capturedText != nil else { return }
        capturedText! += string
    }

    func parser(_ parser: XMLParser,
                didEndElement elementName: String,
                namespaceURI: String?,
                qualifiedName qName: String?) {
        switch localName(elementName) {
        case "t":
            output += capturedText ?? ""
            capturedText = nil
        case "p":
            output += "\n"
        default:
            break
        }
    }
}

private final class SharedStringsXMLParser: NSObject, XMLParserDelegate {
    private var strings: [String] = []
    private var currentString: String?
    private var capturedText: String?

    func parse(_ data: Data) throws -> [String] {
        strings = []
        currentString = nil
        capturedText = nil
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Excel document")
        }
        return strings
    }

    func parser(_ parser: XMLParser,
                didStartElement elementName: String,
                namespaceURI: String?,
                qualifiedName qName: String?,
                attributes attributeDict: [String: String] = [:]) {
        switch localName(elementName) {
        case "si":
            currentString = ""
        case "t":
            capturedText = ""
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard capturedText != nil else { return }
        capturedText! += string
    }

    func parser(_ parser: XMLParser,
                didEndElement elementName: String,
                namespaceURI: String?,
                qualifiedName qName: String?) {
        switch localName(elementName) {
        case "t":
            currentString? += capturedText ?? ""
            capturedText = nil
        case "si":
            strings.append(currentString ?? "")
            currentString = nil
        default:
            break
        }
    }
}

private final class WorksheetXMLParser: NSObject, XMLParserDelegate {
    private struct Cell {
        var reference = ""
        var type: String?
        var formula = ""
        var value = ""
        var inlineText = ""
    }

    private let sharedStrings: [String]
    private var output = ""
    private var rowCells: [String] = []
    private var cell: Cell?
    private var capturedElement: String?
    private var capturedText = ""

    init(sharedStrings: [String]) {
        self.sharedStrings = sharedStrings
    }

    func parse(_ data: Data) throws -> String {
        output = ""
        rowCells = []
        cell = nil
        capturedElement = nil
        capturedText = ""
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Excel worksheet")
        }
        return output
    }

    func parser(_ parser: XMLParser,
                didStartElement elementName: String,
                namespaceURI: String?,
                qualifiedName qName: String?,
                attributes attributeDict: [String: String] = [:]) {
        switch localName(elementName) {
        case "row":
            rowCells = []
        case "c":
            cell = Cell(
                reference: attribute(attributeDict, localName: "r") ?? "",
                type: attribute(attributeDict, localName: "t"))
        case "v", "f", "t":
            guard cell != nil else { return }
            capturedElement = localName(elementName)
            capturedText = ""
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard capturedElement != nil else { return }
        capturedText += string
    }

    func parser(_ parser: XMLParser,
                didEndElement elementName: String,
                namespaceURI: String?,
                qualifiedName qName: String?) {
        let name = localName(elementName)
        if name == capturedElement {
            switch name {
            case "v": cell?.value = capturedText
            case "f": cell?.formula = capturedText
            case "t": cell?.inlineText += capturedText
            default: break
            }
            capturedElement = nil
            capturedText = ""
        }

        switch name {
        case "c":
            if let rendered = render(cell) {
                rowCells.append(rendered)
            }
            cell = nil
        case "row":
            if !rowCells.isEmpty {
                output += rowCells.joined(separator: "\t") + "\n"
            }
        default:
            break
        }
    }

    private func render(_ cell: Cell?) -> String? {
        guard let cell else { return nil }
        let value: String
        switch cell.type {
        case "s":
            if let index = Int(cell.value), sharedStrings.indices.contains(index) {
                value = sharedStrings[index]
            } else {
                value = cell.value
            }
        case "inlineStr":
            value = cell.inlineText
        case "b":
            value = cell.value == "1" ? "TRUE" : "FALSE"
        default:
            value = cell.value.isEmpty ? cell.inlineText : cell.value
        }

        let renderedValue: String
        if !cell.formula.isEmpty {
            renderedValue = value.isEmpty
                ? "=\(cell.formula)"
                : "=\(cell.formula) → \(value)"
        } else {
            renderedValue = value
        }
        guard !renderedValue.isEmpty else { return nil }
        return cell.reference.isEmpty
            ? renderedValue
            : "\(cell.reference): \(renderedValue)"
    }
}

private struct WorkbookSheet {
    let name: String
    let entry: String
}

private final class WorkbookXMLParser: NSObject, XMLParserDelegate {
    struct SheetReference {
        let name: String
        let relationshipID: String
    }

    private var sheets: [SheetReference] = []

    func parse(_ data: Data) throws -> [SheetReference] {
        sheets = []
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Excel workbook")
        }
        return sheets
    }

    func parser(_ parser: XMLParser,
                didStartElement elementName: String,
                namespaceURI: String?,
                qualifiedName qName: String?,
                attributes attributeDict: [String: String] = [:]) {
        guard localName(elementName) == "sheet",
              let name = attribute(attributeDict, localName: "name"),
              let relationshipID = attribute(attributeDict, localName: "id") else {
            return
        }
        sheets.append(SheetReference(name: name, relationshipID: relationshipID))
    }
}

private final class RelationshipsXMLParser: NSObject, XMLParserDelegate {
    private var relationships: [String: String] = [:]

    func parse(_ data: Data) throws -> [String: String] {
        relationships = [:]
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Excel relationships")
        }
        return relationships
    }

    func parser(_ parser: XMLParser,
                didStartElement elementName: String,
                namespaceURI: String?,
                qualifiedName qName: String?,
                attributes attributeDict: [String: String] = [:]) {
        guard localName(elementName) == "Relationship",
              let id = attribute(attributeDict, localName: "Id"),
              let target = attribute(attributeDict, localName: "Target") else {
            return
        }
        relationships[id] = target
    }
}

private func localName(_ qualifiedName: String) -> String {
    qualifiedName.split(separator: ":").last.map(String.init) ?? qualifiedName
}

private func attribute(_ attributes: [String: String], localName name: String) -> String? {
    attributes.first { localName($0.key) == name }?.value
}
