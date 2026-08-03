import Foundation
import PDFKit
import UniformTypeIdentifiers

/// Supported document types for text extraction
public enum DocumentType: String, CaseIterable, Sendable, Codable {
    case pdf
    case docx
    case txt
    case md
    case rtf
    
    public var fileExtension: String { rawValue }
    
    public var utType: UTType {
        switch self {
        case .pdf: return .pdf
        case .docx: return UTType("org.openxmlformats.wordprocessingml.document") ?? .data
        case .txt: return .plainText
        case .md: return UTType("net.daringfireball.markdown") ?? .plainText
        case .rtf: return .rtf
        }
    }
    
    public var displayName: String {
        switch self {
        case .pdf: return "PDF"
        case .docx: return "Word"
        case .txt: return "Text"
        case .md: return "Markdown"
        case .rtf: return "RTF"
        }
    }
    
    public var icon: String {
        switch self {
        case .pdf: return "doc.fill"
        case .docx: return "doc.text.fill"
        case .txt, .md: return "doc.plaintext.fill"
        case .rtf: return "doc.richtext.fill"
        }
    }
}

/// Document extraction errors
public enum DocumentError: Error, LocalizedError, Sendable {
    case unsupportedFormat(String)
    case emptyDocument
    case pdfExtractionFailed
    case docxExtractionFailed
    case fileTooLarge(UInt64, maximum: UInt64)
    case readFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .unsupportedFormat(let ext):
            return "Unsupported format: \(ext). Accepted formats: PDF, DOCX, TXT, MD, RTF."
        case .emptyDocument:
            return "The document contains no extractable text."
        case .pdfExtractionFailed:
            return "Could not extract text from the PDF. The file may be protected or corrupted."
        case .docxExtractionFailed:
            return "Could not extract text from the Word document."
        case .fileTooLarge(let size, let max):
            let sizeMB = Double(size) / 1_048_576
            let maxMB = Double(max) / 1_048_576
            return String(format: "File too large (%.1f MB). Maximum: %.1f MB.", sizeMB, maxMB)
        case .readFailed(let reason):
            return "Read error: \(reason)"
        }
    }
}

/// Metadata for an attached document
public struct DocumentAttachment: Identifiable, Sendable, Equatable, Codable {
    public let id: UUID
    public let filename: String
    public let type: DocumentType
    public let fileSize: UInt64
    public let extractedText: String
    public let pageCount: Int?
    public let truncated: Bool
    public let originalLength: Int
    
    public init(id: UUID = UUID(),
                filename: String,
                type: DocumentType,
                fileSize: UInt64,
                extractedText: String,
                pageCount: Int? = nil,
                truncated: Bool = false,
                originalLength: Int = 0) {
        self.id = id
        self.filename = filename
        self.type = type
        self.fileSize = fileSize
        self.extractedText = extractedText
        self.pageCount = pageCount
        self.truncated = truncated
        self.originalLength = originalLength
    }
    
    /// Formatted size for display
    public var formattedSize: String {
        ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file)
    }
    
    /// Truncated preview of the extracted text
    public func preview(maxLength: Int = 200) -> String {
        if extractedText.count <= maxLength {
            return extractedText
        }
        let endIndex = extractedText.index(extractedText.startIndex, offsetBy: maxLength)
        return String(extractedText[..<endIndex]) + "…"
    }
    
    /// Text injected into the prompt, wrapped in context markers
    public func promptContext() -> String {
        var context = "\n\n[Document: \(filename)]\n"
        context += extractedText
        if truncated {
            context += "\n[... document truncated ...]"
        }
        context += "\n[End of document]\n"
        return context
    }
}

/// Text extractor for documents
public struct DocumentExtractor: Sendable {
    /// Default maximum file size (50 MB)
    public static let maximumFileSize: UInt64 = 50 * 1_048_576
    /// Default maximum extracted text length (before truncation)
    public static let maximumTextLength: Int = 500_000

    /// Maximum file size for this extractor
    public let maximumFileSize: UInt64
    /// Maximum extracted text length (before truncation)
    public let maximumTextLength: Int

    public init(maximumFileSize: UInt64 = DocumentExtractor.maximumFileSize,
                maximumTextLength: Int = DocumentExtractor.maximumTextLength) {
        self.maximumFileSize = maximumFileSize
        self.maximumTextLength = maximumTextLength
    }
    
    /// Extracts text from a file
    public func extract(from url: URL) throws -> DocumentAttachment {
        let fileSize = try fileSize(at: url)
        guard fileSize <= maximumFileSize else {
            throw DocumentError.fileTooLarge(fileSize, maximum: maximumFileSize)
        }
        
        let filename = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        guard let type = DocumentType(rawValue: ext) else {
            throw DocumentError.unsupportedFormat(ext)
        }
        
        let (text, pageCount) = try extractText(from: url, type: type)
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DocumentError.emptyDocument
        }
        
        // Truncate if needed
        let truncated = text.count > maximumTextLength
        let finalText = truncated ? String(text.prefix(maximumTextLength)) : text
        
        return DocumentAttachment(
            filename: filename,
            type: type,
            fileSize: fileSize,
            extractedText: finalText,
            pageCount: pageCount,
            truncated: truncated,
            originalLength: text.count
        )
    }
    
    /// Extracts text according to the document type
    private func extractText(from url: URL, type: DocumentType) throws -> (String, Int?) {
        switch type {
        case .pdf:
            return try extractPDF(from: url)
        case .docx:
            return try extractDOCX(from: url)
        case .txt, .md, .rtf:
            let text = try extractPlainText(from: url)
            return (text, nil)
        }
    }
    
    /// PDF extraction via PDFKit. Pages are separated with a `[Page N]`
    /// marker so the model can cite where information came from.
    private func extractPDF(from url: URL) throws -> (String, Int?) {
        guard let document = PDFDocument(url: url) else {
            throw DocumentError.pdfExtractionFailed
        }
        var text = ""
        for pageIndex in 0..<document.pageCount {
            guard let page = document.page(at: pageIndex) else { continue }
            if let pageText = page.string {
                if pageIndex > 0 {
                    text += "\n\n[Page \(pageIndex + 1)]\n"
                }
                text += pageText
            }
        }
        return (text, document.pageCount)
    }
    
    /// DOCX extraction via NSAttributedString
    private func extractDOCX(from url: URL) throws -> (String, Int?) {
        guard let data = try? Data(contentsOf: url),
              let attributed = try? NSAttributedString(
                data: data,
                options: [.documentType: NSAttributedString.DocumentType.officeOpenXML],
                documentAttributes: nil
              ) else {
            throw DocumentError.docxExtractionFailed
        }
        return (attributed.string, nil)
    }
    
    /// Plain text extraction
    private func extractPlainText(from url: URL) throws -> String {
        guard let data = try? Data(contentsOf: url) else {
            throw DocumentError.readFailed("Could not read the file")
        }
        guard let text = String(data: data, encoding: .utf8) ??
                         String(data: data, encoding: .utf16) else {
            throw DocumentError.readFailed("Unrecognized text encoding")
        }
        return text
    }
    
    /// File size
    private func fileSize(at url: URL) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = attrs[.size] as? NSNumber else {
            throw DocumentError.readFailed("File size unavailable")
        }
        return size.uint64Value
    }
    
    /// Accepted file types for the picker
    public static var supportedTypes: [UTType] {
        DocumentType.allCases.map(\.utType)
    }
}