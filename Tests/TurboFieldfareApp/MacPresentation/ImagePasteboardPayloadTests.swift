import AppKit
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct ImagePasteboardPayloadTests {
    private func board() -> NSPasteboard {
        let board = NSPasteboard(name: NSPasteboard.Name(UUID().uuidString))
        board.clearContents()
        return board
    }

    private func png() throws -> Data {
        let image = NSImage(size: NSSize(width: 1, height: 1))
        image.lockFocus()
        NSColor.red.setFill()
        NSRect(x: 0, y: 0, width: 1, height: 1).fill()
        image.unlockFocus()
        let tiff = try #require(image.tiffRepresentation)
        let bitmap = try #require(NSBitmapImageRep(data: tiff))
        return try #require(bitmap.representation(using: .png, properties: [:]))
    }

    private func bitmap() throws -> NSBitmapImageRep {
        let image = try #require(NSImage(data: png()))
        let tiff = try #require(image.tiffRepresentation)
        return try #require(NSBitmapImageRep(data: tiff))
    }

    @Test func incidentalStringDoesNotDefeatPureImage() throws {
        let board = board()
        board.declareTypes([.png, .string], owner: nil)
        board.setData(try png(), forType: .png)
        board.setString("Preview", forType: .string)

        guard case .image(_, let name) = ImagePasteboardPayload.read(from: board)
        else { Issue.record("a decodable image was not classified as an image"); return }
        #expect(name == "Pasted image.png")
    }

    @Test func richTextWithImageRemainsText() throws {
        let board = board()
        board.declareTypes([.png, .rtf, .string], owner: nil)
        board.setData(try png(), forType: .png)
        board.setData(Data("{\\rtf1 selected}".utf8), forType: .rtf)
        board.setString("selected", forType: .string)
        guard case .text = ImagePasteboardPayload.read(from: board)
        else { Issue.record("a genuine rich-text selection did not remain text"); return }
    }

    @Test func malformedDeclaredImageIsReportedUnsupported() {
        let board = board()
        board.declareTypes([.png], owner: nil)
        board.setData(Data("not an image".utf8), forType: .png)
        guard case .unsupportedImage = ImagePasteboardPayload.read(from: board)
        else { Issue.record("malformed image bytes were treated as usable"); return }
    }

    @Test func jpegAndTiffAreDecodedAndTiffIsNormalizedToPNG() throws {
        let jpegBoard = board()
        let jpegType = NSPasteboard.PasteboardType("public.jpeg")
        jpegBoard.declareTypes([jpegType], owner: nil)
        jpegBoard.setData(
            try bitmap().representation(using: .jpeg, properties: [:]),
            forType: jpegType)
        guard case .image(_, let jpegName) = ImagePasteboardPayload.read(from: jpegBoard)
        else { Issue.record("JPEG was not accepted"); return }
        #expect(jpegName.hasSuffix(".jpeg"))

        let tiffBoard = board()
        tiffBoard.declareTypes([.tiff], owner: nil)
        tiffBoard.setData(try #require(NSImage(data: png())?.tiffRepresentation), forType: .tiff)
        guard case .image(let normalized, let tiffName) = ImagePasteboardPayload.read(from: tiffBoard)
        else { Issue.record("TIFF was not accepted"); return }
        #expect(NSBitmapImageRep(data: normalized)?.representation(using: .png, properties: [:]) != nil)
        #expect(tiffName.hasSuffix(".png"))
    }

    @Test func supportedFileURLWinsAndPlainTextRemainsText() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("paste-\(UUID()).png")
        try png().write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let fileBoard = board()
        fileBoard.writeObjects([url as NSURL])
        guard case .fileURLs(let urls) = ImagePasteboardPayload.read(from: fileBoard)
        else { Issue.record("supported file URL was not accepted"); return }
        #expect(urls == [url])

        let textBoard = board()
        textBoard.setString("plain text", forType: .string)
        guard case .text = ImagePasteboardPayload.read(from: textBoard)
        else { Issue.record("plain text was not preserved"); return }
    }

    @Test func unsupportedImageFlavorAndEmptyBoardAreDistinct() {
        let unsupported = board()
        let gif = NSPasteboard.PasteboardType("com.compuserve.gif")
        unsupported.declareTypes([gif], owner: nil)
        unsupported.setData(Data("GIF89a".utf8), forType: gif)
        guard case .unsupportedImage = ImagePasteboardPayload.read(from: unsupported)
        else { Issue.record("unsupported image flavor was not diagnosed"); return }

        guard case .none = ImagePasteboardPayload.read(from: board())
        else { Issue.record("an empty pasteboard was not empty"); return }
    }
}
