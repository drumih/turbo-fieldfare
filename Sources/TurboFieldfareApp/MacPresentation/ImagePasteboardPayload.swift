import AppKit
import UniformTypeIdentifiers
import TurboFieldfare

public enum ImagePasteboardPayload {
    case fileURLs([URL])
    case image(data: Data, name: String)
    case text
    case unsupportedImage
    case none

    public static func read(from pasteboard: NSPasteboard) -> Self {
        let urls = pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [
                .urlReadingFileURLsOnly: true,
                .urlReadingContentsConformToTypes:
                    Array(VisionImageLimits().allowedTypeIdentifiers),
            ]) as? [NSURL]
        if let urls, !urls.isEmpty { return .fileURLs(urls.map { $0 as URL }) }

        let hasText = pasteboard.string(forType: .string)?.isEmpty == false
        let hasRichText = pasteboard.data(forType: .rtf) != nil
            || pasteboard.data(forType: .rtfd) != nil
        if hasText && hasRichText { return .text }

        if let data = pasteboard.data(forType: .png), NSImage(data: data) != nil {
            return .image(data: data, name: "Pasted image.png")
        }
        if let data = pasteboard.data(
            forType: NSPasteboard.PasteboardType("public.jpeg")),
           NSImage(data: data) != nil {
            return .image(data: data, name: "Pasted image.jpeg")
        }
        if let tiff = pasteboard.data(forType: .tiff),
           let representation = NSBitmapImageRep(data: tiff),
           let png = representation.representation(using: .png, properties: [:]) {
            return .image(data: png, name: "Pasted image.png")
        }

        let declaresImage = (pasteboard.types ?? []).contains {
            UTType($0.rawValue)?.conforms(to: .image) == true
        }
        if declaresImage { return .unsupportedImage }
        if hasText { return .text }
        return .none
    }
}
