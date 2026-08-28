import AppKit
import Foundation
import Testing

struct TranscriptFrameError: Error, CustomStringConvertible {
    let description: String
}

/// Offscreen renderer for transcript frames. The text view is configured to
/// match `IncrementalTranscriptView.makeNSView` so a frame shows what the app
/// shows; the only deliberate differences are an opaque background (the app
/// lets the SwiftUI surface show through, which would render as transparent
/// pixels here) and a fixed width so frames stay comparable across runs.
@MainActor
enum TranscriptFrameRenderer {
    static let defaultWidth: CGFloat = 560

    /// Beyond this the bitmap allocation and the hosting window stop being
    /// reliable; taller documents are cropped rather than dropped.
    static let maximumHeight: CGFloat = 4000

    private static let verticalInset: CGFloat = 4

    static func image(
        _ attributed: NSAttributedString,
        width: CGFloat = defaultWidth,
        dark: Bool
    ) throws -> NSImage {
        let textView = NSTextView(frame: NSRect(x: 0, y: 0, width: width, height: 10))
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = true
        textView.drawsBackground = true
        textView.backgroundColor = .textBackgroundColor
        textView.textContainerInset = NSSize(width: 0, height: verticalInset)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = 0
        textView.isAutomaticLinkDetectionEnabled = false
        textView.isAutomaticDataDetectionEnabled = false

        guard let storage = textView.textStorage,
              let container = textView.textContainer,
              let layoutManager = textView.layoutManager else {
            throw TranscriptFrameError(description: "text view has no TextKit 1 stack")
        }
        storage.setAttributedString(attributed)
        layoutManager.ensureLayout(for: container)
        let used = layoutManager.usedRect(for: container)
        let height = min(max(used.height + verticalInset * 2, 12), maximumHeight)
        textView.frame.size.height = height

        let window = NSWindow(
            contentRect: textView.bounds,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false)
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentView = textView
        window.layoutIfNeeded()

        let bounds = textView.bounds
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(bounds.width * 2),
            pixelsHigh: Int(bounds.height * 2),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0) else {
            throw TranscriptFrameError(
                description: "cannot allocate a 2x bitmap for \(bounds.size)")
        }
        rep.size = bounds.size
        textView.cacheDisplay(in: bounds, to: rep)

        let image = NSImage(size: bounds.size)
        image.addRepresentation(rep)
        return image
    }

    /// Records the frame as a test attachment and, when
    /// `TURBO_FIELDFARE_FRAME_DIR` is set, also writes it there so frames can
    /// be reviewed without passing `--attachments-path`.
    static func record(_ image: NSImage, named name: String) throws {
        #if compiler(>=6.3)
        Attachment.record(image, named: name, as: .png)
        #else
        // `NSImage` gained its `Attachable` conformance in the Testing library
        // shipped with Swift 6.3, and this suite has to build on the 6.2
        // toolchains the README supports. The bytes are the same; what is lost
        // is the lazy serialization, so a 6.2 run pays a PNG encode per frame
        // even without `--attachments-path`. Isolating a lazy wrapper is not
        // available either: a struct holding an `NSImage` cannot conform
        // without the conformance crossing into main-actor code.
        if let rep = image.representations.first as? NSBitmapImageRep,
           let data = rep.representation(using: .png, properties: [:]) {
            Attachment.record(data, named: name)
        } else {
            Issue.record("frame \(name) has no PNG representation to attach")
        }
        #endif
        guard let directory = ProcessInfo.processInfo
            .environment["TURBO_FIELDFARE_FRAME_DIR"], !directory.isEmpty else {
            return
        }
        guard let rep = image.representations.first as? NSBitmapImageRep,
              let data = rep.representation(using: .png, properties: [:]) else {
            throw TranscriptFrameError(description: "frame \(name) has no PNG representation")
        }
        let root = URL(fileURLWithPath: directory, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try data.write(to: root.appendingPathComponent(name))
    }

    static func record(
        _ attributed: NSAttributedString,
        named name: String,
        width: CGFloat = defaultWidth,
        dark: Bool
    ) throws {
        try record(try image(attributed, width: width, dark: dark), named: name)
    }
}
