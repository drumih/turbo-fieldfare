import CoreGraphics
import Darwin
import Foundation
import ImageIO
import Metal
import TurboFieldfare
import TurboFieldfareRepackCore
import UniformTypeIdentifiers

public enum ConversationImageWriterError: Error, CustomStringConvertible {
    case noMetalDevice
    case modelInputUnavailable(name: String)
    case encodeFailed(path: String)

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "no Metal device, so an image cannot be prepared for storage"
        case .modelInputUnavailable(let name):
            return "preprocessing \(name) produced no model-input copy"
        case .encodeFailed(let path):
            return "could not write \(path)"
        }
    }
}

/// Writes the two copies of an image a conversation keeps.
///
/// The model-input copy is a lossless PNG of the post-resize RGB8 at the
/// tower's geometry. Not the file the user attached: decode, EXIF transform and
/// resampling all sit between that file and the model, none of them has a
/// documented pixel-stability contract across macOS versions, and one changed
/// pixel changes a 768-value patch vector, its activation, and after pooling at
/// least one soft-token embedding — which is a different KV and a different
/// continuation. It is already downscaled, because the tower never sees more
/// than 645,120 pixels; the "store it smaller" instinct is satisfied by the
/// model input itself.
///
/// The thumbnail is lossy on purpose: it is only ever looked at.
public struct ConversationImageWriter: Sendable {
    /// The size the transcript already asks for, so the composer's 96 px tile
    /// can be derived from it rather than from the model-input copy.
    public static let thumbnailMaxPixelSize = 720
    public static let thumbnailQuality = 0.8

    private let device: MTLDevice

    public init(device: MTLDevice) {
        self.device = device
    }

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ConversationImageWriterError.noMetalDevice
        }
        self.device = device
    }

    /// Preprocesses `attachment` once and writes both copies into `imagesDirectory`.
    ///
    /// One preprocess, not two: the stored bytes are the ones this call
    /// patchified, so there is no second decode to disagree with the first.
    public func write(
        attachment: StagedImage,
        into imagesDirectory: URL
    ) throws -> ConversationImageRecord {
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let plan = try preprocessor.plan(fileURL: attachment.fileURL)
        let output = try preprocessor.preprocess(plan, capturingModelInput: true)
        guard let modelInput = output.modelInput else {
            throw ConversationImageWriterError.modelInputUnavailable(
                name: attachment.displayName)
        }

        try FileManager.default.createDirectory(
            at: imagesDirectory, withIntermediateDirectories: true)
        // A later OS can preprocess the same source differently. Its new
        // pixels must not replace a previous turn's digest-bound replay input.
        let stem = String(attachment.sha256.prefix(16))
        let pixelsName = "\(stem)-\(modelInput.digest).png"
        let thumbnailName = "\(stem).thumb.jpg"

        try Self.writeAtomically(to: imagesDirectory.appendingPathComponent(pixelsName)) {
            try writeModelInputPNG(modelInput, to: $0)
        }
        try Self.writeAtomically(to: imagesDirectory.appendingPathComponent(thumbnailName)) {
            try writeThumbnail(from: attachment.fileURL, to: $0)
        }

        return ConversationImageRecord(
            id: attachment.id,
            displayName: attachment.displayName,
            pixelsFile: "images/\(pixelsName)",
            thumbnailFile: "images/\(thumbnailName)",
            sourceDigest: attachment.sha256,
            modelInputDigest: modelInput.digest,
            width: modelInput.width,
            height: modelInput.height,
            softTokens: output.pixels.geometry.softTokenCount)
    }

    static func writeAtomically(to url: URL, encode: (URL) throws -> Void) throws {
        let directory = url.deletingLastPathComponent()
        let temporary = directory.appendingPathComponent("\(UUID()).tmp")
        // Encode to disk, not a second image-sized heap buffer. A non-hidden
        // temporary left by a crash is collected by the store's orphan sweep.
        defer {
            if Darwin.unlink(temporary.path) != 0 {
                let code = errno
                if code != ENOENT {
                    FileHandle.standardError.write(Data(
                        "Could not remove image temporary \(temporary.lastPathComponent): errno \(code)\n".utf8))
                }
            }
        }
        try encode(temporary)
        let descriptor = try Posix.openReadNoFollow(temporary.path)
        defer { Darwin.close(descriptor) }
        try Posix.fsync(descriptor, path: temporary.path)
        try Posix.rename(from: temporary.path, to: url.path)
        try Posix.fsyncDirectory(directory.path)
    }

    /// 24-bit sRGB, no alpha, no resampling: the bytes as they are, so decoding
    /// the file hands them straight back.
    private func writeModelInputPNG(
        _ pixels: VisionModelInputPixels, to url: URL
    ) throws {
        guard let provider = CGDataProvider(data: Data(pixels.rgb8) as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(
                width: pixels.width, height: pixels.height,
                bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: pixels.width * 3,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent) else {
            throw ConversationImageWriterError.encodeFailed(path: url.path)
        }
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw ConversationImageWriterError.encodeFailed(path: url.path)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw ConversationImageWriterError.encodeFailed(path: url.path)
        }
    }

    /// From the original rather than the model input, so the display copy keeps
    /// the aspect ratio the user recognises rather than the tower's 48-aligned
    /// one. `kCGImageSourceThumbnailMaxPixelSize` is always passed: without it
    /// `CGImageSourceCreateThumbnailAtIndex` has returned nil since iOS 17.4.
    private func writeThumbnail(from source: URL, to url: URL) throws {
        guard let imageSource = CGImageSourceCreateWithURL(source as CFURL, nil),
              let thumbnail = CGImageSourceCreateThumbnailAtIndex(imageSource, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: Self.thumbnailMaxPixelSize,
                kCGImageSourceShouldCacheImmediately: true,
              ] as CFDictionary) else {
            throw ConversationImageWriterError.encodeFailed(path: url.path)
        }
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw ConversationImageWriterError.encodeFailed(path: url.path)
        }
        CGImageDestinationAddImage(destination, thumbnail, [
            kCGImageDestinationLossyCompressionQuality: Self.thumbnailQuality,
        ] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw ConversationImageWriterError.encodeFailed(path: url.path)
        }
    }
}
