import CoreGraphics
import Darwin
import Foundation
import ImageIO

public enum VisionImageError: Error, CustomStringConvertible {
    case invalidSource(String)
    case invalidMetadata(String)
    case unsupportedFrameCount(Int)
    case sourceTooLarge(bytes: Int, limit: Int)
    case dimensionsTooLarge(width: Int, height: Int, pixelLimit: Int)
    case sideTooLarge(width: Int, height: Int, sideLimit: Int)
    case decodedBytesTooLarge(width: Int, height: Int, byteLimit: Int)
    case decodeFailed
    case allocationFailed

    public var description: String {
        switch self {
        case .invalidSource(let detail): "invalid image source: \(detail)"
        case .invalidMetadata(let detail): "invalid image metadata: \(detail)"
        case .unsupportedFrameCount(let count): "image must contain one frame, found \(count)"
        case .sourceTooLarge(let bytes, let limit):
            "encoded image has \(bytes) bytes, limit is \(limit)"
        case .dimensionsTooLarge(let width, let height, let limit):
            "image dimensions \(width)x\(height) exceed the \(limit)-pixel limit"
        case .sideTooLarge(let width, let height, let limit):
            "image dimensions \(width)x\(height) exceed the \(limit)-pixel per-side limit"
        case .decodedBytesTooLarge(let width, let height, let limit):
            "image \(width)x\(height) would decode to more than \(limit) bytes"
        case .decodeFailed: "ImageIO could not decode the image"
        case .allocationFailed: "image preprocessing buffer allocation failed"
        }
    }
}

public struct VisionImageSource: Sendable {
    public let fileURL: URL

    public init(fileURL: URL) throws {
        guard fileURL.isFileURL else {
            throw VisionImageError.invalidSource("a file URL is required")
        }
        self.fileURL = fileURL.standardizedFileURL
    }

    package func open(maximumEncodedBytes: Int) throws -> OpenedVisionImage {
        let descriptor = fileURL.path.withCString {
            // O_NONBLOCK or a FIFO blocks here forever: the regular-file check
            // below runs only once open returns. pread on a regular file ignores
            // it, so nothing downstream changes. Matches every other open here.
            Darwin.open($0, O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW)
        }
        guard descriptor >= 0 else {
            throw VisionImageError.invalidSource(
                "open failed for \(fileURL.lastPathComponent): \(String(cString: strerror(errno)))")
        }
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            let savedErrno = errno
            Darwin.close(descriptor)
            throw VisionImageError.invalidSource(
                "fstat failed: \(String(cString: strerror(savedErrno)))")
        }
        guard (status.st_mode & S_IFMT) == S_IFREG, status.st_size >= 0,
              let encodedBytes = Int(exactly: status.st_size) else {
            Darwin.close(descriptor)
            throw VisionImageError.invalidSource("a regular file is required")
        }
        guard encodedBytes <= maximumEncodedBytes else {
            Darwin.close(descriptor)
            throw VisionImageError.sourceTooLarge(
                bytes: encodedBytes, limit: maximumEncodedBytes)
        }

        let fileProvider = VisionFileProvider(
            descriptor: descriptor, encodedBytes: encodedBytes)
        let context = Unmanaged.passRetained(fileProvider)
        var callbacks = CGDataProviderDirectCallbacks(
            version: 0,
            getBytePointer: nil,
            releaseBytePointer: nil,
            getBytesAtPosition: { opaque, buffer, position, count in
                guard let opaque else { return 0 }
                return Unmanaged<VisionFileProvider>.fromOpaque(opaque)
                    .takeUnretainedValue()
                    .read(into: buffer, position: position, count: count)
            },
            releaseInfo: { opaque in
                guard let opaque else { return }
                Unmanaged<VisionFileProvider>.fromOpaque(opaque).release()
            })
        guard let provider = CGDataProvider(
            directInfo: context.toOpaque(), size: off_t(encodedBytes), callbacks: &callbacks) else {
            context.release()
            throw VisionImageError.invalidSource("could not create image data provider")
        }
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithDataProvider(provider, options) else {
            throw VisionImageError.invalidSource("ImageIO rejected \(fileURL.lastPathComponent)")
        }
        return OpenedVisionImage(
            source: source,
            provider: provider,
            fileProvider: fileProvider,
            encodedBytes: encodedBytes)
    }
}

package struct OpenedVisionImage {
    let source: CGImageSource
    private let provider: CGDataProvider
    private let fileProvider: VisionFileProvider
    let encodedBytes: Int

    fileprivate init(
        source: CGImageSource,
        provider: CGDataProvider,
        fileProvider: VisionFileProvider,
        encodedBytes: Int
    ) {
        self.source = source
        self.provider = provider
        self.fileProvider = fileProvider
        self.encodedBytes = encodedBytes
    }


    /// Whether the JPEG's entropy-coded scan is terminated by an end-of-image
    /// marker, which is the only question that separates a truncated file from
    /// a complete one.
    ///
    /// Neither simpler test works. ImageIO reports `statusComplete` for a file
    /// truncated to a fifth of its length — the incomplete statuses only apply
    /// to incremental sources — and it then decodes the partial scan with gray
    /// filler, so the model describes an image the file does not contain.
    /// Scanning the tail for `FF D9` fails in both directions: a truncated photo
    /// still carries the EOI of its EXIF thumbnail near the front, and a Motion
    /// Photo's appended MP4 pushes the real EOI outside any fixed window.
    ///
    /// Markers are walked from the front to find the start of scan; anything
    /// before it (thumbnails, ICC, XMP) is skipped by construction. A single
    /// `FF D9` anywhere after that point means the scan was terminated, which
    /// tolerates trailing video and padding.
    func hasCompleteJPEGStream() -> Bool {
        guard let scanStart = startOfScanOffset() else { return false }
        var position = scanStart
        var carry: UInt8 = 0
        var buffer = [UInt8](repeating: 0, count: Self.scanChunkBytes)
        while position < encodedBytes {
            let count = min(Self.scanChunkBytes, encodedBytes - position)
            let read = buffer.withUnsafeMutableBytes {
                fileProvider.read(into: $0.baseAddress!,
                                  position: off_t(position), count: count)
            }
            guard read == count else { return false }
            if carry == 0xFF, buffer[0] == 0xD9 { return true }
            var index = 0
            while index + 1 < count {
                if buffer[index] == 0xFF, buffer[index + 1] == 0xD9 { return true }
                index += 1
            }
            carry = buffer[count - 1]
            position += count
        }
        return false
    }

    /// Whether the PNG ends with its `IEND` chunk, which is the same question
    /// `hasCompleteJPEGStream` asks of a JPEG.
    ///
    /// A PNG cut short decodes exactly as badly and just as silently: ImageIO
    /// returns the rows it has and leaves the rest black, so the model
    /// confidently describes a black square. `IEND` is a fixed 12-byte trailer
    /// with a constant CRC, so the check is a single tail read rather than a
    /// walk, and appended data after `IEND` is not something PNG permits.
    func hasCompletePNGStream() -> Bool {
        let trailer = 12
        guard encodedBytes >= 8 + trailer else { return false }
        var tail = [UInt8](repeating: 0, count: trailer)
        let read = tail.withUnsafeMutableBytes {
            fileProvider.read(into: $0.baseAddress!,
                              position: off_t(encodedBytes - trailer),
                              count: trailer)
        }
        guard read == trailer else { return false }
        return tail == [0x00, 0x00, 0x00, 0x00,
                        0x49, 0x45, 0x4E, 0x44,
                        0xAE, 0x42, 0x60, 0x82]
    }

    /// Offset just past the SOS segment header, or nil if the file ends before
    /// one appears.
    private func startOfScanOffset() -> Int? {
        var position = 2  // past SOI
        var header = [UInt8](repeating: 0, count: 4)
        while position + 4 <= min(encodedBytes, Self.markerScanLimitBytes) {
            let read = header.withUnsafeMutableBytes {
                fileProvider.read(into: $0.baseAddress!,
                                  position: off_t(position), count: 4)
            }
            guard read == 4, header[0] == 0xFF else { return nil }
            // Fill bytes: any number of 0xFF may precede a marker (ITU T.81
            // B.1.1.2), and scanners and older camera firmware emit them.
            // Reading one as the marker made its length the real marker byte
            // and half the real length — a jump to nothing, and an intact photo
            // reported as truncated.
            if header[1] == 0xFF {
                position += 1
                continue
            }
            let marker = header[1]
            // Standalone markers carry no length. SOI and EOI belong here too:
            // treating them as length-prefixed walked off into the entropy data.
            if marker == 0x01 || marker == 0xD8 || marker == 0xD9
                || (0xD0...0xD7).contains(marker) {
                position += 2
                continue
            }
            let length = Int(header[2]) << 8 | Int(header[3])
            guard length >= 2 else { return nil }
            if marker == 0xDA { return position + 2 + length }
            position += 2 + length
        }
        return nil
    }

    /// Read granularity for the scan search.
    static let scanChunkBytes = 256 * 1_024
    /// How far into the file the marker walk may go before giving up. Headers
    /// live at the front; a file whose SOS is beyond this is not one we serve.
    static let markerScanLimitBytes = 8 * 1_024 * 1_024
}

private final class VisionFileProvider {
    private let descriptor: Int32
    private let encodedBytes: Int

    init(descriptor: Int32, encodedBytes: Int) {
        self.descriptor = descriptor
        self.encodedBytes = encodedBytes
    }

    deinit {
        Darwin.close(descriptor)
    }

    func read(into buffer: UnsafeMutableRawPointer, position: off_t, count: Int) -> Int {
        guard position >= 0, let offset = Int(exactly: position),
              offset <= encodedBytes else { return 0 }
        let requested = min(count, encodedBytes - offset)
        var completed = 0
        while completed < requested {
            let result = pread(
                descriptor,
                buffer.advanced(by: completed),
                requested - completed,
                off_t(offset + completed))
            if result > 0 {
                completed += result
            } else if result == 0 {
                break
            } else if errno != EINTR {
                break
            }
        }
        return completed
    }
}
