import CoreGraphics
import Foundation
import ImageIO
import Metal

public struct Gemma4ImagePreprocessingOptions: Sendable, Equatable {
    public var maximumFileBytes: UInt64
    public var maximumSourcePixels: UInt64
    public var maximumSoftTokens: Int

    public init(maximumFileBytes: UInt64 = 25 * 1_024 * 1_024,
                maximumSourcePixels: UInt64 = 64_000_000,
                maximumSoftTokens: Int = 280) {
        precondition(maximumFileBytes > 0)
        precondition(maximumSourcePixels > 0)
        precondition((1...280).contains(maximumSoftTokens))
        self.maximumFileBytes = maximumFileBytes
        self.maximumSourcePixels = maximumSourcePixels
        self.maximumSoftTokens = maximumSoftTokens
    }

    public static let gemma4Default = Gemma4ImagePreprocessingOptions()
}

public enum Gemma4ImagePreprocessingError: Error, CustomStringConvertible, Equatable {
    case missingFile
    case fileTooLarge(actual: UInt64, maximum: UInt64)
    case unsupportedOrCorruptImage
    case invalidDimensions(width: Int, height: Int)
    case sourcePixelLimitExceeded(actual: UInt64, maximum: UInt64)
    case drawingFailed
    case bufferAllocationFailed(bytes: Int)

    public var description: String {
        switch self {
        case .missingFile:
            return "image file does not exist"
        case .fileTooLarge(let actual, let maximum):
            return "image file is \(actual) bytes; maximum is \(maximum)"
        case .unsupportedOrCorruptImage:
            return "image format is unsupported or the image is corrupt"
        case .invalidDimensions(let width, let height):
            return "invalid image dimensions \(width)x\(height)"
        case .sourcePixelLimitExceeded(let actual, let maximum):
            return "image declares \(actual) pixels; maximum is \(maximum)"
        case .drawingFailed:
            return "ImageIO could not render the bounded RGB thumbnail"
        case .bufferAllocationFailed(let bytes):
            return "Metal could not allocate a \(bytes)-byte image patch buffer"
        }
    }
}

public struct Gemma4PreparedImage: @unchecked Sendable {
    public static let patchSize = 16
    public static let poolingKernelSize = 3
    public static let patchVectorSize = 16 * 16 * 3

    public let contentHash: String
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let resizedWidth: Int
    public let resizedHeight: Int
    public let patchColumns: Int
    public let patchRows: Int
    public let softTokenColumns: Int
    public let softTokenRows: Int
    public let patchValues: MTLBuffer

    public var patchCount: Int { patchColumns * patchRows }
    public var softTokenCount: Int { softTokenColumns * softTokenRows }

    /// Row-major `(x, y)` coordinates corresponding to `patchValues`.
    /// This is intentionally generated on demand because at most 2,520 pairs
    /// are needed and the encoder normally consumes the dimensions directly.
    public var patchPositionIDs: [SIMD2<Int32>] {
        var positions: [SIMD2<Int32>] = []
        positions.reserveCapacity(patchCount)
        for y in 0..<patchRows {
            for x in 0..<patchColumns {
                positions.append(SIMD2(Int32(x), Int32(y)))
            }
        }
        return positions
    }
}

/// Bounded Gemma 4 image preprocessing.
///
/// ImageIO is asked for a thumbnail before any bitmap allocation, then the
/// thumbnail is drawn bicubically into an exact multiple-of-48 RGB canvas.
/// Pixel values are patchified in row-major HWC order and transformed to
/// `2 * (pixel / 255 - 0.5)`, matching the model-side reference operation.
public enum Gemma4ImagePreprocessor {
    private static let alignmentPixels = Gemma4PreparedImage.patchSize
        * Gemma4PreparedImage.poolingKernelSize

    public static func prepare(imageAt imageURL: URL,
                               device: MTLDevice,
                               options: Gemma4ImagePreprocessingOptions = .gemma4Default) throws
        -> Gemma4PreparedImage {
        let fileAttributes: [FileAttributeKey: Any]
        do {
            fileAttributes = try FileManager.default.attributesOfItem(atPath: imageURL.path)
        } catch {
            throw Gemma4ImagePreprocessingError.missingFile
        }
        guard let fileSizeNumber = fileAttributes[.size] as? NSNumber else {
            throw Gemma4ImagePreprocessingError.missingFile
        }
        let fileSize = fileSizeNumber.uint64Value
        guard fileSize <= options.maximumFileBytes else {
            throw Gemma4ImagePreprocessingError.fileTooLarge(
                actual: fileSize,
                maximum: options.maximumFileBytes)
        }

        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, [
            kCGImageSourceShouldCache: false,
        ] as CFDictionary), CGImageSourceGetCount(source) > 0 else {
            throw Gemma4ImagePreprocessingError.unsupportedOrCorruptImage
        }
        guard let properties = CGImageSourceCopyPropertiesAtIndex(
            source,
            0,
            [kCGImageSourceShouldCache: false] as CFDictionary) as? [CFString: Any],
              let widthNumber = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let heightNumber = properties[kCGImagePropertyPixelHeight] as? NSNumber else {
            throw Gemma4ImagePreprocessingError.unsupportedOrCorruptImage
        }
        let sourceWidth = widthNumber.intValue
        let sourceHeight = heightNumber.intValue
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw Gemma4ImagePreprocessingError.invalidDimensions(
                width: sourceWidth,
                height: sourceHeight)
        }
        let sourcePixels = UInt64(sourceWidth) * UInt64(sourceHeight)
        guard sourcePixels <= options.maximumSourcePixels else {
            throw Gemma4ImagePreprocessingError.sourcePixelLimitExceeded(
                actual: sourcePixels,
                maximum: options.maximumSourcePixels)
        }

        let grid = outputGrid(sourceWidth: sourceWidth,
                              sourceHeight: sourceHeight,
                              maximumSoftTokens: options.maximumSoftTokens)
        let resizedWidth = grid.columns * alignmentPixels
        let resizedHeight = grid.rows * alignmentPixels
        let thumbnailMaximum = max(resizedWidth, resizedHeight)
        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: thumbnailMaximum,
            kCGImageSourceShouldCacheImmediately: true,
        ] as CFDictionary) else {
            throw Gemma4ImagePreprocessingError.unsupportedOrCorruptImage
        }

        let bytesPerPixel = 4
        let rowBytes = resizedWidth * bytesPerPixel
        let bitmapBytes = rowBytes * resizedHeight
        var bitmap = [UInt8](repeating: 0, count: bitmapBytes)
        let drewImage = bitmap.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(
                    data: base,
                    width: resizedWidth,
                    height: resizedHeight,
                    bitsPerComponent: 8,
                    bytesPerRow: rowBytes,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                        | CGBitmapInfo.byteOrder32Big.rawValue) else {
                return false
            }
            context.interpolationQuality = .high
            context.setBlendMode(.copy)
            context.translateBy(x: 0, y: CGFloat(resizedHeight))
            context.scaleBy(x: 1, y: -1)
            context.draw(thumbnail,
                         in: CGRect(x: 0, y: 0,
                                    width: resizedWidth,
                                    height: resizedHeight))
            return true
        }
        guard drewImage else {
            throw Gemma4ImagePreprocessingError.drawingFailed
        }

        let patchColumns = resizedWidth / Gemma4PreparedImage.patchSize
        let patchRows = resizedHeight / Gemma4PreparedImage.patchSize
        let patchCount = patchColumns * patchRows
        let elementCount = patchCount * Gemma4PreparedImage.patchVectorSize
        let outputBytes = elementCount * MemoryLayout<Float16>.stride
        guard let patchBuffer = device.makeBuffer(
            length: outputBytes,
            options: .storageModeShared) else {
            throw Gemma4ImagePreprocessingError.bufferAllocationFailed(bytes: outputBytes)
        }
        let output = patchBuffer.contents().bindMemory(to: Float16.self,
                                                       capacity: elementCount)
        let scale = Float(2.0 / 255.0)
        bitmap.withUnsafeBytes { raw in
            let pixels = raw.bindMemory(to: UInt8.self)
            var destination = 0
            for patchY in 0..<patchRows {
                for patchX in 0..<patchColumns {
                    for localY in 0..<Gemma4PreparedImage.patchSize {
                        let y = patchY * Gemma4PreparedImage.patchSize + localY
                        for localX in 0..<Gemma4PreparedImage.patchSize {
                            let x = patchX * Gemma4PreparedImage.patchSize + localX
                            let sourceIndex = y * rowBytes + x * bytesPerPixel
                            output[destination] = Float16(Float(pixels[sourceIndex]) * scale - 1)
                            output[destination + 1] = Float16(Float(pixels[sourceIndex + 1]) * scale - 1)
                            output[destination + 2] = Float16(Float(pixels[sourceIndex + 2]) * scale - 1)
                            destination += 3
                        }
                    }
                }
            }
        }

        return Gemma4PreparedImage(
            contentHash: try Sha256Verifier.hashFile(at: imageURL),
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            resizedWidth: resizedWidth,
            resizedHeight: resizedHeight,
            patchColumns: patchColumns,
            patchRows: patchRows,
            softTokenColumns: grid.columns,
            softTokenRows: grid.rows,
            patchValues: patchBuffer)
    }

    /// Computes the largest approximately aspect-preserving pooled-token grid
    /// under the requested budget. The resulting pixel dimensions are exact
    /// multiples of `3 * 16 == 48` and patch count is `softTokens * 9`.
    static func outputGrid(sourceWidth: Int,
                           sourceHeight: Int,
                           maximumSoftTokens: Int) -> (columns: Int, rows: Int) {
        precondition(sourceWidth > 0 && sourceHeight > 0)
        precondition(maximumSoftTokens > 0)
        let aspect = Double(sourceWidth) / Double(sourceHeight)
        var columns = max(1, Int(floor(sqrt(Double(maximumSoftTokens) * aspect))))
        var rows = max(1, Int(floor(sqrt(Double(maximumSoftTokens) / aspect))))
        while columns * rows > maximumSoftTokens {
            if Double(columns) / Double(rows) > aspect {
                columns -= 1
            } else {
                rows -= 1
            }
        }

        // The reference processor strictly floors each ideal dimension to a
        // multiple of 48; it does not fill spare budget with an extra row or
        // column because doing so changes the reference aspect calculation.
        return (columns, rows)
    }
}
