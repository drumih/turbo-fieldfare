import CoreGraphics
import Foundation
import ImageIO
import Metal

public final class Gemma4ImagePreprocessor {
    private static let unitBF16 = (0...255).map {
        Quantization.bf16Bits(Float($0) / 255)
    }

    private let device: MTLDevice
    private let config: VisionConfig
    private let metadataReader: ImageMetadataReader
    /// Set when the reduction runs on the GPU. Building it needs a full
    /// `MetalContext`, so callers that only have a device keep the CPU filter.
    private let gpuResize: VisionResize?

    public init(
        device: MTLDevice,
        config: VisionConfig = VisionConfig(),
        limits: VisionImageLimits = VisionImageLimits(),
        gpuResize: VisionResize? = nil
    ) {
        self.device = device
        self.config = config
        self.metadataReader = ImageMetadataReader(limits: limits)
        self.gpuResize = gpuResize
    }

    public func preprocess(fileURL: URL) throws -> VisionPixelBuffer {
        try preprocess(VisionImageSource(fileURL: fileURL))
    }

    public func preprocess(_ image: VisionImageSource) throws -> VisionPixelBuffer {
        try preprocess(plan(image))
    }

    public func plan(fileURL: URL) throws -> VisionImagePlan {
        try plan(VisionImageSource(fileURL: fileURL))
    }

    /// The geometry a request's admission needs, from headers alone. The soft
    /// token count follows from oriented dimensions, so admission has no
    /// reason to pay the JPEG scan-termination walk a full plan performs —
    /// the encode that follows re-plans with full verification anyway.
    public func admissionGeometry(fileURL: URL) throws -> Gemma4ImageGeometry {
        let image = try VisionImageSource(fileURL: fileURL)
        let opened = try image.open(
            maximumEncodedBytes: metadataReader.limits.maximumEncodedBytes)
        let metadata = try metadataReader.read(
            opened: opened, verifyStreamCompleteness: false)
        return try Gemma4ImageGeometry(
            sourceWidth: metadata.orientedWidth,
            sourceHeight: metadata.orientedHeight,
            config: config)
    }

    public func plan(_ image: VisionImageSource) throws -> VisionImagePlan {
        let started = ContinuousClock.now
        let opened = try image.open(
            maximumEncodedBytes: metadataReader.limits.maximumEncodedBytes)
        let metadata = try metadataReader.read(opened: opened)
        let geometry = try Gemma4ImageGeometry(
            sourceWidth: metadata.orientedWidth,
            sourceHeight: metadata.orientedHeight,
            config: config)
        return VisionImagePlan(
            metadata: metadata, geometry: geometry, opened: opened, started: started)
    }

    /// The size ImageIO is asked to decode at.
    ///
    /// **Known gap, deliberately left open.** For a source larger than the
    /// target, ImageIO's own scaler does the reduction and `TorchBicubicResize`
    /// — the stage that exists to match the reference implementation — runs on
    /// pre-scaled pixels. A 4032x3024 photo bound for 912x672 is reduced
    /// 4032->912 by ImageIO and only 684->672 by bicubic, so downscales are not
    /// bit-comparable with the pinned Transformers/PIL reference. Every corpus
    /// fixture is an upscale, so this is unmeasured as well as unfixed.
    ///
    /// Handing the whole reduction to bicubic restores parity and was tried:
    /// it costs 4.5 s for a 1600x1200 source, 18.6 s at 12 MP and 61 s with
    /// 240 MB of footprint at 48 MP, because the resampler is CPU and runs at
    /// about 1.5 us per source pixel. That is far worse than the gap it closes.
    /// Closing it properly needs a fast resampler (vImage or GPU) first.
    static func decodeMaxPixelSize(
        metadata: VisionImageMetadata,
        geometry: Gemma4ImageGeometry
    ) -> Int {
        // The whole reduction belongs to our filter, which is the pinned
        // reference's. Decoding at the target instead handed it to ImageIO's
        // undocumented scaler: on a UI screenshot that changed the model's
        // transcription, dropping whole lines of text. It was reverted once for
        // costing seconds — a debug-build measurement. Release, on the GPU, it
        // is a few milliseconds.
        max(metadata.orientedWidth, metadata.orientedHeight)
    }

    public func preprocess(_ plan: VisionImagePlan) throws -> VisionPixelBuffer {
        let source = plan.opened.source
        let metadata = plan.metadata
        let geometry = plan.geometry
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize:
                Self.decodeMaxPixelSize(metadata: metadata, geometry: geometry),
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceShouldAllowFloat: false,
        ]
        guard let decoded = CGImageSourceCreateThumbnailAtIndex(
            source, 0, thumbnailOptions as CFDictionary) else {
            throw VisionImageError.decodeFailed
        }

        let sourceRowBytes = try checkedMultiply(decoded.width, 4)
        let sourceBytes = try checkedMultiply(sourceRowBytes, decoded.height)
        // Decoded straight into GPU-visible memory when the GPU filter will
        // read it, which removes one copy but not the fundamental cost: ImageIO
        // materialises a full-resolution CGImage and `draw` needs both it and
        // this surface live at once. For a 48 MP source that is two ~192 MB
        // surfaces, about 400 MB, on a machine already holding the model — the
        // reason the decoded-byte ceiling is what it is, and why both are
        // counted in `allocatedBytes` below rather than only ours.
        let sourceBuffer = gpuResize == nil
            ? nil
            : device.makeBuffer(length: sourceBytes, options: .storageModeShared)
        let sourceRGBA = sourceBuffer?.contents()
            ?? UnsafeMutableRawPointer.allocate(byteCount: sourceBytes, alignment: 64)
        sourceRGBA.initializeMemory(as: UInt8.self, repeating: 255, count: sourceBytes)
        defer { if sourceBuffer == nil { sourceRGBA.deallocate() } }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let sourceDrawing = CGContext(
                data: sourceRGBA,
                width: decoded.width,
                height: decoded.height,
                bitsPerComponent: 8,
                bytesPerRow: sourceRowBytes,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw VisionImageError.allocationFailed
        }
        sourceDrawing.setFillColor(CGColor(gray: 1, alpha: 1))
        sourceDrawing.fill(CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
        sourceDrawing.draw(
            decoded, in: CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
        withExtendedLifetime(sourceDrawing) {}

        let rowBytes = try checkedMultiply(geometry.processedWidth, 4)
        let rgbaBytes = try checkedMultiply(rowBytes, geometry.processedHeight)
        let rgba = UnsafeMutableRawPointer.allocate(byteCount: rgbaBytes, alignment: 64)
        defer { rgba.deallocate() }
        // Same filter either way — the GPU path applies the weights this one
        // builds, and the parity tests assert byte equality — so this chooses
        // only where the arithmetic runs.
        // Both surfaces plus whatever the resize holds. Reporting only our own
        // allocation understated the peak by about half, and a test uses this
        // number as a footprint guard.
        let decodedSurfaceBytes = sourceBytes
        var resizeScratchBytes = 0
        if let gpuResize, let sourceBuffer {
            resizeScratchBytes = gpuResize.scratchBytes(
                sourceWidth: decoded.width, sourceHeight: decoded.height,
                destinationWidth: geometry.processedWidth,
                destinationHeight: geometry.processedHeight)
            try gpuResize.resize(
                sourceBuffer: sourceBuffer,
                sourceWidth: decoded.width,
                sourceHeight: decoded.height,
                sourceRowBytes: sourceRowBytes,
                destination: rgba,
                destinationWidth: geometry.processedWidth,
                destinationHeight: geometry.processedHeight,
                destinationRowBytes: rowBytes)
        } else {
            resizeScratchBytes = TorchBicubicResize.resize(
                source: sourceRGBA.assumingMemoryBound(to: UInt8.self),
                sourceWidth: decoded.width,
                sourceHeight: decoded.height,
                sourceRowBytes: sourceRowBytes,
                destination: rgba.assumingMemoryBound(to: UInt8.self),
                destinationWidth: geometry.processedWidth,
                destinationHeight: geometry.processedHeight,
                destinationRowBytes: rowBytes)
        }

        let patchElements = try checkedMultiply(geometry.patchCount, config.patchDimension)
        let patchBytes = try checkedMultiply(patchElements, MemoryLayout<UInt16>.stride)
        let positionElements = try checkedMultiply(geometry.patchCount, 2)
        let positionBytes = try checkedMultiply(
            positionElements, MemoryLayout<Int32>.stride)
        guard let patches = device.makeBuffer(length: patchBytes, options: .storageModeShared),
              let positions = device.makeBuffer(
                length: positionBytes, options: .storageModeShared) else {
            throw VisionImageError.allocationFailed
        }
        patchify(
            rgba: rgba.assumingMemoryBound(to: UInt8.self),
            rowBytes: rowBytes,
            geometry: geometry,
            patches: patches,
            positions: positions)
        return VisionPixelBuffer(
            patchesBF16: patches,
            positionsInt32x2: positions,
            metadata: metadata,
            geometry: geometry,
            wallNanoseconds: nanoseconds(plan.started.duration(to: .now)),
            // Both full-resolution surfaces: ImageIO's decoded CGImage and
            // ours, which `draw` holds simultaneously.
            allocatedBytes: decodedSurfaceBytes + sourceBytes + rgbaBytes
                + resizeScratchBytes
                + patchBytes + positionBytes)
    }

    private func patchify(
        rgba: UnsafePointer<UInt8>,
        rowBytes: Int,
        geometry: Gemma4ImageGeometry,
        patches: MTLBuffer,
        positions: MTLBuffer
    ) {
        let patchPointer = patches.contents().bindMemory(
            to: UInt16.self, capacity: geometry.patchCount * config.patchDimension)
        let positionPointer = positions.contents().bindMemory(
            to: Int32.self, capacity: geometry.patchCount * 2)
        let size = config.patchSize
        for patchY in 0..<geometry.patchGridHeight {
            for patchX in 0..<geometry.patchGridWidth {
                let row = patchY * geometry.patchGridWidth + patchX
                positionPointer[row * 2] = Int32(patchX)
                positionPointer[row * 2 + 1] = Int32(patchY)
                var output = row * config.patchDimension
                for pixelY in 0..<size {
                    let inputRow = (patchY * size + pixelY) * rowBytes
                    for pixelX in 0..<size {
                        let input = inputRow + (patchX * size + pixelX) * 4
                        patchPointer[output] = Self.unitBF16[Int(rgba[input])]
                        patchPointer[output + 1] = Self.unitBF16[Int(rgba[input + 1])]
                        patchPointer[output + 2] = Self.unitBF16[Int(rgba[input + 2])]
                        output += 3
                    }
                }
            }
        }
    }

    private func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw VisionImageError.invalidMetadata("buffer size overflow")
        }
        return value
    }

    private func nanoseconds(_ duration: Duration) -> UInt64 {
        let components = duration.components
        let seconds = UInt64(max(0, components.seconds))
        let fractional = UInt64(max(0, components.attoseconds / 1_000_000_000))
        return seconds * 1_000_000_000 + fractional
    }
}
