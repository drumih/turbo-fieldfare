import Foundation
import Metal

/// GPU separable bicubic resize, byte-identical to `TorchBicubicResize`.
///
/// The image front-end is the one place where a wrong filter changes what the
/// model reads: measured on a UI screenshot, letting ImageIO do the reduction
/// instead of this filter changed the transcription, dropping whole lines of
/// text. Doing the reduction ourselves on the CPU fixes that but costs 0.1 to
/// 0.2 s per image, which is most of the encode budget.
///
/// So the same arithmetic runs here. The weights come from
/// `TorchBicubicResize.weightTable`, so the two paths cannot drift, and the
/// tests assert byte equality rather than a tolerance.
public final class VisionResize {
    private struct Params {
        var sourceWidth: UInt32
        var sourceHeight: UInt32
        var destinationWidth: UInt32
        var destinationHeight: UInt32
        var sourceRowBytes: UInt32
        var destinationRowBytes: UInt32
        var tapStride: UInt32
        var precision: Int32
    }

    private let context: MetalContext
    private let horizontal: MTLComputePipelineState
    private let vertical: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try MetalContext.privateLibrary(
            device: context.device, module: "vision_resize")
        guard let horizontalFunction = library.makeFunction(name: "vision_resize_horizontal"),
              let verticalFunction = library.makeFunction(name: "vision_resize_vertical") else {
            throw MetalError.missingFunction("vision resize")
        }
        horizontal = try context.device.makeComputePipelineState(function: horizontalFunction)
        vertical = try context.device.makeComputePipelineState(function: verticalFunction)
    }

    /// Bytes this resize will hold in GPU-visible buffers, so the caller can
    /// report a peak that includes them.
    public func scratchBytes(
        sourceWidth: Int, sourceHeight: Int,
        destinationWidth: Int, destinationHeight: Int
    ) -> Int {
        guard sourceWidth != destinationWidth || sourceHeight != destinationHeight else {
            return 0
        }
        let intermediate = sourceHeight * destinationWidth * 4
        let output = destinationHeight * destinationWidth * 4
        return intermediate + output
    }

    /// Resizes RGBA8 pixels. `destination` must hold
    /// `destinationHeight * destinationRowBytes` bytes.
    public func resize(
        source: UnsafeRawPointer,
        sourceWidth: Int,
        sourceHeight: Int,
        sourceRowBytes: Int,
        destination: UnsafeMutableRawPointer,
        destinationWidth: Int,
        destinationHeight: Int,
        destinationRowBytes: Int
    ) throws {
        guard sourceWidth > 0, sourceHeight > 0,
              destinationWidth > 0, destinationHeight > 0 else { return }
        if sourceWidth == destinationWidth && sourceHeight == destinationHeight {
            for row in 0..<sourceHeight {
                memcpy(destination.advanced(by: row * destinationRowBytes),
                       source.advanced(by: row * sourceRowBytes),
                       sourceWidth * 4)
            }
            return
        }

        let device = context.device
        guard let sourceBuffer = device.makeBuffer(
            bytes: source, length: sourceHeight * sourceRowBytes,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        try resize(sourceBuffer: sourceBuffer,
                   sourceWidth: sourceWidth, sourceHeight: sourceHeight,
                   sourceRowBytes: sourceRowBytes,
                   destination: destination,
                   destinationWidth: destinationWidth,
                   destinationHeight: destinationHeight,
                   destinationRowBytes: destinationRowBytes)
    }

    /// The same resize, reading pixels the caller already put in GPU-visible
    /// memory.
    ///
    /// A native-resolution decode is the largest transient in the whole image
    /// path — 192 MB for a 48 MP source — and copying it into a buffer doubled
    /// that. Callers that can decode straight into an `MTLBuffer` should.
    public func resize(
        sourceBuffer: MTLBuffer,
        sourceWidth: Int,
        sourceHeight: Int,
        sourceRowBytes: Int,
        destination: UnsafeMutableRawPointer,
        destinationWidth: Int,
        destinationHeight: Int,
        destinationRowBytes: Int
    ) throws {
        let device = context.device
        let source = UnsafeRawPointer(sourceBuffer.contents())

        // Horizontal first, into an intermediate that keeps every source row:
        // the vertical pass then reads columns of the already-narrowed image,
        // which is the cheaper order whenever the width shrinks.
        let intermediateRowBytes = destinationWidth * 4
        guard let intermediate = device.makeBuffer(
            length: max(1, sourceHeight * intermediateRowBytes),
            options: .storageModeShared),
              let output = device.makeBuffer(
                length: max(1, destinationHeight * destinationRowBytes),
                options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.noDevice
        }

        if sourceWidth != destinationWidth {
            try encode(commandBuffer: commandBuffer,
                       pipeline: horizontal,
                       table: TorchBicubicResize.weightTable(
                        input: sourceWidth, output: destinationWidth),
                       source: sourceBuffer,
                       destination: intermediate,
                       params: Params(
                        sourceWidth: UInt32(sourceWidth),
                        sourceHeight: UInt32(sourceHeight),
                        destinationWidth: UInt32(destinationWidth),
                        destinationHeight: UInt32(sourceHeight),
                        sourceRowBytes: UInt32(sourceRowBytes),
                        destinationRowBytes: UInt32(intermediateRowBytes),
                        tapStride: 0, precision: 0),
                       gridWidth: destinationWidth,
                       gridHeight: sourceHeight)
        } else {
            for row in 0..<sourceHeight {
                memcpy(intermediate.contents().advanced(by: row * intermediateRowBytes),
                       source.advanced(by: row * sourceRowBytes),
                       sourceWidth * 4)
            }
        }

        if sourceHeight != destinationHeight {
            try encode(commandBuffer: commandBuffer,
                       pipeline: vertical,
                       table: TorchBicubicResize.weightTable(
                        input: sourceHeight, output: destinationHeight),
                       source: intermediate,
                       destination: output,
                       params: Params(
                        sourceWidth: UInt32(destinationWidth),
                        sourceHeight: UInt32(sourceHeight),
                        destinationWidth: UInt32(destinationWidth),
                        destinationHeight: UInt32(destinationHeight),
                        sourceRowBytes: UInt32(intermediateRowBytes),
                        destinationRowBytes: UInt32(destinationRowBytes),
                        tapStride: 0, precision: 0),
                       gridWidth: destinationWidth * 4,
                       gridHeight: destinationHeight)
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let finished = sourceHeight == destinationHeight ? intermediate : output
        let rowBytes = sourceHeight == destinationHeight
            ? intermediateRowBytes : destinationRowBytes
        for row in 0..<destinationHeight {
            memcpy(destination.advanced(by: row * destinationRowBytes),
                   finished.contents().advanced(by: row * rowBytes),
                   destinationWidth * 4)
        }
    }

    private func encode(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        table: TorchBicubicResize.WeightTable,
        source: MTLBuffer,
        destination: MTLBuffer,
        params: Params,
        gridWidth: Int,
        gridHeight: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.noQueue
        }
        var params = params
        params.tapStride = UInt32(table.tapStride)
        params.precision = Int32(table.precision)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        // Buffers, not `setBytes`: that path is capped at 4 KB and a 4x
        // reduction of a 4,000-pixel axis needs about 32 KB of taps, which
        // aborted the process rather than failing a call.
        guard let starts = context.device.makeBuffer(
            bytes: table.starts,
            length: table.starts.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared),
              let weights = context.device.makeBuffer(
                bytes: table.weights,
                length: table.weights.count * MemoryLayout<Int16>.stride,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        encoder.setBuffer(starts, offset: 0, index: 2)
        encoder.setBuffer(weights, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 4)

        let width = min(pipeline.threadExecutionWidth, gridWidth)
        let height = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / width, gridHeight))
        encoder.dispatchThreads(
            MTLSize(width: gridWidth, height: gridHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()
    }
}
