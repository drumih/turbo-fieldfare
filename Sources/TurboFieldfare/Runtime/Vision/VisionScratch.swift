import Metal

package final class VisionScratch {
    package let paddedRows: Int
    /// Live rows, so the clear can skip everything a dispatch will overwrite.
    package let rows: Int
    private let hiddenSize: Int
    package let normalizedPatches: MTLBuffer
    package let hiddenA: MTLBuffer
    package let hiddenB: MTLBuffer
    package let q: MTLBuffer
    /// Nil under the fused padded attention layout, where the QKV projections
    /// write the attention-owned padded buffers instead of hidden-width scratch.
    package let k: MTLBuffer?
    package let v: MTLBuffer?
    package let attention: MTLBuffer
    package let gate: MTLBuffer
    /// Nil when the up projection folds GeGLU into its epilogue.
    package let up: MTLBuffer?

    package init(device: MTLDevice, rows: Int, config: VisionConfig,
                 needsSeparateKV: Bool = true,
                 needsSeparateUp: Bool = true) throws {
        paddedRows = ((rows + 63) / 64) * 64
        self.rows = rows
        self.hiddenSize = config.hiddenSize
        func buffer(_ elements: Int) throws -> MTLBuffer {
            guard let value = device.makeBuffer(
                length: elements * MemoryLayout<UInt16>.stride,
                options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return value
        }
        normalizedPatches = try buffer(paddedRows * config.patchDimension)
        hiddenA = try buffer(paddedRows * config.hiddenSize)
        hiddenB = try buffer(paddedRows * config.hiddenSize)
        q = try buffer(paddedRows * config.hiddenSize)
        k = needsSeparateKV ? try buffer(paddedRows * config.hiddenSize) : nil
        v = needsSeparateKV ? try buffer(paddedRows * config.hiddenSize) : nil
        attention = try buffer(paddedRows * config.hiddenSize)
        gate = try buffer(paddedRows * config.intermediateSize + 64)
        up = needsSeparateUp
            ? try buffer(paddedRows * config.intermediateSize + 64) : nil
    }

    private var buffers: [MTLBuffer] {
        [normalizedPatches, hiddenA, hiddenB, q, attention, gate]
            + [k, v, up].compactMap(\.self)
    }

    /// Zeroes only what a later read can actually see.
    ///
    /// Clearing every buffer meant about 47 MiB of blit fills per image at the
    /// 2,520-patch maximum, nearly all of it dead: `normalizedPatches`, `q` and
    /// `gate` are fully overwritten by the dispatch that follows, which zeroes
    /// its own pad region. What genuinely has to start clean is the padding
    /// rows of the two buffers whose tails are read back — `hiddenB` and
    /// `attention` — which is under 300 KB.
    package func encodeClear(commandBuffer: MTLCommandBuffer) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.noDevice
        }
        let liveBytes = rows * hiddenSize * MemoryLayout<UInt16>.stride
        for buffer in [hiddenB, attention] {
            guard buffer.length > liveBytes else { continue }
            blit.fill(buffer: buffer, range: liveBytes..<buffer.length, value: 0)
        }
        blit.endEncoding()
    }

    package var allocatedBytes: Int {
        buffers.reduce(0) { $0 + $1.length }
    }
}
