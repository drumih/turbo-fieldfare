import Metal

/// Decode's single-row GEMVs for up to nine activation rows, with each weight
/// row read once (`multirow_gemv.metal`).
///
/// The wide verify runs W rows of one sequence and must produce, for every
/// row, the bytes scalar decode would. The prefill tile kernels reorder the
/// reduction; looping decode's GEMV per row keeps the order but reads every
/// weight matrix W times. This keeps both: per row it is the decode kernel's
/// arithmetic, operation for operation, and the weights are read once per
/// dispatch. The tests hold it to that byte for byte against the decode
/// kernels (`MultiRowGEMVTests`).
///
/// One dispatch takes up to four segments over a concatenated output-row
/// space -- the fused GDN input projection's qkv/z/a/b, or q/k/v -- each with
/// its own weights, width and output, all reading the same activation rows.
/// More rows than `maxRowsPerDispatch` are tiled; each tile reads the weights
/// again.
final class MultiRowGEMV {
    static let maxRowsPerDispatch = 9
    static let maxSegments = 4
    private static let rowsPerThreadgroup = 8

    /// How a segment's weights are stored, and so which decode kernel's
    /// arithmetic its rows follow.
    enum Format: Equatable, Sendable {
        /// `dequant_int4_gemv_simd`: MLX affine 4-bit, group 64.
        case int4
        /// `affine_quant_gemv_simd` at 2 or 8 bits.
        case affine(bits: Int)
        /// `bf16_gemv_simd`: the checkpoint's own bf16, no scales or biases.
        case bf16

        fileprivate var code: UInt32 {
            switch self {
            case .int4: 4
            case .affine(let bits): UInt32(bits)
            case .bf16: 16
            }
        }
    }

    /// One weight matrix and where its `rows` outputs for every activation row
    /// go: activation row `r` writes `y[yOffset + r * yRowStride ...]`.
    struct Segment {
        let weights: MTLBuffer
        let weightsOffset: Int
        let scales: MTLBuffer
        let scalesOffset: Int
        let biases: MTLBuffer
        let biasesOffset: Int
        let format: Format
        let rows: Int
        let y: MTLBuffer
        let yOffset: Int
        /// In elements.
        let yRowStride: Int

        /// A resident tensor: scales and biases live in the weights' buffer.
        init(
            view: TensorView, format: Format, rows: Int,
            y: MTLBuffer, yOffset: Int = 0, yRowStride: Int
        ) {
            self.weights = view.buffer
            self.weightsOffset = Int(view.offset)
            self.scales = view.buffer
            self.scalesOffset = Int(view.scaleOffset)
            self.biases = view.buffer
            self.biasesOffset = Int(view.biasOffset)
            self.format = format
            self.rows = rows
            self.y = y
            self.yOffset = yOffset
            self.yRowStride = yRowStride
        }

        init(
            weights: MTLBuffer, weightsOffset: Int,
            scales: MTLBuffer, scalesOffset: Int,
            biases: MTLBuffer, biasesOffset: Int,
            format: Format, rows: Int,
            y: MTLBuffer, yOffset: Int = 0, yRowStride: Int
        ) {
            self.weights = weights
            self.weightsOffset = weightsOffset
            self.scales = scales
            self.scalesOffset = scalesOffset
            self.biases = biases
            self.biasesOffset = biasesOffset
            self.format = format
            self.rows = rows
            self.y = y
            self.yOffset = yOffset
            self.yRowStride = yRowStride
        }
    }

    private let context: MetalContext

    init(context: MetalContext) {
        self.context = context
    }

    /// `y_s[r, i] = sum_k W_s[i, k] * x[r, k]` for every segment `s`, activation
    /// row `r < activationRows`, output row `i < segment.rows`.
    ///
    /// Row strides are in elements. `x` is read as `half4`, so its offset and
    /// row stride must keep 8-byte alignment.
    func encode(
        commandBuffer: MTLCommandBuffer,
        segments: [Segment],
        x: MTLBuffer, xOffset: Int = 0, xRowStride: Int,
        activationRows: Int,
        columns n: Int
    ) throws {
        precondition(
            (1...Self.maxSegments).contains(segments.count),
            "MultiRowGEMV takes 1...\(Self.maxSegments) segments")
        precondition(
            n > 0 && n.isMultiple(of: Quantization.groupSize),
            "MultiRowGEMV needs a column count that is a multiple of \(Quantization.groupSize)")
        let half = MemoryLayout<Float16>.stride
        precondition(
            xOffset.isMultiple(of: 4 * half) && xRowStride.isMultiple(of: 4),
            "MultiRowGEMV reads x as half4: offset and row stride must be 4-element aligned")
        for segment in segments {
            precondition(segment.rows > 0, "a MultiRowGEMV segment needs rows")
            switch segment.format {
            case .int4, .bf16:
                precondition(segment.weightsOffset.isMultiple(of: 2))
            case .affine(let bits):
                precondition([2, 8].contains(bits), "affine segments are 2 or 8 bits")
                precondition(segment.weightsOffset.isMultiple(of: 4))
            }
        }
        var first = 0
        while first < activationRows {
            let count = min(Self.maxRowsPerDispatch, activationRows - first)
            try encodeTile(
                commandBuffer: commandBuffer, segments: segments,
                x: x, xOffset: xOffset + first * xRowStride * half,
                xRowStride: xRowStride,
                firstRow: first, rows: count, columns: n)
            first += count
        }
    }

    private func encodeTile(
        commandBuffer: MTLCommandBuffer,
        segments: [Segment],
        x: MTLBuffer, xOffset: Int, xRowStride: Int,
        firstRow: Int, rows: Int, columns n: Int
    ) throws {
        var constants = [
            MetalFunctionConstant(index: 130, value: .uint32(UInt32(rows))),
            MetalFunctionConstant(index: 131, value: .uint32(UInt32(n))),
        ]
        for (index, segment) in segments.enumerated() {
            constants.append(
                MetalFunctionConstant(index: 132 + index, value: .uint32(segment.format.code)))
        }
        let pipeline = try context.pipeline(
            "multirow_gemv_simd", constants: constants,
            maxTotalThreadsPerThreadgroup: 32 * Self.rowsPerThreadgroup)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        let half = MemoryLayout<Float16>.stride
        var params = [UInt32](repeating: 0, count: 2 * Self.maxSegments + 1)
        for slot in 0..<Self.maxSegments {
            // Unused slots bind segment 0's buffers and get zero rows, so the
            // kernel never reads them.
            let segment = slot < segments.count ? segments[slot] : segments[0]
            encoder.setBuffer(segment.weights, offset: segment.weightsOffset, index: slot * 3)
            encoder.setBuffer(segment.scales, offset: segment.scalesOffset, index: slot * 3 + 1)
            encoder.setBuffer(segment.biases, offset: segment.biasesOffset, index: slot * 3 + 2)
            encoder.setBuffer(
                segment.y,
                offset: segment.yOffset + firstRow * segment.yRowStride * half,
                index: 13 + slot)
            if slot < segments.count {
                params[slot] = UInt32(segment.rows)
                params[Self.maxSegments + slot] = UInt32(segment.yRowStride)
            }
        }
        params[2 * Self.maxSegments] = UInt32(xRowStride)
        encoder.setBuffer(x, offset: xOffset, index: 12)
        encoder.setBytes(
            &params, length: params.count * MemoryLayout<UInt32>.stride, index: 17)
        let totalRows = segments.reduce(0) { $0 + $1.rows }
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (totalRows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
                height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
