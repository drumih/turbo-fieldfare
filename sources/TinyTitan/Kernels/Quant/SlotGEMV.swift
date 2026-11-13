import Metal

/// The GEMV for one quantization slot, chosen by that slot's width.
///
/// `DequantInt4GEMV` and `AffineQuantGEMV` present the same `encode`
/// signature, so a caller needs nothing but to be handed the right one. The
/// reason this type exists is that three kernels did not: `HyperConnection`,
/// `PLEBlock` and `QSAIndexer` each constructed `DequantInt4GEMV`
/// unconditionally, and an 8-bit install then handed them 8-bit weights that
/// they read as nibbles -- same buffers, half the bytes, no error. The model
/// loaded, answered " Paris", and degenerated.
///
/// Anything reading weights whose width comes from the manifest should hold
/// one of these rather than a concrete class.
struct SlotGEMV {
    private enum Quantized {
        case int4(DequantInt4GEMV)
        case affine(AffineQuantGEMV)
    }

    private let quantized: Quantized
    /// Built alongside the quantized kernel because promotion is per *tensor*,
    /// not per slot: a family kept at the checkpoint's bf16 sits inside an
    /// otherwise-8-bit slot, so both kernels have to be available at once and
    /// the choice made when the weights are known.
    private let bf16: BF16GEMV
    /// Several activation rows with one weight read, in the single-row
    /// kernel's per-row arithmetic (the wide verify's decode-order pass).
    private let multiRow: MultiRowGEMV
    let weightBits: Int

    init(context: MetalContext, weightBits: Int) throws {
        precondition(
            [4, 8].contains(weightBits),
            "unsupported slot width \(weightBits)")
        self.quantized =
            weightBits == 4
            ? .int4(try DequantInt4GEMV(context: context))
            : .affine(try AffineQuantGEMV(context: context, weightBits: weightBits))
        self.bf16 = try BF16GEMV(context: context)
        self.multiRow = MultiRowGEMV(context: context)
        self.weightBits = weightBits
    }

    /// `encode` for `rows` activation rows at once: row `r` reads
    /// `x[xOffset + r * n]` and writes `y[yOffset + r * m]`, with the bytes
    /// `encode` would give that row alone. The weights are read once per nine
    /// rows rather than once per row.
    /// Compiles `encodeRows`'s pipelines at `columns` for every row count in
    /// `rows`, for this slot's width and for a promoted bf16 tensor.
    func prewarmRows(columns: [Int], rows: ClosedRange<Int>) throws {
        let quantized: MultiRowGEMV.Format = weightBits == 4 ? .int4 : .affine(bits: weightBits)
        for n in columns {
            try multiRow.prewarm(formats: [quantized], columns: n, rows: rows)
            try multiRow.prewarm(formats: [.bf16], columns: n, rows: rows)
        }
    }

    func encodeRows(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: Int, n: Int, rows: Int,
        isBF16: Bool = false
    ) throws {
        let format: MultiRowGEMV.Format =
            isBF16 ? .bf16 : (weightBits == 4 ? .int4 : .affine(bits: weightBits))
        try multiRow.encode(
            commandBuffer: commandBuffer,
            segments: [
                .init(
                    weights: weights, weightsOffset: weightsOffset,
                    scales: scales, scalesOffset: scalesOffset,
                    biases: biases, biasesOffset: biasesOffset,
                    format: format, rows: m,
                    y: y, yOffset: yOffset, yRowStride: m)
            ],
            x: x, xOffset: xOffset, xRowStride: n,
            activationRows: rows, columns: n)
    }

    /// `isBF16` comes from the tensor's own dtype in the resident index, not
    /// from the slot. Passing it wrongly is silent in the same way the
    /// INT4-only kernels were: the buffers are the right size for whichever
    /// reading you pick, and only the numbers come out wrong.
    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32,
        isBF16: Bool = false
    ) throws {
        if isBF16 {
            try bf16.encode(
                commandBuffer: commandBuffer,
                weights: weights, weightsOffset: weightsOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
            return
        }
        switch quantized {
        case .int4(let gemv):
            try gemv.encode(
                commandBuffer: commandBuffer,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
        case .affine(let gemv):
            try gemv.encode(
                commandBuffer: commandBuffer,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
        }
    }
}
