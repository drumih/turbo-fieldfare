import Foundation
import Metal

final class PrefillFinalRowHeadInt4 {
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV?
    private let affine: AffineQuantGEMV?
    /// A head kept at the checkpoint's bf16 (`weightBits == 16`): no scales or
    /// biases, and neither quantized GEMV is built.
    private let bf16: BF16GEMV?
    private let normed: MTLBuffer
    private let normedPair: MTLBuffer
    private let maxD: Int

    init(context: MetalContext, maxD: Int = 2816, weightBits: Int = 4) throws {
        precondition(maxD > 0, "maxD must be positive")
        precondition([4, 8, 16].contains(weightBits))
        self.rms = try RMSNorm(context: context)
        self.int4 = weightBits == 4 ? try DequantInt4GEMV(context: context) : nil
        self.affine =
            weightBits == 8
            ? try AffineQuantGEMV(context: context, weightBits: weightBits)
            : nil
        self.bf16 = weightBits == 16 ? try BF16GEMV(context: context) : nil
        self.maxD = maxD
        guard
            let normed = context.device.makeBuffer(
                length: maxD * MemoryLayout<Float16>.size,
                options: .storageModePrivate),
            let normedPair = context.device.makeBuffer(
                length: 2 * maxD * MemoryLayout<Float16>.size,
                options: .storageModePrivate)
        else {
            throw MetalError.noDevice
        }
        self.normed = normed
        self.normedPair = normedPair
    }

    /// Logits for two consecutive hidden rows with one lm_head weight read.
    ///
    /// The single-row `encodeLogits` reads the full lm_head once per call, so
    /// an MTP verify paid the model's largest weight twice per pass. The
    /// two-row GEMV serves both rows from one traversal — the same
    /// amortization `useTwoRowProjection` already applies to attention.
    /// Writes `[2, vocab]` FP16 logits contiguously at `logits`.
    func encodeLogitsPair(
        commandBuffer: MTLCommandBuffer,
        hiddenBlock: MTLBuffer,
        rowStrideElements: Int,
        normWeight: MTLBuffer,
        normWeightOffset: Int = 0,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        scales: MTLBuffer,
        scalesOffset: Int = 0,
        biases: MTLBuffer,
        biasesOffset: Int = 0,
        logits: MTLBuffer,
        d: UInt32,
        vocab: UInt32,
        rmsEps: Float
    ) throws {
        precondition(rowStrideElements >= Int(d), "row stride must cover d")
        precondition(Int(d) <= maxD, "d=\(d) exceeds maxD=\(maxD)")
        precondition(
            d % UInt32(Quantization.groupSize) == 0,
            "d must be a multiple of \(Quantization.groupSize)")
        let halfBytes = MemoryLayout<Float16>.size
        for row in 0..<2 {
            try rms.encodeBF16W(
                commandBuffer: commandBuffer,
                x: hiddenBlock,
                xOffset: row * rowStrideElements * halfBytes,
                weight: normWeight,
                weightOffset: normWeightOffset,
                out: normedPair,
                outOffset: row * Int(d) * halfBytes,
                d: d,
                eps: rmsEps)
        }
        if let bf16 {
            for row in 0..<2 {
                try bf16.encode(
                    commandBuffer: commandBuffer,
                    weights: weights, weightsOffset: weightsOffset,
                    x: normedPair, xOffset: row * Int(d) * halfBytes,
                    y: logits, yOffset: row * Int(vocab) * halfBytes,
                    m: vocab, n: d)
            }
            return
        }
        if let affine {
            try affine.encodeTwoRows(
                commandBuffer: commandBuffer,
                weights: weights,
                weightsOffset: weightsOffset,
                scales: scales,
                scalesOffset: scalesOffset,
                biases: biases,
                biasesOffset: biasesOffset,
                x: normedPair,
                y: logits,
                m: vocab,
                n: d)
        } else {
            guard let int4 else {
                preconditionFailure(
                    "PrefillFinalRowHeadInt4 has neither affine nor int4 GEMV (weightBits init contract broken)"
                )
            }
            try int4.encodeTwoRows(
                commandBuffer: commandBuffer,
                weights: weights,
                weightsOffset: weightsOffset,
                scales: scales,
                scalesOffset: scalesOffset,
                biases: biases,
                biasesOffset: biasesOffset,
                x: normedPair,
                y: logits,
                m: vocab,
                n: d)
        }
    }

    func encodeLogits(
        commandBuffer: MTLCommandBuffer,
        hiddenBlock: MTLBuffer,
        row: Int,
        rowStrideElements: Int,
        normWeight: MTLBuffer,
        normWeightOffset: Int = 0,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        scales: MTLBuffer,
        scalesOffset: Int = 0,
        biases: MTLBuffer,
        biasesOffset: Int = 0,
        logits: MTLBuffer,
        logitsOffset: Int = 0,
        d: UInt32,
        vocab: UInt32,
        rmsEps: Float
    ) throws {
        precondition(row >= 0, "row must be non-negative")
        precondition(rowStrideElements >= Int(d), "row stride must cover d")
        precondition(Int(d) <= maxD, "d=\(d) exceeds maxD=\(maxD)")
        precondition(
            d % UInt32(Quantization.groupSize) == 0,
            "d must be a multiple of \(Quantization.groupSize)")
        let hiddenOffset = (row * rowStrideElements) * MemoryLayout<Float16>.size
        try rms.encodeBF16W(
            commandBuffer: commandBuffer,
            x: hiddenBlock,
            xOffset: hiddenOffset,
            weight: normWeight,
            weightOffset: normWeightOffset,
            out: normed,
            d: d,
            eps: rmsEps)
        if let bf16 {
            try bf16.encode(
                commandBuffer: commandBuffer,
                weights: weights, weightsOffset: weightsOffset,
                x: normed, y: logits, yOffset: logitsOffset,
                m: vocab, n: d)
            return
        }
        if let affine {
            try affine.encode(
                commandBuffer: commandBuffer,
                weights: weights,
                weightsOffset: weightsOffset,
                scales: scales,
                scalesOffset: scalesOffset,
                biases: biases,
                biasesOffset: biasesOffset,
                x: normed,
                y: logits,
                yOffset: logitsOffset,
                m: vocab,
                n: d)
        } else {
            // K20: int4 is nil exactly when affine is non-nil (weightBits != 4),
            // so this branch implies int4 is present; assert instead of force
            // unwrapping so a future init change fails with a clear message.
            guard let int4 else {
                preconditionFailure(
                    "PrefillFinalRowHeadInt4 has neither affine nor int4 GEMV (weightBits init contract broken)"
                )
            }
            try int4.encode(
                commandBuffer: commandBuffer,
                weights: weights,
                weightsOffset: weightsOffset,
                scales: scales,
                scalesOffset: scalesOffset,
                biases: biases,
                biasesOffset: biasesOffset,
                x: normed,
                y: logits,
                yOffset: logitsOffset,
                m: vocab,
                n: d)
        }
    }
}
