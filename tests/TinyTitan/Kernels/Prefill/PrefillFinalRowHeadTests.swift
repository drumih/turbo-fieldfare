import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct PrefillFinalRowHeadTests {
    private static func packedAffine(bits: Int, rows: Int, columns: Int) -> [UInt8] {
        let rowBytes = columns * bits / 8
        let mask = UInt32((1 << bits) - 1)
        var packed = [UInt8](repeating: 0, count: rows * rowBytes)
        for row in 0..<rows {
            for column in 0..<columns {
                let value = UInt32((row * 7 + column * 11 + 5) & Int(mask))
                let bitOffset = column * bits
                let byteOffset = row * rowBytes + bitOffset / 8
                let shift = bitOffset % 8
                var word = value << shift
                var remaining = bits + shift
                var index = byteOffset
                while remaining > 0 {
                    packed[index] |= UInt8(truncatingIfNeeded: word)
                    word >>= 8
                    remaining -= 8
                    index += 1
                }
            }
        }
        return packed
    }

    private static func packRows(_ rows: [Quantization.Int4AffineRow])
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let rowBytes = rows[0].packed.count
        let groups = rows[0].scales.count
        var packed = [UInt8](repeating: 0, count: rows.count * rowBytes)
        var scales = [UInt16](repeating: 0, count: rows.count * groups)
        var biases = [UInt16](repeating: 0, count: rows.count * groups)

        for row in 0..<rows.count {
            for i in 0..<rowBytes {
                packed[row * rowBytes + i] = rows[row].packed[i]
            }
            for g in 0..<groups {
                scales[row * groups + g] = rows[row].scales[g]
                biases[row * groups + g] = rows[row].biases[g]
            }
        }
        return (packed, scales, biases)
    }

    @Test func finalRowHeadLogitsMatchesScalarOffsetPath() throws {
        let rows = 5
        let selectedRow = 3
        let d = 128
        let rowStride = d + 17
        let vocab = 96
        let eps: Float = 1e-6
        var rng = SeedTree(0xB501).key("prefill-final-row-head")

        var hidden = [Float16](repeating: 0, count: rows * rowStride)
        for row in 0..<rows {
            for i in 0..<d {
                hidden[row * rowStride + i] = Float16(rng.uniform(-1.0, 1.0))
            }
        }
        let normBits = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let weightRows = (0..<vocab).map { _ -> Quantization.Int4AffineRow in
            let raw = (0..<d).map { _ in rng.uniform(-0.5, 0.5) }
            return Quantization.quantizeInt4Affine(raw)
        }
        let (packed, scales, biases) = Self.packRows(weightRows)

        let ctx = try MetalContext()
        let scalarNorm = try RMSNorm(context: ctx)
        let scalarHead = try DequantInt4GEMV(context: ctx)
        let finalRowHead = try PrefillFinalRowHeadInt4(context: ctx, maxD: d)

        guard let hiddenBuf = Fp16Buffer.make(ctx.device, halves: hidden),
            let normBuf = ctx.device.makeBuffer(
                bytes: normBits,
                length: normBits.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
            let wBuf = ctx.device.makeBuffer(
                bytes: packed,
                length: packed.count,
                options: .storageModeShared),
            let sBuf = ctx.device.makeBuffer(
                bytes: scales,
                length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
            let bBuf = ctx.device.makeBuffer(
                bytes: biases,
                length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
            let scalarNormed = Fp16Buffer.make(ctx.device, count: d),
            let scalarLogits = Fp16Buffer.make(ctx.device, count: vocab),
            let blockLogits = Fp16Buffer.make(ctx.device, count: vocab)
        else {
            Issue.record("alloc failed")
            return
        }

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try scalarNorm.encodeBF16W(
            commandBuffer: cb,
            x: hiddenBuf,
            xOffset: selectedRow * rowStride * MemoryLayout<Float16>.size,
            weight: normBuf,
            out: scalarNormed,
            d: UInt32(d),
            eps: eps)
        try scalarHead.encode(
            commandBuffer: cb,
            weights: wBuf,
            scales: sBuf,
            biases: bBuf,
            x: scalarNormed,
            y: scalarLogits,
            m: UInt32(vocab),
            n: UInt32(d))
        try finalRowHead.encodeLogits(
            commandBuffer: cb,
            hiddenBlock: hiddenBuf,
            row: selectedRow,
            rowStrideElements: rowStride,
            normWeight: normBuf,
            weights: wBuf,
            scales: sBuf,
            biases: bBuf,
            logits: blockLogits,
            d: UInt32(d),
            vocab: UInt32(vocab),
            rmsEps: eps)
        cb.commit()
        cb.waitUntilCompleted()

        let scalar = Fp16Buffer.read(scalarLogits, count: vocab)
        let block = Fp16Buffer.read(blockLogits, count: vocab)
        let maxAbs = RelError.maxAbsDiff(block, scalar)
        let rel = RelError.compute(actual: block, reference: scalar)
        #expect(maxAbs <= 1e-3, "maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 1e-4, "rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test(arguments: [8])
    func affineFinalRowHeadMatchesScalarOffsetPath(bits: Int) throws {
        let rows = 5
        let selectedRow = 3
        let d = 128
        let rowStride = d + 17
        let vocab = 96
        let eps: Float = 1e-6
        var hidden = [Float16](repeating: 0, count: rows * rowStride)
        for row in 0..<rows {
            for column in 0..<d {
                hidden[row * rowStride + column] = Float16(
                    Float((row * 13 + column * 3) % 29 - 14) / 32)
            }
        }
        let normBits = (0..<d).map { index in
            Quantization.bf16Bits(0.75 + Float(index % 5) * 0.05)
        }
        let packed = Self.packedAffine(bits: bits, rows: vocab, columns: d)
        let groups = d / Quantization.groupSize
        let scales = [UInt16](
            repeating: Quantization.bf16Bits(0.002),
            count: vocab * groups)
        let biases = [UInt16](
            repeating: Quantization.bf16Bits(-0.01),
            count: vocab * groups)

        let context = try MetalContext()
        let scalarNorm = try RMSNorm(context: context)
        let scalarHead = try AffineQuantGEMV(context: context, weightBits: bits)
        let finalRowHead = try PrefillFinalRowHeadInt4(
            context: context,
            maxD: d,
            weightBits: bits)
        guard let hiddenBuffer = Fp16Buffer.make(context.device, halves: hidden),
            let normBuffer = context.device.makeBuffer(
                bytes: normBits,
                length: normBits.count * MemoryLayout<UInt16>.stride),
            let weights = context.device.makeBuffer(
                bytes: packed,
                length: packed.count),
            let scaleBuffer = context.device.makeBuffer(
                bytes: scales,
                length: scales.count * MemoryLayout<UInt16>.stride),
            let biasBuffer = context.device.makeBuffer(
                bytes: biases,
                length: biases.count * MemoryLayout<UInt16>.stride),
            let normed = Fp16Buffer.make(context.device, count: d),
            let expected = Fp16Buffer.make(context.device, count: vocab),
            let actual = Fp16Buffer.make(context.device, count: vocab),
            let commandBuffer = context.queue.makeCommandBuffer()
        else {
            Issue.record("allocation failed")
            return
        }

        try scalarNorm.encodeBF16W(
            commandBuffer: commandBuffer,
            x: hiddenBuffer,
            xOffset: selectedRow * rowStride * MemoryLayout<Float16>.stride,
            weight: normBuffer,
            out: normed,
            d: UInt32(d),
            eps: eps)
        try scalarHead.encode(
            commandBuffer: commandBuffer,
            weights: weights,
            scales: scaleBuffer,
            biases: biasBuffer,
            x: normed,
            y: expected,
            m: UInt32(vocab),
            n: UInt32(d))
        try finalRowHead.encodeLogits(
            commandBuffer: commandBuffer,
            hiddenBlock: hiddenBuffer,
            row: selectedRow,
            rowStrideElements: rowStride,
            normWeight: normBuffer,
            weights: weights,
            scales: scaleBuffer,
            biases: biasBuffer,
            logits: actual,
            d: UInt32(d),
            vocab: UInt32(vocab),
            rmsEps: eps)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        #expect(
            Fp16Buffer.read(actual, count: vocab)
                == Fp16Buffer.read(expected, count: vocab))
    }
}
