import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillFinalRowHeadTests {
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
              let normBuf = ctx.device.makeBuffer(bytes: normBits,
                                                  length: normBits.count * MemoryLayout<UInt16>.size,
                                                  options: .storageModeShared),
              let wBuf = ctx.device.makeBuffer(bytes: packed,
                                               length: packed.count,
                                               options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: scales,
                                               length: scales.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(bytes: biases,
                                               length: biases.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let scalarNormed = Fp16Buffer.make(ctx.device, count: d),
              let scalarLogits = Fp16Buffer.make(ctx.device, count: vocab),
              let blockLogits = Fp16Buffer.make(ctx.device, count: vocab) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        scalarNorm.encodeBF16W(commandBuffer: cb,
                               x: hiddenBuf,
                               xOffset: selectedRow * rowStride * MemoryLayout<Float16>.size,
                               weight: normBuf,
                               out: scalarNormed,
                               d: UInt32(d),
                               eps: eps)
        scalarHead.encode(commandBuffer: cb,
                          weights: wBuf,
                          scales: sBuf,
                          biases: bBuf,
                          x: scalarNormed,
                          y: scalarLogits,
                          m: UInt32(vocab),
                          n: UInt32(d))
        finalRowHead.encodeLogits(commandBuffer: cb,
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
}
