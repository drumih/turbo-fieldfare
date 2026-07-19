import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillQKVEpilogueTests {
    @Test func blockQKVEpilogueMatchesRepeatedScalarSWA() throws {
        _ = try Self.expectMatchesRepeatedScalar(rows: 3,
                                                 numQHeads: 16,
                                                 numKVHeads: 8,
                                                 headDim: 256,
                                                 startPosition: 1023,
                                                 theta: 10_000.0,
                                                 rotatedPairs: 128,
                                                 sharedKVRaw: false,
                                                 seed: 0x7A50_5A01)
    }

    @Test func blockQKVEpiloguePreservesFullLayerSharedRawKVQuirk() throws {
        let result = try Self.expectMatchesRepeatedScalar(rows: 2,
                                                          numQHeads: 16,
                                                          numKVHeads: 2,
                                                          headDim: 512,
                                                          startPosition: 4095,
                                                          theta: 1_000_000.0,
                                                          rotatedPairs: 64,
                                                          sharedKVRaw: true,
                                                          seed: 0x7A50_F011)
        #expect(result.kUsed != result.vUsed,
                "full-layer K/V must diverge after K norm+RoPE and V no-scale norm")
    }

    private static func expectMatchesRepeatedScalar(rows: Int,
                                                    numQHeads: Int,
                                                    numKVHeads: Int,
                                                    headDim: Int,
                                                    startPosition: Int,
                                                    theta: Float,
                                                    rotatedPairs: UInt32,
                                                    sharedKVRaw: Bool,
                                                    seed: UInt64) throws -> (kUsed: [Float], vUsed: [Float]) {
        let qUsed = numQHeads * headDim
        let kvUsed = numKVHeads * headDim
        let qStride = qUsed + 13
        let kvStride = kvUsed + 17
        let q = Self.makeBlock(rows: rows,
                               rowStride: qStride,
                               usedPerRow: qUsed,
                               seed: seed,
                               label: "q")
        let k = Self.makeBlock(rows: rows,
                               rowStride: kvStride,
                               usedPerRow: kvUsed,
                               seed: seed &+ 1,
                               label: "k")
        let v = sharedKVRaw
            ? k
            : Self.makeBlock(rows: rows,
                             rowStride: kvStride,
                             usedPerRow: kvUsed,
                             seed: seed &+ 2,
                             label: "v")
        var rng = SeedTree(seed).key("norm-weights")
        let qWeight = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let kWeight = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let scalarNorm = try RMSNorm(context: ctx)
        let scalarRoPE = try RoPE(context: ctx)
        let block = try PrefillQKVEpilogue(context: ctx)
        guard
            let qWeightBuf = ctx.device.makeBuffer(bytes: qWeight,
                                                   length: qWeight.count * MemoryLayout<UInt16>.size,
                                                   options: .storageModeShared),
            let kWeightBuf = ctx.device.makeBuffer(bytes: kWeight,
                                                   length: kWeight.count * MemoryLayout<UInt16>.size,
                                                   options: .storageModeShared),
            let qRef = Fp16Buffer.make(ctx.device, halves: q),
            let kRef = Fp16Buffer.make(ctx.device, halves: k),
            let vRef = Fp16Buffer.make(ctx.device, halves: v),
            let qBlock = Fp16Buffer.make(ctx.device, halves: q),
            let kBlock = Fp16Buffer.make(ctx.device, halves: k),
            let vBlock = Fp16Buffer.make(ctx.device, halves: v)
        else {
            Issue.record("alloc failed")
            return ([], [])
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            let qOffset = row * qStride * MemoryLayout<Float16>.size
            let kvOffset = row * kvStride * MemoryLayout<Float16>.size
            scalarNorm.encodeBF16WPerHead(commandBuffer: cb,
                                          x: qRef,
                                          xOffset: qOffset,
                                          weight: qWeightBuf,
                                          out: qRef,
                                          outOffset: qOffset,
                                          headDim: UInt32(headDim),
                                          numHeads: numQHeads,
                                          eps: 1e-6)
            scalarNorm.encodeBF16WPerHead(commandBuffer: cb,
                                          x: kRef,
                                          xOffset: kvOffset,
                                          weight: kWeightBuf,
                                          out: kRef,
                                          outOffset: kvOffset,
                                          headDim: UInt32(headDim),
                                          numHeads: numKVHeads,
                                          eps: 1e-6)
            scalarNorm.encodeNoScalePerHead(commandBuffer: cb,
                                            x: vRef,
                                            xOffset: kvOffset,
                                            out: vRef,
                                            outOffset: kvOffset,
                                            headDim: UInt32(headDim),
                                            numHeads: numKVHeads,
                                            eps: 1e-6)
            if rotatedPairs * 2 == UInt32(headDim) {
                scalarRoPE.encodeDefaultNeox(commandBuffer: cb,
                                             data: qRef,
                                             dataOffset: qOffset,
                                             position: UInt32(startPosition + row),
                                             headDim: UInt32(headDim),
                                             numHeads: UInt32(numQHeads),
                                             numTokens: 1,
                                             theta: theta)
                scalarRoPE.encodeDefaultNeox(commandBuffer: cb,
                                             data: kRef,
                                             dataOffset: kvOffset,
                                             position: UInt32(startPosition + row),
                                             headDim: UInt32(headDim),
                                             numHeads: UInt32(numKVHeads),
                                             numTokens: 1,
                                             theta: theta)
            } else {
                scalarRoPE.encodeProportionalNeox(commandBuffer: cb,
                                                  data: qRef,
                                                  dataOffset: qOffset,
                                                  position: UInt32(startPosition + row),
                                                  headDim: UInt32(headDim),
                                                  numHeads: UInt32(numQHeads),
                                                  rotatedPairs: rotatedPairs,
                                                  numTokens: 1,
                                                  theta: theta)
                scalarRoPE.encodeProportionalNeox(commandBuffer: cb,
                                                  data: kRef,
                                                  dataOffset: kvOffset,
                                                  position: UInt32(startPosition + row),
                                                  headDim: UInt32(headDim),
                                                  numHeads: UInt32(numKVHeads),
                                                  rotatedPairs: rotatedPairs,
                                                  numTokens: 1,
                                                  theta: theta)
            }
        }
        block.encode(commandBuffer: cb,
                     q: qBlock,
                     k: kBlock,
                     v: vBlock,
                     qWeight: qWeightBuf,
                     kWeight: kWeightBuf,
                     startPosition: UInt32(startPosition),
                     queryCount: UInt32(rows),
                     headDim: UInt32(headDim),
                     numQHeads: UInt32(numQHeads),
                     numKVHeads: UInt32(numKVHeads),
                     qTokenStrideElements: UInt32(qStride),
                     kvTokenStrideElements: UInt32(kvStride),
                     theta: theta,
                     rotatedPairs: rotatedPairs,
                     eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            Issue.record("command buffer failed: \(err)")
            return ([], [])
        }

        Self.expectClose(Fp16Buffer.read(qBlock, count: rows * qStride),
                         Fp16Buffer.read(qRef, count: rows * qStride),
                         label: "Q")
        Self.expectClose(Fp16Buffer.read(kBlock, count: rows * kvStride),
                         Fp16Buffer.read(kRef, count: rows * kvStride),
                         label: "K")
        Self.expectClose(Fp16Buffer.read(vBlock, count: rows * kvStride),
                         Fp16Buffer.read(vRef, count: rows * kvStride),
                         label: "V")
        Self.assertPaddingUnchanged(qBlock,
                                    original: q,
                                    rows: rows,
                                    rowStride: qStride,
                                    usedPerRow: qUsed)
        Self.assertPaddingUnchanged(kBlock,
                                    original: k,
                                    rows: rows,
                                    rowStride: kvStride,
                                    usedPerRow: kvUsed)
        Self.assertPaddingUnchanged(vBlock,
                                    original: v,
                                    rows: rows,
                                    rowStride: kvStride,
                                    usedPerRow: kvUsed)

        let kActual = Fp16Buffer.read(kBlock, count: rows * kvStride)
        let vActual = Fp16Buffer.read(vBlock, count: rows * kvStride)
        return (Self.usedRows(kActual, rows: rows, rowStride: kvStride, usedPerRow: kvUsed),
                Self.usedRows(vActual, rows: rows, rowStride: kvStride, usedPerRow: kvUsed))
    }

    private static func makeBlock(rows: Int,
                                  rowStride: Int,
                                  usedPerRow: Int,
                                  seed: UInt64,
                                  label: String) -> [Float16] {
        var rng = SeedTree(seed).key("prefill-qkv-\(label)")
        var values = [Float16](repeating: Float16(-9.0), count: rows * rowStride)
        for row in 0..<rows {
            for i in 0..<usedPerRow {
                values[row * rowStride + i] = Float16(rng.uniform(-1.0, 1.0))
            }
        }
        return values
    }

    private static func usedRows(_ values: [Float],
                                 rows: Int,
                                 rowStride: Int,
                                 usedPerRow: Int) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(rows * usedPerRow)
        for row in 0..<rows {
            out.append(contentsOf: values[(row * rowStride)..<(row * rowStride + usedPerRow)])
        }
        return out
    }

    private static func expectClose(_ actual: [Float],
                                    _ reference: [Float],
                                    label: String) {
        let rel = RelError.compute(actual: actual, reference: reference)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        #expect(rel < Tolerance.fp16Reduction, "\(label) rel=\(rel) maxAbs=\(maxAbs)")
        #expect(maxAbs <= 1e-3, "\(label) maxAbs=\(maxAbs) rel=\(rel)")
    }

    private static func assertPaddingUnchanged(_ buffer: MTLBuffer,
                                               original: [Float16],
                                               rows: Int,
                                               rowStride: Int,
                                               usedPerRow: Int) {
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: rows * rowStride)
        for row in 0..<rows {
            for i in usedPerRow..<rowStride {
                #expect(ptr[row * rowStride + i] == original[row * rowStride + i],
                        "padding changed at row=\(row) i=\(i)")
            }
        }
    }
}
