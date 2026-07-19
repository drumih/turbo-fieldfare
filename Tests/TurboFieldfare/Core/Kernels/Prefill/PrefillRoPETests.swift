import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillRoPETests {
    private static func makeBlock(rows: Int,
                                  rowStride: Int,
                                  usedPerRow: Int,
                                  seed: UInt64,
                                  label: String) -> [Float16] {
        var rng = SeedTree(seed).key(label)
        var values = [Float16](repeating: Float16(-7.0), count: rows * rowStride)
        for row in 0..<rows {
            for i in 0..<usedPerRow {
                values[row * rowStride + i] = Float16(rng.uniform(-1.0, 1.0))
            }
        }
        return values
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

    @Test func blockDefaultNeoxMatchesRepeatedScalarAbsolutePositions() throws {
        let rows = 5
        let heads = 8
        let headDim = 256
        let used = heads * headDim
        let rowStride = used + 37
        let startPosition = 1023
        let theta: Float = 10_000.0
        let input = Self.makeBlock(rows: rows,
                                   rowStride: rowStride,
                                   usedPerRow: used,
                                   seed: 0xA501,
                                   label: "prefill-rope-swa")

        let ctx = try MetalContext()
        let scalar = try RoPE(context: ctx)
        let block = try PrefillRoPE(context: ctx)
        guard let ref = Fp16Buffer.make(ctx.device, halves: input),
              let candidate = Fp16Buffer.make(ctx.device, halves: input) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            scalar.encodeDefaultNeox(commandBuffer: cb,
                                     data: ref,
                                     dataOffset: row * rowStride * MemoryLayout<Float16>.size,
                                     position: UInt32(startPosition + row),
                                     headDim: UInt32(headDim),
                                     numHeads: UInt32(heads),
                                     numTokens: 1,
                                     theta: theta)
        }
        block.encodeDefaultNeox(commandBuffer: cb,
                                data: candidate,
                                startPosition: UInt32(startPosition),
                                queryCount: UInt32(rows),
                                headDim: UInt32(headDim),
                                numHeads: UInt32(heads),
                                tokenStrideElements: UInt32(rowStride),
                                theta: theta)
        cb.commit()
        cb.waitUntilCompleted()

        let reference = Fp16Buffer.read(ref, count: rows * rowStride)
        let actual = Fp16Buffer.read(candidate, count: rows * rowStride)
        let rel = RelError.compute(actual: actual, reference: reference)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        #expect(rel < Tolerance.fp16Reduction, "block SWA RoPE rel=\(rel) maxAbs=\(maxAbs)")
        #expect(maxAbs <= 1e-3, "block SWA RoPE maxAbs=\(maxAbs) rel=\(rel)")
        Self.assertPaddingUnchanged(candidate,
                                    original: input,
                                    rows: rows,
                                    rowStride: rowStride,
                                    usedPerRow: used)
    }

    @Test func blockProportionalNeoxMatchesRepeatedScalarAbsolutePositions() throws {
        let rows = 4
        let heads = 2
        let headDim = 512
        let rotatedPairs = 64
        let used = heads * headDim
        let rowStride = used + 19
        let startPosition = 4095
        let theta: Float = 1_000_000.0
        let input = Self.makeBlock(rows: rows,
                                   rowStride: rowStride,
                                   usedPerRow: used,
                                   seed: 0xA502,
                                   label: "prefill-rope-full")

        let ctx = try MetalContext()
        let scalar = try RoPE(context: ctx)
        let block = try PrefillRoPE(context: ctx)
        guard let ref = Fp16Buffer.make(ctx.device, halves: input),
              let candidate = Fp16Buffer.make(ctx.device, halves: input) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            scalar.encodeProportionalNeox(commandBuffer: cb,
                                          data: ref,
                                          dataOffset: row * rowStride * MemoryLayout<Float16>.size,
                                          position: UInt32(startPosition + row),
                                          headDim: UInt32(headDim),
                                          numHeads: UInt32(heads),
                                          rotatedPairs: UInt32(rotatedPairs),
                                          numTokens: 1,
                                          theta: theta)
        }
        block.encodeProportionalNeox(commandBuffer: cb,
                                     data: candidate,
                                     startPosition: UInt32(startPosition),
                                     queryCount: UInt32(rows),
                                     headDim: UInt32(headDim),
                                     numHeads: UInt32(heads),
                                     rotatedPairs: UInt32(rotatedPairs),
                                     tokenStrideElements: UInt32(rowStride),
                                     theta: theta)
        cb.commit()
        cb.waitUntilCompleted()

        let reference = Fp16Buffer.read(ref, count: rows * rowStride)
        let actual = Fp16Buffer.read(candidate, count: rows * rowStride)
        let rel = RelError.compute(actual: actual, reference: reference)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        #expect(rel < Tolerance.fp16Reduction, "block full RoPE rel=\(rel) maxAbs=\(maxAbs)")
        #expect(maxAbs <= 1e-3, "block full RoPE maxAbs=\(maxAbs) rel=\(rel)")
        Self.assertPaddingUnchanged(candidate,
                                    original: input,
                                    rows: rows,
                                    rowStride: rowStride,
                                    usedPerRow: used)
    }
}
