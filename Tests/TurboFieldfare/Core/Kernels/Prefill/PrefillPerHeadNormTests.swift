import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillPerHeadNormTests {
    private static func makeBlock(rows: Int,
                                  rowStride: Int,
                                  usedPerRow: Int,
                                  seed: UInt64,
                                  label: String) -> [Float16] {
        var rng = SeedTree(seed).key(label)
        var values = [Float16](repeating: Float16(-5.0), count: rows * rowStride)
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

    private static func runBF16Case(rows: Int,
                                    heads: Int,
                                    headDim: Int,
                                    seed: UInt64,
                                    label: String) throws {
        let used = heads * headDim
        let rowStride = used + 23
        let input = makeBlock(rows: rows,
                              rowStride: rowStride,
                              usedPerRow: used,
                              seed: seed,
                              label: label)
        var rng = SeedTree(seed).key("\(label)-weight")
        let weight = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let scalar = try RMSNorm(context: ctx)
        let block = try PrefillPerHeadNorm(context: ctx)
        guard let weightBuf = ctx.device.makeBuffer(bytes: weight,
                                                    length: weight.count * MemoryLayout<UInt16>.size,
                                                    options: .storageModeShared),
              let ref = Fp16Buffer.make(ctx.device, halves: input),
              let candidate = Fp16Buffer.make(ctx.device, halves: input) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            scalar.encodeBF16WPerHead(commandBuffer: cb,
                                      x: ref,
                                      xOffset: row * rowStride * MemoryLayout<Float16>.size,
                                      weight: weightBuf,
                                      out: ref,
                                      outOffset: row * rowStride * MemoryLayout<Float16>.size,
                                      headDim: UInt32(headDim),
                                      numHeads: heads,
                                      eps: 1e-6)
        }
        block.encodeBF16W(commandBuffer: cb,
                          x: candidate,
                          weight: weightBuf,
                          out: candidate,
                          queryCount: UInt32(rows),
                          headDim: UInt32(headDim),
                          numHeads: UInt32(heads),
                          tokenStrideElements: UInt32(rowStride),
                          eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()

        let reference = Fp16Buffer.read(ref, count: rows * rowStride)
        let actual = Fp16Buffer.read(candidate, count: rows * rowStride)
        #expect(actual == reference, "block BF16 per-head norm must match repeated scalar per-token path for \(label)")
        assertPaddingUnchanged(candidate,
                               original: input,
                               rows: rows,
                               rowStride: rowStride,
                               usedPerRow: used)
    }

    private static func runNoScaleCase(rows: Int,
                                       heads: Int,
                                       headDim: Int,
                                       seed: UInt64,
                                       label: String) throws {
        let used = heads * headDim
        let rowStride = used + 17
        let input = makeBlock(rows: rows,
                              rowStride: rowStride,
                              usedPerRow: used,
                              seed: seed,
                              label: label)

        let ctx = try MetalContext()
        let scalar = try RMSNorm(context: ctx)
        let block = try PrefillPerHeadNorm(context: ctx)
        guard let ref = Fp16Buffer.make(ctx.device, halves: input),
              let candidate = Fp16Buffer.make(ctx.device, halves: input) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            scalar.encodeNoScalePerHead(commandBuffer: cb,
                                        x: ref,
                                        xOffset: row * rowStride * MemoryLayout<Float16>.size,
                                        out: ref,
                                        outOffset: row * rowStride * MemoryLayout<Float16>.size,
                                        headDim: UInt32(headDim),
                                        numHeads: heads,
                                        eps: 1e-6)
        }
        block.encodeNoScale(commandBuffer: cb,
                            x: candidate,
                            out: candidate,
                            queryCount: UInt32(rows),
                            headDim: UInt32(headDim),
                            numHeads: UInt32(heads),
                            tokenStrideElements: UInt32(rowStride),
                            eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()

        let reference = Fp16Buffer.read(ref, count: rows * rowStride)
        let actual = Fp16Buffer.read(candidate, count: rows * rowStride)
        #expect(actual == reference, "block no-scale per-head norm must match repeated scalar per-token path for \(label)")
        assertPaddingUnchanged(candidate,
                               original: input,
                               rows: rows,
                               rowStride: rowStride,
                               usedPerRow: used)
    }

    @Test func blockBF16PerHeadNormMatchesRepeatedScalarQKShapes() throws {
        try Self.runBF16Case(rows: 3, heads: 16, headDim: 256, seed: 0xB501, label: "swa-q")
        try Self.runBF16Case(rows: 3, heads: 8, headDim: 256, seed: 0xB502, label: "swa-k")
        try Self.runBF16Case(rows: 2, heads: 16, headDim: 512, seed: 0xB503, label: "full-q")
        try Self.runBF16Case(rows: 2, heads: 2, headDim: 512, seed: 0xB504, label: "full-k")
    }

    @Test func blockNoScalePerHeadNormMatchesRepeatedScalarVShapes() throws {
        try Self.runNoScaleCase(rows: 3, heads: 8, headDim: 256, seed: 0xB601, label: "swa-v")
        try Self.runNoScaleCase(rows: 2, heads: 2, headDim: 512, seed: 0xB602, label: "full-v")
    }
}
