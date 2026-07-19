import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillPrimitiveTests {
    private static let vocab = 16
    private static let d = 128
    private static let groupsPerRow = d / Quantization.groupSize

    private static func buildInt4Table(seed: UInt64) -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        var rng = SeedTree(seed).key("prefill-int4-embed-table")
        var packed = [UInt8](repeating: 0, count: vocab * (d / 2))
        var scales = [UInt16](repeating: 0, count: vocab * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: vocab * groupsPerRow)

        for v in 0..<vocab {
            let row = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
            let q = Quantization.quantizeInt4Affine(row)
            for i in 0..<(d / 2) {
                packed[v * (d / 2) + i] = q.packed[i]
            }
            for g in 0..<groupsPerRow {
                scales[v * groupsPerRow + g] = q.scales[g]
                biases[v * groupsPerRow + g] = q.biases[g]
            }
        }
        return (packed, scales, biases)
    }

    @Test func embedBlockMatchesPerTokenEmbed() throws {
        let (packed, scales, biases) = Self.buildInt4Table(seed: 0x5101)
        let tokens: [UInt32] = [3, 11, 2, 9, 14]
        let outScale = Float(Self.d).squareRoot()
        let ctx = try MetalContext()
        let scalar = try EmbedLookupInt4(context: ctx)
        let block = try PrefillEmbedLookupInt4(context: ctx)

        guard let tableBuf = ctx.device.makeBuffer(bytes: packed, length: packed.count, options: .storageModeShared),
              let scalesBuf = ctx.device.makeBuffer(bytes: scales,
                                                    length: scales.count * MemoryLayout<UInt16>.size,
                                                    options: .storageModeShared),
              let biasesBuf = ctx.device.makeBuffer(bytes: biases,
                                                    length: biases.count * MemoryLayout<UInt16>.size,
                                                    options: .storageModeShared),
              let tokenBuf = ctx.device.makeBuffer(bytes: tokens,
                                                   length: tokens.count * MemoryLayout<UInt32>.size,
                                                   options: .storageModeShared),
              let scalarOut = Fp16Buffer.make(ctx.device, count: tokens.count * Self.d),
              let blockOut = Fp16Buffer.make(ctx.device, count: tokens.count * Self.d) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for (row, token) in tokens.enumerated() {
            scalar.encode(commandBuffer: cb,
                          table: tableBuf,
                          scales: scalesBuf,
                          biases: biasesBuf,
                          out: scalarOut,
                          outOffset: row * Self.d * MemoryLayout<Float16>.size,
                          tokenId: token,
                          d: UInt32(Self.d),
                          outScale: outScale)
        }
        block.encode(commandBuffer: cb,
                     table: tableBuf,
                     scales: scalesBuf,
                     biases: biasesBuf,
                     tokens: tokenBuf,
                     out: blockOut,
                     t: UInt32(tokens.count),
                     d: UInt32(Self.d),
                     outScale: outScale)
        cb.commit()
        cb.waitUntilCompleted()

        let scalarRows = Fp16Buffer.read(scalarOut, count: tokens.count * Self.d)
        let blockRows = Fp16Buffer.read(blockOut, count: tokens.count * Self.d)
        #expect(blockRows == scalarRows)
    }

    @Test func blockRMSNormMatchesScalarRows() throws {
        let rows = 7
        let dim = 256
        let eps: Float = 1e-6
        var rng = SeedTree(0x5202).key("prefill-rms-block")
        let x = (0..<(rows * dim)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let wBits = (0..<dim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let scalar = try RMSNorm(context: ctx)
        let block = try PrefillRMSNorm(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let scalarOut = Fp16Buffer.make(ctx.device, count: rows * dim),
              let blockOut = Fp16Buffer.make(ctx.device, count: rows * dim),
              let wBuf = ctx.device.makeBuffer(length: wBits.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared) else {
            Issue.record("alloc failed")
            return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }

        let rowBytes = dim * MemoryLayout<Float16>.size
        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            scalar.encodeBF16W(commandBuffer: cb,
                               x: xBuf,
                               xOffset: row * rowBytes,
                               weight: wBuf,
                               out: scalarOut,
                               outOffset: row * rowBytes,
                               d: UInt32(dim),
                               eps: eps)
        }
        block.encodeBF16W(commandBuffer: cb,
                          x: xBuf,
                          weight: wBuf,
                          out: blockOut,
                          t: UInt32(rows),
                          d: UInt32(dim),
                          eps: eps)
        cb.commit()
        cb.waitUntilCompleted()

        let scalarRows = Fp16Buffer.read(scalarOut, count: rows * dim)
        let blockRows = Fp16Buffer.read(blockOut, count: rows * dim)
        let maxAbs = RelError.maxAbsDiff(blockRows, scalarRows)
        let rel = RelError.compute(actual: blockRows, reference: scalarRows)
        #expect(maxAbs <= 1e-3, "maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 1e-4, "rel=\(rel) maxAbs=\(maxAbs)")
    }
}
