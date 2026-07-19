import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillLayerTailTests {
    private static let d = ArchConfig.gemma4_26B_A4B.hiddenSize
    private static let eps: Float = 1e-6

    @Test func blockLayerTailMatchesRepeatedFusedRows() throws {
        let rows = 3
        let h2Stride = Self.d + 9
        let h1Stride = Self.d + 13
        let hiddenStride = Self.d + 17
        var rng = SplitMix64(seed: 0x7A11_B10C)
        let h2 = Self.makeBlock(rows: rows,
                                rowStride: h2Stride,
                                usedPerRow: Self.d,
                                rng: &rng,
                                sentinel: -21.0)
        let h1 = Self.makeBlock(rows: rows,
                                rowStride: h1Stride,
                                usedPerRow: Self.d,
                                rng: &rng,
                                sentinel: -23.0)
        let hidden = Self.makeBlock(rows: rows,
                                    rowStride: hiddenStride,
                                    usedPerRow: Self.d,
                                    rng: &rng,
                                    sentinel: -25.0)
        let wPostFFN2 = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let wPostFFN = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let layerScalar = rng.uniform(0.75, 1.25)

        let ctx = try MetalContext()
        let fused = try FusedLayerTail(context: ctx)
        let block = try PrefillLayerTail(context: ctx)
        guard
            let wPostFFN2Buf = ctx.device.makeBuffer(bytes: wPostFFN2,
                                                     length: wPostFFN2.count * MemoryLayout<UInt16>.size,
                                                     options: .storageModeShared),
            let wPostFFNBuf = ctx.device.makeBuffer(bytes: wPostFFN,
                                                    length: wPostFFN.count * MemoryLayout<UInt16>.size,
                                                    options: .storageModeShared),
            let h2Block = Fp16Buffer.make(ctx.device, halves: h2),
            let h1Block = Fp16Buffer.make(ctx.device, halves: h1),
            let hiddenBlock = Fp16Buffer.make(ctx.device, halves: hidden)
        else {
            Issue.record("alloc failed")
            return
        }

        var hiddenRef = hidden
        for row in 0..<rows {
            let h2Row = Array(h2[(row * h2Stride)..<(row * h2Stride + Self.d)])
            let h1Row = Array(h1[(row * h1Stride)..<(row * h1Stride + Self.d)])
            let hiddenRow = Array(hidden[(row * hiddenStride)..<(row * hiddenStride + Self.d)])
            guard
                let h2Buf = Fp16Buffer.make(ctx.device, halves: h2Row),
                let h1Buf = Fp16Buffer.make(ctx.device, halves: h1Row),
                let hiddenBuf = Fp16Buffer.make(ctx.device, halves: hiddenRow)
            else {
                Issue.record("row alloc failed")
                return
            }
            let cb = ctx.queue.makeCommandBuffer()!
            fused.encode(commandBuffer: cb,
                         h2: h2Buf,
                         h1: h1Buf,
                         hidden: hiddenBuf,
                         postFFN2Weight: wPostFFN2Buf,
                         postFFNWeight: wPostFFNBuf,
                         d: UInt32(Self.d),
                         eps: Self.eps,
                         layerScalar: layerScalar)
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error {
                Issue.record("fused row command failed: \(err)")
                return
            }
            Self.copyUsed(Fp16Buffer.readHalf(hiddenBuf, count: Self.d),
                          into: &hiddenRef,
                          row: row,
                          rowStride: hiddenStride)
        }

        let cb = ctx.queue.makeCommandBuffer()!
        block.encode(commandBuffer: cb,
                     h2: h2Block,
                     h1: h1Block,
                     hidden: hiddenBlock,
                     postFFN2Weight: wPostFFN2Buf,
                     postFFNWeight: wPostFFNBuf,
                     queryCount: UInt32(rows),
                     d: UInt32(Self.d),
                     h2StrideElements: UInt32(h2Stride),
                     h1StrideElements: UInt32(h1Stride),
                     hiddenStrideElements: UInt32(hiddenStride),
                     eps: Self.eps,
                     layerScalar: layerScalar)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            Issue.record("block command failed: \(err)")
            return
        }

        #expect(Fp16Buffer.readHalf(hiddenBlock, count: rows * hiddenStride) == hiddenRef)
        #expect(Fp16Buffer.readHalf(h2Block, count: rows * h2Stride) == h2)
        #expect(Fp16Buffer.readHalf(h1Block, count: rows * h1Stride) == h1)
        Self.assertPaddingUnchanged(hiddenBlock,
                                    original: hidden,
                                    rows: rows,
                                    rowStride: hiddenStride,
                                    usedPerRow: Self.d)
    }

    private static func makeBlock(rows: Int,
                                  rowStride: Int,
                                  usedPerRow: Int,
                                  rng: inout SplitMix64,
                                  sentinel: Float16) -> [Float16] {
        var values = [Float16](repeating: sentinel, count: rows * rowStride)
        for row in 0..<rows {
            for i in 0..<usedPerRow {
                values[row * rowStride + i] = Float16(rng.uniform(-1.0, 1.0))
            }
        }
        return values
    }

    private static func copyUsed(_ source: [Float16],
                                 into destination: inout [Float16],
                                 row: Int,
                                 rowStride: Int) {
        for i in 0..<source.count {
            destination[row * rowStride + i] = source[i]
        }
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
