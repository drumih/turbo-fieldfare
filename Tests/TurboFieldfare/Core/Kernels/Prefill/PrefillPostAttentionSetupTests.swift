import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillPostAttentionSetupTests {
    private static let d = ArchConfig.gemma4_26B_A4B.hiddenSize
    private static let eps: Float = 1e-6

    @Test func blockPostAttentionSetupMatchesRepeatedScalarFusedRows() throws {
        let rows = 3
        let hiddenStride = Self.d + 11
        let attnStride = Self.d + 17
        let outStride = Self.d + 23
        var rng = SplitMix64(seed: 0x5057_A77E)
        let hidden = Self.makeBlock(rows: rows,
                                    rowStride: hiddenStride,
                                    usedPerRow: Self.d,
                                    rng: &rng,
                                    sentinel: -11.0)
        let attn = Self.makeBlock(rows: rows,
                                  rowStride: attnStride,
                                  usedPerRow: Self.d,
                                  rng: &rng,
                                  sentinel: -13.0)
        let outSentinel = [Float16](repeating: Float16(-17.0), count: rows * outStride)
        let wPost = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let wPre = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let wPre2 = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let scalar = try FusedPostAttentionSetup(context: ctx)
        let block = try PrefillPostAttentionSetup(context: ctx)
        guard
            let wPostBuf = ctx.device.makeBuffer(bytes: wPost,
                                                 length: wPost.count * MemoryLayout<UInt16>.size,
                                                 options: .storageModeShared),
            let wPreBuf = ctx.device.makeBuffer(bytes: wPre,
                                                length: wPre.count * MemoryLayout<UInt16>.size,
                                                options: .storageModeShared),
            let wPre2Buf = ctx.device.makeBuffer(bytes: wPre2,
                                                 length: wPre2.count * MemoryLayout<UInt16>.size,
                                                 options: .storageModeShared),
            let hiddenBlock = Fp16Buffer.make(ctx.device, halves: hidden),
            let attnBlock = Fp16Buffer.make(ctx.device, halves: attn),
            let denseBlock = Fp16Buffer.make(ctx.device, halves: outSentinel),
            let routedBlock = Fp16Buffer.make(ctx.device, halves: outSentinel),
            let routerBlock = Fp16Buffer.make(ctx.device, halves: outSentinel)
        else {
            Issue.record("alloc failed")
            return
        }

        var hiddenRef = hidden
        var denseRef = outSentinel
        var routedRef = outSentinel
        var routerRef = outSentinel

        for row in 0..<rows {
            let h = Array(hidden[(row * hiddenStride)..<(row * hiddenStride + Self.d)])
            let a = Array(attn[(row * attnStride)..<(row * attnStride + Self.d)])
            guard
                let hBuf = Fp16Buffer.make(ctx.device, halves: h),
                let aBuf = Fp16Buffer.make(ctx.device, halves: a),
                let denseBuf = Fp16Buffer.make(ctx.device, count: Self.d),
                let routedBuf = Fp16Buffer.make(ctx.device, count: Self.d),
                let routerBuf = Fp16Buffer.make(ctx.device, count: Self.d)
            else {
                Issue.record("row alloc failed")
                return
            }
            let cb = ctx.queue.makeCommandBuffer()!
            scalar.encode(commandBuffer: cb,
                          hidden: hBuf,
                          attn: aBuf,
                          denseX: denseBuf,
                          routedX: routedBuf,
                          routerX: routerBuf,
                          postAttentionWeight: wPostBuf,
                          preFFNWeight: wPreBuf,
                          preFFN2Weight: wPre2Buf,
                          d: UInt32(Self.d),
                          eps: Self.eps)
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error {
                Issue.record("scalar row command failed: \(err)")
                return
            }
            Self.copyUsed(Fp16Buffer.readHalf(hBuf, count: Self.d),
                          into: &hiddenRef,
                          row: row,
                          rowStride: hiddenStride)
            Self.copyUsed(Fp16Buffer.readHalf(denseBuf, count: Self.d),
                          into: &denseRef,
                          row: row,
                          rowStride: outStride)
            Self.copyUsed(Fp16Buffer.readHalf(routedBuf, count: Self.d),
                          into: &routedRef,
                          row: row,
                          rowStride: outStride)
            Self.copyUsed(Fp16Buffer.readHalf(routerBuf, count: Self.d),
                          into: &routerRef,
                          row: row,
                          rowStride: outStride)
        }

        let cb = ctx.queue.makeCommandBuffer()!
        block.encode(commandBuffer: cb,
                     hidden: hiddenBlock,
                     attn: attnBlock,
                     denseX: denseBlock,
                     routedX: routedBlock,
                     routerX: routerBlock,
                     postAttentionWeight: wPostBuf,
                     preFFNWeight: wPreBuf,
                     preFFN2Weight: wPre2Buf,
                     queryCount: UInt32(rows),
                     d: UInt32(Self.d),
                     hiddenStrideElements: UInt32(hiddenStride),
                     attnStrideElements: UInt32(attnStride),
                     denseStrideElements: UInt32(outStride),
                     routedStrideElements: UInt32(outStride),
                     routerStrideElements: UInt32(outStride),
                     eps: Self.eps)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            Issue.record("block command failed: \(err)")
            return
        }

        #expect(Fp16Buffer.readHalf(hiddenBlock, count: rows * hiddenStride) == hiddenRef)
        #expect(Fp16Buffer.readHalf(denseBlock, count: rows * outStride) == denseRef)
        #expect(Fp16Buffer.readHalf(routedBlock, count: rows * outStride) == routedRef)
        #expect(Fp16Buffer.readHalf(routerBlock, count: rows * outStride) == routerRef)
        Self.assertPaddingUnchanged(hiddenBlock,
                                    original: hidden,
                                    rows: rows,
                                    rowStride: hiddenStride,
                                    usedPerRow: Self.d)
        Self.assertPaddingUnchanged(attnBlock,
                                    original: attn,
                                    rows: rows,
                                    rowStride: attnStride,
                                    usedPerRow: Self.d)
        Self.assertPaddingUnchanged(denseBlock,
                                    original: outSentinel,
                                    rows: rows,
                                    rowStride: outStride,
                                    usedPerRow: Self.d)
        Self.assertPaddingUnchanged(routedBlock,
                                    original: outSentinel,
                                    rows: rows,
                                    rowStride: outStride,
                                    usedPerRow: Self.d)
        Self.assertPaddingUnchanged(routerBlock,
                                    original: outSentinel,
                                    rows: rows,
                                    rowStride: outStride,
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
