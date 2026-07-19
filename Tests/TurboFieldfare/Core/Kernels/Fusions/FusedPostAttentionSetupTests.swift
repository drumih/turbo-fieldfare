import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct FusedPostAttentionSetupTests {
    private static let d = ArchConfig.gemma4_26B_A4B.hiddenSize
    private static let eps: Float = 1e-6

    @Test func fusedPostAttentionSetup_matchesPrimitiveChainBitwise_realShape() throws {
        var rng = SplitMix64(seed: 0x9057_A77E)
        let hidden = (0..<Self.d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let attn = (0..<Self.d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let wPost = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let wPre = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let wPre2 = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let rms = try RMSNorm(context: ctx)
        let fused = try FusedPostAttentionSetup(context: ctx)

        guard
            let hiddenLegacy = ctx.device.makeBuffer(bytes: hidden, length: Self.bytes(Self.d), options: .storageModeShared),
            let attnLegacy = ctx.device.makeBuffer(bytes: attn, length: Self.bytes(Self.d), options: .storageModeShared),
            let denseLegacy = ctx.device.makeBuffer(length: Self.bytes(Self.d), options: .storageModeShared),
            let routedLegacy = ctx.device.makeBuffer(length: Self.bytes(Self.d), options: .storageModeShared),
            let routerLegacy = ctx.device.makeBuffer(length: Self.bytes(Self.d), options: .storageModeShared),
            let hiddenFused = ctx.device.makeBuffer(bytes: hidden, length: Self.bytes(Self.d), options: .storageModeShared),
            let attnFused = ctx.device.makeBuffer(bytes: attn, length: Self.bytes(Self.d), options: .storageModeShared),
            let denseFused = ctx.device.makeBuffer(length: Self.bytes(Self.d), options: .storageModeShared),
            let routedFused = ctx.device.makeBuffer(length: Self.bytes(Self.d), options: .storageModeShared),
            let routerFused = ctx.device.makeBuffer(length: Self.bytes(Self.d), options: .storageModeShared),
            let wPostBuf = ctx.device.makeBuffer(bytes: wPost,
                                                 length: wPost.count * MemoryLayout<UInt16>.size,
                                                 options: .storageModeShared),
            let wPreBuf = ctx.device.makeBuffer(bytes: wPre,
                                                length: wPre.count * MemoryLayout<UInt16>.size,
                                                options: .storageModeShared),
            let wPre2Buf = ctx.device.makeBuffer(bytes: wPre2,
                                                 length: wPre2.count * MemoryLayout<UInt16>.size,
                                                 options: .storageModeShared)
        else {
            Issue.record("Failed to allocate buffers")
            return
        }

        guard let attentionNormCB = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate command buffer")
            return
        }
        rms.encodeBF16W(commandBuffer: attentionNormCB,
                        x: attnLegacy,
                        weight: wPostBuf,
                        out: attnLegacy,
                        d: UInt32(Self.d),
                        eps: Self.eps)
        attentionNormCB.commit()
        attentionNormCB.waitUntilCompleted()
        if let error = attentionNormCB.error {
            Issue.record("Command buffer failed: \(error)")
            return
        }

        let hiddenPointer = hiddenLegacy.contents().assumingMemoryBound(to: Float16.self)
        let attentionPointer = attnLegacy.contents().assumingMemoryBound(to: Float16.self)
        for index in 0..<Self.d {
            hiddenPointer[index] += attentionPointer[index]
        }

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate command buffer")
            return
        }
        rms.encodeBF16W(commandBuffer: cb,
                        x: hiddenLegacy,
                        weight: wPreBuf,
                        out: denseLegacy,
                        d: UInt32(Self.d),
                        eps: Self.eps)
        rms.encodeBF16W(commandBuffer: cb,
                        x: hiddenLegacy,
                        weight: wPre2Buf,
                        out: routedLegacy,
                        d: UInt32(Self.d),
                        eps: Self.eps)
        rms.encodeNoScale(commandBuffer: cb,
                          x: hiddenLegacy,
                          out: routerLegacy,
                          d: UInt32(Self.d),
                          eps: Self.eps)
        fused.encode(commandBuffer: cb,
                     hidden: hiddenFused,
                     attn: attnFused,
                     denseX: denseFused,
                     routedX: routedFused,
                     routerX: routerFused,
                     postAttentionWeight: wPostBuf,
                     preFFNWeight: wPreBuf,
                     preFFN2Weight: wPre2Buf,
                     d: UInt32(Self.d),
                     eps: Self.eps)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            Issue.record("Command buffer failed: \(err)")
            return
        }

        #expect(Self.bufferBytes(hiddenLegacy) == Self.bufferBytes(hiddenFused))
        #expect(Self.bufferBytes(denseLegacy) == Self.bufferBytes(denseFused))
        #expect(Self.bufferBytes(routedLegacy) == Self.bufferBytes(routedFused))
        #expect(Self.bufferBytes(routerLegacy) == Self.bufferBytes(routerFused))
    }

    private static func bytes(_ count: Int) -> Int {
        count * MemoryLayout<Float16>.size
    }

    private static func bufferBytes(_ buffer: MTLBuffer) -> [UInt8] {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: ptr, count: Self.bytes(Self.d)))
    }
}
