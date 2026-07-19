import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct FusedQKVEpilogueTests {
    @Test func fusedQKVEpilogue_matchesLegacyChainBitwise_swaShape() throws {
        try Self.expectMatchesLegacy(numQHeads: 16,
                                     numKVHeads: 8,
                                     headDim: 256,
                                     position: 17,
                                     theta: 10_000.0,
                                     rotatedPairs: 128,
                                     seed: 0x714D_E911)
    }

    @Test func fusedQKVEpilogue_matchesLegacyChainBitwise_fullShape() throws {
        try Self.expectMatchesLegacy(numQHeads: 16,
                                     numKVHeads: 2,
                                     headDim: 512,
                                     position: 23,
                                     theta: 1_000_000.0,
                                     rotatedPairs: 64,
                                     seed: 0x714D_F211)
    }

    private static func expectMatchesLegacy(numQHeads: Int,
                                            numKVHeads: Int,
                                            headDim: Int,
                                            position: Int,
                                            theta: Float,
                                            rotatedPairs: UInt32,
                                            seed: UInt64) throws {
        var rng = SplitMix64(seed: seed)
        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * headDim
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let k = (0..<kvCount).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let v = (0..<kvCount).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let qWeight = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let kWeight = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let rms = try RMSNorm(context: ctx)
        let rope = try RoPE(context: ctx)
        let fused = try FusedQKVEpilogue(context: ctx)

        guard
            let qLegacy = ctx.device.makeBuffer(bytes: q, length: Self.bytes(qCount), options: .storageModeShared),
            let kLegacy = ctx.device.makeBuffer(bytes: k, length: Self.bytes(kvCount), options: .storageModeShared),
            let vLegacy = ctx.device.makeBuffer(bytes: v,
                                                length: Self.bytes(kvCount),
                                                options: .storageModeShared),
            let qFused = ctx.device.makeBuffer(bytes: q, length: Self.bytes(qCount), options: .storageModeShared),
            let kFused = ctx.device.makeBuffer(bytes: k, length: Self.bytes(kvCount), options: .storageModeShared),
            let vFused = ctx.device.makeBuffer(bytes: v, length: Self.bytes(kvCount), options: .storageModeShared),
            let qW = ctx.device.makeBuffer(bytes: qWeight,
                                           length: qWeight.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared),
            let kW = ctx.device.makeBuffer(bytes: kWeight,
                                           length: kWeight.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared)
        else {
            Issue.record("Failed to allocate buffers")
            return
        }

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate command buffer")
            return
        }
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: qLegacy,
                               weight: qW,
                               out: qLegacy,
                               headDim: UInt32(headDim),
                               numHeads: numQHeads,
                               eps: 1e-6)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: kLegacy,
                               weight: kW,
                               out: kLegacy,
                               headDim: UInt32(headDim),
                               numHeads: numKVHeads,
                               eps: 1e-6)
        rms.encodeNoScalePerHead(commandBuffer: cb,
                                 x: vLegacy,
                                 out: vLegacy,
                                 headDim: UInt32(headDim),
                                 numHeads: numKVHeads,
                                 eps: 1e-6)
        if rotatedPairs * 2 == UInt32(headDim) {
            rope.encodeDefaultNeox(commandBuffer: cb,
                                   data: qLegacy,
                                   position: UInt32(position),
                                   headDim: UInt32(headDim),
                                   numHeads: UInt32(numQHeads),
                                   theta: theta)
            rope.encodeDefaultNeox(commandBuffer: cb,
                                   data: kLegacy,
                                   position: UInt32(position),
                                   headDim: UInt32(headDim),
                                   numHeads: UInt32(numKVHeads),
                                   theta: theta)
        } else {
            rope.encodeProportionalNeox(commandBuffer: cb,
                                        data: qLegacy,
                                        position: UInt32(position),
                                        headDim: UInt32(headDim),
                                        numHeads: UInt32(numQHeads),
                                        rotatedPairs: rotatedPairs,
                                        theta: theta)
            rope.encodeProportionalNeox(commandBuffer: cb,
                                        data: kLegacy,
                                        position: UInt32(position),
                                        headDim: UInt32(headDim),
                                        numHeads: UInt32(numKVHeads),
                                        rotatedPairs: rotatedPairs,
                                        theta: theta)
        }
        fused.encode(commandBuffer: cb,
                     q: qFused,
                     k: kFused,
                     v: vFused,
                     qWeight: qW,
                     kWeight: kW,
                     headDim: UInt32(headDim),
                     numQHeads: UInt32(numQHeads),
                     numKVHeads: UInt32(numKVHeads),
                     position: UInt32(position),
                     theta: theta,
                     rotatedPairs: rotatedPairs,
                     eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            Issue.record("Command buffer failed: \(err)")
            return
        }

        #expect(Self.bufferBytes(qLegacy, count: qCount) == Self.bufferBytes(qFused, count: qCount),
                "Q mismatch for headDim=\(headDim) kvHeads=\(numKVHeads)")
        #expect(Self.bufferBytes(kLegacy, count: kvCount) == Self.bufferBytes(kFused, count: kvCount),
                "K mismatch for headDim=\(headDim) kvHeads=\(numKVHeads)")
        #expect(Self.bufferBytes(vLegacy, count: kvCount) == Self.bufferBytes(vFused, count: kvCount),
                "V mismatch for headDim=\(headDim) kvHeads=\(numKVHeads)")
    }

    private static func bytes(_ count: Int) -> Int {
        count * MemoryLayout<Float16>.size
    }

    private static func bufferBytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self),
                                  count: bytes(count)))
    }
}
