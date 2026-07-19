import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct FusedQKVPipelineTests {
    @Test func fullShapeFusedPipelineMatchesThreeGEMVChainBitwise() throws {
        let qRows = 8192
        let kvRows = 1024
        let n = 2816
        let headDim = 512
        let numQHeads = 16
        let numKVHeads = 2
        let qProjection = Self.makeProjection(rows: qRows, n: n, seed: 0x5156_5049)
        let kvProjection = Self.makeProjection(rows: kvRows, n: n, seed: 0x4B56_5049)
        var rng = SplitMix64(seed: 0x585F_5049)
        let x = (0..<n).map { _ in Float16(rng.uniform(-1, 1)) }
        let qNorm = (0..<headDim).map { _ in
            Quantization.bf16Bits(rng.uniform(0.5, 1.5))
        }
        let kNorm = (0..<headDim).map { _ in
            Quantization.bf16Bits(rng.uniform(0.5, 1.5))
        }

        let ctx = try MetalContext()
        let gemv = try DequantInt4GEMV(context: ctx)
        let fusedGEMV = try FusedQKVGEMV(context: ctx)
        let rms = try RMSNorm(context: ctx)
        let rope = try RoPE(context: ctx)
        let fusedEpilogue = try FusedQKVEpilogue(context: ctx)

        guard
            let qWeights = ctx.device.makeBuffer(bytes: qProjection.weights,
                                                 length: qProjection.weights.count,
                                                 options: .storageModeShared),
            let qScales = ctx.device.makeBuffer(bytes: qProjection.scales,
                                                length: qProjection.scales.count * 2,
                                                options: .storageModeShared),
            let qBiases = ctx.device.makeBuffer(bytes: qProjection.biases,
                                                length: qProjection.biases.count * 2,
                                                options: .storageModeShared),
            let kvWeights = ctx.device.makeBuffer(bytes: kvProjection.weights,
                                                  length: kvProjection.weights.count,
                                                  options: .storageModeShared),
            let kvScales = ctx.device.makeBuffer(bytes: kvProjection.scales,
                                                 length: kvProjection.scales.count * 2,
                                                 options: .storageModeShared),
            let kvBiases = ctx.device.makeBuffer(bytes: kvProjection.biases,
                                                 length: kvProjection.biases.count * 2,
                                                 options: .storageModeShared),
            let xBuffer = Fp16Buffer.make(ctx.device, halves: x),
            let qNormBuffer = ctx.device.makeBuffer(bytes: qNorm,
                                                    length: qNorm.count * 2,
                                                    options: .storageModeShared),
            let kNormBuffer = ctx.device.makeBuffer(bytes: kNorm,
                                                    length: kNorm.count * 2,
                                                    options: .storageModeShared),
            let qLegacy = Fp16Buffer.make(ctx.device, count: qRows),
            let kLegacy = Fp16Buffer.make(ctx.device, count: kvRows),
            let vLegacy = Fp16Buffer.make(ctx.device, count: kvRows),
            let qFused = Fp16Buffer.make(ctx.device, count: qRows),
            let kFused = Fp16Buffer.make(ctx.device, count: kvRows),
            let vFused = Fp16Buffer.make(ctx.device, count: kvRows),
            let commandBuffer = ctx.queue.makeCommandBuffer()
        else {
            Issue.record("Failed to allocate full-shape QKV pipeline resources")
            return
        }

        gemv.encode(commandBuffer: commandBuffer,
                    weights: qWeights, scales: qScales, biases: qBiases,
                    x: xBuffer, y: qLegacy,
                    m: UInt32(qRows), n: UInt32(n))
        gemv.encode(commandBuffer: commandBuffer,
                    weights: kvWeights, scales: kvScales, biases: kvBiases,
                    x: xBuffer, y: kLegacy,
                    m: UInt32(kvRows), n: UInt32(n))
        gemv.encode(commandBuffer: commandBuffer,
                    weights: kvWeights, scales: kvScales, biases: kvBiases,
                    x: xBuffer, y: vLegacy,
                    m: UInt32(kvRows), n: UInt32(n))
        rms.encodeBF16WPerHead(commandBuffer: commandBuffer,
                               x: qLegacy, weight: qNormBuffer, out: qLegacy,
                               headDim: UInt32(headDim), numHeads: numQHeads, eps: 1e-6)
        rms.encodeBF16WPerHead(commandBuffer: commandBuffer,
                               x: kLegacy, weight: kNormBuffer, out: kLegacy,
                               headDim: UInt32(headDim), numHeads: numKVHeads, eps: 1e-6)
        rms.encodeNoScalePerHead(commandBuffer: commandBuffer,
                                 x: vLegacy, out: vLegacy,
                                 headDim: UInt32(headDim), numHeads: numKVHeads, eps: 1e-6)
        rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                    data: qLegacy,
                                    position: 12,
                                    headDim: UInt32(headDim),
                                    numHeads: UInt32(numQHeads),
                                    rotatedPairs: 64,
                                    theta: 1_000_000)
        rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                    data: kLegacy,
                                    position: 12,
                                    headDim: UInt32(headDim),
                                    numHeads: UInt32(numKVHeads),
                                    rotatedPairs: 64,
                                    theta: 1_000_000)

        fusedGEMV.encode(commandBuffer: commandBuffer,
                         qWeights: qWeights, qScales: qScales, qBiases: qBiases,
                         kWeights: kvWeights, kScales: kvScales, kBiases: kvBiases,
                         vWeights: kvWeights, vScales: kvScales, vBiases: kvBiases,
                         x: xBuffer,
                         qOut: qFused, kOut: kFused, vOut: vFused,
                         qRows: UInt32(qRows), kvRows: UInt32(kvRows), n: UInt32(n))
        fusedEpilogue.encode(commandBuffer: commandBuffer,
                             q: qFused, k: kFused, v: vFused,
                             qWeight: qNormBuffer, kWeight: kNormBuffer,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(numQHeads),
                             numKVHeads: UInt32(numKVHeads),
                             position: 12,
                             theta: 1_000_000,
                             rotatedPairs: 64,
                             eps: 1e-6)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            Issue.record("Command buffer failed: \(error)")
            return
        }

        #expect(Self.bytes(qLegacy, count: qRows) == Self.bytes(qFused, count: qRows))
        #expect(Self.bytes(kLegacy, count: kvRows) == Self.bytes(kFused, count: kvRows))
        #expect(Self.bytes(vLegacy, count: kvRows) == Self.bytes(vFused, count: kvRows))
    }

    private static func makeProjection(rows: Int,
                                       n: Int,
                                       seed: UInt64) -> (weights: [UInt8],
                                                        scales: [UInt16],
                                                        biases: [UInt16]) {
        let packedCount = rows * n / 2
        let groupCount = rows * n / Quantization.groupSize
        let weights = (0..<packedCount).map { index in
            UInt8(truncatingIfNeeded: (UInt64(index) &* 0x9E37_79B9) &+ seed)
        }
        let scales = (0..<groupCount).map { index in
            Quantization.bf16Bits(0.015625 + Float(index % 7) * 0.001953125)
        }
        let biases = (0..<groupCount).map { index in
            Quantization.bf16Bits(-0.125 + Float(index % 5) * 0.03125)
        }
        return (weights, scales, biases)
    }

    private static func bytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: UInt8.self),
            count: count * MemoryLayout<Float16>.size))
    }
}
