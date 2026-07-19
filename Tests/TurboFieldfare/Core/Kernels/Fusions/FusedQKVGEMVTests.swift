import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct FusedQKVGEMVTests {
    @Test func fusedQKVGEMV_matchesThreeGEMVs_smallShape() throws {
        try Self.expectMatches(qRows: 64, kvRows: 32, n: 128, seed: 0x5156_4745)
    }

    @Test func fusedQKVGEMV_matchesThreeGEMVs_offsetWeights() throws {
        try Self.expectMatches(qRows: 128, kvRows: 64, n: 2816, seed: 0x5156_4746,
                               weightOffset: 2)
    }

    private static func expectMatches(qRows: Int,
                                      kvRows: Int,
                                      n: Int,
                                      seed: UInt64,
                                      weightOffset: Int = 0) throws {
        precondition(n % Quantization.groupSize == 0)
        var rng = SplitMix64(seed: seed)
        let q = Self.makeProjection(rows: qRows, n: n, rng: &rng, weightOffset: weightOffset)
        let k = Self.makeProjection(rows: kvRows, n: n, rng: &rng, weightOffset: weightOffset)
        let v = Self.makeProjection(rows: kvRows, n: n, rng: &rng, weightOffset: weightOffset)
        let x = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

        let ctx = try MetalContext()
        let gemv = try DequantInt4GEMV(context: ctx)
        let fused = try FusedQKVGEMV(context: ctx)
        guard
            let qW = ctx.device.makeBuffer(bytes: q.weights, length: q.weights.count,
                                           options: .storageModeShared),
            let qS = ctx.device.makeBuffer(bytes: q.scales,
                                           length: q.scales.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared),
            let qB = ctx.device.makeBuffer(bytes: q.biases,
                                           length: q.biases.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared),
            let kW = ctx.device.makeBuffer(bytes: k.weights, length: k.weights.count,
                                           options: .storageModeShared),
            let kS = ctx.device.makeBuffer(bytes: k.scales,
                                           length: k.scales.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared),
            let kB = ctx.device.makeBuffer(bytes: k.biases,
                                           length: k.biases.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared),
            let vW = ctx.device.makeBuffer(bytes: v.weights, length: v.weights.count,
                                           options: .storageModeShared),
            let vS = ctx.device.makeBuffer(bytes: v.scales,
                                           length: v.scales.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared),
            let vB = ctx.device.makeBuffer(bytes: v.biases,
                                           length: v.biases.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared),
            let xBuf = Fp16Buffer.make(ctx.device, halves: x),
            let qLegacy = Fp16Buffer.make(ctx.device, count: qRows),
            let kLegacy = Fp16Buffer.make(ctx.device, count: kvRows),
            let vLegacy = Fp16Buffer.make(ctx.device, count: kvRows),
            let qFused = Fp16Buffer.make(ctx.device, count: qRows),
            let kFused = Fp16Buffer.make(ctx.device, count: kvRows),
            let vFused = Fp16Buffer.make(ctx.device, count: kvRows)
        else {
            Issue.record("Failed to allocate buffers")
            return
        }

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate command buffer")
            return
        }
        gemv.encode(commandBuffer: cb,
                    weights: qW, weightsOffset: weightOffset,
                    scales: qS, biases: qB,
                    x: xBuf, y: qLegacy,
                    m: UInt32(qRows), n: UInt32(n))
        gemv.encode(commandBuffer: cb,
                    weights: kW, weightsOffset: weightOffset,
                    scales: kS, biases: kB,
                    x: xBuf, y: kLegacy,
                    m: UInt32(kvRows), n: UInt32(n))
        gemv.encode(commandBuffer: cb,
                    weights: vW, weightsOffset: weightOffset,
                    scales: vS, biases: vB,
                    x: xBuf, y: vLegacy,
                    m: UInt32(kvRows), n: UInt32(n))
        fused.encode(commandBuffer: cb,
                     qWeights: qW, qWeightsOffset: weightOffset,
                     qScales: qS, qBiases: qB,
                     kWeights: kW, kWeightsOffset: weightOffset,
                     kScales: kS, kBiases: kB,
                     vWeights: vW, vWeightsOffset: weightOffset,
                     vScales: vS, vBiases: vB,
                     x: xBuf,
                     qOut: qFused,
                     kOut: kFused,
                     vOut: vFused,
                     qRows: UInt32(qRows),
                     kvRows: UInt32(kvRows),
                     n: UInt32(n))
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            Issue.record("Command buffer failed: \(error)")
            return
        }

        #expect(Self.bytes(qLegacy, count: qRows) == Self.bytes(qFused, count: qRows))
        #expect(Self.bytes(kLegacy, count: kvRows) == Self.bytes(kFused, count: kvRows))
        #expect(Self.bytes(vLegacy, count: kvRows) == Self.bytes(vFused, count: kvRows))
    }

    private static func makeProjection(rows: Int,
                                       n: Int,
                                       rng: inout SplitMix64,
                                       weightOffset: Int) -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
        let packedPerRow = n / 2
        let groups = n / Quantization.groupSize
        var weights = [UInt8](repeating: 0, count: rows * packedPerRow + weightOffset)
        var scales = [UInt16](repeating: 0, count: rows * groups)
        var biases = [UInt16](repeating: 0, count: rows * groups)
        for row in 0..<rows {
            let values = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
            let q = Quantization.quantizeInt4Affine(values)
            for i in 0..<packedPerRow { weights[weightOffset + row * packedPerRow + i] = q.packed[i] }
            for i in 0..<groups {
                scales[row * groups + i] = q.scales[i]
                biases[row * groups + i] = q.biases[i]
            }
        }
        return (weights, scales, biases)
    }

    private static func bytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self),
                                  count: count * MemoryLayout<Float16>.size))
    }
}
