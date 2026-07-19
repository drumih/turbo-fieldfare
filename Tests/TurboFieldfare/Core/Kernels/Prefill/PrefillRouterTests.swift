import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillRouterTests {
    private static let experts = 16
    private static let d = 128
    private static let topK = 8

    @Test func tokenExpertPairLayoutIsStable() {
        #expect(MemoryLayout<PrefillTokenExpertPair>.size == 16)
        #expect(MemoryLayout<PrefillTokenExpertPair>.stride == 16)
        #expect(MemoryLayout.offset(of: \PrefillTokenExpertPair.token) == .some(0))
        #expect(MemoryLayout.offset(of: \PrefillTokenExpertPair.expert) == .some(4))
        #expect(MemoryLayout.offset(of: \PrefillTokenExpertPair.rank) == .some(8))
        #expect(MemoryLayout.offset(of: \PrefillTokenExpertPair.weightBitsAndReserved) == .some(12))
    }

    @Test func tokenExpertPairsAreTokenMajor() {
        let weights: [Float16] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map { Float16($0) }
        let pairs = PrefillRouter.makeTokenExpertPairs(indices: [3, 1, 2, 4, 0, 5],
                                                       weights: weights,
                                                       queryCount: 2,
                                                       topK: 3)
        #expect(pairs.map(\.token) == [0, 0, 0, 1, 1, 1])
        #expect(pairs.map(\.rank) == [0, 1, 2, 0, 1, 2])
        #expect(pairs.map(\.expert) == [3, 1, 2, 4, 0, 5])
        #expect(pairs.map { Float($0.weight) } == weights.map { Float($0) })
    }

    @Test func blockGemma4RouterMatchesRepeatedScalarRows() throws {
        var rng = SplitMix64(seed: 0x9A7E_2026)
        let rows = 3
        let hiddenStride = Self.d + 13
        let weights = Self.makeStableWeights(rng: &rng)
        let hidden = Self.makeHiddenBlock(rows: rows,
                                          rowStride: hiddenStride,
                                          d: Self.d,
                                          rng: &rng,
                                          sentinel: Float16(-31.0))
        let effectiveScale = (0..<Self.d).map { _ in
            rng.uniform(0.5, 1.5) / Float(Self.d).squareRoot()
        }
        let perExpertScale = (0..<Self.experts).map { _ in rng.uniform(0.6, 1.4) }

        let ctx = try MetalContext()
        let moe = try MoE(context: ctx)
        let prefill = try PrefillRouter(context: ctx)
        let buffers = try Self.makeBuffers(ctx: ctx,
                                           weights: weights,
                                           hidden: hidden,
                                           effectiveScale: effectiveScale,
                                           perExpertScale: perExpertScale,
                                           rows: rows)

        var expectedIndices = [UInt32]()
        var expectedWeights = [Float16]()
        for row in 0..<rows {
            let compactHidden = Array(hidden[(row * hiddenStride)..<(row * hiddenStride + Self.d)])
            let (idx, wt) = try Self.runScalarRouter(ctx: ctx,
                                                     moe: moe,
                                                     buffers: buffers,
                                                     hiddenRow: compactHidden)
            expectedIndices.append(contentsOf: idx)
            expectedWeights.append(contentsOf: wt)
        }

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer")
            return
        }
        prefill.encodeGemma4Block(commandBuffer: cb,
                                  weights: buffers.weights,
                                  scales: buffers.scales,
                                  biases: buffers.biases,
                                  hidden: buffers.hidden,
                                  effectiveScale: buffers.effectiveScale,
                                  perExpertScale: buffers.perExpertScale,
                                  outIndices: buffers.blockIndices,
                                  outWeights: buffers.blockWeights,
                                  queryCount: UInt32(rows),
                                  numExperts: UInt32(Self.experts),
                                  d: UInt32(Self.d),
                                  topK: UInt32(Self.topK),
                                  hiddenStrideElements: UInt32(hiddenStride))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let gotIndices = Self.readUInt32(buffers.blockIndices, count: rows * Self.topK)
        let gotWeights = Fp16Buffer.readHalf(buffers.blockWeights, count: rows * Self.topK)
        #expect(gotIndices == expectedIndices)
        Self.assertWeightsClose(gotWeights, expectedWeights, tolerance: 5e-3)
        Self.assertPaddingUnchanged(buffer: buffers.hidden,
                                    original: hidden,
                                    rows: rows,
                                    rowStride: hiddenStride,
                                    used: Self.d)
    }

    @Test func blockGemma4RouterNearTieMatchesScalarPath() throws {
        let weights = Self.makeNearTieWeights()
        let hidden = [Float16](repeating: 1, count: Self.d)
        let effectiveScale = [Float](repeating: 1, count: Self.d)
        let perExpertScale = [Float](repeating: 1, count: Self.experts)
        let ctx = try MetalContext()
        let moe = try MoE(context: ctx)
        let prefill = try PrefillRouter(context: ctx)
        let buffers = try Self.makeBuffers(ctx: ctx,
                                           weights: weights,
                                           hidden: hidden,
                                           effectiveScale: effectiveScale,
                                           perExpertScale: perExpertScale,
                                           rows: 1)
        let (scalar, _) = try Self.runScalarRouter(ctx: ctx,
                                                   moe: moe,
                                                   buffers: buffers,
                                                   hiddenRow: hidden)

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer")
            return
        }
        prefill.encodeGemma4Block(commandBuffer: cb,
                                  weights: buffers.weights,
                                  scales: buffers.scales,
                                  biases: buffers.biases,
                                  hidden: buffers.hidden,
                                  effectiveScale: buffers.effectiveScale,
                                  perExpertScale: buffers.perExpertScale,
                                  outIndices: buffers.blockIndices,
                                  outWeights: buffers.blockWeights,
                                  queryCount: 1,
                                  numExperts: UInt32(Self.experts),
                                  d: UInt32(Self.d),
                                  topK: UInt32(Self.topK),
                                  hiddenStrideElements: UInt32(Self.d))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let block = Self.readUInt32(buffers.blockIndices, count: Self.topK)
        #expect(block == scalar)
    }

    private struct RouterBuffers {
        let weights: MTLBuffer
        let scales: MTLBuffer
        let biases: MTLBuffer
        let hidden: MTLBuffer
        let effectiveScale: MTLBuffer
        let perExpertScale: MTLBuffer
        let blockIndices: MTLBuffer
        let blockWeights: MTLBuffer
    }

    private static func makeBuffers(ctx: MetalContext,
                                    weights: [[Float]],
                                    hidden: [Float16],
                                    effectiveScale: [Float],
                                    perExpertScale: [Float],
                                    rows: Int) throws -> RouterBuffers {
        let packed = packWeights(weights)
        let effBits = effectiveScale.map { Quantization.bf16Bits($0) }
        let pesBits = perExpertScale.map { Quantization.bf16Bits($0) }
        guard let wBuf = ctx.device.makeBuffer(bytes: packed.packed,
                                               length: packed.packed.count,
                                               options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: packed.scales,
                                               length: packed.scales.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(bytes: packed.biases,
                                               length: packed.biases.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let hBuf = ctx.device.makeBuffer(bytes: hidden,
                                               length: hidden.count * MemoryLayout<Float16>.size,
                                               options: .storageModeShared),
              let eBuf = ctx.device.makeBuffer(bytes: effBits,
                                               length: effBits.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let pBuf = ctx.device.makeBuffer(bytes: pesBits,
                                               length: pesBits.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let idxBuf = ctx.device.makeBuffer(length: rows * Self.topK * MemoryLayout<UInt32>.size,
                                                 options: .storageModeShared),
              let wtBuf = ctx.device.makeBuffer(length: rows * Self.topK * MemoryLayout<Float16>.size,
                                                options: .storageModeShared) else {
            throw RouterTestError.allocationFailed
        }
        return RouterBuffers(weights: wBuf,
                             scales: sBuf,
                             biases: bBuf,
                             hidden: hBuf,
                             effectiveScale: eBuf,
                             perExpertScale: pBuf,
                             blockIndices: idxBuf,
                             blockWeights: wtBuf)
    }

    private static func runScalarRouter(ctx: MetalContext,
                                        moe: MoE,
                                        buffers: RouterBuffers,
                                        hiddenRow: [Float16]) throws -> ([UInt32], [Float16]) {
        guard let hBuf = ctx.device.makeBuffer(bytes: hiddenRow,
                                               length: hiddenRow.count * MemoryLayout<Float16>.size,
                                               options: .storageModeShared),
              let idxBuf = ctx.device.makeBuffer(length: Self.topK * MemoryLayout<UInt32>.size,
                                                 options: .storageModeShared),
              let wtBuf = ctx.device.makeBuffer(length: Self.topK * MemoryLayout<Float16>.size,
                                                options: .storageModeShared),
              let cb = ctx.queue.makeCommandBuffer() else {
            throw RouterTestError.allocationFailed
        }
        moe.encodeRouterGemma4(commandBuffer: cb,
                               weights: buffers.weights,
                               scales: buffers.scales,
                               biases: buffers.biases,
                               hidden: hBuf,
                               effectiveScale: buffers.effectiveScale,
                               perExpertScale: buffers.perExpertScale,
                               outIndices: idxBuf,
                               outWeights: wtBuf,
                               numExperts: UInt32(Self.experts),
                               d: UInt32(Self.d),
                               topK: UInt32(Self.topK))
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw error
        }
        return (readUInt32(idxBuf, count: Self.topK),
                Fp16Buffer.readHalf(wtBuf, count: Self.topK))
    }

    private static func makeStableWeights(rng: inout SplitMix64) -> [[Float]] {
        (0..<Self.experts).map { expert in
            let bias = Float(expert) * 0.015
            return (0..<Self.d).map { _ in rng.uniform(-0.04, 0.04) + bias }
        }
    }

    private static func makeNearTieWeights() -> [[Float]] {
        var weights = [[Float]](repeating: [Float](repeating: 0, count: Self.d),
                                count: Self.experts)
        for expert in 0..<Self.experts {
            let gain: Float
            if expert < 7 {
                gain = 1.0 - Float(expert) * 0.05
            } else if expert == 7 {
                gain = 0.50
            } else if expert == 8 {
                gain = 0.50 * (1.0 + 1e-4)
            } else {
                gain = 0.40 - Float(expert - 9) * 0.02
            }
            for k in 0..<Self.d {
                let pattern = 0.25 + Float((k * 37 + 11) % 97) / 97.0
                weights[expert][k] = pattern * gain
            }
        }
        return weights
    }

    private static func makeHiddenBlock(rows: Int,
                                        rowStride: Int,
                                        d: Int,
                                        rng: inout SplitMix64,
                                        sentinel: Float16) -> [Float16] {
        var values = [Float16](repeating: sentinel, count: rows * rowStride)
        for row in 0..<rows {
            for i in 0..<d {
                values[row * rowStride + i] = Float16(rng.uniform(-1.0, 1.0))
            }
        }
        return values
    }

    private static func packWeights(_ weights: [[Float]])
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let eCount = weights.count
        let d = weights[0].count
        let groupsPerRow = d / Quantization.groupSize
        var packed = [UInt8](repeating: 0, count: eCount * d)
        var scales = [UInt16](repeating: 0, count: eCount * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: eCount * groupsPerRow)
        for e in 0..<eCount {
            let q = Quantization.quantizeInt8Affine(weights[e])
            for i in 0..<d {
                packed[e * d + i] = q.packed[i]
            }
            for g in 0..<groupsPerRow {
                scales[e * groupsPerRow + g] = q.scales[g]
                biases[e * groupsPerRow + g] = q.biases[g]
            }
        }
        return (packed, scales, biases)
    }

    private static func readUInt32(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
        (0..<count).map {
            buffer.contents().load(fromByteOffset: $0 * MemoryLayout<UInt32>.size,
                                   as: UInt32.self)
        }
    }

    private static func assertWeightsClose(_ got: [Float16],
                                           _ expected: [Float16],
                                           tolerance: Float) {
        var maxAbs: Float = 0
        for i in got.indices {
            maxAbs = max(maxAbs, abs(Float(got[i]) - Float(expected[i])))
        }
        #expect(maxAbs <= tolerance, "route weight maxAbsDiff=\(maxAbs)")
    }

    private static func assertPaddingUnchanged(buffer: MTLBuffer,
                                               original: [Float16],
                                               rows: Int,
                                               rowStride: Int,
                                               used: Int) {
        let got = Fp16Buffer.readHalf(buffer, count: original.count)
        for row in 0..<rows {
            for i in used..<rowStride {
                let idx = row * rowStride + i
                #expect(got[idx] == original[idx], "padding changed at row \(row) element \(i)")
            }
        }
    }

    private enum RouterTestError: Error {
        case allocationFailed
    }
}
