import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// K2-Horizon's router is not the Qwen softmax router: sigmoid scores, the
/// bias applied only to the ranking, the selected sigmoid scores normalized
/// over top-k and scaled by `router_scaling_factor`. The same selection serves
/// the 100-expert MoE (top-8) and the 64-expert MoVA router (top-4), so both
/// shapes are pinned here.
@Suite struct K2RouterSelectionTests {
    private struct Result {
        let indices: [UInt32]
        let weights: [Float]
    }

    private static func reference(
        logits: [Float],
        bias: [Float],
        topK: Int,
        scaling: Float
    ) -> Result {
        let scores = logits.map { 1 / (1 + exp(-$0)) }
        let ranks = scores.indices.map { scores[$0] + bias[$0] }
        let order = ranks.indices.sorted {
            ranks[$0] == ranks[$1] ? $0 < $1 : ranks[$0] > ranks[$1]
        }
        let chosen = Array(order.prefix(topK))
        let total = chosen.reduce(Float(0)) { $0 + scores[$1] }
        return Result(
            indices: chosen.map(UInt32.init),
            weights: chosen.map { scores[$0] / total * scaling })
    }

    private static func makeBuffer<T>(_ ctx: MetalContext, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return nil }
            return ctx.device.makeBuffer(
                bytes: base,
                length: ptr.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }

    private static func run(
        logits: [Float],
        bias: [Float],
        topK: Int,
        scaling: Float
    ) throws -> Result {
        let ctx = try MetalContext()
        let selection = try K2RouterSelection(context: ctx)
        let logitsBuf = try #require(Self.makeBuffer(ctx, logits))
        let biasBits = bias.map { Quantization.bf16Bits($0) }
        let biasBuf = try #require(Self.makeBuffer(ctx, biasBits))
        let indexBuf = try #require(
            ctx.device.makeBuffer(
                length: topK * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let weightBuf = try #require(Fp16Buffer.make(ctx.device, count: topK))

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try selection.encode(
            commandBuffer: cb, logits: logitsBuf, bias: biasBuf,
            outIndices: indexBuf, outWeights: weightBuf,
            numExperts: logits.count, topK: topK, scaling: scaling)
        cb.commit()
        cb.waitUntilCompleted()

        var indices = [UInt32](repeating: 0, count: topK)
        let base = indexBuf.contents()
        for i in 0..<topK {
            indices[i] = base.load(
                fromByteOffset: i * MemoryLayout<UInt32>.stride, as: UInt32.self)
        }
        return Result(indices: indices, weights: Fp16Buffer.read(weightBuf, count: topK))
    }

    private static func biasInKernelUnits(_ values: [Float]) -> [Float] {
        values.map { Quantization.bf16ToFloat(Quantization.bf16Bits($0)) }
    }

    @Test func moeShapeMatchesReference() throws {
        var rng = SeedTree(0x100E_2A11).key("k2-router-moe")
        let logits = (0..<100).map { _ in rng.uniform(-8, 8) }
        let bias = Self.biasInKernelUnits((0..<100).map { _ in rng.uniform(-2, 2) })
        let expected = Self.reference(
            logits: logits, bias: bias, topK: 8, scaling: 2.5)
        let actual = try Self.run(
            logits: logits, bias: bias, topK: 8, scaling: 2.5)
        #expect(actual.indices == expected.indices)
        #expect(
            RelError.compute(actual: actual.weights, reference: expected.weights)
                < Tolerance.fp16Reduction)
    }

    @Test func movaShapeMatchesReference() throws {
        var rng = SeedTree(0x64A0_0F4).key("k2-router-mova")
        let logits = (0..<64).map { _ in rng.uniform(-6, 6) }
        let bias = Self.biasInKernelUnits((0..<64).map { _ in rng.uniform(-1, 1) })
        let expected = Self.reference(
            logits: logits, bias: bias, topK: 4, scaling: 2.5)
        let actual = try Self.run(
            logits: logits, bias: bias, topK: 4, scaling: 2.5)
        #expect(actual.indices == expected.indices)
        #expect(
            RelError.compute(actual: actual.weights, reference: expected.weights)
                < Tolerance.fp16Reduction)
    }

    /// All ranks equal: the reference's tie-break keeps the lowest expert ids,
    /// and the weights are the renormalized sigmoid of the chosen set.
    @Test func exactTiesKeepTheLowestIds() throws {
        let logits = [Float](repeating: 0.5, count: 16)
        let bias = Self.biasInKernelUnits([Float](repeating: 0.25, count: 16))
        let expected = Self.reference(
            logits: logits, bias: bias, topK: 4, scaling: 2.5)
        let actual = try Self.run(
            logits: logits, bias: bias, topK: 4, scaling: 2.5)
        #expect(actual.indices == [0, 1, 2, 3])
        #expect(actual.indices == expected.indices)
        #expect(
            RelError.compute(actual: actual.weights, reference: expected.weights)
                < Tolerance.fp16Reduction)
    }
}
