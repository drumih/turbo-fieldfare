import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Drives the K3 FP32 sigmoid router (`k3_router_gemv_f32` + the CPU
/// top-16/renormalize in `K3Router.selectTopK`) against
/// `K3RouterReference`, which mirrors `KimiMoEGate`: sigmoid scores, bias
/// only in the selection key, unbiased scores renormalized by sum + 1e-20,
/// scaling factor 1.0.
@Suite struct K3RouterKernelTests {

    // MARK: helpers

    private static func makeF32Buffer(_ device: MTLDevice, _ values: [Float])
        -> MTLBuffer?
    {
        values.withUnsafeBufferPointer {
            device.makeBuffer(bytes: $0.baseAddress!,
                              length: values.count * MemoryLayout<Float>.stride,
                              options: .storageModeShared)
        }
    }

    private static func readF32Buffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let base = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    private struct Fixture {
        var numExperts: Int
        var d: Int
        var gate: [Float]     // row-major [numExperts, d]
        var x: [Float]
        var bias: [Float]
    }

    private static func makeFixture(seed: UInt64, numExperts: Int, d: Int)
        -> Fixture
    {
        var rng = SeedTree(seed).key("k3-router-e\(numExperts)-d\(d)")
        // Gate entries small so logits (and sigmoid) cover the linear
        // region, not just the saturated tails.
        let gate = (0..<(numExperts * d)).map { _ in
            rng.uniform(-0.02, 0.02)
        }
        let x = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let bias = (0..<numExperts).map { _ in rng.uniform(-0.1, 0.1) }
        return Fixture(numExperts: numExperts, d: d, gate: gate, x: x,
                       bias: bias)
    }

    private static func gpuScores(fixture: Fixture)
        throws -> (scores: [Float], biased: [Float])
    {
        let ctx = try MetalContext()
        let router = try K3Router(context: ctx)
        guard let gateBuf = makeF32Buffer(ctx.device, fixture.gate),
              let xBuf = makeF32Buffer(ctx.device, fixture.x),
              let biasBuf = makeF32Buffer(ctx.device, fixture.bias),
              let scoresBuf = ctx.device.makeBuffer(
                length: fixture.numExperts * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let biasedBuf = ctx.device.makeBuffer(
                length: fixture.numExperts * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers")
            return ([], [])
        }
        router.encodeScores(commandBuffer: cmd,
                            gate: gateBuf, x: xBuf, bias: biasBuf,
                            scores: scoresBuf, scoresBiased: biasedBuf,
                            numExperts: UInt32(fixture.numExperts),
                            d: UInt32(fixture.d))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
        return (readF32Buffer(scoresBuf, count: fixture.numExperts),
                readF32Buffer(biasedBuf, count: fixture.numExperts))
    }

    // MARK: kernel vs reference

    @Test func scoresAndBiasedMatchReferenceSmall() throws {
        let fixture = Self.makeFixture(seed: 0xD1, numExperts: 16, d: 256)
        let (scores, biased) = try Self.gpuScores(fixture: fixture)
        let reference = K3RouterReference.route(
            x: fixture.x, gate: fixture.gate, bias: fixture.bias,
            numExperts: fixture.numExperts, d: fixture.d, topK: 4)
        let relScores = RelError.compute(actual: scores, reference: reference.scores)
        let relKeys = RelError.compute(actual: biased, reference: reference.keys)
        #expect(relScores < Tolerance.identity, "scores: rel=\(relScores)")
        #expect(relKeys < Tolerance.identity, "keys: rel=\(relKeys)")
    }

    /// Canonical router shape: 896 x 7168 FP32 — also exercises the
    /// function-constant specialized pipeline.
    @Test func scoresMatchReferenceCanonicalShape() throws {
        let fixture = Self.makeFixture(seed: 0xD2, numExperts: 896, d: 7168)
        let (scores, _) = try Self.gpuScores(fixture: fixture)
        let reference = K3RouterReference.route(
            x: fixture.x, gate: fixture.gate, bias: fixture.bias,
            numExperts: fixture.numExperts, d: fixture.d, topK: 16)
        let rel = RelError.compute(actual: scores, reference: reference.scores)
        let maxAbs = RelError.maxAbsDiff(scores, reference.scores)
        #expect(rel < Tolerance.identity, "scores: rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// numExperts not a multiple of the 4-rows-per-threadgroup shape.
    @Test func scoresMatchReferenceOddExpertCount() throws {
        let fixture = Self.makeFixture(seed: 0xD3, numExperts: 13, d: 160)
        let (scores, _) = try Self.gpuScores(fixture: fixture)
        let reference = K3RouterReference.route(
            x: fixture.x, gate: fixture.gate, bias: fixture.bias,
            numExperts: fixture.numExperts, d: fixture.d, topK: 4)
        let rel = RelError.compute(actual: scores, reference: reference.scores)
        #expect(rel < Tolerance.identity, "scores: rel=\(rel)")
    }

    // MARK: CPU top-k + renormalize

    /// The dispatcher's CPU selection must match the reference gate exactly
    /// on identical scores (both implement: keys = scores + bias, top-k by
    /// key, weights = unbiased scores / (sum + 1e-20)).
    @Test func selectTopKMatchesReference() throws {
        let fixture = Self.makeFixture(seed: 0xD4, numExperts: 64, d: 128)
        let reference = K3RouterReference.route(
            x: fixture.x, gate: fixture.gate, bias: fixture.bias,
            numExperts: fixture.numExperts, d: fixture.d, topK: 16)
        let (indices, weights) = K3Router.selectTopK(
            scores: reference.scores, bias: fixture.bias, topK: 16)
        #expect(indices == reference.indices.map { UInt32($0) })
        #expect(weights == reference.weights)
        // Renormalized: sum of weights is 1 (scaling factor 1.0).
        let weightSum = weights.reduce(0, +)
        #expect(abs(weightSum - 1) < 1e-6, "weight sum \(weightSum)")
    }

    /// Bias must be able to change the selection: an expert outside the raw
    /// score top-k gets promoted by a large correction bias, with the
    /// UNBIASED score still producing its weight.
    @Test func biasPromotesIntoSelection() throws {
        var scores = [Float](repeating: 0, count: 8)
        for i in 0..<8 { scores[i] = Float(8 - i) / 10 }  // 0.8, 0.7, ..., 0.1
        var bias = [Float](repeating: 0, count: 8)
        bias[6] = 0.75  // key 0.2 + 0.75 = 0.95 — highest
        let (indices, weights) = K3Router.selectTopK(
            scores: scores, bias: bias, topK: 3)
        #expect(indices == [6, 0, 1])
        let sum = scores[6] + scores[0] + scores[1]
        #expect(abs(weights[0] - scores[6] / sum) < 1e-7)
        #expect(abs(weights[1] - scores[0] / sum) < 1e-7)
        #expect(abs(weights[2] - scores[1] / sum) < 1e-7)
    }

    /// Exact key ties keep the lower expert index first.
    @Test func selectTopKTieBreaksByLowerIndex() {
        let scores: [Float] = [0.5, 0.5, 0.9, 0.2]
        let bias: [Float] = [0.1, 0.1, -1.0, 0.4]  // keys: 0.6, 0.6, -0.1, 0.6
        let (indices, _) = K3Router.selectTopK(scores: scores, bias: bias,
                                               topK: 3)
        #expect(indices == [0, 1, 3])
    }

    /// End-to-end: GPU scores feed the CPU selection; compared against the
    /// full reference route from the same inputs.
    @Test func gpuScoresDriveCpuSelectionEndToEnd() throws {
        let fixture = Self.makeFixture(seed: 0xD5, numExperts: 32, d: 192)
        let (scores, _) = try Self.gpuScores(fixture: fixture)
        let reference = K3RouterReference.route(
            x: fixture.x, gate: fixture.gate, bias: fixture.bias,
            numExperts: fixture.numExperts, d: fixture.d, topK: 16)
        let (indices, weights) = K3Router.selectTopK(
            scores: scores, bias: fixture.bias, topK: 16)
        #expect(indices == reference.indices.map { UInt32($0) })
        let rel = RelError.compute(actual: weights, reference: reference.weights)
        #expect(rel < Tolerance.identity, "weights: rel=\(rel)")
    }
}
