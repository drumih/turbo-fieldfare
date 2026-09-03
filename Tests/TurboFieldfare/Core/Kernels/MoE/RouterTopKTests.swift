import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct RouterTopKTests {
    private static let experts = 16
    private static let dimension = 128
    private static let topK = 8

    private struct Result {
        let indices: [UInt32]
        let weights: [Float]
    }

    private struct KernelResult {
        let indices: [UInt32]
        let weights: [Float]
        /// The FP16 bit patterns the kernel wrote, not the decoded floats: the
        /// pre-generalisation fixture is an identity claim, and a comparison
        /// through `Float` would hide a one-ulp change.
        let weightBits: [UInt16]
    }

    @Test func productionRouterMatchesReference() throws {
        var rng = SplitMix64(seed: 0xA5B6_1234)
        let weights = (0..<Self.experts).map { expert in
            (0..<Self.dimension).map { _ in
                rng.uniform(-0.05, 0.05) + Float(expert) * 0.01
            }
        }
        let hidden = (0..<Self.dimension).map { _ in rng.uniform(-1.0, 1.0) }
        let invSqrtD = 1.0 / Float(Self.dimension).squareRoot()
        let effectiveScale = (0..<Self.dimension).map { _ in
            rng.uniform(0.5, 1.5) * invSqrtD
        }
        let expertScale = (0..<Self.experts).map { _ in rng.uniform(0.6, 1.4) }

        let expected = Self.reference(weights: weights,
                                      hidden: hidden,
                                      effectiveScale: effectiveScale,
                                      expertScale: expertScale)
        let actual = try Self.run(weights: weights,
                                  hidden: hidden,
                                  effectiveScale: effectiveScale,
                                  expertScale: expertScale)
        #expect(actual.indices == expected.indices)
        let maxError = zip(actual.weights, expected.weights)
            .map { abs($0 - $1) }
            .max() ?? 0
        #expect(maxError < 5e-3)
    }

    @Test func productionRouterResolvesNearTieLikeReference() throws {
        var rng = SplitMix64(seed: 0x71E_0F4A)
        let pattern = (0..<Self.dimension).map { _ in rng.uniform(0.2, 1.0) }
        let hidden = (0..<Self.dimension).map { _ in rng.uniform(0.5, 1.5) }
        var gains = [Float](repeating: 0, count: Self.experts)
        for expert in 0..<7 { gains[expert] = 1.0 - Float(expert) * 0.05 }
        gains[7] = 0.5
        gains[8] = 0.5 * (1.0 + 1e-4)
        for expert in 9..<Self.experts {
            gains[expert] = 0.4 - Float(expert - 9) * 0.02
        }
        let weights = gains.map { gain in pattern.map { $0 * gain } }
        let effectiveScale = [Float](repeating: 1.0, count: Self.dimension)
        let expertScale = [Float](repeating: 1.0, count: Self.experts)

        let expected = Self.reference(weights: weights,
                                      hidden: hidden,
                                      effectiveScale: effectiveScale,
                                      expertScale: expertScale)
        let actual = try Self.run(weights: weights,
                                  hidden: hidden,
                                  effectiveScale: effectiveScale,
                                  expertScale: expertScale)
        #expect(actual.indices == expected.indices)
    }

    // MARK: - Routed width

    private static let widthExperts = 128

    private struct WidthFixture {
        let weights: [[Float]]
        let hidden: [Float]
        let effectiveScale: [Float]
        let expertScale: [Float]
    }

    /// A production-width router input, 128 experts, whose logits fall
    /// monotonically with the expert index and whose rows 3 and 4 are the same
    /// weights. The duplicate puts an exact tie astride the K=4 boundary:
    /// expert 3 is the last selected and expert 4 the first rejected, so at the
    /// narrowest width the kernel's `e < top_idx[i]` tie-break decides
    /// membership rather than only order. The per-expert gains are exactly
    /// representable in BF16, which is how the kernel reads them, so dividing
    /// the output weights by them recovers the softmax.
    private static func widthFixture() -> WidthFixture {
        var rng = SplitMix64(seed: 0x0167_0903)
        let pattern = (0..<dimension).map { _ in rng.uniform(0.2, 1.0) }
        var gains = (0..<widthExperts).map { 1.0 - Float($0) * 0.005 }
        gains[4] = gains[3]
        let hidden = (0..<dimension).map { _ in rng.uniform(0.5, 1.5) }
        return WidthFixture(
            weights: gains.map { gain in pattern.map { $0 * gain } },
            hidden: hidden,
            effectiveScale: [Float](repeating: 1.0 / Float(dimension).squareRoot(),
                                    count: dimension),
            expertScale: (0..<widthExperts).map { 0.5 + Float($0 % 5) * 0.25 })
    }

    /// Recorded from `router_topk_select_k8` as it stands in private `main` at
    /// `680aa478`, where `public/.../Metal/MoE/moe.metal` is still the
    /// pre-generalisation file; check out that commit's `public/` tree to re-run
    /// the capture. The generalised kernel must reproduce these bits exactly at
    /// the checkpoint's width: the arithmetic is the same instruction sequence,
    /// and this is the gate that says so. Regenerate only alongside a deliberate
    /// change to the router's math.
    private static let widthFixtureIndicesAtEight: [UInt32] =
        [0, 1, 2, 3, 4, 5, 6, 7]
    private static let widthFixtureWeightBitsAtEight: [UInt16] =
        [0x2C7E, 0x2E82, 0x3030, 0x310D, 0x320F, 0x2B87, 0x2D73, 0x2F03]

    @Test func routerAtTheCheckpointWidthReproducesThePreGeneralisationFixture() throws {
        let fixture = Self.widthFixture()
        let actual = try Self.run(weights: fixture.weights,
                                  hidden: fixture.hidden,
                                  effectiveScale: fixture.effectiveScale,
                                  expertScale: fixture.expertScale,
                                  experts: Self.widthExperts,
                                  topK: 8)
        #expect(actual.indices == Self.widthFixtureIndicesAtEight)
        #expect(actual.weightBits == Self.widthFixtureWeightBitsAtEight)
    }

    /// Renormalisation is over the experts actually used, not over eight with
    /// the tail dropped: consuming the first K of an eight-wide softmax would
    /// scale the routed branch down by the mass the dropped experts carried.
    /// The kernel selects K and softmaxes over those K, so the weights divided
    /// by their per-expert gains sum to one at every width.
    @Test(arguments: [4, 6, 8])
    func routerRenormalisesOverTheExpertsItActuallyUses(_ topK: Int) throws {
        let fixture = Self.widthFixture()
        let expected = Self.reference(weights: fixture.weights,
                                      hidden: fixture.hidden,
                                      effectiveScale: fixture.effectiveScale,
                                      expertScale: fixture.expertScale,
                                      experts: Self.widthExperts,
                                      topK: topK)
        let actual = try Self.run(weights: fixture.weights,
                                  hidden: fixture.hidden,
                                  effectiveScale: fixture.effectiveScale,
                                  expertScale: fixture.expertScale,
                                  experts: Self.widthExperts,
                                  topK: topK)

        #expect(actual.indices == expected.indices)
        // A narrower width keeps the same experts in the same order as the
        // checkpoint's width, which is what makes the tie-break observable.
        #expect(actual.indices == Array(Self.widthFixtureIndicesAtEight.prefix(topK)))

        let softmax = zip(actual.indices, actual.weights).map { index, weight in
            weight / fixture.expertScale[Int(index)]
        }
        // The bound is derived, not chosen. Each stored weight is
        // `round_fp16(s_i * g_i)`, and FP16 round-to-nearest has unit roundoff
        // 2^-11, so the stored value is within `2^-11 * s_i * g_i` of the exact
        // product. The gains are exactly representable in BF16 and in FP32, so
        // dividing by one leaves each term within `2^-11 * s_i`; the terms sum
        // to one, so their errors sum to at most 2^-11. The FP32 division and
        // the accumulation below add at most a few 2^-24, which 2^-20 covers.
        let sumTolerance: Float = 0x1p-11 + 0x1p-20
        let sum = softmax.reduce(0, +)
        #expect(abs(sum - 1.0) < sumTolerance,
                "routed weights at K=\(topK) sum to \(sum)")
        // Against the CPU reference the allowance is the one
        // `productionRouterMatchesReference` already uses: the kernel takes
        // `fast::exp` where the reference takes `exp`, and that difference is
        // far larger than the FP16 storage error bounded above.
        let maxError = zip(actual.weights, expected.weights)
            .map { abs($0 - $1) }
            .max() ?? 0
        #expect(maxError < 5e-3)
    }

    private static func reference(weights: [[Float]],
                                  hidden: [Float],
                                  effectiveScale: [Float],
                                  expertScale: [Float],
                                  experts: Int = RouterTopKTests.experts,
                                  topK: Int = RouterTopKTests.topK) -> Result {
        let scaled = zip(hidden, effectiveScale).map { $0 * $1 }
        let rows = weights.map { Quantization.quantizeInt8Affine($0) }
        let logits = DequantInt8GemvRef.apply(weightRows: rows,
                                              x: scaled,
                                              n: Self.dimension)
        var paired: [(Float, UInt32)] = []
        paired.reserveCapacity(experts)
        for expert in 0..<experts {
            paired.append((logits[expert], UInt32(expert)))
        }
        paired.sort { lhs, rhs in
            lhs.0 == rhs.0 ? lhs.1 < rhs.1 : lhs.0 > rhs.0
        }
        let selected = Array(paired.prefix(topK))
        let maximum = selected.first?.0 ?? 0
        let exponents = selected.map { exp($0.0 - maximum) }
        let sum = exponents.reduce(0, +)
        let outputWeights = zip(selected, exponents).map { item, value in
            value / sum * expertScale[Int(item.1)]
        }
        return Result(indices: selected.map { $0.1 }, weights: outputWeights)
    }

    private static func run(weights: [[Float]],
                            hidden: [Float],
                            effectiveScale: [Float],
                            expertScale: [Float],
                            experts: Int = RouterTopKTests.experts,
                            topK: Int = RouterTopKTests.topK) throws -> KernelResult {
        let packedRows = weights.map { Quantization.quantizeInt8Affine($0) }
        let groupsPerRow = Self.dimension / Quantization.groupSize
        let packed = packedRows.flatMap(\.packed)
        let scales = packedRows.flatMap(\.scales)
        let biases = packedRows.flatMap(\.biases)
        precondition(scales.count == experts * groupsPerRow)

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        guard let weightBuffer = context.device.makeBuffer(
                  bytes: packed, length: packed.count, options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: scales,
                  length: scales.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                  bytes: biases,
                  length: biases.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let hiddenBuffer = Fp16Buffer.make(context.device, values: hidden),
              let effectiveBuffer = context.device.makeBuffer(
                  bytes: effectiveScale.map(Quantization.bf16Bits),
                  length: effectiveScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let expertScaleBuffer = context.device.makeBuffer(
                  bytes: expertScale.map(Quantization.bf16Bits),
                  length: expertScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let indexBuffer = context.device.makeBuffer(
                  length: topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let outputWeightBuffer = Fp16Buffer.make(context.device, count: topK),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        kernel.encodeRouterGemma4(
            commandBuffer: commandBuffer,
            weights: weightBuffer,
            scales: scaleBuffer,
            biases: biasBuffer,
            hidden: hiddenBuffer,
            effectiveScale: effectiveBuffer,
            perExpertScale: expertScaleBuffer,
            outIndices: indexBuffer,
            outWeights: outputWeightBuffer,
            numExperts: UInt32(experts),
            d: UInt32(Self.dimension),
            topK: UInt32(topK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: topK)
        let bitPointer = outputWeightBuffer.contents().bindMemory(
            to: UInt16.self, capacity: topK)
        return KernelResult(
            indices: (0..<topK).map { indexPointer[$0] },
            weights: Fp16Buffer.read(outputWeightBuffer, count: topK),
            weightBits: (0..<topK).map { bitPointer[$0] })
    }
}
