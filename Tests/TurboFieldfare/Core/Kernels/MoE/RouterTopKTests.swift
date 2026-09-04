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

    private static func reference(weights: [[Float]],
                                  hidden: [Float],
                                  effectiveScale: [Float],
                                  expertScale: [Float]) -> Result {
        let scaled = zip(hidden, effectiveScale).map { $0 * $1 }
        let rows = weights.map { Quantization.quantizeInt8Affine($0) }
        let logits = DequantInt8GemvRef.apply(weightRows: rows,
                                              x: scaled,
                                              n: Self.dimension)
        var paired: [(Float, UInt32)] = []
        paired.reserveCapacity(Self.experts)
        for expert in 0..<Self.experts {
            paired.append((logits[expert], UInt32(expert)))
        }
        paired.sort { lhs, rhs in
            lhs.0 == rhs.0 ? lhs.1 < rhs.1 : lhs.0 > rhs.0
        }
        let selected = Array(paired.prefix(Self.topK))
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
                            expertScale: [Float]) throws -> Result {
        let packedRows = weights.map { Quantization.quantizeInt8Affine($0) }
        let groupsPerRow = Self.dimension / Quantization.groupSize
        let packed = packedRows.flatMap(\.packed)
        let scales = packedRows.flatMap(\.scales)
        let biases = packedRows.flatMap(\.biases)
        precondition(scales.count == Self.experts * groupsPerRow)

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
                  length: Self.topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let outputWeightBuffer = Fp16Buffer.make(context.device, count: Self.topK),
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
            numExperts: UInt32(Self.experts),
            d: UInt32(Self.dimension),
            topK: UInt32(Self.topK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: Self.topK)
        return Result(
            indices: (0..<Self.topK).map { indexPointer[$0] },
            weights: Fp16Buffer.read(outputWeightBuffer, count: Self.topK))
    }
}
