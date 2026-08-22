import Foundation
import Metal
import Testing
@testable import TurboFieldfare

private let visionMPPTensorOpsAvailable = MTLCreateSystemDefaultDevice()?
    .supportsFamily(.apple8) == true

@Suite struct VisionLinearBF16Tests {
    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision linear kernels require Apple8 or newer"))
    func nativePathsProduceExpectedConstantProduct() throws {
        let context = try MetalContext()
        let m = 128, n = 256, k = 1_152
        func buffer(count: Int, value: Float) throws -> MTLBuffer {
            let bits = [UInt16](repeating: Quantization.bf16Bits(value), count: count)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        let input = try buffer(count: m * k, value: 0.125)
        let weights = try buffer(count: n * k, value: 0.25)

        for environment in [
            [String: String](),
            ["TURBO_FIELDFARE_VISION_REGISTER_GEMM": "1"],
        ] {
            // A fresh output per variant, poisoned rather than zeroed: sharing
            // one buffer meant the second variant was compared against what the
            // first had written, so a kernel that dispatched nothing passed.
            let output = try buffer(count: m * n, value: -1)
            let linear = try VisionLinearBF16(context: context, environment: environment)
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            linear.encode(commandBuffer: commandBuffer,
                          input: input, weights: weights, output: output,
                          m: m, n: n, k: k)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            let values = output.contents().bindMemory(to: UInt16.self, capacity: m * n)
            var maximumError: Float = 0
            for index in 0..<(m * n) {
                maximumError = max(
                    maximumError,
                    abs(Quantization.bf16ToFloat(values[index]) - 36))
            }
            #expect(maximumError == 0, "environment: \(environment)")
        }
    }

    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision linear kernels require Apple8 or newer"))
    func registerPathHandlesNonconstantTailShape() throws {
        let context = try MetalContext()
        let m = 79, n = 272, k = 23
        let inputBits = (0..<(m * k)).map { index in
            Quantization.bf16Bits(Float((index * 17) % 31 - 15) / 32)
        }
        let weightBits = (0..<(n * k)).map { index in
            Quantization.bf16Bits(Float((index * 13) % 29 - 14) / 64)
        }
        let inputValues = inputBits.map(Quantization.bf16ToFloat)
        let weightValues = weightBits.map(Quantization.bf16ToFloat)
        let input = try #require(context.device.makeBuffer(
            bytes: inputBits,
            length: inputBits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let weights = try #require(context.device.makeBuffer(
            bytes: weightBits,
            length: weightBits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let output = try #require(context.device.makeBuffer(
            length: m * n * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let linear = try VisionLinearBF16(
            context: context,
            environment: ["TURBO_FIELDFARE_VISION_REGISTER_GEMM": "1"])
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        linear.encode(commandBuffer: commandBuffer,
                      input: input, weights: weights, output: output,
                      m: m, n: n, k: k)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actual = output.contents().bindMemory(to: UInt16.self, capacity: m * n)
        var maximumWideError: Float = 0
        var maximumTailError: Float = 0
        for row in 0..<m {
            for column in 0..<n {
                var expected: Float = 0
                for inner in 0..<k {
                    expected += inputValues[row * k + inner] * weightValues[column * k + inner]
                }
                let error = abs(
                    Quantization.bf16ToFloat(actual[row * n + column]) - expected)
                if column < 256 {
                    maximumWideError = max(maximumWideError, error)
                } else {
                    maximumTailError = max(maximumTailError, error)
                }
            }
        }
        #expect(maximumWideError < 0.016)
        #expect(maximumTailError < 0.016)
    }

    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision linear kernels require Apple8 or newer"))
    func registerMatchesMPPReductionAtProductionK() throws {
        let context = try MetalContext()
        let m = 64, n = 32, k = 1_152
        let inputBits = (0..<(m * k)).map {
            Quantization.bf16Bits(Float(($0 * 17) % 127 - 63) / 64)
        }
        let weightBits = (0..<(n * k)).map {
            Quantization.bf16Bits(Float(($0 * 29) % 113 - 56) / 128)
        }
        let input = try #require(context.device.makeBuffer(
            bytes: inputBits, length: inputBits.count * 2, options: .storageModeShared))
        let weights = try #require(context.device.makeBuffer(
            bytes: weightBits, length: weightBits.count * 2, options: .storageModeShared))
        let baseline = try #require(context.device.makeBuffer(
            length: m * n * 2, options: .storageModeShared))
        let register = try #require(context.device.makeBuffer(
            length: m * n * 2, options: .storageModeShared))

        for (environment, output) in [
            ([String: String](), baseline),
            (["TURBO_FIELDFARE_VISION_REGISTER_GEMM": "1"], register),
        ] {
            let linear = try VisionLinearBF16(context: context, environment: environment)
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            linear.encode(commandBuffer: commandBuffer,
                          input: input, weights: weights, output: output,
                          m: m, n: n, k: k)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
        }

        #expect(Data(bytes: baseline.contents(), count: m * n * 2)
            == Data(bytes: register.contents(), count: m * n * 2))
    }
}
