import Foundation
import Metal
import Testing

@testable import DenseGPU
@testable import TinyTitanCore

/// The dense path's threshold must be invisible below `n̂` and must match the
/// extended table above it. These are the kernel-level pins for TT-111; the
/// end-to-end cap invariance is checked against the stored dense goldens.
@Suite struct DenseGPURoPETests {
    private func parameters(threshold: Int) -> YaRNRoPEParameters {
        YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 1_048_576, threshold: threshold)
    }

    private func makeBuffer(_ device: MTLDevice, _ values: [Float16]) throws -> MTLBuffer {
        let buffer = try #require(
            device.makeBuffer(
                length: values.count * MemoryLayout<Float16>.size,
                options: .storageModeShared))
        values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            buffer.contents().copyMemory(from: baseAddress, byteCount: bytes.count)
        }
        return buffer
    }

    private func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Float16] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float16.self, capacity: count),
                count: count))
    }

    @Test func belowThresholdIsTheNativeRotationBitForBit() throws {
        let context = try MetalContext()
        let thresholdRope = try RoPE(context: context, yarn: parameters(threshold: 4_096))
        let nativeRope = try RoPE(context: context)
        let input = (0..<256).map { Float16(Float($0 % 19) / 19 - 0.5) }
        let thresholdBuffer = try makeBuffer(context.device, input)
        let nativeBuffer = try makeBuffer(context.device, input)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try thresholdRope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: thresholdBuffer,
            position: 128, headDim: 256, numHeads: 1, rotaryDim: 64,
            theta: 10_000_000)
        try nativeRope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: nativeBuffer,
            position: 128, headDim: 256, numHeads: 1, rotaryDim: 64,
            theta: 10_000_000)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(
            readBuffer(thresholdBuffer, count: input.count)
                == readBuffer(nativeBuffer, count: input.count),
            "a position below the threshold must take the native branch exactly")
    }

    @Test func aboveThresholdUsesTheExtendedTable() throws {
        let parameters = parameters(threshold: 4_096)
        let context = try MetalContext()
        let rope = try RoPE(context: context, yarn: parameters)
        let input = (0..<256).map { Float16(Float($0 % 19) / 19 - 0.5) }
        let buffer = try makeBuffer(context.device, input)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: buffer,
            position: 5_000, headDim: 256, numHeads: 1, rotaryDim: 64,
            theta: 10_000_000)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        var expected = input
        for pair in 0..<32 {
            let angle = Float(5_000) * parameters.inverseFrequencies[pair]
            let cosine = cos(angle) * parameters.attentionFactor
            let sine = sin(angle) * parameters.attentionFactor
            let x0 = Float(input[pair])
            let x1 = Float(input[32 + pair])
            expected[pair] = Float16(x0 * cosine - x1 * sine)
            expected[32 + pair] = Float16(x0 * sine + x1 * cosine)
        }
        let actual = readBuffer(buffer, count: input.count)
        var worst = Float(0)
        for index in actual.indices {
            worst = max(worst, abs(Float(actual[index]) - Float(expected[index])))
        }
        // Same Metal-vs-libm transcendental allowance as the shared YaRN test.
        #expect(worst <= 0.02, "extended branch differs from the CPU reference by \(worst)")
    }

    @Test func prefillAndScalarAgreeOnBothSidesOfTheThreshold() throws {
        let parameters = parameters(threshold: 4_096)
        let context = try MetalContext()
        let scalar = try RoPE(context: context, yarn: parameters)
        let prefill = try PrefillRoPE(context: context, yarn: parameters)
        let tokens = 3
        let heads = 2
        let headDim = 256
        let input = (0..<(tokens * heads * headDim)).map {
            Float16(Float($0 % 23) / 23 - 0.5)
        }
        for start in [128, 5_000] {
            let scalarBuffer = try makeBuffer(context.device, input)
            let prefillBuffer = try makeBuffer(context.device, input)
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            try scalar.encodeNeoxSubdim(
                commandBuffer: commandBuffer, data: scalarBuffer,
                position: UInt32(start), headDim: 256, numHeads: UInt32(heads),
                rotaryDim: 64, numTokens: UInt32(tokens), theta: 10_000_000)
            try prefill.encodeNeoxSubdim(
                commandBuffer: commandBuffer, data: prefillBuffer,
                startPosition: UInt32(start), queryCount: UInt32(tokens),
                headDim: 256, numHeads: UInt32(heads), rotaryDim: 64,
                tokenStrideElements: UInt32(heads * headDim), theta: 10_000_000)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            #expect(
                readBuffer(scalarBuffer, count: input.count)
                    == readBuffer(prefillBuffer, count: input.count),
                "scalar and prefill disagree at start \(start)")
        }
    }

    @Test func theCapDoesNotHaveToEqualTheExtensionProfile() throws {
        let runtime = try RuntimeConfiguration(
            forceLogitsHead: true,
            ropeScalingMode: .yarn,
            yarnContextTokens: 1_048_576)
        try runtime.validate(maxContext: 1_048_576)
        try runtime.validate(maxContext: 4_194_304)
        try runtime.validate(maxContext: 16_777_216)
        #expect(runtime.yarnContextTokens == 1_048_576)
        #expect(runtime.ropeScalingMode == .yarn)
        let native = try RuntimeConfiguration(forceLogitsHead: true)
        #expect(throws: RuntimeConfigurationError.self) {
            try native.validate(maxContext: 524_288)
        }
    }

    @Test func theDefaultThresholdIsTheNativeWindow() {
        let parameters = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 1_048_576)
        #expect(parameters.threshold == RuntimeConfiguration.nativeMaximumContextTokens)
    }
}
