import Darwin
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct YaRNRoPETests {
    @Test func referenceParametersMatchTransformersYaRN() {
        let twoX = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 524_288)
        let fourX = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 1_048_576)
        #expect(abs(twoX.attentionFactor - 1.0693147) < 1e-6)
        #expect(abs(fourX.attentionFactor - 1.1386294) < 1e-6)
        #expect(abs(twoX.inverseFrequencies[16] - 0.0002766993) < 1e-10)
        #expect(abs(fourX.inverseFrequencies[16] - 0.0002569351) < 1e-10)
        #expect(abs(twoX.inverseFrequencies[31] - 8.2740855e-8) < 1e-12)
        #expect(abs(fourX.inverseFrequencies[31] - 4.1370427e-8) < 1e-12)
        // The 2M rung is the LongRoPE-scale 8x over the native 262144. Its last
        // interpolated frequency is exactly half of the 4x rung's, and the
        // attention factor follows 0.1*ln(factor)+1.
        let eightX = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 2_097_152)
        #expect(eightX.factor == 8)
        #expect(abs(eightX.attentionFactor - 1.2079442) < 1e-6)
        #expect(abs(eightX.inverseFrequencies[31] - fourX.inverseFrequencies[31] / 2) < 1e-12)
    }

    @Test(arguments: [524_287, 1_048_575, 2_097_151])
    func scalarKernelMatchesCPUAtExtendedPositions(_ position: Int) throws {
        let target =
            position < 1_000_000
            ? 524_288 : (position < 2_000_000 ? 1_048_576 : 2_097_152)
        let parameters = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: target)
        let context = try MetalContext()
        let rope = try RoPE(context: context, yarn: parameters)
        let input = (0..<256).map { Float16(Float($0 % 19) / 19 - 0.5) }
        guard let buffer = Fp16Buffer.make(context.device, halves: input),
            let commandBuffer = context.queue.makeCommandBuffer()
        else {
            Issue.record("buffer allocation failed")
            return
        }
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: buffer,
            position: UInt32(position), headDim: 256,
            numHeads: 1, rotaryDim: 64,
            theta: 10_000_000)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        var expected = input
        for pair in 0..<32 {
            let angle = Float(position) * parameters.inverseFrequencies[pair]
            let cosine = cos(angle) * parameters.attentionFactor
            let sine = sin(angle) * parameters.attentionFactor
            let x0 = Float(input[pair])
            let x1 = Float(input[32 + pair])
            expected[pair] = Float16(x0 * cosine - x1 * sine)
            expected[32 + pair] = Float16(x0 * sine + x1 * cosine)
        }
        let actual = Fp16Buffer.read(buffer, count: input.count).map(Float16.init)

        // Agreement, not identity.
        //
        // This compares a Metal kernel against a CPU reference, and the two do
        // not round alike: both evaluate cos/sin at a large angle — position
        // 524287 against this pair's inverse frequency is about 145 radians —
        // and Metal's transcendentals are the fast variants while the reference
        // goes through libm, so the argument reduction differs. On an M3 the two
        // agree bit for bit; on the GitHub `macos-26` runner they differ by up
        // to 0.017 of full scale at 524287 and 1048575, and by 0.0325 at
        // 2,097,151 (index 32, the highest-frequency pair). The drift grows with
        // the phase, which is why the bound is position-dependent. That
        // measurement is why this asserts a tolerance instead of `==`:
        // bit-exactness against a CPU reference is not a property of the kernel,
        // and the repository already treats GPU-family numerics as
        // machine-specific — see the SCOPE note at the top of
        // `tools/golden-baseline.sh`.
        //
        // The bound still fails a wrong rotation: a swapped pair, a sign error
        // or a wrong partial-rotary split moves a component by O(full scale),
        // not by a few percent of it. `chunkedPrefillMatchesScalarYaRN` below
        // compares two Metal paths, so it stays exact.
        var worst = Float(0)
        var worstIndex = 0
        for index in actual.indices {
            let deviation = abs(Float(actual[index]) - Float(expected[index]))
            if deviation > worst {
                worst = deviation
                worstIndex = index
            }
        }
        // 0.05 leaves ~50% headroom over the measured 0.0325 at 2M and is still
        // far below the O(1) error a wrong rotation produces.
        let tolerance: Float = position >= 2_000_000 ? 0.05 : 0.02
        #expect(
            worst <= tolerance,
            "scalar RoPE differs from the CPU reference by \(worst) at index \(worstIndex) at position \(position)"
        )
    }

    @Test func chunkedPrefillMatchesScalarYaRN() throws {
        let parameters = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 1_048_576)
        let context = try MetalContext()
        let scalar = try RoPE(context: context, yarn: parameters)
        let prefill = try PrefillRoPE(context: context, yarn: parameters)
        let tokens = 3
        let heads = 2
        let headDim = 256
        let input = (0..<(tokens * heads * headDim)).map {
            Float16(Float($0 % 23) / 23 - 0.5)
        }
        guard let scalarBuffer = Fp16Buffer.make(context.device, halves: input),
            let prefillBuffer = Fp16Buffer.make(context.device, halves: input),
            let commandBuffer = context.queue.makeCommandBuffer()
        else {
            Issue.record("buffer allocation failed")
            return
        }
        try scalar.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: scalarBuffer,
            position: 524_288, headDim: UInt32(headDim),
            numHeads: UInt32(heads), rotaryDim: 64,
            numTokens: UInt32(tokens), theta: 10_000_000)
        try prefill.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: prefillBuffer,
            startPosition: 524_288, queryCount: UInt32(tokens),
            headDim: UInt32(headDim), numHeads: UInt32(heads), rotaryDim: 64,
            tokenStrideElements: UInt32(heads * headDim), theta: 10_000_000)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(
            Fp16Buffer.read(scalarBuffer, count: input.count)
                == Fp16Buffer.read(prefillBuffer, count: input.count))
    }
}
