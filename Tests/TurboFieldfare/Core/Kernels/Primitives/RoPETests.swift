import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct RoPETests {
    @Test func defaultNeoxMatchesSWAReference() throws {
        let tokens = 1
        let heads = 16
        let headDim = 256
        let position = 7
        let theta: Float = 10_000
        let count = tokens * heads * headDim
        let input = Self.randomInputs(count: count, seed: 0xF1, label: "rope-neox-default")
        let context = try MetalContext()
        let kernel = try RoPE(context: context)
        let buffer = try #require(Fp16Buffer.make(context.device, values: input))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeDefaultNeox(
            commandBuffer: commandBuffer,
            data: buffer,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(heads),
            numTokens: UInt32(tokens),
            theta: theta)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actual = Fp16Buffer.read(buffer, count: count)
        let reference = RopeRef.applyNeox(
            input: input,
            numTokens: tokens,
            numHeads: heads,
            headDim: headDim,
            rotatedPairs: headDim / 2,
            position: position,
            theta: theta)
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.fp16Reduction)
    }

    @Test func proportionalNeoxMatchesFullAttentionReference() throws {
        let tokens = 1
        let heads = 16
        let headDim = 512
        let rotatedPairs = 64
        let position = 11
        let theta: Float = 1_000_000
        let count = tokens * heads * headDim
        let input = Self.randomInputs(count: count, seed: 0xF3, label: "rope-neox-full")
        let context = try MetalContext()
        let kernel = try RoPE(context: context)
        let buffer = try #require(Fp16Buffer.make(context.device, values: input))
        let before = Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: UInt8.self),
            count: count * MemoryLayout<Float16>.size))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeProportionalNeox(
            commandBuffer: commandBuffer,
            data: buffer,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(heads),
            rotatedPairs: UInt32(rotatedPairs),
            theta: theta)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actual = Fp16Buffer.read(buffer, count: count)
        let reference = RopeRef.applyNeox(
            input: input,
            numTokens: tokens,
            numHeads: heads,
            headDim: headDim,
            rotatedPairs: rotatedPairs,
            position: position,
            theta: theta)
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.fp16Reduction)

        let halfDim = headDim / 2
        let stride = MemoryLayout<Float16>.size
        let after = UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: UInt8.self),
            count: count * stride)
        for head in 0..<heads {
            let base = head * headDim * stride
            for index in rotatedPairs..<halfDim {
                let offset = base + index * stride
                #expect(after[offset] == before[offset])
                #expect(after[offset + 1] == before[offset + 1])
            }
            for index in (halfDim + rotatedPairs)..<headDim {
                let offset = base + index * stride
                #expect(after[offset] == before[offset])
                #expect(after[offset + 1] == before[offset + 1])
            }
        }
    }

    private static func randomInputs(count: Int,
                                     seed: UInt64,
                                     label: String) -> [Float] {
        var random = SeedTree(seed).key(label)
        return (0..<count).map { _ in random.uniform(-1, 1) }
    }
}
