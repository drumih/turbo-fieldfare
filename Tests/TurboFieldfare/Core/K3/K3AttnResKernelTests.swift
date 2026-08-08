import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Drives `k3_attnres` against `K3AttnResReference` (the direct port of
/// `_apply_attn_res`): fp16 block slab + prefix, fp32 fused score vector,
/// fp32 max-subtracted softmax, output REPLACES the stream.
@Suite struct K3AttnResKernelTests {

    private static func makeF32Buffer(_ device: MTLDevice, _ values: [Float])
        -> MTLBuffer?
    {
        values.withUnsafeBufferPointer {
            device.makeBuffer(bytes: $0.baseAddress!,
                              length: values.count * MemoryLayout<Float>.stride,
                              options: .storageModeShared)
        }
    }

    private struct Fixture {
        var blocks: [[Float16]]
        var prefix: [Float16]
        var scoreVector: [Float]
    }

    private static func makeFixture(seed: UInt64, hidden: Int, numBlocks: Int,
                                    scoreScale: Float = 1) -> Fixture {
        var rng = SeedTree(seed).key("k3-attnres-h\(hidden)-b\(numBlocks)-s\(scoreScale)")
        return Fixture(
            blocks: (0..<numBlocks).map { _ in
                (0..<hidden).map { _ in Float16(rng.uniform(-1.0, 1.0)) } },
            prefix: (0..<hidden).map { _ in Float16(rng.uniform(-1.0, 1.0)) },
            scoreVector: (0..<hidden).map { _ in
                scoreScale * rng.uniform(-1.0, 1.0) })
    }

    private static func runGPU(_ fixture: Fixture, hidden: Int,
                               numBlocks: Int) throws -> [Float] {
        let ctx = try MetalContext()
        let attnRes = try K3AttnRes(context: ctx)
        let flatBlocks = fixture.blocks.flatMap { $0 }
        guard let blocksBuf = Fp16Buffer.make(
                ctx.device,
                halves: flatBlocks.isEmpty ? [Float16](repeating: 0, count: hidden)
                                          : flatBlocks),
              let prefixBuf = Fp16Buffer.make(ctx.device, halves: fixture.prefix),
              let svBuf = makeF32Buffer(ctx.device, fixture.scoreVector),
              let outBuf = Fp16Buffer.make(ctx.device, count: hidden),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers")
            return []
        }
        attnRes.encode(commandBuffer: cmd,
                       blocks: blocksBuf, prefix: prefixBuf,
                       scoreVector: svBuf, out: outBuf,
                       hidden: UInt32(hidden), numBlocks: UInt32(numBlocks),
                       eps: 1e-5)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
        return Fp16Buffer.read(outBuf, count: hidden)
    }

    private static func reference(_ fixture: Fixture) -> [Float] {
        K3AttnResReference.apply(
            blocks: fixture.blocks.map { $0.map { Float($0) } },
            prefix: fixture.prefix.map { Float($0) },
            scoreVector: fixture.scoreVector,
            eps: 1e-5)
    }

    /// Empty block list: softmax over the prefix alone is [1], so the
    /// output IS the prefix — the streaming contract that block boundaries
    /// depend on. Bit-exact through the kernel's fp32 round-trip.
    @Test func emptyBlocksPassPrefixThroughBitExactly() throws {
        let fixture = Self.makeFixture(seed: 0x61, hidden: 128, numBlocks: 0)
        let actual = try Self.runGPU(fixture, hidden: 128, numBlocks: 0)
        let expected = fixture.prefix.map { Float($0) }
        #expect(actual == expected)
    }

    @Test func singleBlockMatchesReference() throws {
        let fixture = Self.makeFixture(seed: 0x62, hidden: 128, numBlocks: 1)
        let actual = try Self.runGPU(fixture, hidden: 128, numBlocks: 1)
        let expected = Self.reference(fixture)
        let rel = RelError.compute(actual: actual, reference: expected)
        let maxAbs = RelError.maxAbsDiff(actual, expected)
        #expect(rel < Tolerance.fp16Reduction,
                "rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func fiveBlocksMatchReference() throws {
        let fixture = Self.makeFixture(seed: 0x63, hidden: 128, numBlocks: 5)
        let actual = try Self.runGPU(fixture, hidden: 128, numBlocks: 5)
        let expected = Self.reference(fixture)
        let rel = RelError.compute(actual: actual, reference: expected)
        let maxAbs = RelError.maxAbsDiff(actual, expected)
        #expect(rel < Tolerance.fp16Reduction,
                "rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// Canonical hidden size with the maximum realistic block count (8
    /// blocks + prefix = 9 vectors — the final AttnRes shape).
    @Test func canonicalHiddenEightBlocks() throws {
        let fixture = Self.makeFixture(seed: 0x64, hidden: 7168, numBlocks: 8)
        let actual = try Self.runGPU(fixture, hidden: 7168, numBlocks: 8)
        let expected = Self.reference(fixture)
        let rel = RelError.compute(actual: actual, reference: expected)
        let maxAbs = RelError.maxAbsDiff(actual, expected)
        #expect(rel < Tolerance.fp16Reduction,
                "rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// Softmax stability: a huge score vector drives raw scores into the
    /// hundreds; both kernel and reference subtract the max before exp.
    @Test func largeScoresStayStable() throws {
        let fixture = Self.makeFixture(seed: 0x65, hidden: 256, numBlocks: 5,
                                       scoreScale: 1000)
        let actual = try Self.runGPU(fixture, hidden: 256, numBlocks: 5)
        #expect(actual.allSatisfy { $0.isFinite }, "output must stay finite")
        let expected = Self.reference(fixture)
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")
    }
}
