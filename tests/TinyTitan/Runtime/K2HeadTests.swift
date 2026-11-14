import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
import TinyTitanCore

@testable import TinyTitan

/// K2's input and output ends on the synthetic install: the embedding lookup,
/// the grouped final norm and the untied head GEMV. The toy weights are the
/// known 0.01 constant, so the reference is the CPU formula.
@Suite struct K2HeadTests {
    private static let eps: Float = 1e-6

    private static func groupNorm(_ x: [Float], groups: Int) -> [Float] {
        let groupDim = x.count / groups
        var out = [Float](repeating: 0, count: x.count)
        for group in 0..<groups {
            let range = (group * groupDim)..<((group + 1) * groupDim)
            let mean = range.reduce(Float(0)) { $0 + x[$1] * x[$1] } / Float(groupDim)
            let inv = 1 / (mean + eps).squareRoot()
            for i in range { out[i] = Float(Float16(x[i] * inv)) }
        }
        return out
    }

    @Test func embeddingAndHeadMatchTheReference() throws {
        let directory = try K2ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let head = try K2Head(model: model, context: ctx)

        let token: UInt32 = 7
        let embedBuf = try #require(Fp16Buffer.make(ctx.device, count: config.hiddenSize))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try head.encodeEmbedding(commandBuffer: cb, tokenID: token, out: embedBuf)
        cb.commit()
        cb.waitUntilCompleted()
        let embedded = Fp16Buffer.read(embedBuf, count: config.hiddenSize)
        // Toy table: packed code 1, scale 0.01, bias 0.
        let expectedEmbed = [Float](repeating: 0.01, count: config.hiddenSize)
        #expect(
            RelError.compute(actual: embedded, reference: expectedEmbed) < Tolerance.fp16Reduction)

        var rng = SeedTree(0x4EAD_9001).key("k2-head")
        let hidden = (0..<config.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let hiddenBuf = try #require(Fp16Buffer.make(ctx.device, halves: hidden))
        let cb2 = try #require(ctx.queue.makeCommandBuffer())
        try head.encodeLogits(commandBuffer: cb2, hidden: hiddenBuf)
        cb2.commit()
        cb2.waitUntilCompleted()

        let normed = Self.groupNorm(hidden.map(Float.init), groups: config.normGroups)
        let dot = normed.reduce(Float(0), +) * 0.01
        let expected = [Float](repeating: Float(Float16(dot)), count: config.vocabSize)
        let actual = head.readLogits()
        #expect(RelError.compute(actual: actual, reference: expected) < Tolerance.fp16Reduction)
    }
}
