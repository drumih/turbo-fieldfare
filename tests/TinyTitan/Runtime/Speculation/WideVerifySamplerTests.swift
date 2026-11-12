import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// `WideVerifySampler` against a simulated scalar decode loop: synthetic
/// logits rows, the real GPU sampler, and the same seeds and growing penalty
/// history scalar decode would use. No model is loaded.
@Suite struct WideVerifySamplerTests {
    static let vocab = 512
    static let rows = 6

    /// Deterministic logits: row r, token t. Each row gets one clear maximum
    /// so greedy ties cannot depend on FP16 softmax rounding.
    static func logitsRows(seed: UInt64) -> [[Float]] {
        var state = seed
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 24)
        }
        return (0..<rows).map { _ in
            var row = (0..<vocab).map { _ in next() * 8 - 4 }
            row[Int(next() * Float(vocab)) % vocab] += 3
            return row
        }
    }

    static func buffer(_ rows: [[Float]], context: MetalContext) throws -> MTLBuffer {
        let flat = rows.flatMap { $0 }.map { Float16($0) }
        let buffer = try #require(
            context.device.makeBuffer(
                length: flat.count * MemoryLayout<Float16>.stride, options: .storageModeShared))
        write(flat, to: buffer)
        return buffer
    }

    static func write(_ values: [Float16], to buffer: MTLBuffer) {
        let target = buffer.contents().bindMemory(to: Float16.self, capacity: values.count)
        for (index, value) in values.enumerated() { target[index] = value }
    }

    /// The history before the pass, ending with the boundary token, which also
    /// warms the sampler's incremental penalty table the way the scalar step
    /// that emitted the boundary token would have.
    static let history: [Int32] = [3, 17, 17, 250, 499, 42]
    static let firstSampleIndex = 5

    static func warmedScratch(context: MetalContext, config: GenerationConfig) throws
        -> RawCompletionScratch
    {
        let scratch = try RawCompletionScratch(context: context, vocab: vocab)
        scratch.sampler.resetPenaltyHistory()
        write((0..<vocab).map { Float16(Float($0 % 7) - 3) }, to: scratch.logits)
        _ = try sampleOnce(
            scratch: scratch, context: context, history: Array(history.dropLast()),
            config: config, position: firstSampleIndex - 1)
        return scratch
    }

    /// Scalar decode over the rows: sample, append, next position.
    static func scalarLoop(
        _ rows: [[Float]], context: MetalContext, config: GenerationConfig
    ) throws -> [Int32] {
        let scratch = try warmedScratch(context: context, config: config)
        var tokens = history
        var out: [Int32] = []
        for (index, row) in rows.enumerated() {
            write(row.map { Float16($0) }, to: scratch.logits)
            let token = try sampleOnce(
                scratch: scratch, context: context, history: tokens,
                config: config, position: firstSampleIndex + index)
            out.append(token)
            tokens.append(token)
        }
        return out
    }

    static func wide(
        _ rows: [[Float]], drafts: [Int32], context: MetalContext, config: GenerationConfig
    ) throws -> DraftAcceptance.Outcome {
        let scratch = try warmedScratch(context: context, config: config)
        let sampler = try WideVerifySampler(scratch: scratch, context: context, config: config)
        return try sampler.resolve(
            rowLogits: try buffer(rows, context: context), drafts: drafts[...],
            history: history, firstSampleIndex: firstSampleIndex)
    }

    static let configurations: [GenerationConfig] = [
        GenerationConfig(temperature: 0, topK: nil, topP: nil),
        GenerationConfig(temperature: 0.7, topK: 20, topP: 0.8, presencePenalty: 1.5, seed: 11),
        GenerationConfig(temperature: 1.0, topK: 5, topP: 0.9, seed: 12),
        GenerationConfig(temperature: 1.2, topK: nil, topP: 0.95, repetitionPenalty: 1.3, seed: 13),
        GenerationConfig(temperature: 0, topK: nil, topP: nil, presencePenalty: 1.5),
    ]

    /// Full acceptance, rejection at the first draft and in the middle, for
    /// every configuration: the emitted tokens are the scalar loop's.
    @Test(arguments: 0..<configurations.count)
    func matchesTheScalarLoop(configuration: Int) throws {
        let context = try MetalContext()
        let config = Self.configurations[configuration]
        let rows = Self.logitsRows(seed: UInt64(configuration) + 1)
        let truth = try Self.scalarLoop(rows, context: context, config: config)
        let vocab = Int32(Self.vocab)

        let all = try Self.wide(
            rows, drafts: Array(truth.prefix(Self.rows - 1)), context: context, config: config)
        #expect(all.acceptedDrafts == Self.rows - 1)
        #expect(all.emitted == truth)

        for rejected in [0, 2] {
            var drafts = Array(truth.prefix(Self.rows - 1))
            drafts[rejected] = (drafts[rejected] + 1) % vocab
            let outcome = try Self.wide(rows, drafts: drafts, context: context, config: config)
            #expect(outcome.acceptedDrafts == rejected)
            #expect(outcome.emitted == Array(truth.prefix(rejected + 1)))
        }
    }

    /// A fixed seed makes the wide path reproducible run to run.
    @Test func seededResolveIsDeterministic() throws {
        let context = try MetalContext()
        let config = Self.configurations[1]
        let rows = Self.logitsRows(seed: 99)
        let drafts: [Int32] = [1, 2, 3, 4, 5]
        let first = try Self.wide(rows, drafts: drafts, context: context, config: config)
        let second = try Self.wide(rows, drafts: drafts, context: context, config: config)
        #expect(first == second)
    }

    /// Pure greedy reduces to the argmax rule `DraftAcceptance` is tested on.
    @Test func greedyIsTheArgmaxRule() throws {
        let context = try MetalContext()
        let config = Self.configurations[0]
        let rows = Self.logitsRows(seed: 7)
        let argmax = rows.map { row in
            Int32(row.indices.max { Float16(row[$0]) < Float16(row[$1]) } ?? 0)
        }
        for drafts in [Array(argmax.prefix(5)), [argmax[0], argmax[1], 0, 0, 0], [9, 9, 9, 9, 9]] {
            let expected = DraftAcceptance.resolve(drafts: drafts[...]) { row, _ in argmax[row] }
            let outcome = try Self.wide(rows, drafts: drafts, context: context, config: config)
            #expect(outcome == expected)
        }
    }

    @Test func refusesAConstrainedRequest() throws {
        let context = try MetalContext()
        let scratch = try RawCompletionScratch(context: context, vocab: Self.vocab)
        var config = Self.configurations[0]
        config.constraint = JSONConstraint(
            table: JSONTokenTable(vocab: 2, entries: [(0, Array("1".utf8))], stopTokens: [1]),
            node: .scalar([.integer]))
        #expect(throws: GeneratorError.self) {
            _ = try WideVerifySampler(scratch: scratch, context: context, config: config)
        }
    }
}
