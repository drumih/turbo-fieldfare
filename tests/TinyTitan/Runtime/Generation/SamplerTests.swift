import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
import TinyTitanCore

@testable import TinyTitan

/// `Sampler` exercises: greedy=argmax, seeded determinism + per-position
/// reproducibility, top-k / top-p truncation, repetition penalty, temperature
/// spread. Inputs are raw pre-softcap logits
/// (the sampler runs the softcap+softmax front-end itself).
@Suite struct SamplerTests {

    /// Reusable rig — one MetalContext + Sampler + buffers, shared across the
    /// many draws a single test makes (avoids recompiling the shader library
    /// per draw).
    private final class Rig {
        let ctx: MetalContext
        let sampler: Sampler
        let vocab: Int
        let logits: MTLBuffer
        let probs: MTLBuffer
        let outToken: MTLBuffer

        init(vocab: Int) throws {
            self.ctx = try MetalContext()
            self.sampler = try Sampler(context: ctx, vocab: vocab)
            self.vocab = vocab
            guard
                let l = ctx.device.makeBuffer(
                    length: vocab * MemoryLayout<Float16>.size,
                    options: .storageModeShared),
                let p = ctx.device.makeBuffer(
                    length: vocab * MemoryLayout<Float16>.size,
                    options: .storageModeShared),
                let o = ctx.device.makeBuffer(
                    length: MemoryLayout<UInt32>.size,
                    options: .storageModeShared)
            else {
                throw MetalError.noDevice
            }
            self.logits = l
            self.probs = p
            self.outToken = o
        }

        func writeLogits(_ values: [Float]) {
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
            for i in 0..<vocab { ptr[i] = Float16(values[i]) }
        }

        @discardableResult
        func draw(
            _ values: [Float], config: GenerationConfig,
            position: Int = 0, history: [Int32] = []
        ) throws -> (id: UInt32, path: SamplePath) {
            writeLogits(values)
            let cmd = try #require(ctx.queue.makeCommandBuffer())
            let path = try sampler.sample(
                commandBuffer: cmd, logits: logits, probs: probs,
                history: history, config: config,
                position: position, outToken: outToken)
            cmd.commit()
            cmd.waitUntilCompleted()
            return (outToken.contents().load(as: UInt32.self), path)
        }
    }

    /// Pins the invariant the generation loop depends on: whatever the logits,
    /// the sampler returns an index inside the vocabulary. The result is used
    /// to index the vocabulary and to extend the KV history, so an out-of-range
    /// value propagates a garbage token rather than failing cleanly.
    ///
    /// Note on coverage: this does **not** exercise the `kept == 0` branch, and
    /// the reason once given here was wrong. A row with no finite mass does
    /// reach the reduction: `softcap_value` now folds each NaN to -inf so one
    /// bad logit cannot take the row with it (C47), but a row that is
    /// non-finite *throughout* has no finite value left, and the kernels fall
    /// back to token 0 by construction. This test guards the in-range
    /// invariant, not that path. The generation loop reports the empty row
    /// (`GeneratorError.degenerateLogitsRow`) before that fallback could be fed
    /// back as a token, and the kernel keeps the fallback anyway, so no path can
    /// hand an out-of-range id to the vocabulary. The intermittent out-of-range
    /// id seen under full-suite GPU load is still unidentified.
    @Test func degenerateDistributionStillReturnsAnInRangeToken() throws {
        let v = 64
        let rig = try Rig(vocab: v)
        for fill in [-Float.infinity, Float.nan] {
            let logits = [Float](repeating: fill, count: v)
            for seed in UInt64(1)...8 {
                let cfg = GenerationConfig(temperature: 1.0, topK: v, seed: seed)
                let id = try rig.draw(logits, config: cfg, position: Int(seed)).id
                #expect(
                    id < UInt32(v),
                    "sampler returned \(id) (0x\(String(id, radix: 16))) for a distribution with no finite mass; vocab=\(v)"
                )
            }
        }
    }

    /// The sampler keeps its in-range guarantee -- the generation loop indexes
    /// the vocabulary and the KV history with the id -- so an empty row is
    /// *reported*, not answered, and `lastRowHadFiniteLogit` is how. It has to
    /// be false for the empty row and true for a healthy one: a stale or
    /// defaulted buffer would read as "fine" for both, which is the failure this
    /// accessor exists to make impossible.
    @Test func anEmptyRowIsReportedWhileAHealthyRowIsNot() throws {
        let v = 64
        let rig = try Rig(vocab: v)
        _ = try rig.draw(
            [Float](repeating: .nan, count: v),
            config: GenerationConfig(temperature: 1.0, topK: v, seed: 1))
        #expect(
            rig.sampler.lastRowHadFiniteLogit == false,
            "an all-NaN row must report as empty")

        _ = try rig.draw(
            [Float](repeating: 0.5, count: v),
            config: GenerationConfig(temperature: 1.0, topK: v, seed: 1))
        #expect(rig.sampler.lastRowHadFiniteLogit)

        // One NaN logit among finite ones must *not* empty the row; the fold in
        // `softcap_value` is what keeps this true.
        var oneNaN = [Float](repeating: -30, count: v)
        oneNaN[0] = .nan
        _ = try rig.draw(
            oneNaN,
            config: GenerationConfig(temperature: 1.0, topK: v, seed: 1))
        #expect(
            rig.sampler.lastRowHadFiniteLogit,
            "a single NaN logit must not empty the row")
    }

    @Test func greedy_picksArgmax() throws {
        let v = 2048
        let rig = try Rig(vocab: v)
        var logits = [Float](repeating: 0.1, count: v)
        logits[1337] = 9.0
        let (id, path) = try rig.draw(logits, config: GenerationConfig(temperature: 0))
        #expect(id == 1337, "got \(id)")
        #expect(path == .greedyGPU)
    }

    @Test func seeded_isDeterministicAtPosition() throws {
        let v = 1024
        let rig = try Rig(vocab: v)
        var rng = SeedTree(0x51A7_1005).key("sampler-position-determinism")
        let logits = (0..<v).map { _ in rng.uniform(-2, 2) }
        let cfg = GenerationConfig(temperature: 1.0, seed: 42)
        let a = try rig.draw(logits, config: cfg, position: 3).id
        let b = try rig.draw(logits, config: cfg, position: 3).id
        #expect(a == b, "same seed+position gave \(a) vs \(b)")
        #expect(try rig.draw(logits, config: cfg, position: 0).path == .gpuSampled)
    }

    @Test func seeded_reproducibleAcrossPositions() throws {
        let v = 1024
        let rig = try Rig(vocab: v)
        var rng = SeedTree(0x51A7_1006).key("sampler-position-replay")
        let logits = (0..<v).map { _ in rng.uniform(-2, 2) }
        let cfg = GenerationConfig(temperature: 1.0, seed: 42)
        let run1 = try (0..<5).map { try rig.draw(logits, config: cfg, position: $0).id }
        let run2 = try (0..<5).map { try rig.draw(logits, config: cfg, position: $0).id }
        #expect(run1 == run2, "seed=42 not reproducible: \(run1) vs \(run2)")

        // A different seed should diverge on at least one position.
        let cfg43 = GenerationConfig(temperature: 1.0, seed: 43)
        let run3 = try (0..<5).map { try rig.draw(logits, config: cfg43, position: $0).id }
        #expect(run3 != run1, "seed 42 and 43 produced identical sequences")
    }

    @Test func topK_restrictsToTop() throws {
        let v = 1024
        let rig = try Rig(vocab: v)
        var logits = [Float](repeating: -8.0, count: v)
        let top: [Int] = [10, 200, 500, 900]
        logits[top[0]] = 4.0
        logits[top[1]] = 3.0
        logits[top[2]] = 2.0
        logits[top[3]] = 1.0
        let topSet = Set(top.map { UInt32($0) })
        for t in 0..<32 {
            let cfg = GenerationConfig(temperature: 1.0, topK: 4, seed: UInt64(t) &+ 1)
            let id = try rig.draw(logits, config: cfg, position: t).id
            #expect(topSet.contains(id), "trial \(t): id=\(id) outside top-4")
        }
    }

    @Test func topP_restrictsToNucleus() throws {
        let v = 512
        let rig = try Rig(vocab: v)
        // Two tokens hold ~99% of the mass after softmax.
        var logits = [Float](repeating: -10.0, count: v)
        logits[7] = 6.0
        logits[42] = 5.6
        let nucleus: Set<UInt32> = [7, 42]
        for t in 0..<32 {
            let cfg = GenerationConfig(temperature: 1.0, topP: 0.9, seed: UInt64(t) &+ 1)
            let id = try rig.draw(logits, config: cfg, position: t).id
            #expect(nucleus.contains(id), "trial \(t): id=\(id) outside nucleus")
        }
    }

    @Test func repetitionPenalty_suppressesHistory() throws {
        let v = 64
        let rig = try Rig(vocab: v)
        // Large positive logits so penalty 2.0 (logit 8 -> 4) suppresses id 5
        // decisively post-softmax (~0.03x the others); flat logits of 1.0 only
        // scale its mass by ~0.6x, which the statistical bound below cannot
        // separate from noise.
        let logits = [Float](repeating: 8.0, count: v)
        let history: [Int32] = [5, 5, 5]
        var count5 = 0
        let trials = 200
        for t in 0..<trials {
            let cfg = GenerationConfig(
                temperature: 1.0, repetitionPenalty: 2.0, seed: UInt64(t) &+ 1)
            let (id, path) = try rig.draw(logits, config: cfg, position: t, history: history)
            #expect(path == .hostPenalty)
            if id == 5 { count5 += 1 }
        }
        // Uniform would pick 5 about trials/v times; suppression should push it
        // well below that. Generous bound to avoid flakiness.
        let uniformExpect = Double(trials) / Double(v)
        #expect(
            Double(count5) < 0.5 * uniformExpect,
            "id 5 chosen \(count5) times (uniform≈\(uniformExpect))")
    }

    /// Every id appended since the last sample is folded in, not only the last.
    ///
    /// The incremental path added `history.last` per call, which is correct for
    /// the one production caller — it appends exactly one token between samples —
    /// but a speculative path appending two folds one, and an id appearing only as
    /// the *first* of the pair is then never in the penalty table. The frequency
    /// count does not scale the penalty, so being in the table is the whole of the
    /// effect, and the penalty edits the logits buffer in place: the table's
    /// contents are directly observable rather than inferred from a draw.
    @Test func repetitionPenaltyFoldsEveryAppendedId() throws {
        let v = 64
        let rig = try Rig(vocab: v)
        let flat = [Float](repeating: 8.0, count: v)
        let cfg = GenerationConfig(temperature: 1.0, repetitionPenalty: 2.0, seed: 1)

        // Seed a table that does not mention 42, then append two ids at once with
        // 42 first and 9 last.
        _ = try rig.draw(flat, config: cfg, position: 0, history: [7, 7, 7])
        _ = try rig.draw(flat, config: cfg, position: 1, history: [7, 7, 7, 42, 9])

        let ptr = rig.logits.contents().bindMemory(to: Float16.self, capacity: v)
        let untouched = Float(ptr[3])  // never in any history
        #expect(Float(ptr[7]) < untouched * 0.9, "the seeded id is penalized")
        #expect(Float(ptr[9]) < untouched * 0.9, "the last appended id is penalized")
        #expect(
            Float(ptr[42]) < untouched * 0.9,
            Comment(
                rawValue:
                    "id 42 was appended before 9, so it is penalized only if every "
                    + "appended id is folded and not just the last"))
    }

    /// Saturated-logit suppression: raw logits deep in softcap-tanh
    /// saturation must still respond to the repetition penalty. The penalty
    /// acts on the post-softcap value — applied to the raw logit it moves the
    /// capped result by ~nothing and the penalty silently no-ops (the
    /// repetition-loop regression this pins).
    @Test func repetitionPenalty_bitesOnSaturatedLogits() throws {
        let v = 64
        let rig = try Rig(vocab: v)
        // All raw logits deep in tanh saturation; id 5 is the model's strong
        // favorite and also the repeated-history token.
        var logits = [Float](repeating: 300.0, count: v)
        logits[5] = 400.0
        let history: [Int32] = [5]
        var count5 = 0
        let trials = 64
        for t in 0..<trials {
            let cfg = GenerationConfig(
                temperature: 1.0, repetitionPenalty: 1.3, seed: UInt64(t) &+ 1)
            let (id, path) = try rig.draw(logits, config: cfg, position: t, history: history)
            #expect(path == .hostPenalty)
            if id == 5 { count5 += 1 }
        }
        // Post-softcap both land near the 30 cap; penalty 1.3 drops id 5 to
        // ~23, ~e^-7 of the others — it should essentially never win. The raw
        // pre-fix math left id 5 the argmax favorite at >half the draws.
        #expect(count5 < trials / 8, "saturated id 5 drawn \(count5)/\(trials) despite penalty")
    }

    /// Temperature spread: a logit sharp enough that greedy would always pick
    /// index 0 should, under raised temperature, distribute mass across many
    /// tokens. Exercised through the enumerated top-k path (`topK == v`).
    /// The `topK == 0` branch is the Gumbel-max fast path (argmax over noised
    /// log-probs), which cannot return an out-of-range id by construction.
    @Test func raisedTemperature_spreadsMass() throws {
        let v = 32
        let rig = try Rig(vocab: v)
        var logits = [Float](repeating: 0.0, count: v)
        logits[0] = 8.0
        var counts = [Int](repeating: 0, count: v)
        let trials = 1600
        for t in 0..<trials {
            let cfg = GenerationConfig(
                temperature: 2.0, topK: v, topP: 1,
                seed: UInt64(t) &+ 1)
            let raw = try rig.draw(logits, config: cfg, position: t).id
            let id = Int(raw)
            guard id >= 0 && id < v else {
                Issue.record("id \(raw) (0x\(String(raw, radix: 16))) out of range v=\(v)")
                continue
            }
            counts[id] += 1
        }
        let maxShare = Double(counts.max() ?? trials) / Double(trials)
        let distinct = counts.filter { $0 > 0 }.count
        // Greedy would give a top-token share of 1.0; raised temperature must
        // pull it well below that. (Bound kept loose — this asserts spread, not
        // a precise distribution.)
        #expect(maxShare < 0.7, "top token share \(maxShare) — temperature did not spread mass")
        #expect(distinct > v / 4, "only \(distinct)/\(v) tokens drawn — too concentrated")
    }

}
