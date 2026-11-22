import Foundation
import Testing

@testable import TinyTitan

/// The sampler ranks candidates by sorting (logit, index) pairs; this pins
/// that ranking, and every seeded draw built on it, to the index sort it
/// replaced, including over tied logits.
@Suite struct CPUSamplerRankingTests {
    /// Logits with heavy ties (a coarse grid of values) and a spread like a
    /// softcapped head's.
    static func logits(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { Float((($0 &* 7_919 &+ seed &* 104_729) % 61) - 30) * 0.5 }
    }

    /// The previous ranking, verbatim.
    static func indexSorted(_ logits: [Float]) -> [Int] {
        var order = Array(logits.indices)
        order.sort { logits[$0] > logits[$1] }
        return order
    }

    /// The previous `pick` after penalties, verbatim apart from the ranking.
    static func referencePick(
        _ logits: [Float], sampler: CPUSampler, generator: CPUSampler.Generator
    ) -> Int {
        let limit = sampler.topK > 0 ? min(sampler.topK, logits.count) : logits.count
        let order = Array(indexSorted(logits).prefix(limit))
        let peak = logits[order[0]]
        var weights = [Float]()
        var total: Float = 0
        for index in order {
            let value = expf((logits[index] - peak) / max(sampler.temperature, 1e-4))
            weights.append(value)
            total += value
        }
        var cutoff = order.count
        if sampler.topP < 1 {
            var mass: Float = 0
            for position in weights.indices {
                mass += weights[position] / total
                if mass >= sampler.topP {
                    cutoff = position + 1
                    break
                }
            }
        }
        var remaining: Float = 0
        for position in 0..<cutoff { remaining += weights[position] }
        let target = generator.next() * remaining
        var running: Float = 0
        for position in 0..<cutoff {
            running += weights[position]
            if running >= target { return order[position] }
        }
        return order[0]
    }

    @Test func rankingIsTheStableIndexSort() {
        for seed in 0..<4 {
            let values = Self.logits(20_011, seed: seed)
            #expect(CPUSampler.ranked(values).map(\.index) == Self.indexSorted(values))
        }
    }

    @Test(arguments: [(Float(0.8), Float(0.95), 0), (1.0, 1.0, 0), (0.7, 1.0, 40), (1.2, 0.5, 200)])
    func seededDrawsMatchThePreviousAlgorithm(temperature: Float, topP: Float, topK: Int) {
        let sampler = CPUSampler(temperature: temperature, topP: topP, topK: topK, seed: 11)
        let ours = sampler.makeGenerator()
        let reference = sampler.makeGenerator()
        for step in 0..<64 {
            let values = Self.logits(4_099, seed: step)
            #expect(
                sampler.pick(values, using: ours)
                    == Self.referencePick(values, sampler: sampler, generator: reference),
                "step \(step)")
        }
    }
}
