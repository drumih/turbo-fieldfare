import Foundation
import Testing
import TinyTitanCore

@Suite struct StableLogSoftmaxTests {
    @Test func thePaperCaseKeepsTheTinyMargin() {
        // z = [9, -9, -9], target 0. The standard float formulation rounds the
        // 2*exp(-18) term away and returns 0; the margin form returns 3.046e-8.
        let nll = StableLogSoftmax.negativeLogLikelihood([9, -9, -9], target: 0)
        #expect(abs(nll - 3.046e-8) < 1e-12)
    }

    @Test func logSoftmaxNormalisesAndIsStable() {
        let logits: [Float] = [100, 99, 98]
        let logProbs = StableLogSoftmax.logSoftmax(logits)
        #expect(logProbs.count == logits.count)
        let total = logProbs.reduce(0.0) { $0 + exp(Double($1)) }
        #expect(abs(total - 1.0) < 1e-6)
        #expect(logProbs[0] > logProbs[1] && logProbs[1] > logProbs[2])
        #expect(logProbs.allSatisfy { $0.isFinite })
    }

    @Test func matchesTheDirectFormForOrdinaryLogits() {
        let logits: [Float] = [2.5, -1.25, 0.75, 3.5]
        let target = 2
        let peak = logits.max() ?? 0
        let sum = logits.reduce(0.0) { $0 + exp(Double($1 - peak)) }
        let expected = Double(peak) + log(sum) - Double(logits[target])
        let got = StableLogSoftmax.negativeLogLikelihood(logits, target: target)
        #expect(abs(got - expected) < 1e-6)
    }

    @Test func anInvalidTargetIsInfinite() {
        #expect(StableLogSoftmax.negativeLogLikelihood([1, 2], target: 9) == .infinity)
        #expect(StableLogSoftmax.negativeLogLikelihood([.nan, 2], target: 0) == .infinity)
    }
}
