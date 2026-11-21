import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2MoETests {
    private func weights() -> Inkling2ExpertWeights {
        // hidden 2, intermediate 2. gate = identity, up = identity, down = identity.
        let w13: [Float] = [
            1, 0,  // gate row 0
            1, 0,  // up row 0
            0, 1,  // gate row 1
            0, 1,  // up row 1
        ]
        let w2: [Float] = [1, 0, 0, 1]
        return Inkling2ExpertWeights(
            layer: 0, expert: 0, w13: w13, w2: w2, hidden: 2, intermediate: 2)
    }

    @Test func softcappedSwiGLUWithIdentityProjections() {
        let output = Inkling2MoE.expert([1, 2], weights: weights())
        func activation(_ value: Float) -> Float {
            tanh(value / 4) * 4 * (1 / (1 + exp(-value)))
        }
        let g0 = activation(1) * tanh(1 / 25) * 25
        let g1 = activation(2) * tanh(2 / 25) * 25
        #expect(abs(output[0] - g0) < 1e-5)
        #expect(abs(output[1] - g1) < 1e-5)
    }

    @Test func weightedCombinationAddsExperts() {
        let output = Inkling2MoE.forward(
            [1, 2], experts: [weights(), weights()],
            indices: [0, 1], weights: [0.25, 0.75])
        let one = Inkling2MoE.expert([1, 2], weights: weights())
        #expect(abs(output[0] - one[0]) < 1e-5)
        #expect(abs(output[1] - one[1]) < 1e-5)
    }
}
