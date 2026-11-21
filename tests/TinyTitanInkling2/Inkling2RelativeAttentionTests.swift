import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2RelativeAttentionTests {
    @Test func causalSlidingBand() {
        let r: [[Float]] = [[1], [2]]
        let projection: [Float] = [1, 2, 3]
        let biases = Inkling2RelativeAttention.biases(
            r: r, relProjection: projection, relativeDimension: 1,
            extent: 3, sliding: 2, keyCount: 3, newTokens: 2)
        // q0 (position 1): keys 0..1 only; q1 (position 2): keys 1..2.
        #expect(biases[0][0] == 2)
        #expect(biases[0][1] == 1)
        #expect(biases[0][2] == -Float.infinity)
        #expect(biases[1][0] == -Float.infinity)
        #expect(biases[1][1] == 4)
        #expect(biases[1][2] == 2)
    }

    @Test func fullCausalWithoutSliding() {
        let biases = Inkling2RelativeAttention.biases(
            r: [[1]], relProjection: [1, 2], relativeDimension: 1,
            extent: 2, sliding: 0, keyCount: 2, newTokens: 1)
        #expect(biases[0][0] == 2)
        #expect(biases[0][1] == 1)
    }
}
