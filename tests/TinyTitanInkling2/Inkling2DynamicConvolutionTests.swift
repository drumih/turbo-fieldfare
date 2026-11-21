import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2DynamicConvolutionTests {
    @Test func zeroFiltersReduceToResidual() {
        let result = Inkling2DynamicConvolution.apply(
            [1, 2, 3, 4], dynamicInput: [1, 0, 1, 0],
            tokenCount: 2, channels: 2, hidden: 2, heads: 1, kernelSize: 2,
            staticWeight: [0, 0, 0, 0], weightProjection: [0, 0, 0, 0])
        let scale = 1 / Float(5).squareRoot()
        #expect(abs(result.values[0] - 1 * scale) < 1e-5)
        #expect(abs(result.values[3] - 4 * scale) < 1e-5)
        #expect(result.state == [3, 4])
    }

    @Test func carriesThePreviousTokens() {
        let first = Inkling2DynamicConvolution.apply(
            [1, 1], dynamicInput: [1, 1], tokenCount: 1, channels: 2, hidden: 2,
            heads: 1, kernelSize: 2, staticWeight: [2, 2, 0, 0],
            weightProjection: [0, 0, 0, 0])
        #expect(abs(first.values[0] - 3 / Float(5).squareRoot()) < 1e-5)
        let second = Inkling2DynamicConvolution.apply(
            [1, 1], dynamicInput: [1, 1], tokenCount: 1, channels: 2, hidden: 2,
            heads: 1, kernelSize: 2, staticWeight: [2, 2, 0, 0],
            weightProjection: [0, 0, 0, 0], state: first.state)
        // lag0 uses current x with weight 2, lag1 previous x with 0.
        #expect(abs(second.values[0] - 3 / Float(5).squareRoot()) < 1e-5)
    }
}
