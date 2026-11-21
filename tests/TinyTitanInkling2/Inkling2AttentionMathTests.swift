import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2AttentionMathTests {
    @Test func l2ClipScalesTheVector() {
        let output = Inkling2AttentionMath.l2Clip([3, 4], threshold: 1)
        let norm = (output[0] * output[0] + output[1] * output[1]).squareRoot()
        #expect(abs(norm - 1) < 1e-5)
        #expect(abs(output[0] / output[1] - 0.75) < 1e-6)
    }

    @Test func softmaxSumsToOne() {
        let output = Inkling2AttentionMath.softmax([1, 2, 3])
        #expect(abs(output.reduce(0, +) - 1) < 1e-6)
        #expect(output[2] > output[0])
    }

    @Test func shortConvolutionCarriesState() {
        // One channel, kernel 2, weight [1, 0]: output is the current input.
        let first = Inkling2AttentionMath.shortConvolution(
            [1, 2, 3], length: 3, channels: 1, kernel: [1, 0],
            kernelSize: 2, state: [0])
        let scale = 1 / Float(3).squareRoot()
        #expect(abs(first.values[0] - 1 * scale) < 1e-5)
        #expect(abs(first.values[2] - 5 * scale) < 1e-5)
        let second = Inkling2AttentionMath.shortConvolution(
            [4], length: 1, channels: 1, kernel: [1, 0],
            kernelSize: 2, state: first.state)
        #expect(abs(second.values[0] - 7 * scale) < 1e-5)
    }
}
