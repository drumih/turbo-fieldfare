import Foundation

/// Blends the previous loop pass's message into one sublayer input.
public struct Inkling2MessageController: Sendable {
    private let down: [Float]
    private let forget: [Float]
    private let gain: [Float]
    private let rank: Int
    private let hidden: Int

    public init(tensors: Inkling2DenseTensors, layer: Int, name: String) throws {
        let prefix = "model.llm.layers.\(layer).controllers.\(name)."
        self.down = try tensors.matrix(prefix + "down_weight")
        self.forget = try tensors.matrix(prefix + "forget_weight")
        self.gain = try tensors.matrix(prefix + "gain_weight")
        let downShape = try tensors.matrixShape(prefix + "down_weight")
        self.rank = downShape[0]
        self.hidden = downShape[1] / 2
        precondition(forget.count == hidden * rank)
        precondition(gain.count == hidden * rank)
    }

    public func forward(_ x: [Float], message: [Float]) -> [Float] {
        precondition(x.count == hidden)
        precondition(message.count == hidden)
        let xNorm = Inkling2Ops.rmsNorm(x, weight: [Float](repeating: 1, count: hidden))
        let mNorm = Inkling2Ops.rmsNorm(
            message, weight: [Float](repeating: 1, count: hidden))
        let pair = xNorm + mNorm
        var compressed = [Float](repeating: 0, count: rank)
        for row in 0..<rank {
            var dot: Float = 0
            for column in 0..<(2 * hidden) {
                dot += down[row * 2 * hidden + column] * pair[column]
            }
            compressed[row] = dot
        }
        var forgetValue = [Float](repeating: 0, count: hidden)
        var gainValue = [Float](repeating: 0, count: hidden)
        for row in 0..<hidden {
            var f: Float = 0
            var g: Float = 0
            for column in 0..<rank {
                f += forget[row * rank + column] * compressed[column]
                g += gain[row * rank + column] * compressed[column]
            }
            forgetValue[row] = tanh(f)
            gainValue[row] = tanh(g)
        }
        return (0..<hidden).map {
            (1 - forgetValue[$0]) * x[$0] + gainValue[$0] * message[$0]
        }
    }
}
