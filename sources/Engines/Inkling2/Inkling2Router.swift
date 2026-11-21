import Foundation

/// The sparse router: a sigmoid over 128 gate rows, top-8 by score, normalized
/// to the configured route scale. Inkling 2 Mini stores no correction bias, so
/// the stored `gate.bias` is the only bias term.
public struct Inkling2Router: Sendable {
    public let experts: Int
    public let hidden: Int
    public let topK: Int
    public let routeScale: Float
    private let gate: [Float]
    private let bias: [Float]

    public init(tensors: Inkling2DenseTensors, layer: Int, topK: Int = 8, routeScale: Float = 8)
        throws
    {
        let prefix = "model.llm.layers.\(layer).mlp"
        let gate = try tensors.layerMatrix(layer, "mlp.gate.weight")
        let shape = try tensors.layerMatrixShape(layer, "mlp.gate.weight")
        guard shape.count == 2 else {
            throw Inkling2Safetensors.Error.invalid("router gate is not 2-D")
        }
        self.experts = shape[0]
        self.hidden = shape[1]
        self.gate = gate
        self.bias = try tensors.layerMatrix(layer, "mlp.gate.bias")
        self.topK = topK
        self.routeScale = routeScale
        guard bias.count == experts else {
            throw Inkling2Safetensors.Error.invalid("router bias width mismatch")
        }
        _ = prefix
    }

    public func route(_ x: [Float]) -> (indices: [Int], weights: [Float]) {
        precondition(x.count == hidden)
        var scores = [Float](repeating: 0, count: experts)
        for expert in 0..<experts {
            let row = expert * hidden
            var dot: Float = 0
            for index in 0..<hidden {
                dot += gate[row + index] * x[index]
            }
            scores[expert] = 1 / (1 + exp(-(dot + bias[expert])))
        }
        let ranked = scores.indices.sorted { a, b in
            scores[a] == scores[b] ? a < b : scores[a] > scores[b]
        }
        let indices = Array(ranked.prefix(topK))
        let total = indices.reduce(Float(0)) { $0 + scores[$1] }
        let weights = indices.map { scores[$0] / (total + 1e-20) * routeScale }
        return (indices, weights)
    }
}
