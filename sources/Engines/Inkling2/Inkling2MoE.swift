import Foundation

/// The routed-expert computation for one token: the softcapped SwiGLU and the
/// down projection, followed by the router-weighted combination.
public enum Inkling2MoE {
    public static let gateCap: Float = 4
    public static let upCap: Float = 25

    public static func expert(
        _ x: [Float], weights: Inkling2ExpertWeights
    ) -> [Float] {
        precondition(x.count == weights.hidden)
        let intermediate = weights.intermediate
        var gateUp = [Float](repeating: 0, count: 2 * intermediate)
        for row in 0..<(2 * intermediate) {
            var dot: Float = 0
            let base = row * weights.hidden
            for column in 0..<weights.hidden {
                dot += weights.w13[base + column] * x[column]
            }
            gateUp[row] = dot
        }
        let activated = Self.activate(
            gateUp, hidden: weights.hidden, intermediate: intermediate)
        var output = [Float](repeating: 0, count: weights.hidden)
        for row in 0..<weights.hidden {
            var dot: Float = 0
            let base = row * intermediate
            for column in 0..<intermediate {
                dot += weights.w2[base + column] * activated[column]
            }
            output[row] = dot
        }
        return output
    }

    public static func activate(
        _ gateUp: [Float], hidden: Int, intermediate: Int
    ) -> [Float] {
        precondition(gateUp.count == 2 * intermediate)
        var activated = [Float](repeating: 0, count: intermediate)
        for index in 0..<intermediate {
            let gate = gateUp[index]
            let up = gateUp[intermediate + index]
            let g = tanh(gate / gateCap) * gateCap * (1 / (1 + exp(-gate)))
            let u = tanh(up / upCap) * upCap
            activated[index] = g * u
        }
        _ = hidden
        return activated
    }

    public static func forward(
        _ x: [Float],
        experts: [Inkling2ExpertWeights],
        indices: [Int],
        weights: [Float]
    ) -> [Float] {
        precondition(experts.count == indices.count)
        precondition(weights.count == indices.count)
        var output = [Float](repeating: 0, count: x.count)
        for (expert, weight) in zip(experts, weights) {
            let result = self.expert(x, weights: expert)
            for index in output.indices {
                output[index] += result[index] * weight
            }
        }
        return output
    }
}
