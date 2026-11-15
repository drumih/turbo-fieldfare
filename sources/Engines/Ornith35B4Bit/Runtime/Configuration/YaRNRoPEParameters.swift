import Foundation

/// YaRN parameters following the reference implementation used by
/// Transformers: factor-based interpolation, beta-fast/beta-slow correction,
/// and the recommended attention magnitude scale.
struct YaRNRoPEParameters: Sendable, Equatable {
    let factor: Float
    let attentionFactor: Float
    let inverseFrequencies: [Float]

    init(
        headDim: Int,
        partialRotaryFactor: Double,
        theta: Double,
        targetContextTokens: Int,
        originalContextTokens: Int = RuntimeConfiguration.nativeMaximumContextTokens,
        betaFast: Double = 32,
        betaSlow: Double = 1
    ) {
        precondition(headDim > 0)
        precondition(partialRotaryFactor > 0 && partialRotaryFactor <= 1)
        precondition(theta > 1)
        precondition(targetContextTokens > originalContextTokens)
        let dimension = Int(Double(headDim) * partialRotaryFactor)
        precondition(dimension > 0 && dimension.isMultiple(of: 2))
        let factor = Double(targetContextTokens) / Double(originalContextTokens)
        self.factor = Float(factor)
        self.attentionFactor = Float(0.1 * log(factor) + 1.0)

        func correctionDimension(_ rotations: Double) -> Double {
            Double(dimension)
                * log(Double(originalContextTokens) / (rotations * 2.0 * Double.pi))
                / (2.0 * log(theta))
        }
        let low = max(floor(correctionDimension(betaFast)), 0)
        let high = min(ceil(correctionDimension(betaSlow)), Double(dimension - 1))
        let rampDenominator = low == high ? 0.001 : high - low
        self.inverseFrequencies = (0..<(dimension / 2)).map { pair in
            let positionFrequency = pow(theta, Double(2 * pair) / Double(dimension))
            let extrapolated = 1.0 / positionFrequency
            let interpolated = 1.0 / (factor * positionFrequency)
            let ramp = min(max((Double(pair) - low) / rampDenominator, 0), 1)
            let extrapolationFactor = 1.0 - ramp
            return Float(
                interpolated * (1.0 - extrapolationFactor)
                    + extrapolated * extrapolationFactor)
        }
    }
}
