import Foundation

/// Small numerical pieces shared by the attention block. They are kept in
/// FP32: the checkpoint's BF16 weights are expanded at the boundary, and the
/// convolution/residual are summed before casting back.
public enum Inkling2AttentionMath {
    public static func l2Clip(
        _ values: [Float], threshold: Float, scale: [Float]? = nil
    ) -> [Float] {
        var output = values
        let width = values.count
        if let scale { precondition(scale.count == width) }
        var sum: Float = 0
        for index in 0..<width {
            let scaled = scale.map { values[index] * $0[index] } ?? values[index]
            output[index] = scaled
            sum += scaled * scaled
        }
        let norm = sum.squareRoot()
        let factor = min(threshold / (norm + 1e-6), 1)
        return output.map { $0 * factor }
    }

    public static func softmax(_ values: [Float]) -> [Float] {
        guard let peak = values.max() else { return [] }
        let exponentials = values.map { exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    /// Sliding short convolution: four taps per channel, a residual and the
    /// `1/sqrt(kernel + 1)` scale from the checkpoint's layer.
    public static func shortConvolution(
        _ x: [Float],
        length: Int,
        channels: Int,
        kernel: [Float],
        kernelSize: Int,
        state: [Float]? = nil
    ) -> (values: [Float], state: [Float]) {
        precondition(x.count == length * channels)
        precondition(kernel.count == channels * kernelSize)
        let history = state ?? [Float](repeating: 0, count: channels * (kernelSize - 1))
        var context = history
        context.append(contentsOf: x)
        var output = [Float](repeating: 0, count: length * channels)
        let scale = 1 / Float(kernelSize + 1).squareRoot()
        for token in 0..<length {
            for channel in 0..<channels {
                var convolution: Float = 0
                for tap in 0..<kernelSize {
                    let position = token + tap
                    let value = context[position * channels + channel]
                    convolution += value * kernel[channel * kernelSize + tap]
                }
                let index = token * channels + channel
                output[index] = (convolution + x[index]) * scale
            }
        }
        let keep = (kernelSize - 1) * channels
        return (output, Array(context.suffix(keep)))
    }
}
