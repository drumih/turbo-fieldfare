import Foundation

/// The global layers' dynamic short convolution: a per-head, per-token filter
/// generated from the hidden state plus a static `[kernel, channels]` table.
public enum Inkling2DynamicConvolution {
    public static func apply(
        _ x: [Float],
        dynamicInput: [Float],
        tokenCount: Int,
        channels: Int,
        hidden: Int,
        heads: Int,
        kernelSize: Int,
        staticWeight: [Float],
        weightProjection: [Float],
        state: [Float]? = nil
    ) -> (values: [Float], state: [Float]) {
        precondition(x.count == tokenCount * channels)
        precondition(dynamicInput.count == tokenCount * hidden)
        precondition(channels % heads == 0)
        precondition(staticWeight.count == kernelSize * channels)
        precondition(weightProjection.count == heads * kernelSize * hidden)
        let width = channels / heads
        let history = state ?? [Float](repeating: 0, count: (kernelSize - 1) * channels)
        var context = history
        context.append(contentsOf: x)
        var output = [Float](repeating: 0, count: tokenCount * channels)
        let scale = 1 / Float(2 * kernelSize + 1).squareRoot()
        var dynamic = [Float](repeating: 0, count: heads * kernelSize)
        for token in 0..<tokenCount {
            for row in 0..<(heads * kernelSize) {
                var dot: Float = 0
                let base = row * hidden
                for column in 0..<hidden {
                    dot += weightProjection[base + column] * dynamicInput[token * hidden + column]
                }
                dynamic[row] = dot
            }
            for head in 0..<heads {
                for element in 0..<width {
                    let channel = head * width + element
                    var accumulator: Float = 0
                    for lag in 0..<kernelSize {
                        let contextIndex = token + kernelSize - 1 - lag
                        let value = context[contextIndex * channels + channel]
                        let staticValue = staticWeight[lag * channels + channel]
                        accumulator += value * (staticValue + dynamic[head * kernelSize + lag])
                    }
                    output[token * channels + channel] =
                        (accumulator + x[token * channels + channel]) * scale
                }
            }
        }
        let keep = (kernelSize - 1) * channels
        return (output, Array(context.suffix(keep)))
    }
}
