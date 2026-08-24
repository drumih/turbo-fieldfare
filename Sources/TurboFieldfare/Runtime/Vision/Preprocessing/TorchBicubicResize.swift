import Foundation

package enum TorchBicubicResize {
    /// The weights for one axis, flattened for a GPU that cannot chase arrays
    /// of arrays: `starts[i]` is where output `i` begins reading, and its taps
    /// are `weights[i * tapStride ..< (i + 1) * tapStride]`, zero-padded.
    ///
    /// Exposed so the Metal path applies the same numbers rather than its own —
    /// the weights depend only on the axis ratio, so building them stays on the
    /// CPU where the arithmetic is already proven.
    package struct WeightTable {
        package let starts: [Int32]
        package let weights: [Int16]
        package let tapStride: Int
        package let precision: Int
    }

    package static func weightTable(input: Int, output: Int) -> WeightTable {
        let axis = axis(input: input, output: output)
        let stride = axis.samples.map(\.weights.count).max() ?? 0
        var starts: [Int32] = []
        var weights: [Int16] = []
        starts.reserveCapacity(axis.samples.count)
        weights.reserveCapacity(axis.samples.count * max(stride, 1))
        for sample in axis.samples {
            starts.append(Int32(sample.start))
            weights.append(contentsOf: sample.weights)
            weights.append(contentsOf: repeatElement(0, count: stride - sample.weights.count))
        }
        return WeightTable(starts: starts, weights: weights,
                           tapStride: max(stride, 1), precision: axis.precision)
    }

    private struct Sample {
        let start: Int
        let weights: [Int16]
    }

    private struct Axis {
        let samples: [Sample]
        let precision: Int
    }

    package static func resize(
        source: UnsafePointer<UInt8>,
        sourceWidth: Int,
        sourceHeight: Int,
        sourceRowBytes: Int,
        destination: UnsafeMutablePointer<UInt8>,
        destinationWidth: Int,
        destinationHeight: Int,
        destinationRowBytes: Int
    ) -> Int {
        if sourceWidth == destinationWidth && sourceHeight == destinationHeight {
            for row in 0..<sourceHeight {
                memcpy(
                    destination.advanced(by: row * destinationRowBytes),
                    source.advanced(by: row * sourceRowBytes),
                    sourceWidth * 4)
            }
            return 0
        }

        if sourceWidth == destinationWidth {
            let vertical = axis(input: sourceHeight, output: destinationHeight)
            resizeVertical(
                source: source,
                sourceWidth: sourceWidth,
                sourceRowBytes: sourceRowBytes,
                destination: destination,
                destinationHeight: destinationHeight,
                destinationRowBytes: destinationRowBytes,
                axis: vertical)
            return 0
        }
        let horizontal = axis(input: sourceWidth, output: destinationWidth)
        if sourceHeight == destinationHeight {
            resizeHorizontal(
                source: source,
                sourceHeight: sourceHeight,
                sourceRowBytes: sourceRowBytes,
                destination: destination,
                destinationWidth: destinationWidth,
                destinationRowBytes: destinationRowBytes,
                axis: horizontal)
            return 0
        }

        let temporaryRowBytes = destinationWidth * 4
        let temporaryBytes = temporaryRowBytes * sourceHeight
        let temporary = UnsafeMutablePointer<UInt8>.allocate(capacity: temporaryBytes)
        defer { temporary.deallocate() }
        resizeHorizontal(
            source: source,
            sourceHeight: sourceHeight,
            sourceRowBytes: sourceRowBytes,
            destination: temporary,
            destinationWidth: destinationWidth,
            destinationRowBytes: temporaryRowBytes,
            axis: horizontal)
        let vertical = axis(input: sourceHeight, output: destinationHeight)
        resizeVertical(
            source: UnsafePointer(temporary),
            sourceWidth: destinationWidth,
            sourceRowBytes: temporaryRowBytes,
            destination: destination,
            destinationHeight: destinationHeight,
            destinationRowBytes: destinationRowBytes,
            axis: vertical)
        return temporaryBytes
    }

    private static func axis(input: Int, output: Int) -> Axis {
        let scale = Double(input) / Double(output)
        let support = scale >= 1 ? 2 * scale : 2
        let inverseScale = scale >= 1 ? 1 / scale : 1
        let maximumCount = Int(ceil(support)) * 2 + 1
        var floating: [(start: Int, weights: [Double])] = []
        floating.reserveCapacity(output)
        var maximumWeight = 0.0
        for outputIndex in 0..<output {
            let center = scale * (Double(outputIndex) + 0.5)
            let start = max(Int(center - support + 0.5), 0)
            let count = min(
                max(min(Int(center + support + 0.5), input) - start, 0),
                maximumCount)
            var weights: [Double] = []
            weights.reserveCapacity(count)
            var total = 0.0
            for offset in 0..<count {
                let distance = (Double(offset + start) - center + 0.5) * inverseScale
                let weight = cubic(distance)
                weights.append(weight)
                total += weight
            }
            if total != 0 {
                for index in weights.indices {
                    weights[index] /= total
                    maximumWeight = max(maximumWeight, weights[index])
                }
            }
            floating.append((start, weights))
        }

        var precision = 0
        while precision < 22 {
            let next = Int(0.5 + maximumWeight * Double(1 << (precision + 1)))
            if next >= 1 << 15 { break }
            precision += 1
        }
        let scaleFactor = Double(1 << precision)
        let samples = floating.map { item in
            Sample(start: item.start, weights: item.weights.map { value in
                let scaled = value * scaleFactor
                return Int16(scaled < 0 ? Int(scaled - 0.5) : Int(scaled + 0.5))
            })
        }
        return Axis(samples: samples, precision: precision)
    }

    private static func cubic(_ value: Double) -> Double {
        let x = abs(value)
        let a = -0.5
        if x < 1 {
            return ((a + 2) * x - (a + 3)) * x * x + 1
        }
        if x < 2 {
            return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
        }
        return 0
    }

    private static func resizeHorizontal(
        source: UnsafePointer<UInt8>,
        sourceHeight: Int,
        sourceRowBytes: Int,
        destination: UnsafeMutablePointer<UInt8>,
        destinationWidth: Int,
        destinationRowBytes: Int,
        axis: Axis
    ) {
        let rounding = 1 << (axis.precision - 1)
        for row in 0..<sourceHeight {
            let inputRow = source.advanced(by: row * sourceRowBytes)
            let outputRow = destination.advanced(by: row * destinationRowBytes)
            for x in 0..<destinationWidth {
                let sample = axis.samples[x]
                for channel in 0..<4 {
                    var value = rounding
                    for index in sample.weights.indices {
                        value += Int(inputRow[(sample.start + index) * 4 + channel])
                            * Int(sample.weights[index])
                    }
                    outputRow[x * 4 + channel] = UInt8(clamping: value >> axis.precision)
                }
            }
        }
    }

    private static func resizeVertical(
        source: UnsafePointer<UInt8>,
        sourceWidth: Int,
        sourceRowBytes: Int,
        destination: UnsafeMutablePointer<UInt8>,
        destinationHeight: Int,
        destinationRowBytes: Int,
        axis: Axis
    ) {
        let rounding = 1 << (axis.precision - 1)
        for y in 0..<destinationHeight {
            let sample = axis.samples[y]
            let outputRow = destination.advanced(by: y * destinationRowBytes)
            for x in 0..<(sourceWidth * 4) {
                var value = rounding
                for index in sample.weights.indices {
                    value += Int(source[(sample.start + index) * sourceRowBytes + x])
                        * Int(sample.weights[index])
                }
                outputRow[x] = UInt8(clamping: value >> axis.precision)
            }
        }
    }
}
