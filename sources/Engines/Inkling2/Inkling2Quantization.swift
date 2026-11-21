import Foundation

/// The exact dequantization of Inkling 2 Mini's two expert formats.
///
/// NVFP4 stores packed E2M1 (low nibble first), one FP8 E4M3 scale per 16
/// input channels, and one outer scale per expert equal to
/// `amax[expert] / outer_scale_denominator`. MXFP8 stores E4M3 values with one
/// UE8M0 scale per 32 channels. Both decode to the checkpoint's values without
/// any requantization.
public enum Inkling2Quantization {
    public static let nvfp4Block = 16
    public static let mxfp8Block = 32

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case invalid(String)

        public var description: String {
            switch self {
            case .invalid(let detail): "invalid Inkling quantization: \(detail)"
            }
        }
    }

    private static let e2m1: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    public static func e2m1Value(_ nibble: UInt8) -> Float {
        let magnitude = e2m1[Int(nibble & 0x07)]
        return (nibble & 0x08) == 0 ? magnitude : -magnitude
    }

    /// Decode one FP8 E4M3FN byte. `NaN` is returned for the reserved 0x7F/0xFF.
    public static func e4m3Value(_ byte: UInt8) -> Float {
        let bits = Int(byte)
        let exponent = (bits >> 3) & 0xF
        let mantissa = Float(bits & 0x7)
        let magnitude: Float
        if (bits & 0x7F) == 0x7F {
            return .nan
        } else if exponent == 0 {
            magnitude = mantissa * 0.001953125
        } else {
            magnitude = (1 + mantissa * 0.125) * exp2(Float(exponent - 7))
        }
        return (bits & 0x80) == 0 ? magnitude : -magnitude
    }

    /// `[experts, rows, columns]` packed NVFP4 for one expert slice.
    ///
    /// `packed` is `columns / 2` bytes per row, `scales` is `columns / 16`
    /// bytes per row, and `outerScale` is `amax[expert] / denominator`.
    public static func dequantizeNVFP4(
        packed: Data,
        scales: Data,
        outerScale: Float,
        rows: Int,
        columns: Int
    ) throws -> [Float] {
        guard columns % nvfp4Block == 0 else {
            throw Error.invalid("NVFP4 columns \(columns) not divisible by 16")
        }
        let packedBytes = columns / 2
        let scaleBytes = columns / nvfp4Block
        guard packed.count == rows * packedBytes else {
            throw Error.invalid("packed size \(packed.count) != \(rows * packedBytes)")
        }
        guard scales.count == rows * scaleBytes else {
            throw Error.invalid("scale size \(scales.count) != \(rows * scaleBytes)")
        }
        var output = [Float](repeating: 0, count: rows * columns)
        packed.withUnsafeBytes { packedRaw in
            scales.withUnsafeBytes { scaleRaw in
                let bytes = packedRaw.bindMemory(to: UInt8.self)
                let blocks = scaleRaw.bindMemory(to: UInt8.self)
                for row in 0..<rows {
                    for block in 0..<scaleBytes {
                        let scale = e4m3Value(blocks[row * scaleBytes + block]) * outerScale
                        let base = row * packedBytes + block * (nvfp4Block / 2)
                        for element in 0..<nvfp4Block {
                            let byte = bytes[base + element / 2]
                            let nibble = element.isMultiple(of: 2) ? byte & 0x0F : byte >> 4
                            output[row * columns + block * nvfp4Block + element] =
                                e2m1Value(nibble) * scale
                        }
                    }
                }
            }
        }
        return output
    }

    /// `[rows, columns]` MXFP8 values plus `[rows, columns / 32]` UE8M0 scales.
    public static func dequantizeMXFP8(
        weights: Data,
        scales: Data,
        rows: Int,
        columns: Int
    ) throws -> [Float] {
        guard columns % mxfp8Block == 0 else {
            throw Error.invalid("MXFP8 columns \(columns) not divisible by 32")
        }
        let scaleBytes = columns / mxfp8Block
        guard weights.count == rows * columns else {
            throw Error.invalid("weight size \(weights.count) != \(rows * columns)")
        }
        guard scales.count == rows * scaleBytes else {
            throw Error.invalid("scale size \(scales.count) != \(rows * scaleBytes)")
        }
        var output = [Float](repeating: 0, count: rows * columns)
        weights.withUnsafeBytes { weightRaw in
            scales.withUnsafeBytes { scaleRaw in
                let bytes = weightRaw.bindMemory(to: UInt8.self)
                let blockScales = scaleRaw.bindMemory(to: UInt8.self)
                for row in 0..<rows {
                    for block in 0..<scaleBytes {
                        let scale = exp2(Float(blockScales[row * scaleBytes + block]) - 127)
                        for element in 0..<mxfp8Block {
                            let index = row * columns + block * mxfp8Block + element
                            output[index] = e4m3Value(bytes[index]) * scale
                        }
                    }
                }
            }
        }
        return output
    }
}
