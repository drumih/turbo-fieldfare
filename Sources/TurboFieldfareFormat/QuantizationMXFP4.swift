import Foundation

// Pure-Swift MXFP4 (E2M1 values + E8M0 group scales) reference primitives,
// shared by the repack pipeline and the test suites. Never the runtime path.
package enum QuantizationMXFP4 {
    package static let groupSize = 32
    package static let bytesPerGroup = 16

    // E2M1 decode table indexed by the raw nibble (sign << 3 | exponent << 1
    // | mantissa). Index 8 is negative zero.
    package static let e2m1DecodeLUT: [Float] = [
        0, 0.5, 1, 1.5, 2, 3, 4, 6,
        -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
    ]

    // Low nibble holds the even element index, matching the repo's int4
    // packing convention. Nibble order verified against compressed-tensors
    // mxfp4-pack-quantized layout; single place to flip if real-checkpoint
    // validation disagrees.
    @inline(__always)
    package static func valueIndex(_ elementIndex: Int, in byte: UInt8) -> UInt8 {
        elementIndex & 1 == 0 ? byte & 0x0F : byte >> 4
    }

    // E8M0 scale: 2^(byte - 127). The exponent-field bit trick is exact for
    // bytes 1...254; byte 0 is the subnormal 2^-127. Byte 255 is NaN in the
    // E8M0 spec — decode it as 0 so a corrupt scale zeroes its group instead
    // of poisoning the row, matching kimi-k3-in-c's corruption guard.
    @inline(__always)
    package static func decodeScale(_ byte: UInt8) -> Float {
        if byte == 255 { return 0 }
        if byte == 0 { return Float(bitPattern: 0x0040_0000) }
        return Float(bitPattern: UInt32(byte) << 23)
    }

    /// Dequantize one group of 32 values from 16 packed bytes + one E8M0
    /// scale byte.
    package static func dequantizeMxfp4Group(packed: [UInt8], scale: UInt8) -> [Float] {
        precondition(packed.count == bytesPerGroup,
                     "group is \(packed.count) bytes, expected \(bytesPerGroup)")
        let groupScale = decodeScale(scale)
        var out = [Float](repeating: 0, count: groupSize)
        for element in 0..<groupSize {
            out[element] = e2m1DecodeLUT[Int(valueIndex(element, in: packed[element / 2]))]
                * groupScale
        }
        return out
    }

    /// Reference row-major GEMV: `out[r] = sum_c dequant(w[r, c]) * x[c]`.
    /// `packedWeights` is rows x columns/2 bytes, `scales` is rows x
    /// columns/32 E8M0 bytes, `vector` has columns fp32 elements.
    package static func matvecMxfp4(packedWeights: [UInt8], scales: [UInt8],
                                    rows: Int, columns: Int, vector: [Float]) -> [Float] {
        precondition(columns > 0 && columns % groupSize == 0,
                     "columns \(columns) is not a positive multiple of \(groupSize)")
        precondition(vector.count == columns, "vector length mismatch")
        precondition(packedWeights.count == rows * columns / 2, "packed size mismatch")
        precondition(scales.count == rows * columns / groupSize, "scale size mismatch")
        let groupsPerRow = columns / groupSize
        var out = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            let packedBase = row * columns / 2
            let scaleBase = row * groupsPerRow
            var acc: Float = 0
            for group in 0..<groupsPerRow {
                let groupScale = decodeScale(scales[scaleBase + group])
                for k in 0..<groupSize {
                    let column = group * groupSize + k
                    let byte = packedWeights[packedBase + column / 2]
                    let value = e2m1DecodeLUT[Int(valueIndex(k, in: byte))]
                    acc += value * groupScale * vector[column]
                }
            }
            out[row] = acc
        }
        return out
    }
}
