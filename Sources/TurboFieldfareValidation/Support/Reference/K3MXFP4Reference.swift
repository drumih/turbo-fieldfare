import Foundation
import Accelerate
import TurboFieldfareFormat

/// FP32 reference for the K3 MXFP4 (E2M1 + E8M0 group-of-32) GEMV.
///
/// Dequantization wraps `QuantizationMXFP4` directly — that file is the
/// canonical byte-level contract, so the reference shares it on purpose. The
/// matvec then bulk-dequantizes each row and dots with `vDSP_dotpr`: the
/// kernel interleaves nibble unpack + group scale + FMA inside one per-lane
/// loop with a SIMD reduction, so staging and summation order differ and
/// in-flight-dequant bugs don't replicate here.
public enum K3MXFP4Reference {
    /// Whole-matrix dequant, row-major `rows x columns`, via the canonical
    /// per-group decode.
    public static func dequantize(packedWeights: [UInt8], scales: [UInt8],
                                  rows: Int, columns: Int) -> [Float] {
        precondition(rows > 0)
        precondition(columns > 0 && columns % QuantizationMXFP4.groupSize == 0,
                     "columns \(columns) is not a positive multiple of 32")
        precondition(packedWeights.count == rows * columns / 2,
                     "packed size mismatch")
        precondition(scales.count == rows * columns / QuantizationMXFP4.groupSize,
                     "scale size mismatch")
        let groupsPerRow = columns / QuantizationMXFP4.groupSize
        var out = [Float](repeating: 0, count: rows * columns)
        for row in 0..<rows {
            let packedBase = row * columns / 2
            let scaleBase = row * groupsPerRow
            for group in 0..<groupsPerRow {
                let lo = packedBase + group * QuantizationMXFP4.bytesPerGroup
                let hi = lo + QuantizationMXFP4.bytesPerGroup
                let packed = Array(packedWeights[lo..<hi])
                let values = QuantizationMXFP4.dequantizeMxfp4Group(
                    packed: packed, scale: scales[scaleBase + group])
                for k in 0..<QuantizationMXFP4.groupSize {
                    out[row * columns + group * QuantizationMXFP4.groupSize + k] = values[k]
                }
            }
        }
        return out
    }

    /// `out[r] = sum_c dequant(w[r, c]) * vector[c]`, one vDSP dot per row
    /// over the bulk-dequantized matrix.
    public static func matvec(packedWeights: [UInt8], scales: [UInt8],
                              rows: Int, columns: Int, vector: [Float]) -> [Float] {
        precondition(vector.count == columns, "vector length mismatch")
        let dequantized = dequantize(packedWeights: packedWeights, scales: scales,
                                     rows: rows, columns: columns)
        var out = [Float](repeating: 0, count: rows)
        dequantized.withUnsafeBufferPointer { pw in
            vector.withUnsafeBufferPointer { px in
                for row in 0..<rows {
                    var dot: Float = 0
                    vDSP_dotpr(pw.baseAddress! + row * columns, 1,
                               px.baseAddress!, 1,
                               &dot, vDSP_Length(columns))
                    out[row] = dot
                }
            }
        }
        return out
    }
}
