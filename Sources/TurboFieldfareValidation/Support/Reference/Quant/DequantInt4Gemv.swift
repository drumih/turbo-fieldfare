import Foundation
import Accelerate
import TurboFieldfare

/// FP32 reference for the INT4-affine groupwise GEMV `y = W * x`.
///
/// Bulk-dequantizes each row to FP32 via `Quantization.dequantizeInt4Affine`,
/// then dots with `vDSP_dotpr`. The kernel interleaves nibble unpack + scale +
/// bias + FMA inside one inner loop per thread — different staging, different
/// summation order. Bugs that depend on the in-flight dequant pattern won't
/// replicate here.
public enum DequantInt4GemvRef {
    public static func apply(
        weightRows: [Quantization.Int4AffineRow],
        x: [Float],
        n: Int
    ) -> [Float] {
        precondition(!weightRows.isEmpty)
        precondition(x.count == n)
        precondition(n % Quantization.groupSize == 0,
                     "N must be a multiple of \(Quantization.groupSize)")

        let m = weightRows.count
        var y = [Float](repeating: 0, count: m)

        for row in 0..<m {
            let wRow = Quantization.dequantizeInt4Affine(weightRows[row], n: n)
            var dot: Float = 0
            wRow.withUnsafeBufferPointer { pw in
                x.withUnsafeBufferPointer { px in
                    vDSP_dotpr(
                        pw.baseAddress!, 1,
                        px.baseAddress!, 1,
                        &dot,
                        vDSP_Length(n)
                    )
                }
            }
            y[row] = dot
        }
        return y
    }
}
