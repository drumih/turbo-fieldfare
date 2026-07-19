import Foundation
import Accelerate
import TurboFieldfare

/// FP32 reference for the INT8-affine groupwise GEMV `y = W * x`.
///
/// Bulk-dequantizes each row to FP32 via `Quantization.dequantizeInt8Affine`,
/// then computes the inner product with `vDSP_dotpr`. The kernel interleaves
/// byte unpack + per-group scale + bias + FMA in the inner loop per thread;
/// this reference splits dequant from accumulation. Different staging,
/// different summation order.
public enum DequantInt8GemvRef {
    public static func apply(
        weightRows: [Quantization.Int8AffineRow],
        x: [Float],
        n: Int
    ) -> [Float] {
        precondition(!weightRows.isEmpty)
        precondition(x.count == n)
        precondition(n % Quantization.groupSize == 0)

        let m = weightRows.count
        var y = [Float](repeating: 0, count: m)
        for row in 0..<m {
            let wRow = Quantization.dequantizeInt8Affine(weightRows[row], n: n)
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
