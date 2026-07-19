import Foundation
import Accelerate

/// FP32 RMSNorm reference computed with Accelerate vectorized primitives.
///
/// Deliberately reaches the same answer via a *different* sequence of
/// operations than the Metal kernel: `vDSP_svesq` for the sum-of-squares,
/// `vDSP_vmul` + `vDSP_vsmul` for the per-element scale. The kernel uses a
/// per-thread block reduction with `simd_sum`; this reference uses
/// Accelerate's pipeline. Different summation tree, different rounding chain,
/// same math — which is what makes this a useful comparator.
public enum RmsNormRef {
    /// y[i] = x[i] * weight[i] * 1 / sqrt(mean(x^2) + eps)
    public static func apply(x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count, "x and weight must match length")
        let d = x.count

        var sumSq: Float = 0
        x.withUnsafeBufferPointer { p in
            vDSP_svesq(p.baseAddress!, 1, &sumSq, vDSP_Length(d))
        }
        let invRms = 1.0 / (sumSq / Float(d) + eps).squareRoot()

        var y = [Float](repeating: 0, count: d)
        x.withUnsafeBufferPointer { px in
            weight.withUnsafeBufferPointer { pw in
                y.withUnsafeMutableBufferPointer { py in
                    vDSP_vmul(
                        px.baseAddress!, 1,
                        pw.baseAddress!, 1,
                        py.baseAddress!, 1,
                        vDSP_Length(d)
                    )
                    var s = invRms
                    vDSP_vsmul(
                        py.baseAddress!, 1,
                        &s,
                        py.baseAddress!, 1,
                        vDSP_Length(d)
                    )
                }
            }
        }
        return y
    }
}
