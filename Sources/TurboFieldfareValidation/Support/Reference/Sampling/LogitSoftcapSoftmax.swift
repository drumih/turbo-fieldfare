import Foundation
import Accelerate

/// FP32 reference for `softmax(softcap * tanh(x / softcap))`.
///
/// The kernel runs a single-pass online safe softmax with the softcap
/// fused into the running-max update. This reference is deliberately
/// *two-pass*: apply the softcap to a temp buffer, find the max, subtract,
/// exp via `vForce.exp`, sum via `vDSP_sve`, divide. Different staging,
/// different summation order — any bug that depends on the kernel's online
/// (m, d) merge logic won't replicate here.
///
/// Softcap is `c * tanh(x / c)` with c=30.0 by default for Gemma 4.
public enum LogitSoftcapSoftmaxRef {
    public static func apply(x: [Float], softcap: Float) -> [Float] {
        let v = x.count
        let invC = 1.0 / softcap

        // 1. Apply softcap: y = softcap * tanh(x / softcap)
        //    Two-step: y = x * invC; y = tanh(y); y = y * softcap.
        var y = [Float](repeating: 0, count: v)
        var s = invC
        x.withUnsafeBufferPointer { px in
            y.withUnsafeMutableBufferPointer { py in
                vDSP_vsmul(px.baseAddress!, 1, &s, py.baseAddress!, 1, vDSP_Length(v))
            }
        }
        y = vForce.tanh(y)
        var c = softcap
        y.withUnsafeMutableBufferPointer { py in
            vDSP_vsmul(py.baseAddress!, 1, &c, py.baseAddress!, 1, vDSP_Length(v))
        }

        // 2. Numerically stable softmax: subtract max, exp, divide by sum.
        var mx: Float = -.infinity
        y.withUnsafeBufferPointer { py in
            vDSP_maxv(py.baseAddress!, 1, &mx, vDSP_Length(v))
        }
        var negMax = -mx
        y.withUnsafeMutableBufferPointer { py in
            vDSP_vsadd(py.baseAddress!, 1, &negMax, py.baseAddress!, 1, vDSP_Length(v))
        }
        y = vForce.exp(y)

        var sum: Float = 0
        y.withUnsafeBufferPointer { py in
            vDSP_sve(py.baseAddress!, 1, &sum, vDSP_Length(v))
        }
        var invSum = 1.0 / sum
        y.withUnsafeMutableBufferPointer { py in
            vDSP_vsmul(py.baseAddress!, 1, &invSum, py.baseAddress!, 1, vDSP_Length(v))
        }
        return y
    }
}
