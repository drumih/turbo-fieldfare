import Foundation
import Accelerate
import TurboFieldfare

/// FP32 reference for the fused MoE FFN: top-k routed experts + 1 shared
/// expert, each with gate / up / down projections, GeGLU activation, and a
/// weighted combine into the residual stream.
///
/// Composed from `DequantInt4GemvRef` (vDSP_dotpr after bulk affine dequant)
/// + a vForce-based `gelu_pytorch_tanh`. The kernel inlines the dequant +
/// gate*up*down chain in one threadgroup per output element; this reference
/// materializes the F-dim intermediate per expert and uses Accelerate
/// primitives for the inner products. Different code shape, different
/// summation order — bugs that depend on the in-flight dequant pattern won't
/// replicate here.
public enum MoeRef {
    public static let geluCoeff: Float = 0.7978845608028654  // sqrt(2/π)
    public static let geluCubic: Float = 0.044715

    /// Per-element `gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(c * (x + k * x^3)))`.
    public static func geluTanh(_ x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count {
            let xv = x[i]
            let inner = geluCoeff * (xv + geluCubic * xv * xv * xv)
            y[i] = 0.5 * xv * (1.0 + Foundation.tanh(inner))
        }
        return y
    }

    /// Runs one FFN block: `down(gelu(gate(x)) * up(x))`.
    /// `gateRows` / `upRows` are F-by-D affine INT4. `downRows` is D-by-F.
    public static func runFFN(
        gateRows: [Quantization.Int4AffineRow],
        upRows:   [Quantization.Int4AffineRow],
        downRows: [Quantization.Int4AffineRow],
        x: [Float],
        d: Int,
        f: Int
    ) -> [Float] {
        precondition(gateRows.count == f, "gateRows must be F=\(f)")
        precondition(upRows.count == f, "upRows must be F=\(f)")
        precondition(downRows.count == d, "downRows must be D=\(d)")
        precondition(x.count == d)

        let gateOut = DequantInt4GemvRef.apply(weightRows: gateRows, x: x, n: d)
        let upOut   = DequantInt4GemvRef.apply(weightRows: upRows,   x: x, n: d)
        let gated   = geluTanh(gateOut)
        var act = [Float](repeating: 0, count: f)
        vDSP_vmul(gated, 1, upOut, 1, &act, 1, vDSP_Length(f))
        return DequantInt4GemvRef.apply(weightRows: downRows, x: act, n: f)
    }

    /// Routed-only sibling of `applyStreamed`. Gemma 4's parallel-MoE block
    /// computes dense MLP and routed branches separately, then sums; this is
    /// the routed-only half.
    /// `y = residual + Σ wₑ · routed_e(x)`.
    public static func applyStreamedRouted(
        x: [Float],
        residual: [Float],
        routedGate: [[Quantization.Int4AffineRow]],
        routedUp:   [[Quantization.Int4AffineRow]],
        routedDown: [[Quantization.Int4AffineRow]],
        indices: [Int],
        routingWeights: [Float],
        d: Int,
        f: Int
    ) -> [Float] {
        precondition(indices.count == routingWeights.count)
        precondition(residual.count == d)

        var y = residual
        for slot in 0..<indices.count {
            let e = indices[slot]
            let w = routingWeights[slot]
            let out = runFFN(
                gateRows: routedGate[e],
                upRows:   routedUp[e],
                downRows: routedDown[e],
                x: x, d: d, f: f
            )
            var scale = w
            var scaled = [Float](repeating: 0, count: d)
            vDSP_vsmul(out, 1, &scale, &scaled, 1, vDSP_Length(d))
            vDSP_vadd(y, 1, scaled, 1, &y, 1, vDSP_Length(d))
        }
        return y
    }

}
