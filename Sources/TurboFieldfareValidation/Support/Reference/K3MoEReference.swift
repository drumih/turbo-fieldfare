import Foundation
import Accelerate
import TurboFieldfare

/// FP32 reference for the K3 fused LatentMoE decode path (per selected
/// expert: fused w1/w3 GEMV + SiTU, then w2 GEMV + weighted reduce), as
/// specified in docs/K3_DATAFLOW.md "MoE block" steps 3-4:
///
///   g, u   = w1.x_lat, w3.x_lat            (MXFP4, FP32)
///   h      = (4·tanh(g/4)·sigmoid(g)) ⊙ (25·tanh(u/25))
///   e      = w2.h
///   y_lat  = Σ_k weight_k · e_k            (FP32 accumulate, no residual)
///
/// Composed from `K3MXFP4Reference.matvec` (bulk dequant + vDSP) and scalar
/// FP32 SiTU — a different code shape from the fused phase1/phase2 kernels.
public enum K3MoEReference {
    /// SiTU-GLU gate half: `4 * tanh(g/4) * sigmoid(g)`; sigmoid reads the
    /// uncapped gate, matching `SituAndMul` (β1=4).
    public static func situGate(_ g: Float) -> Float {
        4 * tanh(g / 4) * (1 / (1 + exp(-g)))
    }

    /// SiTU-GLU up half: `25 * tanh(u/25)` (β2=25).
    public static func situUp(_ u: Float) -> Float {
        25 * tanh(u / 25)
    }

    /// Full SiTU: `situGate(g) * situUp(u)`, FP32.
    public static func situAndMul(g: Float, u: Float) -> Float {
        situGate(g) * situUp(u)
    }

    /// One expert's forward pass. `blob` is the raw packed expert blob;
    /// `offsets` locates the six MXFP4 sub-tensors inside it. Returns the
    /// intermediate activations `h` and the expert output `e = w2.h`.
    public static func expertForward(
        blob: [UInt8],
        offsets: K3ExpertSubtensorOffsets,
        xLat: [Float],
        dLatent: Int,
        intermediate: Int
    ) -> (h: [Float], e: [Float]) {
        precondition(xLat.count == dLatent)
        let w1PackedCount = intermediate * dLatent / 2
        let w1ScalesCount = intermediate * dLatent / 32
        let w2PackedCount = dLatent * intermediate / 2
        let w2ScalesCount = dLatent * intermediate / 32
        let w3PackedCount = w1PackedCount
        let w3ScalesCount = w1ScalesCount
        func slice(_ offset: UInt32, _ count: Int) -> [UInt8] {
            let base = Int(offset)
            precondition(base + count <= blob.count, "sub-tensor overruns blob")
            return Array(blob[base..<(base + count)])
        }
        let g = K3MXFP4Reference.matvec(
            packedWeights: slice(offsets.w1PackedOff, w1PackedCount),
            scales: slice(offsets.w1ScalesOff, w1ScalesCount),
            rows: intermediate, columns: dLatent, vector: xLat)
        let u = K3MXFP4Reference.matvec(
            packedWeights: slice(offsets.w3PackedOff, w3PackedCount),
            scales: slice(offsets.w3ScalesOff, w3ScalesCount),
            rows: intermediate, columns: dLatent, vector: xLat)
        var h = [Float](repeating: 0, count: intermediate)
        for f in 0..<intermediate {
            h[f] = situAndMul(g: g[f], u: u[f])
        }
        let e = K3MXFP4Reference.matvec(
            packedWeights: slice(offsets.w2PackedOff, w2PackedCount),
            scales: slice(offsets.w2ScalesOff, w2ScalesCount),
            rows: dLatent, columns: intermediate, vector: h)
        return (h, e)
    }

    /// Full routed latent path: `y_lat = Σ_k weight_k · expert_k(x_lat)`.
    /// `blobs[k]` is the blob of the k-th selected expert (selection itself
    /// happens in the router); weights are the renormalized router weights.
    public static func apply(
        xLat: [Float],
        blobs: [[UInt8]],
        offsets: K3ExpertSubtensorOffsets,
        routingWeights: [Float],
        dLatent: Int,
        intermediate: Int
    ) -> [Float] {
        precondition(blobs.count == routingWeights.count)
        var yLat = [Float](repeating: 0, count: dLatent)
        for (blob, weight) in zip(blobs, routingWeights) {
            let (_, e) = expertForward(
                blob: blob, offsets: offsets,
                xLat: xLat, dLatent: dLatent, intermediate: intermediate)
            var w = weight
            e.withUnsafeBufferPointer { pe in
                yLat.withUnsafeMutableBufferPointer { py in
                    vDSP_vsma(pe.baseAddress!, 1, &w,
                              py.baseAddress!, 1, py.baseAddress!, 1,
                              vDSP_Length(dLatent))
                }
            }
        }
        return yLat
    }
}
