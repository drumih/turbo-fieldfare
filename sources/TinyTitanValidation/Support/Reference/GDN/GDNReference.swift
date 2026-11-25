import Foundation
import TinyTitanCore

/// CPU reference for the gated-DeltaNet kernels.
///
/// Lifted out of GDNKernelTests so it sits with the other nine references and
/// is reviewed alongside them. GDN is the most novel maths in Qwen 3.6 and its
/// reference was previously the least visible one in the suite.

public struct GDNReference {
    /// Kept alongside the reference rather than in the test file: they are part
    /// of the definition of what the kernel is supposed to compute.
    public static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    public static func softplus(_ x: Float) -> Float {
        x > 20 ? x : logf(1 + expf(x))
    }

    public let cfg: LinearAttentionConfig
    public let convW: [Float]  // [C, K], bf16-representable
    public let aLog: [Float]  // [Hv]
    public let dtBias: [Float]  // [Hv]
    public let normW: [Float]  // [Dv]

    public var tail: [[Float]]  // K-1 rows of C
    public var state: [Float]  // [Hv, Dv, Dk]

    public init(
        cfg: LinearAttentionConfig, convW: [Float], aLog: [Float],
        dtBias: [Float], normW: [Float]
    ) {
        self.cfg = cfg
        self.convW = convW
        self.aLog = aLog
        self.dtBias = dtBias
        self.normW = normW
        self.tail = Array(
            repeating: [Float](repeating: 0, count: cfg.qkvDim),
            count: cfg.convKernelSize - 1)
        self.state = [Float](
            repeating: 0,
            count: cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim)
    }

    public mutating func step(
        qkvRaw: [Float], a: [Float], b: [Float],
        z: [Float]
    ) -> [Float] {
        let C = cfg.qkvDim
        let K = cfg.convKernelSize
        let Hk = cfg.numKHeads
        let Hv = cfg.numVHeads
        let Dk = cfg.keyHeadDim
        let Dv = cfg.valueHeadDim

        // Conv + SiLU (fp32; row order [tail..., current]).
        var conv = [Float](repeating: 0, count: C)
        for ch in 0..<C {
            var acc = qkvRaw[ch] * convW[ch * K + (K - 1)]
            for j in 0..<(K - 1) {
                acc += tail[j][ch] * convW[ch * K + j]
            }
            conv[ch] = Float(Float16(GDNReference.silu(acc)))
        }
        tail.removeFirst()
        tail.append(qkvRaw.map { Float(Float16($0)) })

        // q/k norm with folded scales (per key head).
        var normed = conv
        for headIndex in 0..<(2 * Hk) {
            let isQ = headIndex < Hk
            let head = isQ ? headIndex : headIndex - Hk
            let base = (isQ ? 0 : Hk * Dk) + head * Dk
            var sumsq: Float = 0
            for i in 0..<Dk { sumsq += conv[base + i] * conv[base + i] }
            let invRms = 1 / sqrtf(sumsq / Float(Dk) + 1e-6)
            let scale = isQ ? (1 / Float(Dk)) : (1 / sqrtf(Float(Dk)))
            for i in 0..<Dk {
                normed[base + i] = Float(Float16(conv[base + i] * invRms * scale))
            }
        }

        // Delta recurrence per value head.
        var y = [Float](repeating: 0, count: Hv * Dv)
        for h in 0..<Hv {
            let hk = h / (Hv / Hk)
            let qBase = hk * Dk
            let kBase = Hk * Dk + hk * Dk
            let vBase = 2 * Hk * Dk + h * Dv
            let g = expf(-expf(aLog[h]) * GDNReference.softplus(a[h] + dtBias[h]))
            let beta = 1 / (1 + expf(-b[h]))
            for dv in 0..<Dv {
                let srow = (h * Dv + dv) * Dk
                var kv: Float = 0
                for i in 0..<Dk {
                    state[srow + i] *= g
                    kv += state[srow + i] * normed[kBase + i]
                }
                let delta = (normed[vBase + dv] - kv) * beta
                var out: Float = 0
                for i in 0..<Dk {
                    state[srow + i] += normed[kBase + i] * delta
                    out += state[srow + i] * normed[qBase + i]
                }
                y[h * Dv + dv] = Float(Float16(out))
            }
        }

        // Gated output norm.
        var gated = [Float](repeating: 0, count: Hv * Dv)
        for h in 0..<Hv {
            let base = h * Dv
            var sumsq: Float = 0
            for i in 0..<Dv { sumsq += y[base + i] * y[base + i] }
            let invRms = 1 / sqrtf(sumsq / Float(Dv) + 1e-6)
            for i in 0..<Dv {
                let normedY = y[base + i] * invRms * normW[i]
                gated[base + i] = normedY * GDNReference.silu(z[base + i])
            }
        }
        return gated
    }
}
