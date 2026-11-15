import Foundation
import TinyTitanKernelsC

/// CPU evaluation of one routed expert, reading the packed `.ssdai` expert
/// block directly.
///
/// This exists because decode leaves most of the machine unused: the GPU is
/// idle over half of every token, seven of eight cores are idle while the
/// orchestrator blocks on GPU completion, and roughly half the memory bus is
/// unclaimed. Measured on the development machine, a 6-thread NEON streaming
/// read sustains 45-60 GB/s *during* inference without slowing the GPU, and an
/// int4 dequant+GEMV of one layer's eight active experts (13.5 MiB) completes
/// in 0.773 ms against a 1.7 ms per-layer budget. So a share of the routed
/// experts can be computed on the CPU inside time the GPU is idle anyway.
///
/// The arithmetic mirrors `moe.metal`: weights are affine INT4 with a BF16
/// scale and bias per group of 64, dequantised as `w = q * scale + bias`, and
/// the dot product is factored the same way the shader factors it --
/// `acc += scale * Σ(q·x) + bias * Σ(x)` -- so the two paths agree on more
/// than just the algebra. It does not agree bit-for-bit: accumulation order
/// differs, so a layer computed here will not reproduce the GPU's last-bit
/// rounding. See `docs/cpu-coexecution-plan.md` for what that means for the
/// golden baseline.
public enum CPUExpertFFN {
    public static let groupSize = 64

    /// Byte offsets within one expert's packed block.
    ///
    /// The packer emits a fixed order -- gate, gate_scales, gate_biases, up,
    /// up_scales, up_biases, down, down_scales, down_biases -- with every
    /// tensor contiguous, so the offsets follow from the shapes alone rather
    /// than needing `layout.json` at read time.
    public struct Offsets: Equatable, Sendable {
        public let gate: Int, gateScales: Int, gateBiases: Int
        public let up: Int, upScales: Int, upBiases: Int
        public let down: Int, downScales: Int, downBiases: Int
        public let strideBytes: Int

        /// - Parameters:
        ///   - d: hidden size (gate/up input, down output)
        ///   - f: MoE intermediate size (gate/up output, down input)
        ///   - weightBits: width the packed weights use. The divisors below are
        ///     what encode it, so it is a parameter rather than a comment: a
        ///     caller laying out 8-bit weights with the old `/ 2` would get a
        ///     block half the size it needs, and every offset after the first
        ///     wrong, with the shapes still plausible.
        public init(d: Int, f: Int, weightBits: Int = 4) {
            precondition(
                [4, 8].contains(weightBits),
                "expert layout supports 4- and 8-bit weights, not \(weightBits)")
            precondition(
                d % groupSize == 0 && f % groupSize == 0,
                "d=\(d) and f=\(f) must be multiples of \(groupSize)")
            let gateWeights = f * d * weightBits / 8
            let gateMeta = f * (d / groupSize) * 2  // BF16
            let downWeights = d * f * weightBits / 8
            let downMeta = d * (f / groupSize) * 2
            gate = 0
            gateScales = gate + gateWeights
            gateBiases = gateScales + gateMeta
            up = gateBiases + gateMeta
            upScales = up + gateWeights
            upBiases = upScales + gateMeta
            down = upBiases + gateMeta
            downScales = down + downWeights
            downBiases = downScales + downMeta
            strideBytes = downBiases + downMeta
        }
    }

    /// Floats a caller must provide as `scratch`: gate, up, and down outputs.
    public static func scratchFloats(d: Int, f: Int) -> Int { 2 * f + d }

    @inline(__always)
    private static func bf16(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    /// `y = W · x` for an affine-INT4 `rows`-by-`n` matrix.
    ///
    /// Factored as the shader factors it so the two implementations make the
    /// same rounding choices at the group level: one scale multiply and one
    /// bias multiply per group of 64, rather than dequantising each weight.
    ///
    /// The body lives in `TinyTitanKernelsC` because Swift's vector types do not
    /// lower well here: an equivalent `SIMD8<Float>` version ran 2.3x slower
    /// (1.19 ms vs 0.49 ms per production expert), spending the difference on
    /// scalar inserts to build each vector rather than on arithmetic.
    @inline(__always)
    private static func gemv(
        weights: UnsafePointer<UInt8>,
        scales: UnsafePointer<UInt16>,
        biases: UnsafePointer<UInt16>,
        x: UnsafePointer<Float>,
        rows: Int,
        n: Int,
        out: UnsafeMutablePointer<Float>
    ) {
        // Same contract as the standalone wrapper: the kernel's row stride and
        // its scale/bias groups are both derived from `n`, so a width that is not
        // a whole number of 64-element groups mis-strides every row after the
        // first. This one hardcodes the 4-bit packing, so a width check is the
        // only thing standing between a caller and silently wrong output.
        Int8AffineGEMV.requireWholeGroups(n, "CPUExpertFFN.gemv")
        tinytitan_int4_affine_gemv(weights, scales, biases, x, rows, n, out)
    }

    @inline(__always)
    private static func silu(_ v: Float) -> Float {
        v / (1 + Foundation.exp(-v))
    }

    /// Accumulates `routeWeight * down(silu(gate·x) * (up·x))` into `out`.
    ///
    /// `expert` points at the start of this expert's block; `out` and `x` are
    /// both length `d`. `scratch` must hold at least `2 * f` floats and is
    /// caller-owned so a worker can reuse one allocation across experts.
    public static func accumulate(
        expert: UnsafeRawPointer,
        offsets: Offsets,
        x: UnsafePointer<Float>,
        d: Int,
        f: Int,
        routeWeight: Float,
        scratch: UnsafeMutablePointer<Float>,
        out: UnsafeMutablePointer<Float>
    ) {
        let base = expert.assumingMemoryBound(to: UInt8.self)
        let gateOut = scratch
        let upOut = scratch + f

        gemv(
            weights: base + offsets.gate,
            scales: (expert + offsets.gateScales).assumingMemoryBound(to: UInt16.self),
            biases: (expert + offsets.gateBiases).assumingMemoryBound(to: UInt16.self),
            x: x, rows: f, n: d, out: gateOut)
        gemv(
            weights: base + offsets.up,
            scales: (expert + offsets.upScales).assumingMemoryBound(to: UInt16.self),
            biases: (expert + offsets.upBiases).assumingMemoryBound(to: UInt16.self),
            x: x, rows: f, n: d, out: upOut)

        for i in 0..<f {
            gateOut[i] = silu(gateOut[i]) * upOut[i]
        }

        // `down` goes through the same gemv as gate and up. An earlier version
        // fused the routing weight into a bespoke scalar loop here to avoid
        // this temporary; replacing it changed throughput by under 6% either
        // way, so this keeps the single code path rather than the micro-
        // optimisation. The cost is elsewhere -- see the note on `gemv`.
        let downOut = scratch + 2 * f
        gemv(
            weights: base + offsets.down,
            scales: (expert + offsets.downScales).assumingMemoryBound(to: UInt16.self),
            biases: (expert + offsets.downBiases).assumingMemoryBound(to: UInt16.self),
            x: gateOut, rows: d, n: f, out: downOut)
        for r in 0..<d {
            out[r] += routeWeight * downOut[r]
        }
    }
}
