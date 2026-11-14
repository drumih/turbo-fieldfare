import Accelerate
import Foundation
import TinyTitanKernelsC
import TinyTitanCore

/// The scalar arithmetic between the GEMVs.
///
/// Every one of these is small — a norm over 2048 values, a softmax over the
/// keys, a gate over 2048 — and together they are a rounding error next to
/// the weight reads. So they are written for clarity and checked against the
/// numpy reference, not tuned. The one place that is not true is the GEMV
/// itself, which is a NEON kernel and is where the time actually goes.
public enum CPUOps {

    @inline(__always)
    public static func silu(_ x: Float) -> Float {
        x / (1 + expf(-x))
    }

    /// `x / (1 + exp(-x))` for a whole buffer, through vForce.
    ///
    /// The scalar `silu` is fine for a gate vector in decode; a prefill calls
    /// it once per channel per token, which is tens of millions of `expf`
    /// calls for a long prompt. `scratch` is the caller's buffer of the same
    /// length, reused across rows so this allocates nothing per token.
    public static func siluInPlace(
        _ values: UnsafeMutableBufferPointer<Float>,
        scratch: UnsafeMutableBufferPointer<Float>
    ) {
        let count = values.count
        guard count > 0, scratch.count >= count,
            let input = values.baseAddress, let temp = scratch.baseAddress
        else { return }
        var n = Int32(count)
        var one: Float = 1
        vDSP_vneg(input, 1, temp, 1, vDSP_Length(count))
        vvexpf(temp, temp, &n)
        vDSP_vsadd(temp, 1, &one, temp, 1, vDSP_Length(count))
        vvrecf(temp, temp, &n)
        vDSP_vmul(input, 1, temp, 1, input, 1, vDSP_Length(count))
    }

    @inline(__always)
    public static func sigmoid(_ x: Float) -> Float {
        1 / (1 + expf(-x))
    }

    /// `log(1 + e^x)`, without overflowing for large `x`.
    @inline(__always)
    public static func softplus(_ x: Float) -> Float {
        log1pf(expf(-abs(x))) + max(x, 0)
    }

    /// RMS norm over `values`, scaled by `gamma`.
    ///
    /// The checkpoint stores every norm but one as an offset from one, and
    /// the converter folds the +1 in — so `gamma` here is the real scale and
    /// this must not add anything. The exception is the gated linear
    /// attention norm, which is stored directly and also arrives here ready
    /// to use.
    public static func rmsNorm(_ values: inout [Float], gamma: [Float], epsilon: Float) {
        var sum: Float = 0
        for value in values { sum += value * value }
        let scale = 1 / (sum / Float(values.count) + epsilon).squareRoot()
        for index in values.indices { values[index] = values[index] * scale * gamma[index] }
    }

    /// RMS norm over each `width`-long slice independently, which is how a
    /// per-head norm works: the statistics are the head's, not the layer's.
    public static func rmsNormPerSlice(
        _ values: inout [Float], width: Int,
        gamma: [Float], epsilon: Float
    ) {
        let slices = values.count / width
        for slice in 0..<slices {
            let base = slice * width
            var sum: Float = 0
            for index in 0..<width { sum += values[base + index] * values[base + index] }
            let scale = 1 / (sum / Float(width) + epsilon).squareRoot()
            for index in 0..<width {
                values[base + index] = values[base + index] * scale * gamma[index]
            }
        }
    }

    /// L2-normalize each `width`-long slice. The delta rule wants unit keys
    /// and queries; without this the recurrent state's magnitude drifts with
    /// the input's.
    public static func l2NormalizePerSlice(
        _ values: inout [Float], width: Int,
        epsilon: Float
    ) {
        let slices = values.count / width
        for slice in 0..<slices {
            let base = slice * width
            var sum: Float = 0
            for index in 0..<width { sum += values[base + index] * values[base + index] }
            let scale = 1 / (sum + epsilon).squareRoot()
            for index in 0..<width { values[base + index] *= scale }
        }
    }

    public static func softmaxInPlace(_ values: inout [Float]) {
        guard let peak = values.max() else { return }
        var total: Float = 0
        for index in values.indices {
            let value = expf(values[index] - peak)
            values[index] = value
            total += value
        }
        let inverse = 1 / total
        for index in values.indices { values[index] *= inverse }
    }

    /// NeoX rotation over the first `rotaryDim` dimensions of each head.
    ///
    /// Partial: 64 of 256 for this model. The pairs are `(i, i + rotaryDim/2)`
    /// — a half-split within the rotated prefix, not adjacent elements, and
    /// not the whole head.
    public static func applyRoPE(
        _ values: inout [Float], headDim: Int,
        rotaryDim: Int, position: Int, theta: Float
    ) {
        let half = rotaryDim / 2
        let heads = values.count / headDim
        for head in 0..<heads {
            let base = head * headDim
            for index in 0..<half {
                let frequency = powf(theta, -Float(index) * 2 / Float(rotaryDim))
                let angle = Float(position) * frequency
                let cosine = cosf(angle)
                let sine = sinf(angle)
                let a = values[base + index]
                let b = values[base + half + index]
                values[base + index] = a * cosine - b * sine
                values[base + half + index] = a * sine + b * cosine
            }
        }
    }

    /// `out = W · x` for an affine matrix of either width, threaded over rows.
    ///
    /// Both widths appear in one snapshot: the 4-bit build keeps the tied
    /// embedding and the attention K/V at 8 bits, because that is where the
    /// measured error was.
    public static func gemv(
        _ matrix: AffineSnapshot.Matrix,
        x: UnsafePointer<Float>,
        out: UnsafeMutablePointer<Float>,
        threads: Int
    ) throws {
        // The kernel's own row stride and group count both come from
        // `columns`, so a width that is not a whole number of groups mis-strides
        // every row after the first while still returning a full-length result.
        Int8AffineGEMV.requireWholeGroups(matrix.columns, "CPUTensorOps.gemv")
        guard let weights = matrix.weights.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
            throw ModelError.internalInconsistency(
                detail: "the affine matrix has no packed weights to multiply")
        }
        let bytesPerRow = matrix.columns * matrix.bits / 8
        let groupsPerRow = matrix.columns / matrix.groupSize
        let kernel:
            (
                UnsafePointer<UInt8>, UnsafePointer<UInt16>, UnsafePointer<UInt16>,
                Int, UnsafeMutablePointer<Float>
            ) -> Void
        switch matrix.bits {
        case 8:
            kernel = { w, s, b, rows, o in
                tinytitan_int8_affine_gemv(w, s, b, x, rows, matrix.columns, o)
            }
        case 4:
            kernel = { w, s, b, rows, o in
                tinytitan_int4_affine_gemv(w, s, b, x, rows, matrix.columns, o)
            }
        default:
            // A snapshot with a width no kernel implements must not silently
            // produce zeros; there is no correct answer to give here.
            preconditionFailure("no CPU kernel for \(matrix.bits)-bit weights")
        }

        // Rows are independent, so a worker advances every pointer together
        // and writes a disjoint slice of `out`. Nothing accumulates across
        // workers, so there is no reduction order to get wrong and the
        // threaded result is bit-identical to the single-threaded one.
        //
        // Splitting *columns* instead would read each row twice, and decode
        // here is bound by exactly those reads.
        let usable = max(1, min(threads, matrix.rows / Int8AffineGEMV.minimumRowsPerThread))
        guard usable > 1 else {
            kernel(weights, matrix.scales, matrix.biases, matrix.rows, out)
            return
        }
        let chunk = (matrix.rows + usable - 1) / usable
        // `concurrentPerform` returns only after every iteration has finished,
        // so the buffers and the kernel outlive the closure; Swift cannot see
        // that through the call and neither closures nor unsafe pointers are
        // `Sendable`, which is what these bindings record. Workers write
        // disjoint row ranges, so no byte of `out` is shared.
        nonisolated(unsafe) let capturedKernel = kernel
        nonisolated(unsafe) let capturedWeights = weights
        nonisolated(unsafe) let capturedOut = out
        DispatchQueue.concurrentPerform(iterations: usable) { slice in
            let first = slice * chunk
            guard first < matrix.rows else { return }
            let count = min(chunk, matrix.rows - first)
            capturedKernel(
                capturedWeights + first * bytesPerRow,
                matrix.scales + first * groupsPerRow,
                matrix.biases + first * groupsPerRow,
                count, capturedOut + first)
        }
    }
}
