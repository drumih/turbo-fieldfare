import Foundation
import TinyTitanKernelsC
import TinyTitanCore

/// A quantized matrix multiplied against many token rows at once.
///
/// This is the batched counterpart of `CPUTensorOps.gemv`. The GEMV reads the
/// packed weights in place because one token is one pass over them; a prefill
/// of T tokens would read the same bytes T times, so the weights are
/// unpacked to FP32 once and the matrix engine does the rest. The decode path
/// keeps the GEMV.
///
/// The unpacking is the only new numeric step. It computes the same
/// `q * scale + bias` per group the GEMV kernels factor, in the same order
/// across a row's groups; the multiplication itself reassociates the
/// dot products, which is why a batched result agrees with the per-token one
/// to a rounding error rather than bit for bit.
public enum CPUBatchedMatmul {

    /// `out[t, r] = sum_k x[t, k] * W[r, k]` for T rows of `x`.
    ///
    /// `x` is row-major T-by-`matrix.columns`; the result is row-major
    /// T-by-`matrix.rows`.
    public static func multiply(
        _ matrix: AffineSnapshot.Matrix,
        x: [Float],
        tokenCount: Int,
        threads: Int
    ) throws -> [Float] {
        guard tokenCount > 0 else { return [] }
        guard matrix.rows > 0, matrix.columns > 0 else {
            throw ModelError.internalInconsistency(
                detail: "batched matmul: \(matrix.rows)x\(matrix.columns) matrix")
        }
        guard x.count == tokenCount * matrix.columns else {
            throw ModelError.internalInconsistency(
                detail: "batched matmul: \(x.count) activations for "
                    + "\(tokenCount)x\(matrix.columns)")
        }
        guard matrix.columns % matrix.groupSize == 0 else {
            throw ModelError.internalInconsistency(
                detail: "batched matmul: width \(matrix.columns) is not a whole number "
                    + "of \(matrix.groupSize)-element groups")
        }
        guard matrix.weights.baseAddress != nil else {
            throw ModelError.internalInconsistency(
                detail: "batched matmul: the affine matrix has no packed weights")
        }
        guard matrix.bits == 4 || matrix.bits == 8 else {
            throw ModelError.internalInconsistency(
                detail: "batched matmul: no dequantizer for \(matrix.bits)-bit weights")
        }

        var dequantized = [Float](repeating: 0, count: matrix.rows * matrix.columns)
        dequantized.withUnsafeMutableBufferPointer { buffer in
            guard let out = buffer.baseAddress else { return }
            dequantize(matrix, into: out, threads: threads)
        }

        var result = [Float](repeating: 0, count: tokenCount * matrix.rows)
        let rows = Int32(matrix.rows)
        let columns = Int32(matrix.columns)
        let tokens = Int32(tokenCount)
        x.withUnsafeBufferPointer { xBuffer in
            dequantized.withUnsafeBufferPointer { wBuffer in
                result.withUnsafeMutableBufferPointer { outBuffer in
                    guard let xBase = xBuffer.baseAddress,
                        let wBase = wBuffer.baseAddress,
                        let outBase = outBuffer.baseAddress
                    else { return }
                    // C[T x rows] = X[T x columns] * W^T. W is row-major
                    // rows x columns, so the transposed operand keeps the
                    // same leading dimension as the stored matrix.
                    tinytitan_sgemm_rowmajor(
                        1, tokens, rows, columns, xBase, wBase, outBase)
                }
            }
        }
        return result
    }

    /// Unpack `matrix` into `out` (row-major, rows by columns).
    ///
    /// Rows are split across the performance cores exactly as the GEMV path
    /// splits them; the unpacking of a row touches only that row's bytes, so
    /// there is no shared state between workers.
    static func dequantize(
        _ matrix: AffineSnapshot.Matrix,
        into out: UnsafeMutablePointer<Float>,
        threads: Int
    ) {
        guard let weights = matrix.weights.baseAddress?.assumingMemoryBound(to: UInt8.self)
        else { return }
        let rows = matrix.rows
        let columns = matrix.columns
        let bits = matrix.bits
        let groupsPerRow = columns / matrix.groupSize
        let bytesPerRow = columns * bits / 8
        let scales = matrix.scales
        let biases = matrix.biases

        func unpack(first: Int, count: Int) {
            let w = weights.advanced(by: first * bytesPerRow)
            let s = scales.advanced(by: first * groupsPerRow)
            let b = biases.advanced(by: first * groupsPerRow)
            let o = out.advanced(by: first * columns)
            switch bits {
            case 4:
                tinytitan_dequant_affine4(w, s, b, count, columns, o)
            case 8:
                tinytitan_dequant_affine8(w, s, b, count, columns, o)
            default:
                break
            }
        }

        let usable = max(1, min(threads, rows / 64))
        guard usable > 1 else {
            unpack(first: 0, count: rows)
            return
        }
        let chunk = (rows + usable - 1) / usable
        // The buffers outlive `concurrentPerform`, and each worker writes a
        // disjoint row range, so the shared pointers are not a data race.
        nonisolated(unsafe) let capturedWeights = weights
        nonisolated(unsafe) let capturedScales = scales
        nonisolated(unsafe) let capturedBiases = biases
        nonisolated(unsafe) let capturedOut = out
        DispatchQueue.concurrentPerform(iterations: usable) { slice in
            let first = slice * chunk
            guard first < rows else { return }
            let count = min(chunk, rows - first)
            let w = capturedWeights.advanced(by: first * bytesPerRow)
            let s = capturedScales.advanced(by: first * groupsPerRow)
            let b = capturedBiases.advanced(by: first * groupsPerRow)
            let o = capturedOut.advanced(by: first * columns)
            switch bits {
            case 4:
                tinytitan_dequant_affine4(w, s, b, count, columns, o)
            case 8:
                tinytitan_dequant_affine8(w, s, b, count, columns, o)
            default:
                break
            }
        }
    }
}
