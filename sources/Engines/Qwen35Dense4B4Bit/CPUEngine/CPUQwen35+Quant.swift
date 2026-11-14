import Foundation

// Reading one row of a quantized matrix, and the bfloat widening the kernels
// share.
//
// Split out of `CPUQwen35.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion; both helpers widened
// from `private` to internal because their callers stay behind.
extension CPUQwen35 {

    // MARK: - one row of a quantized matrix

    /// Unpacks a single row. Used for the embedding lookup, where reading two
    /// gigabytes to fetch one vector would be absurd.
    func dequantize(row: Int, of matrix: AffineSnapshot.Matrix) -> [Float] {
        let lanes = 32 / matrix.bits
        let mask = UInt32((1 << matrix.bits) - 1)
        let wordsPerRow = matrix.columns / lanes
        let groupsPerRow = matrix.columns / matrix.groupSize
        var out = [Float](repeating: 0, count: matrix.columns)
        // The words are read through the buffer's own lifetime rather than an
        // escaping base address, and an empty weights block leaves `out` zero
        // instead of trapping.
        matrix.weights.withUnsafeBytes { raw in
            guard let words = raw.baseAddress?.assumingMemoryBound(to: UInt32.self) else {
                return
            }
            for word in 0..<wordsPerRow {
                let packed = words[row * wordsPerRow + word]
                for lane in 0..<lanes {
                    let column = word * lanes + lane
                    let level = Float((packed >> (matrix.bits * lane)) & mask)
                    let group = row * groupsPerRow + column / matrix.groupSize
                    out[column] =
                        level * bfloat(matrix.scales[group])
                        + bfloat(matrix.biases[group])
                }
            }
        }
        return out
    }

    @inline(__always)
    func bfloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }
}
