import Foundation

/// Error-measurement helpers. Every test computed these inline; centralize.
public enum RelError {
    /// Standard relative error used across the test suite:
    ///   max_i |actual[i] - reference[i]|  /  max(max_i |reference[i]|, 1e-6)
    public static func compute(actual: [Float], reference: [Float]) -> Float {
        precondition(actual.count == reference.count, "length mismatch")
        var maxAbsDiff: Float = 0
        var refNorm: Float = 0
        for i in 0..<actual.count {
            maxAbsDiff = max(maxAbsDiff, abs(actual[i] - reference[i]))
            refNorm = max(refNorm, abs(reference[i]))
        }
        return maxAbsDiff / max(refNorm, 1e-6)
    }

    public static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "length mismatch")
        var m: Float = 0
        for i in 0..<a.count {
            m = max(m, abs(a[i] - b[i]))
        }
        return m
    }

    /// Refines `compute` with both a relative and an absolute floor.
    /// Used when very small reference values would inflate relative error
    /// past the meaningful FP16 noise floor.
    public static func boundedRel(
        actual: [Float],
        reference: [Float],
        absFloor: Float
    ) -> Float {
        precondition(actual.count == reference.count, "length mismatch")
        var worst: Float = 0
        for i in 0..<actual.count {
            let diff = abs(actual[i] - reference[i])
            let denom = max(abs(reference[i]), absFloor)
            worst = max(worst, diff / denom)
        }
        return worst
    }
}
