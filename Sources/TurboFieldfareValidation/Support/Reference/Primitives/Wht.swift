import Foundation

public enum WhtRef {
    public static func apply(_ x: [Float]) -> [Float] {
        let d = x.count
        precondition(d > 0 && (d & (d - 1)) == 0, "D must be a power of two")
        if d == 1 { return x }
        return walsh(x)
    }

    private static func walsh(_ x: [Float]) -> [Float] {
        let n = x.count
        if n == 1 { return x }
        let half = n / 2
        let lo = walsh(Array(x[0..<half]))
        let hi = walsh(Array(x[half..<n]))
        let invSqrt2: Float = 1.0 / Float(2.0).squareRoot()
        var y = [Float](repeating: 0, count: n)
        for i in 0..<half {
            y[i] = (lo[i] + hi[i]) * invSqrt2
            y[half + i] = (lo[i] - hi[i]) * invSqrt2
        }
        return y
    }
}
