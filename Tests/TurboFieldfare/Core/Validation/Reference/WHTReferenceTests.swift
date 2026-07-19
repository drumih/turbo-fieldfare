import Testing
import Foundation
import TurboFieldfareValidationSupport

@Suite struct WHTReferenceTests {
    /// Independent iterative WHT — different control flow from recursive ref.
    private static func iterativeWht(_ x: [Float]) -> [Float] {
        let d = x.count
        var y = x
        let invSqrt2: Float = 1.0 / Float(2.0).squareRoot()
        var stride = 1
        while stride < d {
            var i = 0
            while i < d {
                for k in 0..<stride {
                    let a = y[i + k]
                    let b = y[i + k + stride]
                    y[i + k]          = (a + b) * invSqrt2
                    y[i + k + stride] = (a - b) * invSqrt2
                }
                i += stride * 2
            }
            stride *= 2
        }
        return y
    }

    @Test("Recursive WHT matches iterative", arguments: [128, 256, 512])
    func recursiveMatchesIterative(d: Int) {
        var rng = SeedTree(0x1A1).key("wht-ref-d\(d)")
        let x = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let rec = WhtRef.apply(x)
        let itr = Self.iterativeWht(x)
        let rel = RelError.compute(actual: rec, reference: itr)
        #expect(rel < Tolerance.identity, "D=\(d) rel=\(rel)")
    }

    @Test("WHT is an involution", arguments: [128, 256, 512])
    func involution(d: Int) {
        var rng = SeedTree(0x1A2).key("wht-involution-d\(d)")
        let x = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let twice = WhtRef.apply(WhtRef.apply(x))
        let rel = RelError.compute(actual: twice, reference: x)
        #expect(rel < Tolerance.identity, "D=\(d) round-trip rel=\(rel)")
    }
}
