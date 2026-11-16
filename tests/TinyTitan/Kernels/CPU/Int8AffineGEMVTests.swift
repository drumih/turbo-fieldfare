import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The decode inner loop of the CPU side-engine.
///
/// The kernel walks packed bytes and group scales itself; the reference
/// dequantises each row through `Quantization.dequantizeInt8Affine` and does
/// a plain dot product. Driving both from the *same* quantised weights means
/// agreement is evidence about the byte walk and the arithmetic, not about
/// the quantiser — which is the property that made the 4-bit kernel's tests
/// worth having.
@Suite struct Int8AffineGEMVTests {

    private static func pseudorandom(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(bitPattern: UInt32(truncatingIfNeeded: state >> 33)))
                / Float(Int32.max)
        }
    }

    /// Rows laid out as the snapshot lays them: weights row-major, then all
    /// scales row-major, then all biases.
    private static func run(rows rowValues: [[Float]], x: [Float]) -> [Float] {
        let n = x.count
        let quantised = rowValues.map { Quantization.quantizeInt8Affine($0) }
        var weights: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for row in quantised {
            weights.append(contentsOf: row.packed)
            scales.append(contentsOf: row.scales)
            biases.append(contentsOf: row.biases)
        }
        var out = [Float](repeating: .nan, count: rowValues.count)
        weights.withUnsafeBufferPointer { w in
            scales.withUnsafeBufferPointer { s in
                biases.withUnsafeBufferPointer { b in
                    x.withUnsafeBufferPointer { xp in
                        out.withUnsafeMutableBufferPointer { o in
                            // The arrays are non-empty by construction; an empty
                            // one would leave the output untouched rather than
                            // trapping inside the kernel call.
                            guard let wBase = w.baseAddress, let sBase = s.baseAddress,
                                let bBase = b.baseAddress, let xpBase = xp.baseAddress,
                                let oBase = o.baseAddress
                            else { return }
                            Int8AffineGEMV.apply(
                                weights: wBase, scales: sBase,
                                biases: bBase, x: xpBase,
                                rows: rowValues.count, n: n, out: oBase)
                        }
                    }
                }
            }
        }
        return out
    }

    private static func reference(rows rowValues: [[Float]], x: [Float]) -> [Float] {
        rowValues.map { row in
            let dequantised = Quantization.dequantizeInt8Affine(
                Quantization.quantizeInt8Affine(row), n: row.count)
            return zip(dequantised, x).reduce(0) { $0 + $1.0 * $1.1 }
        }
    }

    /// One group: the narrowest shape the format allows, so an off-by-one in
    /// the group stride has nowhere to hide.
    @Test func singleGroupMatchesTheReference() {
        let x = Self.pseudorandom(64, seed: 11)
        let rows = (0..<3).map { Self.pseudorandom(64, seed: UInt64(100 + $0)) }
        let got = Self.run(rows: rows, x: x)
        let want = Self.reference(rows: rows, x: x)
        for (a, b) in zip(got, want) {
            #expect(abs(a - b) <= 1e-3 * max(1, abs(b)))
        }
    }

    /// A production shape: 2048 wide is the attention and gate/up projection,
    /// 6144 the feed-forward's inner dimension.
    @Test func productionWidthsMatchTheReference() {
        for n in [2048, 6144] {
            let x = Self.pseudorandom(n, seed: 7)
            let rows = (0..<5).map { Self.pseudorandom(n, seed: UInt64(n + $0)) }
            let got = Self.run(rows: rows, x: x)
            let want = Self.reference(rows: rows, x: x)
            for (index, (a, b)) in zip(got, want).enumerated() {
                #expect(
                    abs(a - b) <= 1e-3 * max(1, abs(b)),
                    "row \(index) at n=\(n): \(a) vs \(b)")
            }
        }
    }

    /// A constant row quantises to scale 1, bias = value, and must come back
    /// as `value * sum(x)` exactly — the case where the bias term carries the
    /// whole answer and the quantised term contributes nothing.
    @Test func constantRowIsCarriedByTheBiasTerm() {
        let x = Self.pseudorandom(128, seed: 3)
        let rows = [[Float](repeating: 0.375, count: 128)]
        let got = Self.run(rows: rows, x: x)
        let want = 0.375 * x.reduce(0, +)
        #expect(abs(got[0] - want) <= 1e-3 * max(1, abs(want)))
    }

    /// One row block at one width, laid out exactly as a snapshot lays it,
    /// through `CPUOps.gemv` — the wrapper the engine itself calls.
    ///
    /// Scale is BF16 1.0 and, unless `biasBits` says otherwise, bias BF16 0.0;
    /// every lane is 0...15, so both widths multiply exactly the same products
    /// and any difference between the two results can only have come from the
    /// order they were summed in. A non-zero bias is the only way to make the
    /// kernel's per-group `sum(x)` term affect the row at all.
    private static func gemvAtWidth(
        bits: Int,
        lanes: [UInt8],
        x: [Float],
        rows: Int,
        columns: Int,
        biasBits: UInt16 = 0
    ) -> [Float] {
        let groups = columns / Quantization.groupSize
        let bytesPerRow = columns * bits / 8
        var weights = [UInt8](repeating: 0, count: rows * bytesPerRow)
        if bits == 4 {
            for row in 0..<rows {
                for index in 0..<columns {
                    let value = lanes[row * columns + index]
                    let byte = row * bytesPerRow + index / 2
                    weights[byte] = index % 2 == 0 ? value : weights[byte] | (value << 4)
                }
            }
        } else {
            weights = Array(lanes)
        }
        let scales = [UInt16](repeating: 0x3F80, count: rows * groups)
        let biases = [UInt16](repeating: biasBits, count: rows * groups)
        var out = [Float](repeating: .nan, count: rows)
        weights.withUnsafeBytes { raw in
            scales.withUnsafeBufferPointer { s in
                biases.withUnsafeBufferPointer { b in
                    x.withUnsafeBufferPointer { xp in
                        out.withUnsafeMutableBufferPointer { o in
                            guard let sBase = s.baseAddress, let bBase = b.baseAddress,
                                let xpBase = xp.baseAddress, let oBase = o.baseAddress
                            else { return }
                            let matrix = AffineSnapshot.Matrix(
                                weights: raw, scales: sBase, biases: bBase,
                                rows: rows, columns: columns, bits: bits,
                                groupSize: Quantization.groupSize)
                            do {
                                try CPUOps.gemv(
                                    matrix, x: xpBase, out: oBase, threads: 1)
                            } catch {
                                Issue.record("gemv at \(bits)-bit failed: \(error)")
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// Same products, both widths, same wrapper: the two agree to within a
    /// rounding error of the row's total magnitude, and *not* bit for bit.
    ///
    /// The kernels say otherwise. `tinytitan_kernels.h` claims both "round
    /// alike at group boundaries", and `int8_affine_gemv.c` that "the 4-bit
    /// kernel's single accumulator and this one's four reduce to the same
    /// value for the same inputs". Same factoring is not the same reduction
    /// order -- the 4-bit path chains sixteen adds per lane, the 8-bit path
    /// keeps four accumulators and combines them as a tree -- so the last bit
    /// goes different ways. Measured here: 955.4322 against 955.43225; three
    /// randomized probes over 560k rows found 64.2%, 64.5% and 68.9% of them
    /// differing in the last bits, while both stayed inside 2.6e-3 relative of
    /// a double-precision reference, so neither is wrong -- only differently
    /// ordered.
    ///
    /// Nothing depends on the stronger claim: the side-engine is validated
    /// against the numpy oracle (`tools/qwen35_reference.py`), never
    /// bit-for-bit against the GPU or against the other width.
    @Test func bothWidthsAgreeToWithinARoundingErrorNotBitForBit() {
        let columns = 256
        let rows = 3
        let lanes: [UInt8] = (0..<(rows * columns)).map { UInt8($0 % 16) }
        let x = Self.pseudorandom(columns, seed: 5)
        let four = Self.gemvAtWidth(bits: 4, lanes: lanes, x: x, rows: rows, columns: columns)
        let eight = Self.gemvAtWidth(bits: 8, lanes: lanes, x: x, rows: rows, columns: columns)
        // fp32 roundings cannot be bounded against a result that cancels, so
        // each row is bounded by its own total absolute magnitude.
        for index in 0..<rows {
            var total = 0.0
            for column in 0..<columns {
                total += Double(lanes[index * columns + column]) * abs(Double(x[column]))
            }
            let tolerance = Float(1e-6 * total)
            let gap = abs(four[index] - eight[index])
            #expect(
                gap <= tolerance,
                "row \(index): gap \(gap) over \(tolerance) (\(four[index]) vs \(eight[index]))")
        }
    }

    /// Both kernels hoist `sum(x)` per group into a fixed 256-entry stack
    /// table, and a row wider than that is supposed to fall back to
    /// recomputing it inside the group loop rather than read past the table:
    /// `int4_affine_gemv.c:18` promises exactly that ("wider rows fall back to
    /// the per-row path rather than being wrong"). No shipping tensor is that
    /// wide — 256 groups is an input width of 16384, past every install here —
    /// so the branch was measured only by a scratch C probe until this test.
    ///
    /// 257 groups, both widths, against the same products summed in double.
    /// The bias is BF16 0.5 rather than 0: with a zero bias the `sum(x)` term
    /// multiplies out to nothing and the test could not see the fallback at
    /// all, however wrong it was.
    @Test func rowsWiderThanTheHoistedSumTableMatchTheReference() {
        let columns = 257 * 64
        let rows = 2
        let bias: Double = 0.5
        let lanes: [UInt8] = (0..<(rows * columns)).map { UInt8($0 % 16) }
        let x = Self.pseudorandom(columns, seed: 7)
        for bits in [4, 8] {
            let kernel = Self.gemvAtWidth(
                bits: bits, lanes: lanes, x: x, rows: rows, columns: columns,
                biasBits: 0x3F00)
            for index in 0..<rows {
                var reference = 0.0
                var total = 0.0
                for column in 0..<columns {
                    let product = Double(lanes[index * columns + column]) * Double(x[column])
                    let biasTerm = bias * Double(x[column])
                    reference += product + biasTerm
                    total += abs(product) + abs(biasTerm)
                }
                let gap = abs(kernel[index] - Float(reference))
                let tolerance = Float(1e-6 * total)
                let message = "\(bits)-bit row \(index): \(kernel[index]) vs \(reference)"
                #expect(gap <= tolerance, "gap \(gap) over \(tolerance): \(message)")
            }
        }
    }

    /// Rows are independent, so a caller may thread over row ranges by
    /// advancing every pointer together. This pins that contract: the same
    /// weights split into two calls give the same answer as one call.
    @Test func rowRangesAreIndependent() {
        let n = 256
        let x = Self.pseudorandom(n, seed: 21)
        let rows = (0..<8).map { Self.pseudorandom(n, seed: UInt64(500 + $0)) }
        let whole = Self.run(rows: rows, x: x)
        let first = Self.run(rows: Array(rows[0..<3]), x: x)
        let rest = Self.run(rows: Array(rows[3...]), x: x)
        for (a, b) in zip(whole, first + rest) {
            #expect(a == b)
        }
    }

    /// The 8-bit kernel against the 4-bit *reference*: values a 4-bit lane
    /// holds exactly, dequantised at 4 bits and multiplied in Swift, which the
    /// 8-bit kernel must reproduce to within a rounding error. This is a
    /// kernel-versus-oracle check; the kernel-versus-kernel one is
    /// ``bothWidthsAgreeToWithinARoundingErrorNotBitForBit``, and it passes at
    /// tolerance rather than bitwise, because the two widths sum a group in
    /// different orders.
    @Test func matchesTheFourBitReferenceOnRepresentableValues() {
        let n = 64
        let row = (0..<n).map { Float($0 % 16) }  // exact 4-bit levels
        let x = Self.pseudorandom(n, seed: 5)
        let eight = Self.run(rows: [row], x: x)[0]
        let four = Quantization.dequantizeInt4Affine(
            Quantization.quantizeInt4Affine(row), n: n)
        let want = zip(four, x).reduce(0) { $0 + $1.0 * $1.1 }
        #expect(
            abs(eight - want) <= 1e-3 * max(1, abs(want)),
            "8-bit \(eight) vs 4-bit reference \(want)")
    }

    /// Threading splits rows, so its result must be bit-identical to the
    /// single-threaded one -- there is no cross-worker accumulation whose
    /// order could differ. Asserted rather than assumed.
    @Test func threadedMatchesSingleThreadedExactly() {
        let n = 512
        let x = Self.pseudorandom(n, seed: 31)
        let rowValues = (0..<600).map { Self.pseudorandom(n, seed: UInt64(900 + $0)) }
        let quantised = rowValues.map { Quantization.quantizeInt8Affine($0) }
        var weights: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for row in quantised {
            weights.append(contentsOf: row.packed)
            scales.append(contentsOf: row.scales)
            biases.append(contentsOf: row.biases)
        }
        var one = [Float](repeating: .nan, count: rowValues.count)
        var many = [Float](repeating: .nan, count: rowValues.count)
        weights.withUnsafeBufferPointer { w in
            scales.withUnsafeBufferPointer { s in
                biases.withUnsafeBufferPointer { b in
                    x.withUnsafeBufferPointer { xp in
                        guard let wBase = w.baseAddress, let sBase = s.baseAddress,
                            let bBase = b.baseAddress, let xpBase = xp.baseAddress
                        else {
                            return
                        }
                        one.withUnsafeMutableBufferPointer { o in
                            guard let oBase = o.baseAddress else { return }
                            Int8AffineGEMV.apply(
                                weights: wBase, scales: sBase,
                                biases: bBase, x: xpBase,
                                rows: rowValues.count, n: n, out: oBase)
                        }
                        many.withUnsafeMutableBufferPointer { o in
                            guard let oBase = o.baseAddress else { return }
                            Int8AffineGEMV.threaded(
                                weights: wBase, scales: sBase,
                                biases: bBase, x: xpBase,
                                rows: rowValues.count, n: n, out: oBase)
                        }
                    }
                }
            }
        }
        #expect(one == many)
        #expect(Int8AffineGEMV.preferredThreads >= 1)
    }
}
