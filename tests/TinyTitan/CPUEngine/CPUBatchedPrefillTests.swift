import Foundation
import Testing
import TinyTitanKernelsC

@testable import TinyTitan
import TinyTitanCore

/// The batched prefill's new arithmetic, checked without a model file.
///
/// The end-to-end check against the installed 2B is a benchmark (same prompt,
/// batched against `TINYTITAN_CPU_PREFILL=token`), because it needs a
/// two-gigabyte install. What is here is everything that can be pinned on
/// synthetic data: the dequantized product against the shipping GEMV, the
/// delta-rule kernel against its scalar reference, and the row-wise norm.
@Suite struct CPUBatchedPrefillTests {

    private func bf16(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }

    private func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    /// Pack `rows * columns` values at `bits`, low nibble first at 4 bits,
    /// with one affine pair per 64-element group.
    private func packed(
        rows: Int, columns: Int, bits: Int
    ) -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
        var weights: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        let groups = columns / 64
        var state: UInt32 = 0x1234_5678
        func next() -> UInt32 {
            state = state &* 1_664_525 &+ 1_013_904_223
            return state >> 8
        }
        for _ in 0..<rows {
            for _ in 0..<groups {
                scales.append(bf16(0.01 + Float(next() % 7) * 0.001))
                biases.append(bf16(Float(next() % 5) * 0.0005))
            }
            if bits == 8 {
                for _ in 0..<columns { weights.append(UInt8(next() % 256)) }
            } else {
                for _ in 0..<(columns / 2) {
                    let low = UInt8(next() % 16)
                    let high = UInt8(next() % 16)
                    weights.append(low | (high << 4))
                }
            }
        }
        return (weights, scales, biases)
    }

    /// Runs `body` with a Matrix whose buffers outlive it.
    private func withMatrix<T>(
        rows: Int, columns: Int, bits: Int,
        _ body: (AffineSnapshot.Matrix) throws -> T
    ) rethrows -> T {
        let data = packed(rows: rows, columns: columns, bits: bits)
        let weights = UnsafeMutablePointer<UInt8>.allocate(capacity: data.weights.count)
        let scales = UnsafeMutablePointer<UInt16>.allocate(capacity: data.scales.count)
        let biases = UnsafeMutablePointer<UInt16>.allocate(capacity: data.biases.count)
        data.weights.withUnsafeBufferPointer {
            guard let base = $0.baseAddress else { return }
            weights.initialize(from: base, count: $0.count)
        }
        data.scales.withUnsafeBufferPointer {
            guard let base = $0.baseAddress else { return }
            scales.initialize(from: base, count: $0.count)
        }
        data.biases.withUnsafeBufferPointer {
            guard let base = $0.baseAddress else { return }
            biases.initialize(from: base, count: $0.count)
        }
        defer {
            weights.deallocate()
            scales.deallocate()
            biases.deallocate()
        }
        let matrix = AffineSnapshot.Matrix(
            weights: UnsafeRawBufferPointer(start: weights, count: data.weights.count),
            scales: scales, biases: biases,
            rows: rows, columns: columns, bits: bits, groupSize: 64)
        return try body(matrix)
    }

    /// The batched product and the per-token GEMV agree to float32
    /// reassociation, at both widths.
    @Test(arguments: [4, 8])
    func batchedMatmulMatchesTheGEMV(bits: Int) throws {
        let rows = 16
        let columns = 128
        let tokens = 4
        var x = [Float](repeating: 0, count: tokens * columns)
        var seed: UInt32 = 0x9E37_79B9
        for index in x.indices {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            x[index] = Float(Int(seed >> 20) % 2001 - 1000) / 1000
        }

        try withMatrix(rows: rows, columns: columns, bits: bits) { matrix in
            let batched = try CPUBatchedMatmul.multiply(
                matrix, x: x, tokenCount: tokens, threads: 1)
            var expected = [Float](repeating: 0, count: tokens * rows)
            for token in 0..<tokens {
                let inputStart = token * columns
                let outputStart = token * rows
                try x.withUnsafeBufferPointer { input in
                    try expected.withUnsafeMutableBufferPointer { output in
                        guard let inputBase = input.baseAddress,
                            let outputBase = output.baseAddress
                        else { return }
                        try CPUOps.gemv(
                            matrix, x: inputBase + inputStart,
                            out: outputBase + outputStart, threads: 1)
                    }
                }
            }
            var worst: Float = 0
            var peak: Float = 0
            for index in expected.indices {
                worst = max(worst, abs(batched[index] - expected[index]))
                peak = max(peak, abs(expected[index]))
            }
            #expect(worst / max(peak, 1.0) < 1e-4, "bits \(bits): worst \(worst) of \(peak)")
        }
    }

    /// The dequantizer writes `q * scale + bias` in the GEMV kernels' order.
    @Test(arguments: [4, 8])
    func dequantizerMatchesTheAffineFormula(bits: Int) {
        let rows = 2
        let columns = 128
        withMatrix(rows: rows, columns: columns, bits: bits) { matrix in
            var output = [Float](repeating: 0, count: rows * columns)
            output.withUnsafeMutableBufferPointer { buffer in
                guard let base = buffer.baseAddress else { return }
                CPUBatchedMatmul.dequantize(matrix, into: base, threads: 1)
            }
            let groups = columns / 64
            for row in 0..<rows {
                for group in 0..<groups {
                    let scale = bf16ToFloat(matrix.scales[row * groups + group])
                    let bias = bf16ToFloat(matrix.biases[row * groups + group])
                    for offset in 0..<64 {
                        let column = group * 64 + offset
                        let byte = matrix.weights[
                            row * (bits == 8 ? columns : columns / 2)
                                + (bits == 8 ? column : column / 2)]
                        let q = bits == 8 ? byte : (column % 2 == 0 ? byte & 0x0F : byte >> 4)
                        let expected = Float(q) * scale + bias
                        #expect(
                            abs(output[row * columns + column] - expected) < 1e-6,
                            "bits \(bits) row \(row) column \(column)")
                    }
                }
            }
        }
    }

    /// The C delta rule against the Swift scalar arithmetic it replaced.
    @Test func deltaRuleMatchesTheScalarPath() throws {
        let heads = 2
        let keyHeads = 1
        let dk = 8
        let dv = 4
        let inverse: Float = 1 / Float(dv).squareRoot()
        var seed: UInt32 = 0x55AA_1234
        func random() -> Float {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            return Float(Int(seed >> 20) % 2001 - 1000) / 1000
        }
        let query = (0..<(keyHeads * dk)).map { _ in random() }
        let key = (0..<(keyHeads * dk)).map { _ in random() }
        let value = (0..<(heads * dv)).map { _ in random() }
        let a = (0..<heads).map { _ in random() }
        let b = (0..<heads).map { _ in random() }
        let decayLog = (0..<heads).map { _ in random() }
        let decayBias = (0..<heads).map { _ in random() }
        let start = (0..<(heads * dv * dk)).map { _ in random() * 0.1 }

        var kernelState = start
        var kernelReadout = [Float](repeating: 0, count: heads * dv)
        query.withUnsafeBufferPointer { q in
            key.withUnsafeBufferPointer { k in
                value.withUnsafeBufferPointer { v in
                    a.withUnsafeBufferPointer { ab in
                        b.withUnsafeBufferPointer { bb in
                            decayLog.withUnsafeBufferPointer { dl in
                                decayBias.withUnsafeBufferPointer { db in
                                    kernelState.withUnsafeMutableBufferPointer { sb in
                                        kernelReadout.withUnsafeMutableBufferPointer { rb in
                                            tinytitan_delta_rule(
                                                q.baseAddress, k.baseAddress, v.baseAddress,
                                                ab.baseAddress, bb.baseAddress,
                                                dl.baseAddress, db.baseAddress,
                                                heads, keyHeads, dk, dv, inverse,
                                                sb.baseAddress, rb.baseAddress)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        var referenceState = start
        var referenceReadout = [Float](repeating: 0, count: heads * dv)
        let repeats = heads / keyHeads
        for head in 0..<heads {
            let keyHead = head / repeats
            let beta = CPUOps.sigmoid(b[head])
            let decay = expf(
                -expf(decayLog[head]) * CPUOps.softplus(a[head] + decayBias[head]))
            let stateBase = head * dv * dk
            for index in 0..<dv {
                let row = stateBase + index * dk
                var stored: Float = 0
                for column in 0..<dk {
                    let decayed = referenceState[row + column] * decay
                    referenceState[row + column] = decayed
                    stored += decayed * key[keyHead * dk + column]
                }
                let correction = (value[head * dv + index] - stored) * beta
                var sum: Float = 0
                for column in 0..<dk {
                    referenceState[row + column] += key[keyHead * dk + column] * correction
                    sum += referenceState[row + column] * query[keyHead * dk + column]
                }
                referenceReadout[head * dv + index] = sum * inverse
            }
        }
        var worst: Float = 0
        for index in referenceReadout.indices {
            worst = max(worst, abs(kernelReadout[index] - referenceReadout[index]))
        }
        #expect(worst < 1e-5, "readout worst \(worst)")
        for index in referenceState.indices {
            #expect(abs(kernelState[index] - referenceState[index]) < 1e-5)
        }
    }

    /// The chunked delta rule against the token-ordered kernel it replaces.
    @Test(arguments: [8, 16, 32])
    func chunkedDeltaRuleMatchesTheTokenKernel(tokens: Int) {
        let dk = 8
        let dv = 4
        var seed: UInt32 = 0xC0FF_EE01
        func random() -> Float {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            return Float(Int(seed >> 20) % 2001 - 1000) / 1000
        }
        let keys = (0..<(tokens * dk)).map { _ in random() }
        let values = (0..<(tokens * dv)).map { _ in random() }
        let queries = (0..<(tokens * dk)).map { _ in random() }
        let a = (0..<tokens).map { _ in random() }
        let b = (0..<tokens).map { _ in random() }
        let decayLog: [Float] = [0.3]
        let decayBias: [Float] = [0.1]
        let alphas = (0..<tokens).map { index in
            expf(-expf(decayLog[0]) * CPUOps.softplus(a[index] + decayBias[0]))
        }
        let betas = b.map { CPUOps.sigmoid($0) }

        func allocated<T>(_ values: [T]) -> UnsafeMutablePointer<T> {
            let pointer = UnsafeMutablePointer<T>.allocate(capacity: values.count)
            values.withUnsafeBufferPointer { buffer in
                if let base = buffer.baseAddress {
                    pointer.initialize(from: base, count: buffer.count)
                }
            }
            return pointer
        }
        let k = allocated(keys)
        let v = allocated(values)
        let q = allocated(queries)
        let aPointer = allocated(a)
        let bPointer = allocated(b)
        let logPointer = allocated(decayLog)
        let biasPointer = allocated(decayBias)
        let alphaPointer = allocated(alphas)
        let betaPointer = allocated(betas)
        let tokenState = allocated([Float](repeating: 0, count: dv * dk))
        let tokenReadout = allocated([Float](repeating: 0, count: tokens * dv))
        let chunkState = allocated([Float](repeating: 0, count: dv * dk))
        let chunkReadout = allocated([Float](repeating: 0, count: tokens * dv))
        defer {
            for pointer in [
                UnsafeMutableRawPointer(k), UnsafeMutableRawPointer(v),
                UnsafeMutableRawPointer(q), UnsafeMutableRawPointer(aPointer),
                UnsafeMutableRawPointer(bPointer), UnsafeMutableRawPointer(logPointer),
                UnsafeMutableRawPointer(biasPointer), UnsafeMutableRawPointer(alphaPointer),
                UnsafeMutableRawPointer(betaPointer), UnsafeMutableRawPointer(tokenState),
                UnsafeMutableRawPointer(tokenReadout), UnsafeMutableRawPointer(chunkState),
                UnsafeMutableRawPointer(chunkReadout),
            ] {
                pointer.deallocate()
            }
        }

        for index in 0..<tokens {
            tinytitan_delta_rule(
                q + index * dk, k + index * dk, v + index * dv,
                aPointer + index, bPointer + index, logPointer, biasPointer,
                1, 1, dk, dv, 1, tokenState, tokenReadout + index * dv)
        }
        _ = tinytitan_delta_rule_chunk(
            q, dk, k, dk, v, dv, alphaPointer, 1, betaPointer, 1,
            tokens, dk, dv, chunkState, chunkReadout, dv)

        var worst: Float = 0
        for index in 0..<(tokens * dv) {
            worst = max(worst, abs(tokenReadout[index] - chunkReadout[index]))
        }
        #expect(worst < 1e-4, "readout worst \(worst) for \(tokens) tokens")
        for index in 0..<(dv * dk) {
            #expect(abs(tokenState[index] - chunkState[index]) < 1e-4)
        }
    }

    /// Row-wise RMS norm agrees with the existing per-vector one.
    @Test func rowWiseNormMatchesTheSingleRowNorm() {
        let width = 32
        let rows = 3
        let gamma = (0..<width).map { Float($0 % 5) * 0.1 + 0.5 }
        var values = (0..<(rows * width)).map { Float($0 % 11) - 5 }
        var expected = values
        for row in 0..<rows {
            var single = Array(expected[row * width..<(row + 1) * width])
            CPUOps.rmsNorm(&single, gamma: gamma, epsilon: 1e-6)
            expected.replaceSubrange(row * width..<(row + 1) * width, with: single)
        }
        CPUQwen35.rmsNormRows(&values, rows: rows, gamma: gamma, epsilon: 1e-6)
        for index in values.indices {
            #expect(abs(values[index] - expected[index]) < 1e-5)
        }
    }
}
