import Foundation
import Testing
import TinyTitanCore

@testable import Qwen35Dense4B4Bit
@testable import Qwen35Dense4B8Bit
@testable import Qwen35Dense9B4Bit
@testable import Qwen35Dense9B8Bit

/// Quantized projection parity in the 7.0 dense variants, without a model.
///
/// The M=15 cases catch a batched path that misses the per-token fallback: the
/// fallback must be bit-identical to the shipping GEMV. The M=16 cases catch
/// packing, row-stride and transposition errors in dequantize-once + SGEMM.
@Suite struct DenseVariantBatchedMatmulTests {

    private typealias Projection = (name: String, rows: Int, columns: Int)

    // Attention, FFN and gated-DeltaNet quantized projections. Different row
    // counts make a mistaken output stride visible rather than cancelling out.
    private let projections: [Projection] = [
        ("q", 16, 128), ("k", 8, 128), ("v", 8, 128), ("o", 12, 128),
        ("gate", 24, 128), ("up", 24, 128), ("down", 12, 192),
        ("gdn_in_qkv", 32, 128), ("gdn_in_z", 16, 128), ("gdn_out", 12, 192),
    ]

    private func bf16(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }

    private func packed(
        rows: Int, columns: Int, bits: Int, salt: UInt32
    ) -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
        var state = salt
        func next() -> UInt32 {
            state = state &* 1_664_525 &+ 1_013_904_223
            return state >> 8
        }
        var weights: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for _ in 0..<rows {
            for _ in 0..<(columns / 64) {
                scales.append(bf16(0.01 + Float(next() % 7) * 0.001))
                biases.append(bf16((Float(next() % 5) - 2) * 0.0005))
            }
            if bits == 8 {
                for _ in 0..<columns { weights.append(UInt8(next() % 256)) }
            } else {
                for _ in 0..<(columns / 2) {
                    weights.append(UInt8(next() % 16) | UInt8(next() % 16) << 4)
                }
            }
        }
        return (weights, scales, biases)
    }

    private func activations(count: Int, salt: UInt32) -> [Float] {
        var state = salt
        return (0..<count).map { _ in
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(Int(state >> 20) % 2001 - 1000) / 1000
        }
    }

    private func withMatrix<Matrix>(
        rows: Int, columns: Int, bits: Int, salt: UInt32,
        make: (
            UnsafeRawBufferPointer, UnsafePointer<UInt16>, UnsafePointer<UInt16>, Int, Int, Int
        ) -> Matrix,
        _ body: (Matrix) throws -> Void
    ) throws {
        let data = packed(rows: rows, columns: columns, bits: bits, salt: salt)
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
        let matrix = make(
            UnsafeRawBufferPointer(start: weights, count: data.weights.count),
            scales, biases, rows, columns, bits)
        try body(matrix)
    }

    private func check<Matrix>(
        variant: String, bits: Int,
        make: (
            UnsafeRawBufferPointer, UnsafePointer<UInt16>, UnsafePointer<UInt16>, Int, Int, Int
        ) -> Matrix,
        multiply: (Matrix, [Float], Int, Int) throws -> [Float],
        gemv: (Matrix, UnsafePointer<Float>, UnsafeMutablePointer<Float>, Int) throws -> Void
    ) throws {
        for (projectionIndex, projection) in projections.enumerated() {
            for tokens in [15, 16] {
                let salt = UInt32(projectionIndex + 1) &* 0x9E37 &+ UInt32(tokens)
                let x = activations(count: tokens * projection.columns, salt: salt)
                try withMatrix(
                    rows: projection.rows, columns: projection.columns,
                    bits: bits, salt: salt, make: make
                ) { matrix in
                    let actual = try multiply(matrix, x, tokens, 1)
                    var expected = [Float](repeating: 0, count: tokens * projection.rows)
                    for token in 0..<tokens {
                        try x.withUnsafeBufferPointer { input in
                            try expected.withUnsafeMutableBufferPointer { output in
                                guard let inputBase = input.baseAddress,
                                    let outputBase = output.baseAddress
                                else { return }
                                try gemv(
                                    matrix, inputBase + token * projection.columns,
                                    outputBase + token * projection.rows, 1)
                            }
                        }
                    }
                    var worst: Float = 0
                    var peak: Float = 0
                    for index in expected.indices {
                        worst = max(worst, abs(actual[index] - expected[index]))
                        peak = max(peak, abs(expected[index]))
                    }
                    let context = "\(variant) \(bits)-bit \(projection.name) M=\(tokens)"
                    if tokens < 16 {
                        #expect(worst == 0, "\(context): fallback differs from GEMV by \(worst)")
                    } else {
                        #expect(
                            worst <= 1e-4 * max(peak, 1),
                            "\(context): worst \(worst), peak \(peak)")
                        for token in 0..<tokens {
                            var expectedBest = 0
                            var actualBest = 0
                            for row in 1..<projection.rows {
                                let base = token * projection.rows
                                if expected[base + row] > expected[base + expectedBest] {
                                    expectedBest = row
                                }
                                if actual[base + row] > actual[base + actualBest] {
                                    actualBest = row
                                }
                            }
                            #expect(
                                actualBest == expectedBest,
                                "\(context): top-1 diverged at token \(token)")
                        }
                    }
                }
            }
        }
    }

    @Test(arguments: [4, 8])
    func dense4B4Bit(bits: Int) throws {
        try check(
            variant: "4B4Bit", bits: bits,
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense4B4Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense4B4Bit.CPUBatchedMatmul.multiply,
            gemv: Qwen35Dense4B4Bit.CPUOps.gemv)
    }

    @Test(arguments: [4, 8])
    func dense4B8Bit(bits: Int) throws {
        try check(
            variant: "4B8Bit", bits: bits,
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense4B8Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense4B8Bit.CPUBatchedMatmul.multiply,
            gemv: Qwen35Dense4B8Bit.CPUOps.gemv)
    }

    @Test(arguments: [4, 8])
    func dense9B4Bit(bits: Int) throws {
        try check(
            variant: "9B4Bit", bits: bits,
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense9B4Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense9B4Bit.CPUBatchedMatmul.multiply,
            gemv: Qwen35Dense9B4Bit.CPUOps.gemv)
    }

    @Test(arguments: [4, 8])
    func dense9B8Bit(bits: Int) throws {
        try check(
            variant: "9B8Bit", bits: bits,
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense9B8Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense9B8Bit.CPUBatchedMatmul.multiply,
            gemv: Qwen35Dense9B8Bit.CPUOps.gemv)
    }
}
