import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The bf16 GEMV kernels against a CPU reference, and against each other.
///
/// The multirow kernel exists to amortise dispatches, not to change arithmetic:
/// one threadgroup per (row block, token) runs the same two-columns-per-lane
/// loop as the single-row kernel. So the load-bearing assertion here is that
/// the two produce **byte-identical** results on the same inputs -- a stride
/// bug in the token dimension would still produce plausible numbers, and only
/// the comparison against the path it replaced catches it. The CPU dot product
/// then anchors both to the arithmetic a reader expects.
@Suite struct BF16GEMVTests {
    /// A tiny deterministic generator: the fixtures only need repeatability,
    /// and importing the validation module for one would put a test-only
    /// dependency on a kernel this file can generate for itself.
    private struct LCG {
        var state: UInt64
        mutating func uniform(_ lower: Float, _ upper: Float) -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let unit = Float(state >> 40) / Float(1 << 24)
            return lower + (upper - lower) * unit
        }
    }

    private static func bf16Buffer(_ values: [Float], device: MTLDevice) -> MTLBuffer? {
        let bits = values.map { Quantization.bf16Bits($0) }
        return device.makeBuffer(
            bytes: bits, length: bits.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared)
    }

    /// float32 dots in the caller's order; the kernel's per-lane order differs,
    /// so this is a tolerance check, not an identity one.
    private static func reference(
        weights: [Float], rows: Int, columns: Int,
        x: [Float16], tokens: Int, xStride: Int
    ) -> [Double] {
        var y = [Double](repeating: 0, count: tokens * rows)
        for token in 0..<tokens {
            for row in 0..<rows {
                var acc = 0.0
                for column in 0..<columns {
                    acc +=
                        Double(weights[row * columns + column])
                        * Double(Float(x[token * xStride + column]))
                }
                y[token * rows + row] = acc
            }
        }
        return y
    }

    private static func run(
        rows: Int, columns: Int, tokens: Int,
        seed: UInt64, yStride: Int
    ) throws {
        var rng = LCG(state: seed)
        // The reference has to read the bytes the kernel reads: bf16-rounded
        // weights, not the floats they were rounded from. A 0.4% per-weight
        // difference accumulates over 2048 columns into a few percent.
        let weights = (0..<(rows * columns)).map { _ in
            Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(-0.5, 0.5)))
        }
        let x = (0..<(tokens * columns)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let yRef = reference(
            weights: weights, rows: rows, columns: columns,
            x: x, tokens: tokens, xStride: columns)

        let context = try MetalContext()
        let gemv = try BF16GEMV(context: context)
        let device = context.device
        let weightBits = weights.map { Quantization.bf16Bits($0) }
        let xHalf = x

        guard
            let wBuf = device.makeBuffer(
                bytes: weightBits, length: weightBits.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
            let xBuf = device.makeBuffer(
                bytes: xHalf, length: xHalf.count * MemoryLayout<Float16>.size,
                options: .storageModeShared),
            let multiBuf = device.makeBuffer(
                length: tokens * yStride * MemoryLayout<Float16>.size,
                options: .storageModeShared),
            let rowBuf = device.makeBuffer(
                length: tokens * yStride * MemoryLayout<Float16>.size,
                options: .storageModeShared),
            let queue = device.makeCommandQueue()
        else {
            Issue.record("could not allocate the test buffers")
            return
        }

        // Zero both outputs so the padding assertions below can tell "never
        // written" from "written with a stale value".
        memset(multiBuf.contents(), 0, multiBuf.length)
        memset(rowBuf.contents(), 0, rowBuf.length)

        // One dispatch for the chunk, and the single-row kernel once per token.
        guard let multiCB = queue.makeCommandBuffer() else {
            Issue.record("no command buffer")
            return
        }
        try gemv.encodeMultiRow(
            commandBuffer: multiCB,
            weights: wBuf, x: xBuf, y: multiBuf,
            m: UInt32(rows), n: UInt32(columns),
            tokens: tokens,
            xStrideElements: columns, yStrideElements: yStride)
        multiCB.commit()
        multiCB.waitUntilCompleted()

        guard let rowCB = queue.makeCommandBuffer() else {
            Issue.record("no command buffer")
            return
        }
        for token in 0..<tokens {
            try gemv.encode(
                commandBuffer: rowCB,
                weights: wBuf, x: xBuf,
                xOffset: token * columns * MemoryLayout<Float16>.size,
                y: rowBuf, yOffset: token * yStride * MemoryLayout<Float16>.size,
                m: UInt32(rows), n: UInt32(columns))
        }
        rowCB.commit()
        rowCB.waitUntilCompleted()

        let multi = multiBuf.contents().bindMemory(to: Float16.self, capacity: tokens * yStride)
        let single = rowBuf.contents().bindMemory(to: Float16.self, capacity: tokens * yStride)
        for token in 0..<tokens {
            for row in 0..<rows {
                let index = token * yStride + row
                #expect(
                    multi[index].bitPattern == single[index].bitPattern,
                    "multirow differs from the per-row kernel at token \(token) row \(row)")
                // Padding between rows must stay untouched.
                let expected = yRef[token * rows + row]
                #expect(
                    abs(Float(multi[index]) - Float(expected))
                        <= 0.01 * max(1, abs(Float(expected))),
                    "token \(token) row \(row): kernel \(Float(multi[index])) vs reference \(expected)"
                )
            }
            for padding in rows..<yStride {
                #expect(
                    single[token * yStride + padding] == 0,
                    "the per-row kernel wrote past its row")
            }
        }
    }

    @Test func multirowMatchesThePerRowKernel() throws {
        try Self.run(rows: 64, columns: 128, tokens: 5, seed: 1, yStride: 64)
    }

    /// A row count that is not a multiple of the eight rows one threadgroup
    /// covers: the tail block must still write its rows.
    @Test func multirowHandlesARaggedRowCount() throws {
        try Self.run(rows: 13, columns: 64, tokens: 4, seed: 2, yStride: 13)
    }

    /// A row stride wider than the row exercises the y stride; a kernel that
    /// ignored it would write token 1 over token 0.
    @Test func multirowHonoursTheRowStride() throws {
        try Self.run(rows: 64, columns: 64, tokens: 3, seed: 3, yStride: 96)
    }

    /// The projection shapes this actually serves.
    @Test func multirowMatchesAtProjectionShapes() throws {
        try Self.run(rows: 512, columns: 2048, tokens: 3, seed: 4, yStride: 512)
    }
}
