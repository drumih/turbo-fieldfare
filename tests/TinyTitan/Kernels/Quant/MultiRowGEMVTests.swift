import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The multi-row GEMV must equal the decode kernel run once per row, byte for
/// byte: the wide verify's rows are compared against scalar decode with no
/// tolerance. Each test runs the decode kernel (through its own wrapper, so a
/// specialized decode pipeline is the reference where decode would use one)
/// per activation row, then the multi-row kernel once, and `memcmp`s the
/// outputs. Row and output strides are padded so a stride bug cannot hide.
@Suite struct MultiRowGEMVTests {
    typealias Fixtures = MultiRowGEMVFixtures

    /// 12 exceeds one dispatch's nine rows and exercises the tiling.
    static let rowCounts = [1, 2, 5, 8, 9, 12]

    struct Shape: Sendable, CustomTestStringConvertible {
        let m: Int
        let n: Int
        var pad = 0
        var testDescription: String { "\(m)x\(n)+\(pad)" }
    }

    typealias DecodeGEMV = (
        _ commandBuffer: MTLCommandBuffer, _ x: MTLBuffer, _ xOffset: Int,
        _ y: MTLBuffer, _ yOffset: Int
    ) throws -> Void

    private static func decodeKernel(
        _ context: MetalContext, format: MultiRowGEMV.Format,
        view: TensorView, shape: Shape
    ) throws -> DecodeGEMV {
        let m = UInt32(shape.m)
        let n = UInt32(shape.n)
        let w = view.buffer
        let weightsOffset = Int(view.offset)
        let scalesOffset = Int(view.scaleOffset)
        let biasesOffset = Int(view.biasOffset)
        switch format {
        case .int4:
            let kernel = try DequantInt4GEMV(context: context)
            return { cb, x, xOffset, y, yOffset in
                try kernel.encode(
                    commandBuffer: cb, weights: w, weightsOffset: weightsOffset,
                    scales: w, scalesOffset: scalesOffset,
                    biases: w, biasesOffset: biasesOffset,
                    x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
            }
        case .affine(let bits):
            let kernel = try AffineQuantGEMV(context: context, weightBits: bits)
            return { cb, x, xOffset, y, yOffset in
                try kernel.encode(
                    commandBuffer: cb, weights: w, weightsOffset: weightsOffset,
                    scales: w, scalesOffset: scalesOffset,
                    biases: w, biasesOffset: biasesOffset,
                    x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
            }
        case .bf16:
            let kernel = try BF16GEMV(context: context)
            return { cb, x, xOffset, y, yOffset in
                try kernel.encode(
                    commandBuffer: cb, weights: w, weightsOffset: weightsOffset,
                    x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
            }
        }
    }

    private static func expectMatchesDecodePerRow(
        format: MultiRowGEMV.Format, shape: Shape, seed: UInt64
    ) throws {
        let context = try MetalContext()
        var generator = Fixtures.Generator(seed: seed)
        let view = try Fixtures.matrix(
            device: context.device, format: format,
            rows: shape.m, columns: shape.n, pad: shape.pad, generator: &generator)
        let decode = try decodeKernel(context, format: format, view: view, shape: shape)
        let multiRow = MultiRowGEMV(context: context)
        let xStride = shape.n + 4
        let yStride = shape.m + 3
        let half = MemoryLayout<Float16>.stride
        let x = try Fixtures.activations(
            device: context.device, count: (rowCounts.max() ?? 1) * xStride,
            generator: &generator)
        for rows in rowCounts {
            let expected = try Fixtures.zeroed(device: context.device, halves: rows * yStride)
            let actual = try Fixtures.zeroed(device: context.device, halves: rows * yStride)
            try Fixtures.run(context) { cb in
                for row in 0..<rows {
                    try decode(cb, x, row * xStride * half, expected, row * yStride * half)
                }
            }
            try Fixtures.run(context) { cb in
                try multiRow.encode(
                    commandBuffer: cb,
                    segments: [
                        .init(
                            view: view, format: format, rows: shape.m,
                            y: actual, yRowStride: yStride)
                    ],
                    x: x, xRowStride: xStride, activationRows: rows, columns: shape.n)
            }
            Fixtures.expectIdentical(
                expected, actual, halves: rows * yStride,
                "\(format) \(shape.testDescription) W=\(rows)")
        }
    }

    /// 1024x2816 is one of `DequantInt4GEMV`'s constant-folded decode shapes;
    /// 2560x6144 is Qwen3.8's GDN out_proj and o_proj; n = 320 and 192 leave a
    /// scalar remainder after the four-group blocks; pad 2 is the resident
    /// layout's 2-aligned-but-not-4-aligned weight offset.
    @Test(arguments: [
        Shape(m: 64, n: 128), Shape(m: 40, n: 192), Shape(m: 72, n: 320, pad: 2),
        Shape(m: 1024, n: 2816), Shape(m: 2560, n: 6144, pad: 2),
    ])
    func int4MatchesDecodePerRow(shape: Shape) throws {
        try Self.expectMatchesDecodePerRow(
            format: .int4, shape: shape, seed: 0x4D52_0004 &+ UInt64(shape.m * shape.n))
    }

    @Test(arguments: [Shape(m: 64, n: 128), Shape(m: 72, n: 320, pad: 4), Shape(m: 512, n: 2560)])
    func affine8MatchesDecodePerRow(shape: Shape) throws {
        try Self.expectMatchesDecodePerRow(
            format: .affine(bits: 8), shape: shape, seed: 0x4D52_0008 &+ UInt64(shape.m))
    }

    @Test(arguments: [Shape(m: 64, n: 128), Shape(m: 72, n: 320)])
    func affine2MatchesDecodePerRow(shape: Shape) throws {
        try Self.expectMatchesDecodePerRow(
            format: .affine(bits: 2), shape: shape, seed: 0x4D52_0002 &+ UInt64(shape.m))
    }

    /// `SlotGEMV.encodeRows` -- what the hyper-connection and PLE projections
    /// use in a decode-order pass -- must pick the format `encode` would for
    /// the slot width and the tensor's dtype.
    @Test(arguments: [(4, false), (8, false), (4, true), (8, true)] as [(Int, Bool)])
    func slotGEMVRowsMatchEncodePerRow(slotBits: Int, isBF16: Bool) throws {
        let context = try MetalContext()
        let slot = try SlotGEMV(context: context, weightBits: slotBits)
        let (m, n, rows) = (80, 320, 9)
        var generator = Fixtures.Generator(seed: 0x4D52_5107 &+ UInt64(slotBits))
        let format: MultiRowGEMV.Format =
            isBF16 ? .bf16 : (slotBits == 4 ? .int4 : .affine(bits: slotBits))
        let view = try Fixtures.matrix(
            device: context.device, format: format, rows: m, columns: n, pad: 4,
            generator: &generator)
        let x = try Fixtures.activations(
            device: context.device, count: rows * n, generator: &generator)
        let expected = try Fixtures.zeroed(device: context.device, halves: rows * m)
        let actual = try Fixtures.zeroed(device: context.device, halves: rows * m)
        let (w, wOff) = (view.buffer, Int(view.offset))
        let (sOff, bOff) = (Int(view.scaleOffset), Int(view.biasOffset))
        try Fixtures.run(context) { cb in
            for row in 0..<rows {
                try slot.encode(
                    commandBuffer: cb, weights: w, weightsOffset: wOff,
                    scales: w, scalesOffset: sOff, biases: w, biasesOffset: bOff,
                    x: x, xOffset: row * n * 2, y: expected, yOffset: row * m * 2,
                    m: UInt32(m), n: UInt32(n), isBF16: isBF16)
            }
            try slot.encodeRows(
                commandBuffer: cb, weights: w, weightsOffset: wOff,
                scales: w, scalesOffset: sOff, biases: w, biasesOffset: bOff,
                x: x, y: actual, m: m, n: n, rows: rows, isBF16: isBF16)
        }
        Fixtures.expectIdentical(
            expected, actual, halves: rows * m, "SlotGEMV \(slotBits)-bit bf16=\(isBF16)")
    }

    @Test(arguments: [Shape(m: 64, n: 128), Shape(m: 72, n: 320, pad: 2), Shape(m: 48, n: 2560)])
    func bf16MatchesDecodePerRow(shape: Shape) throws {
        try Self.expectMatchesDecodePerRow(
            format: .bf16, shape: shape, seed: 0x4D52_0016 &+ UInt64(shape.m))
    }
}
