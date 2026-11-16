import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// The affine GEMM is the prefill foundation: one row of 4/8-bit weights held
/// in registers while a chunk of tokens is multiplied against it. These tests
/// pin it against the dequantized dot product and against the per-token GEMV it
/// is meant to replace.
@Suite struct AffineQuantGEMMTests {
    private static func makeBuffer<T>(_ ctx: MetalContext, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return nil }
            return ctx.device.makeBuffer(
                bytes: base, length: ptr.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }

    /// Four-bit weights quantized by the runtime's own helper, plus the
    /// dequantized rows read back from exactly the bytes the kernel sees.
    private static func quantizedRows(
        rows: Int, columns: Int
    ) -> (
        packed: [UInt8], scales: [UInt16], biases: [UInt16], values: [[Float]]
    ) {
        var packed: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for row in 0..<rows {
            let floats = (0..<columns).map { column in
                Float((row * 13 + column * 7) % 31) / 10 - 1.5
            }
            let q = Quantization.quantizeInt4Affine(floats)
            packed += q.packed
            scales += q.scales
            biases += q.biases
        }
        let groups = columns / Quantization.groupSize
        var values: [[Float]] = []
        for row in 0..<rows {
            let byteCount = columns / 2
            let base = row * byteCount
            let rowGroups = row * groups
            values.append(
                (0..<columns).map { column in
                    let byte = packed[base + column / 2]
                    let q = (column & 1) == 0 ? byte & 0x0F : byte >> 4
                    let group = column / Quantization.groupSize
                    return Float(q)
                        * Quantization.bf16ToFloat(
                            scales[rowGroups + group])
                        + Quantization.bf16ToFloat(biases[rowGroups + group])
                })
        }
        return (packed, scales, biases, values)
    }

    @Test func fourBitChunkMatchesTheDequantizedDotProduct() throws {
        let rows = 32
        let columns = 256
        let tokens = 5
        let ctx = try MetalContext()
        let kernel = try AffineQuantGEMM(context: ctx, weightBits: 4)
        let q = Self.quantizedRows(rows: rows, columns: columns)

        let x: [Float16] = (0..<(tokens * columns)).map { index in
            Float16(Float((index * 29) % 17) / 8 - 1)
        }
        let w = try #require(Self.makeBuffer(ctx, q.packed))
        let s = try #require(Self.makeBuffer(ctx, q.scales))
        let b = try #require(Self.makeBuffer(ctx, q.biases))
        let xb = try #require(Self.makeBuffer(ctx, x))
        guard let yb = Fp16Buffer.make(ctx.device, count: tokens * rows) else {
            Issue.record("could not allocate the GEMM output")
            return
        }
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encode(
            commandBuffer: cb, weights: w, scales: s, biases: b,
            x: xb, y: yb,
            rows: UInt32(rows), columns: UInt32(columns), tokens: UInt32(tokens))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        let actual = Fp16Buffer.read(yb, count: tokens * rows)
        let reference = (0..<tokens).flatMap { token in
            (0..<rows).map { row -> Float in
                let xrow = (0..<columns).map {
                    Float(x[token * columns + $0])
                }
                return zip(q.values[row], xrow).reduce(Float(0)) {
                    $0 + $1.0 * $1.1
                }
            }
        }
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < 0.002, "relErr=\(rel)")
    }

    /// The same weights through the per-token GEMV: one token of the GEMM must
    /// be the GEMV result, and a chunk must be those results side by side.
    @Test(arguments: [4, 8])
    func chunkMatchesPerTokenGEMV(bits: Int) throws {
        let rows = 21
        let columns = 256
        let tokens = 4
        let ctx = try MetalContext()
        let gemm = try AffineQuantGEMM(context: ctx, weightBits: bits)
        let gemv = try AffineQuantGEMV(context: ctx, weightBits: bits)

        var packed: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for row in 0..<rows {
            let floats = (0..<columns).map { column in
                Float((row * 5 + column * 3) % 23) / 7 - 1.2
            }
            if bits == 4 {
                let q = Quantization.quantizeInt4Affine(floats)
                packed += q.packed
                scales += q.scales
                biases += q.biases
            } else {
                // 8-bit affine by hand: one byte per element, group scales.
                let groups = columns / Quantization.groupSize
                for column in 0..<columns {
                    let group = column / Quantization.groupSize
                    let scale = Float(0.5 + Double(group) * 0.01)
                    let bias = Float(0.05 * Double(group))
                    let value =
                        Int(((floats[column] - bias) / scale).rounded())
                    packed.append(UInt8(max(0, min(255, value))))
                }
                for group in 0..<groups {
                    let scale = Float(0.5 + Double(group) * 0.01)
                    let bias = Float(0.05 * Double(group))
                    scales.append(Quantization.bf16Bits(scale))
                    biases.append(Quantization.bf16Bits(bias))
                }
            }
        }
        let x: [Float16] = (0..<(tokens * columns)).map { index in
            Float16(Float((index * 11) % 13) / 6 - 1)
        }
        let w = try #require(Self.makeBuffer(ctx, packed))
        let s = try #require(Self.makeBuffer(ctx, scales))
        let b = try #require(Self.makeBuffer(ctx, biases))
        let xb = try #require(Self.makeBuffer(ctx, x))
        guard let yGEMM = Fp16Buffer.make(ctx.device, count: tokens * rows),
            let yGEMV = Fp16Buffer.make(ctx.device, count: tokens * rows)
        else {
            Issue.record("could not allocate the outputs")
            return
        }

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try gemm.encode(
            commandBuffer: cb, weights: w, scales: s, biases: b,
            x: xb, y: yGEMM,
            rows: UInt32(rows), columns: UInt32(columns), tokens: UInt32(tokens))
        for token in 0..<tokens {
            try gemv.encode(
                commandBuffer: cb, weights: w, scales: s, biases: b,
                x: xb, xOffset: token * columns * 2,
                y: yGEMV, yOffset: token * rows * 2,
                m: UInt32(rows), n: UInt32(columns))
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        let gemmValues = Fp16Buffer.read(yGEMM, count: tokens * rows)
        let gemvValues = Fp16Buffer.read(yGEMV, count: tokens * rows)
        let rel = RelError.compute(actual: gemmValues, reference: gemvValues)
        #expect(rel < 0.002, "relErr=\(rel) bits=\(bits)")
        // Token-major output: token 0's rows come first.
        #expect(abs(gemmValues[0] - gemvValues[0]) < 0.01)
        #expect(abs(gemmValues[rows + 1] - gemvValues[rows + 1]) < 0.01)
    }

    @Test func shapeBoundIsTheRegisterBudget() {
        #expect(
            AffineQuantGEMM.canEncode(
                rows: 2560, columns: 2560, tokens: 16, weightBits: 4))
        #expect(
            AffineQuantGEMM.canEncode(
                rows: 768, columns: 2560, tokens: 64, weightBits: 8))
        #expect(
            !AffineQuantGEMM.canEncode(
                rows: 4096, columns: 8192, tokens: 16, weightBits: 8))
        #expect(
            !AffineQuantGEMM.canEncode(
                rows: 64, columns: 100, tokens: 4, weightBits: 4))
    }
}
