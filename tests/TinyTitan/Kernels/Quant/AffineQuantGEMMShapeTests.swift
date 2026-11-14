import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
import TinyTitanCore

/// The GEMM is only useful at the real projections' shapes: 2,560 and 4,096
/// columns (20 and 32 register iterations at 8-bit) and 4,096 rows. The toy
/// test's 256-column shape never reaches the register budget's ceiling, so
/// these pin the widest rows the prefill actually dispatches.
@Suite struct AffineQuantGEMMShapeTests {
    private static func makeBuffer<T>(_ ctx: MetalContext, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return nil }
            return ctx.device.makeBuffer(
                bytes: base, length: ptr.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }

    private static func quantize(
        rows: Int, columns: Int, bits: Int
    ) -> (packed: [UInt8], scales: [UInt16], biases: [UInt16], values: [[Float]]) {
        let groups = columns / Quantization.groupSize
        var packed: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        var values: [[Float]] = []
        for row in 0..<rows {
            let floats = (0..<columns).map { column in
                Float((row * 17 + column * 5) % 29) / 9 - 1.4
            }
            if bits == 4 {
                let q = Quantization.quantizeInt4Affine(floats)
                packed += q.packed
                scales += q.scales
                biases += q.biases
            } else {
                for column in 0..<columns {
                    let group = column / Quantization.groupSize
                    let scale = Float(0.35 + Double(group % 7) * 0.03)
                    let bias = Float(0.02 * Double(group % 5))
                    let q = Int(((floats[column] - bias) / scale).rounded())
                    packed.append(UInt8(max(0, min(255, q))))
                }
                for group in 0..<groups {
                    let scale = Float(0.35 + Double(group % 7) * 0.03)
                    let bias = Float(0.02 * Double(group % 5))
                    scales.append(Quantization.bf16Bits(scale))
                    biases.append(Quantization.bf16Bits(bias))
                }
            }
            var rowValues: [Float] = []
            rowValues.reserveCapacity(columns)
            for column in 0..<columns {
                let group = column / Quantization.groupSize
                let scale = Quantization.bf16ToFloat(scales[row * groups + group])
                let bias = Quantization.bf16ToFloat(biases[row * groups + group])
                let q: Int
                if bits == 4 {
                    let byte = packed[row * (columns / 2) + column / 2]
                    q = Int((column & 1) == 0 ? byte & 0x0F : byte >> 4)
                } else {
                    q = Int(packed[row * columns + column])
                }
                rowValues.append(Float(q) * scale + bias)
            }
            values.append(rowValues)
        }
        return (packed, scales, biases, values)
    }

    @Test(arguments: [4, 8], [(2560, 2560), (4096, 2560)])
    func wideRowsMatchTheGEMV(bits: Int, shape: (Int, Int)) throws {
        let rows = shape.0
        let columns = shape.1
        let tokens = 3
        let ctx = try MetalContext()
        let gemm = try AffineQuantGEMM(context: ctx, weightBits: bits)
        let gemv = try AffineQuantGEMV(context: ctx, weightBits: bits)
        let q = Self.quantize(rows: rows, columns: columns, bits: bits)

        let x: [Float16] = (0..<(tokens * columns)).map { index in
            Float16(Float((index * 23) % 19) / 9 - 1)
        }
        let w = try #require(Self.makeBuffer(ctx, q.packed))
        let s = try #require(Self.makeBuffer(ctx, q.scales))
        let b = try #require(Self.makeBuffer(ctx, q.biases))
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
        var maxAbs: Float = 0
        for index in 0..<(tokens * rows) {
            maxAbs = max(maxAbs, abs(gemmValues[index] - gemvValues[index]))
        }
        #expect(
            rel < 0.002,
            "relErr=\(rel) maxAbs=\(maxAbs) bits=\(bits) rows=\(rows) columns=\(columns)")
    }
}
