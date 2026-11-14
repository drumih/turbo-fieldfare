import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
import TinyTitanCore

@Suite struct AffineQuantTests {
    private static func packed(bits: Int, rows: Int, columns: Int) -> ([UInt8], [[UInt32]]) {
        let rowBytes = columns * bits / 8
        var bytes = [UInt8](repeating: 0, count: rows * rowBytes)
        var values = Array(repeating: [UInt32](repeating: 0, count: columns), count: rows)
        let mask = UInt32((1 << bits) - 1)
        for row in 0..<rows {
            for column in 0..<columns {
                let value = UInt32((row * 11 + column * 3) & Int(mask))
                values[row][column] = value
                let bit = column * bits
                let byte = row * rowBytes + bit / 8
                let shift = bit % 8
                var word = UInt32(value) << shift
                var remaining = bits + shift
                var index = byte
                while remaining > 0 {
                    bytes[index] |= UInt8(truncatingIfNeeded: word)
                    word >>= 8
                    remaining -= 8
                    index += 1
                }
            }
        }
        return (bytes, values)
    }

    @Test(arguments: [2, 4, 8])
    func gemvDecodesPackedAffine(bits: Int) throws {
        let rows = 3
        let columns = 128
        let (packed, values) = Self.packed(bits: bits, rows: rows, columns: columns)
        let one = UInt16(truncatingIfNeeded: Float(1).bitPattern >> 16)
        let scales = [UInt16](repeating: one, count: rows * columns / 64)
        let biases = [UInt16](repeating: 0, count: scales.count)
        let input = [Float16](repeating: 1, count: columns)
        let ctx = try MetalContext()
        let kernel = try AffineQuantGEMV(context: ctx, weightBits: bits)
        let w = try #require(ctx.device.makeBuffer(bytes: packed, length: packed.count))
        let s = try #require(ctx.device.makeBuffer(bytes: scales, length: scales.count * 2))
        let b = try #require(ctx.device.makeBuffer(bytes: biases, length: biases.count * 2))
        let x = try #require(Fp16Buffer.make(ctx.device, halves: input))
        let y = try #require(Fp16Buffer.make(ctx.device, count: rows))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encode(
            commandBuffer: cb, weights: w, scales: s, biases: b,
            x: x, y: y, m: UInt32(rows), n: UInt32(columns))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
        let actual = Fp16Buffer.read(y, count: rows)
        let expected = values.map { Float($0.reduce(0, +)) }
        #expect(RelError.compute(actual: actual, reference: expected) < 0.002)
    }

    /// The two-row head reads the weights once for both rows. This pins that
    /// the shared read produces the same two results as two single-row calls,
    /// within the kernel's summation tolerance — the numerical gate for the
    /// batched decode's first fused stage.
    @Test(arguments: [4, 8])
    func twoRowsMatchTwoSingleRows(bits: Int) throws {
        let rows = 8
        let columns = 128
        let (packed, _) = Self.packed(bits: bits, rows: rows, columns: columns)
        let one = UInt16(truncatingIfNeeded: Float(1).bitPattern >> 16)
        let scales = [UInt16](repeating: one, count: rows * columns / 64)
        let biases = [UInt16](repeating: 0, count: scales.count)
        let inputA = (0..<columns).map { Float16(Float(($0 % 7) + 1) * 0.01) }
        let inputB = (0..<columns).map { Float16(Float(($0 % 5) + 1) * 0.02) }
        let ctx = try MetalContext()
        let kernel = try AffineQuantGEMV(context: ctx, weightBits: bits)
        let w = try #require(ctx.device.makeBuffer(bytes: packed, length: packed.count))
        let s = try #require(ctx.device.makeBuffer(bytes: scales, length: scales.count * 2))
        let b = try #require(ctx.device.makeBuffer(bytes: biases, length: biases.count * 2))
        let x = try #require(Fp16Buffer.make(ctx.device, halves: inputA + inputB))
        let y = try #require(Fp16Buffer.make(ctx.device, count: rows * 2))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encodeTwoRows(
            commandBuffer: cb, weights: w, scales: s, biases: b,
            x: x, y: y, m: UInt32(rows), n: UInt32(columns))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
        let actual = Fp16Buffer.read(y, count: rows * 2)

        let single = try #require(Fp16Buffer.make(ctx.device, count: rows))
        var expected: [Float] = []
        for (index, offset) in [0, columns * 2].enumerated() {
            let cb = try #require(ctx.queue.makeCommandBuffer())
            try kernel.encode(
                commandBuffer: cb, weights: w, scales: s, biases: b,
                x: x, xOffset: offset, y: single, m: UInt32(rows),
                n: UInt32(columns))
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            let got = Fp16Buffer.read(single, count: rows)
            if index == 0 { expected = got } else { expected += got }
        }
        #expect(actual.count == expected.count)
        #expect(RelError.compute(actual: actual, reference: expected) < 0.002)
    }

    /// The four-row head is what width 4 uses instead of two two-row launches.
    /// It must match the four single-row results (the numerical gate) and,
    /// exactly, the two two-row launches it replaces — a width-4 batch must not
    /// change its numbers just because one kernel read the weights once.
    @Test(arguments: [4, 8])
    func fourRowsMatchFourSingleRows(bits: Int) throws {
        let rows = 8
        let columns = 128
        let (packed, _) = Self.packed(bits: bits, rows: rows, columns: columns)
        let one = UInt16(truncatingIfNeeded: Float(1).bitPattern >> 16)
        let scales = [UInt16](repeating: one, count: rows * columns / 64)
        let biases = [UInt16](repeating: 0, count: scales.count)
        let inputs = (0..<4).map { row in
            (0..<columns).map { index in
                Float16(Float(((index + row * 3) % 7) + 1) * 0.01)
            }
        }
        let flat = inputs.flatMap { $0 }
        let ctx = try MetalContext()
        let kernel = try AffineQuantGEMV(context: ctx, weightBits: bits)
        let w = try #require(ctx.device.makeBuffer(bytes: packed, length: packed.count))
        let s = try #require(ctx.device.makeBuffer(bytes: scales, length: scales.count * 2))
        let b = try #require(ctx.device.makeBuffer(bytes: biases, length: biases.count * 2))
        let x = try #require(Fp16Buffer.make(ctx.device, halves: flat))
        let y4 = try #require(Fp16Buffer.make(ctx.device, count: rows * 4))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encodeFourRows(
            commandBuffer: cb, weights: w, scales: s, biases: b,
            x: x, y: y4, m: UInt32(rows), n: UInt32(columns))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
        let actual = Fp16Buffer.read(y4, count: rows * 4)

        // The two two-row launches this replaces, back to back.
        let y2 = try #require(Fp16Buffer.make(ctx.device, count: rows * 2))
        var pairs: [Float] = []
        for pair in 0..<2 {
            let cb = try #require(ctx.queue.makeCommandBuffer())
            try kernel.encodeTwoRows(
                commandBuffer: cb, weights: w, scales: s, biases: b,
                x: x, xOffset: pair * 2 * columns * 2, y: y2,
                m: UInt32(rows), n: UInt32(columns))
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            pairs += Fp16Buffer.read(y2, count: rows * 2)
        }
        #expect(actual == pairs)

        // And four single-row calls, which the two-row kernel itself only
        // matches within the reduction tolerance.
        let single = try #require(Fp16Buffer.make(ctx.device, count: rows))
        var expected: [Float] = []
        for row in 0..<4 {
            let cb = try #require(ctx.queue.makeCommandBuffer())
            try kernel.encode(
                commandBuffer: cb, weights: w, scales: s, biases: b,
                x: x, xOffset: row * columns * 2, y: single,
                m: UInt32(rows), n: UInt32(columns))
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            expected += Fp16Buffer.read(single, count: rows)
        }
        #expect(RelError.compute(actual: actual, reference: expected) < 0.002)
    }

    @Test(arguments: [2, 4, 8])
    func embeddingDecodesPackedAffine(bits: Int) throws {
        let rows = 3
        let columns = 128
        let (packed, values) = Self.packed(bits: bits, rows: rows, columns: columns)
        let one = UInt16(truncatingIfNeeded: Float(1).bitPattern >> 16)
        let scales = [UInt16](repeating: one, count: rows * columns / 64)
        let biases = [UInt16](repeating: 0, count: scales.count)
        let ctx = try MetalContext()
        let kernel = try AffineQuantEmbeddingLookup(context: ctx, weightBits: bits)
        let w = try #require(ctx.device.makeBuffer(bytes: packed, length: packed.count))
        let s = try #require(ctx.device.makeBuffer(bytes: scales, length: scales.count * 2))
        let b = try #require(ctx.device.makeBuffer(bytes: biases, length: biases.count * 2))
        let y = try #require(Fp16Buffer.make(ctx.device, count: columns))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encode(
            commandBuffer: cb, table: w, scales: s, biases: b,
            out: y, tokenId: 1, d: UInt32(columns), outScale: 1,
            vocab: UInt32(rows))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
        let actual = Fp16Buffer.read(y, count: columns)
        #expect(actual == values[1].map { Float($0) })
    }
}
