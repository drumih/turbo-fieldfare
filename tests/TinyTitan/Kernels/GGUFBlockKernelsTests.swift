import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The GPU block decoders (`gguf_block_decoders.metal`) behind the UD-Q2_K_XL
/// file's tensors, against the bit-exact host decoders: the expert-indexed
/// decode GEMV, the tiled prefill GEMM (dense and expert-grouped) and the
/// IQ4_XS embedding row.
@Suite struct GGUFBlockKernelsTests {
    static let gemvTypes: [GGUFType] = [
        .iq2XXS, .iq2XS, .iq2S, .iq1S, .iq3XXS, .q2_0, .mxfp4, .q4K, .q5K, .q6K, .iq4XS,
    ]
    static let gemmTypes: [GGUFType] = [
        .iq3XXS, .q4K, .q5K, .q6K, .iq1S, .iq2XXS, .iq2XS, .iq2S, .q2_0, .mxfp4,
    ]
    static let expertTypes: [GGUFType] = [.iq1S, .iq2XXS, .iq2XS, .iq2S, .q2_0, .mxfp4]

    /// Every byte pattern, with each block's scale bytes overwritten by a
    /// small finite scale so the products stay in range.
    static func blocks(type: GGUFType, rows: Int, columns: Int, seed: Int = 0) -> [UInt8] {
        let count = rows * columns / Int(type.blockSize)
        let blockBytes = Int(type.blockBytes)
        var bytes = (0..<(count * blockBytes)).map { UInt8(($0 &* 37 &+ 11 &+ seed) & 0xFF) }
        func half(_ value: Float, at offset: Int) {
            let bits = Float16(value).bitPattern
            bytes[offset] = UInt8(bits & 0xFF)
            bytes[offset + 1] = UInt8(bits >> 8)
        }
        for block in 0..<count {
            let base = block * blockBytes
            let scale = 0.002 + 0.0005 * Float(block % 9)
            switch type {
            case .mxfp4:
                bytes[base] = UInt8(118 + block % 6)  // E8M0: 2^-9 ... 2^-4
            case .q6K:
                half(scale, at: base + 208)
            case .q4K, .q5K:
                half(scale, at: base)
                half(scale * 0.5, at: base + 2)
            default:
                half(scale, at: base)
            }
        }
        return bytes
    }

    static func activation(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { Float((($0 + seed) * 17) % 29) / 29.0 - 0.5 }
    }

    static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
        let made = values.withUnsafeBytes { raw -> MTLBuffer? in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base, length: max(raw.count, 16), options: .storageModeShared)
        }
        guard let made else { throw MetalError.bufferAllocationFailed("test operand") }
        return made
    }

    static func run(
        _ context: MetalContext, _ body: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        try body(encoder)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    static func floats(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count))
    }

    /// y[p][row] against scale(p) * W[matrix(p)][row] . x[xRow(p)] in Double.
    static func check(
        _ got: [Float], decoded: [Float], rows: Int, columns: Int, x: [Float],
        xRow: (Int) -> Int, matrixOf: (Int) -> Int, scaleOf: (Int) -> Double, outputs: Int,
        relative: Double, label: String
    ) {
        var failures = 0
        for p in 0..<outputs {
            for row in 0..<rows {
                let base = (matrixOf(p) * rows + row) * columns
                var value = 0.0
                var magnitude = 0.0
                for c in 0..<columns {
                    let term = Double(decoded[base + c]) * Double(x[xRow(p) * columns + c])
                    value += term
                    magnitude += abs(term)
                }
                value *= scaleOf(p)
                magnitude *= abs(scaleOf(p))
                let error = abs(Double(got[p * rows + row]) - value)
                if error > relative * magnitude + 1e-6, failures < 4 {
                    failures += 1
                    Issue.record(
                        "\(label) output \(p) row \(row): \(got[p * rows + row]) vs \(value)")
                }
            }
        }
    }

    @Test func tableLayoutHoldsEveryGrid() {
        let table = GGUFBlockTables.bytes()
        #expect(table.count == GGUFBlockTables.byteCount)
        #expect(GGUFIQTables.iq2xxsGrid.count * 8 == GGUFBlockTables.iq2xs - GGUFBlockTables.iq2xxs)
        #expect(GGUFIQTables.iq2xsGrid.count * 8 == GGUFBlockTables.iq2s - GGUFBlockTables.iq2xs)
        #expect(GGUFIQTables.iq2sGrid.count * 8 == GGUFBlockTables.iq1s - GGUFBlockTables.iq2s)
        #expect(GGUFIQTables.iq1sGrid.count * 8 == GGUFBlockTables.iq3xxs - GGUFBlockTables.iq1s)
        #expect(
            GGUFIQTables.iq3xxsGrid.count * 4 == GGUFBlockTables.ksigns - GGUFBlockTables.iq3xxs)
        #expect(GGUFIQTables.ksigns.count == GGUFBlockTables.iq4nl - GGUFBlockTables.ksigns)
        #expect(GGUFBlockTables.iq4nl + 16 == GGUFBlockTables.byteCount)
    }

    @Test(arguments: gemvTypes)
    func blockGemvMatchesHostDecodePerSlot(type: GGUFType) throws {
        let context = try MetalContext()
        let gemv = try GGUFGemvBlocks(
            context: context, tables: try GGUFBlockTables.buffer(device: context.device))
        let device = context.device
        let ids: [Int32] = [2, 0, 3, 2]
        let scales: [Float] = [0.5, 1.25, 2.0, 0.75]
        // 37 rows leave a partial row group; 2,304 is the hidden width.
        for (rows, columns) in [(37, 2_304), (6, 512)] {
            let stride = rows * columns / Int(type.blockSize) * Int(type.blockBytes)
            let bytes = Self.blocks(type: type, rows: rows * 4, columns: columns)
            let decoded = try GGUFDequant.decode(type: type, bytes: bytes)
            for shared in [true, false] {
                let x = Self.activation((shared ? 1 : ids.count) * columns, seed: 1)
                let y = try #require(device.makeBuffer(length: ids.count * rows * 4))
                try Self.run(context) { encoder in
                    try gemv.encode(
                        .init(
                            type: type, rows: rows, columns: columns, slots: ids.count,
                            expertStride: stride, xSlotStride: shared ? 0 : columns),
                        weights: .init(try Self.buffer(device, bytes)),
                        x: .init(try Self.buffer(device, x)), y: .init(y),
                        ids: .init(try Self.buffer(device, ids)),
                        scales: .init(try Self.buffer(device, scales)), encoder: encoder)
                }
                Self.check(
                    Self.floats(y, ids.count * rows), decoded: decoded, rows: rows,
                    columns: columns, x: x, xRow: { shared ? 0 : $0 },
                    matrixOf: { Int(ids[$0]) }, scaleOf: { Double(scales[Int(ids[$0])]) },
                    outputs: ids.count, relative: 1e-5,
                    label: "\(type.name) \(rows)x\(columns) shared=\(shared)")
            }
        }
    }

    @Test(arguments: gemmTypes)
    func tiledDenseMatchesHostDecode(type: GGUFType) throws {
        let context = try MetalContext()
        let gemm = try GGUFGemmTiled(context: context)
        // 70 rows and 37 tokens leave partial tiles in both directions.
        let (rows, columns, tokens) = (70, 512, 37)
        let bytes = Self.blocks(type: type, rows: rows, columns: columns)
        let decoded = try GGUFDequant.decode(type: type, bytes: bytes)
        let x = Self.activation(tokens * columns, seed: 3)
        let y = try #require(context.device.makeBuffer(length: tokens * rows * 4))
        try Self.run(context) { encoder in
            try gemm.encodeDense(
                type: type, weights: .init(try Self.buffer(context.device, bytes)), rows: rows,
                columns: columns, tokens: tokens, x: .init(try Self.buffer(context.device, x)),
                y: .init(y), encoder: encoder)
        }
        // Half weights and activations with fp32 accumulation.
        Self.check(
            Self.floats(y, tokens * rows), decoded: decoded, rows: rows, columns: columns, x: x,
            xRow: { $0 }, matrixOf: { _ in 0 }, scaleOf: { _ in 1 }, outputs: tokens,
            relative: 2e-3, label: "\(type.name) dense")
    }

    @Test(arguments: expertTypes, [8, 32])
    func tiledExpertsMatchHostDecode(type: GGUFType, tileTokens: Int) throws {
        let context = try MetalContext()
        let gemm = try GGUFGemmTiled(context: context)
        let device = context.device
        let (rows, columns, experts, tokens, slots) = (40, 256, 6, 23, 3)
        let stride = rows * columns / Int(type.blockSize) * Int(type.blockBytes)
        let bytes = Self.blocks(type: type, rows: rows * experts, columns: columns)
        let decoded = try GGUFDequant.decode(type: type, bytes: bytes)
        let ids = (0..<(tokens * slots)).map { Int32(($0 * 7 + $0 / slots) % experts) }
        let scales = (0..<experts).map { Float($0 + 1) * 0.5 }
        let pairs = tokens * slots
        let lists = try #require(device.makeBuffer(length: experts * pairs * 4))
        let counts = try #require(device.makeBuffer(length: experts * 4))
        let idBuffer = try Self.buffer(device, ids)
        let x = Self.activation(tokens * columns, seed: 1)
        let y = try #require(device.makeBuffer(length: pairs * rows * 4))
        try Self.run(context) { encoder in
            gemm.encodeGrouping(
                ids: .init(idBuffer), pairs: pairs, experts: experts, lists: .init(lists),
                counts: .init(counts), listStride: pairs,
                barrier: { encoder.memoryBarrier(scope: .buffers) }, encoder: encoder)
            encoder.memoryBarrier(scope: .buffers)
            try gemm.encodeExperts(
                type: type, weights: .init(try Self.buffer(device, bytes)), rows: rows,
                columns: columns, experts: experts, expertStride: stride,
                maxPairsPerExpert: pairs, xDivisor: slots, x: .init(try Self.buffer(device, x)),
                y: .init(y), lists: .init(lists), counts: .init(counts),
                scales: .init(try Self.buffer(device, scales)), tileTokens: tileTokens,
                encoder: encoder)
        }
        Self.check(
            Self.floats(y, pairs * rows), decoded: decoded, rows: rows, columns: columns, x: x,
            xRow: { $0 / slots }, matrixOf: { Int(ids[$0]) },
            scaleOf: { Double(scales[Int(ids[$0])]) }, outputs: pairs, relative: 2e-3,
            label: "\(type.name) experts tile=\(tileTokens)")
    }

    @Test func iq4xsEmbeddingRowsMatchHostDecode() throws {
        let context = try MetalContext()
        let ops = try Inkling2GGUFOps(context: context)
        let device = context.device
        let (vocabulary, columns) = (9, 512)
        let bytes = Self.blocks(type: .iq4XS, rows: vocabulary, columns: columns)
        let decoded = try GGUFDequant.decode(type: .iq4XS, bytes: bytes)
        let tokens: [Int32] = [7, 0, 4]
        let table = Inkling2GGUFTensor(
            operand: .init(try Self.buffer(device, bytes)), type: .iq4XS, rows: vocabulary,
            columns: columns, byteCount: bytes.count)
        let out = try #require(device.makeBuffer(length: tokens.count * columns * 4))
        try Self.run(context) { encoder in
            try ops.embed(
                table, token: .init(try Self.buffer(device, tokens)), out: .init(out),
                tokens: tokens.count, encoder: encoder)
        }
        let got = Self.floats(out, tokens.count * columns)
        for (slot, token) in tokens.enumerated() {
            let want = decoded[Int(token) * columns..<(Int(token) + 1) * columns]
            #expect(Array(got[slot * columns..<(slot + 1) * columns]) == Array(want))
        }
    }
}
