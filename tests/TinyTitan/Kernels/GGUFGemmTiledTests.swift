import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct GGUFGemmTiledTests {
    private static var benchmarkEnabled: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_BENCH"] == "1"
    }

    /// Valid blocks of `type` (finite scales over every byte pattern).
    static func matrix(type: GGUFType, rows: Int, columns: Int, seed: Int = 0) -> [UInt8] {
        switch type {
        case .f32:
            return (0..<(rows * columns)).flatMap { index -> [UInt8] in
                let value = Float(((index + seed) * 37) % 41) / 41 - 0.5
                return withUnsafeBytes(of: value.bitPattern.littleEndian) { Array($0) }
            }
        case .bf16:
            return (0..<(rows * columns)).flatMap { index -> [UInt8] in
                let value = Float(((index + seed) * 29) % 43) / 43 - 0.5
                let bits = UInt16(value.bitPattern >> 16)
                return [UInt8(bits & 0xFF), UInt8(bits >> 8)]
            }
        default:
            let blocks = rows * columns / Int(type.blockSize)
            let blockBytes = Int(type.blockBytes)
            var bytes = (0..<(blocks * blockBytes)).map { UInt8(($0 &* 37 &+ 11 &+ seed) & 0xFF) }
            for block in 0..<blocks {
                let base = block * blockBytes
                if type == .nvfp4 {
                    for sub in 0..<4 { bytes[base + sub] = UInt8(0x28 + (block * 4 + sub) % 24) }
                } else {
                    let bits = Float16(0.002 + 0.0005 * Float(block % 9)).bitPattern
                    bytes[base] = UInt8(bits & 0xFF)
                    bytes[base + 1] = UInt8(bits >> 8)
                }
            }
            return bytes
        }
    }

    static func activations(_ count: Int, seed: Int) -> [Float] {
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

    private static func run(
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

    /// Checks y (rows-wide rows) against W · x in Double; the bound covers half
    /// weights and activations with fp32 accumulation.
    private static func check(
        _ got: UnsafeBufferPointer<Float>, decoded: [Float], rows: Int, columns: Int,
        x: [Float], xRow: (Int) -> Int, matrixOf: (Int) -> Int, scaleOf: (Int) -> Double,
        outputs: Int, label: String
    ) {
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
                #expect(
                    error <= 2e-3 * magnitude + 1e-4,
                    "\(label) output \(p) row \(row): \(got[p * rows + row]) vs \(value)")
            }
        }
    }

    @Test(arguments: [GGUFType.q8_0, .nvfp4, .f32, .bf16])
    func denseMatchesHostDecode(type: GGUFType) throws {
        let context = try MetalContext()
        let gemm = try GGUFGemmTiled(context: context)
        // 70 rows and 37 tokens leave partial tiles in both directions.
        let (rows, columns, tokens) = (70, 128, 37)
        let bytes = Self.matrix(type: type, rows: rows, columns: columns)
        let decoded = try GGUFDequant.decode(type: type, bytes: bytes)
        let x = Self.activations(tokens * columns, seed: 3)
        let y = try #require(context.device.makeBuffer(length: tokens * rows * 4))
        let weights = try Self.buffer(context.device, bytes)
        let xs = try Self.buffer(context.device, x)
        try Self.run(context) { encoder in
            try gemm.encodeDense(
                type: type, weights: .init(weights), rows: rows, columns: columns, tokens: tokens,
                x: .init(xs), y: .init(y), encoder: encoder)
        }
        let got = UnsafeBufferPointer(
            start: y.contents().bindMemory(to: Float.self, capacity: tokens * rows),
            count: tokens * rows)
        Self.check(
            got, decoded: decoded, rows: rows, columns: columns, x: x, xRow: { $0 },
            matrixOf: { _ in 0 }, scaleOf: { _ in 1 }, outputs: tokens, label: type.name)
    }

    @Test(arguments: [GGUFType.q8_0, .nvfp4])
    func expertGroupedMatchesHostDecode(type: GGUFType) throws {
        let context = try MetalContext()
        let gemm = try GGUFGemmTiled(context: context)
        let device = context.device
        let (rows, columns, experts, tokens, slots) = (40, 128, 6, 23, 3)
        let stride = rows * columns / Int(type.blockSize) * Int(type.blockBytes)
        let bytes = Self.matrix(type: type, rows: rows * experts, columns: columns)
        let decoded = try GGUFDequant.decode(type: type, bytes: bytes)
        let ids = (0..<(tokens * slots)).map { Int32(($0 * 7 + $0 / slots) % experts) }
        let scales = (0..<experts).map { Float($0 + 1) * 0.5 }
        let pairs = tokens * slots
        let lists = try #require(device.makeBuffer(length: experts * pairs * 4))
        let counts = try #require(device.makeBuffer(length: experts * 4))
        let idBuffer = try Self.buffer(device, ids)
        let sizes = try #require(device.makeBuffer(length: 12))
        for shared in [true, false] {
            // Shared: one input per token (gate_up); per pair: one per pair (down).
            let x = Self.activations((shared ? tokens : pairs) * columns, seed: shared ? 1 : 2)
            let y = try #require(device.makeBuffer(length: pairs * rows * 4))
            try Self.run(context) { encoder in
                gemm.encodeGrouping(
                    ids: .init(idBuffer), pairs: pairs, experts: experts, lists: .init(lists),
                    counts: .init(counts), listStride: pairs,
                    barrier: { encoder.memoryBarrier(scope: .buffers) }, encoder: encoder)
                encoder.memoryBarrier(scope: .buffers)
                gemm.encodeDispatchSizes(
                    counts: .init(counts), experts: experts, rowsOfShapes: [rows],
                    sizes: .init(sizes), encoder: encoder)
                encoder.memoryBarrier(scope: .buffers)
                try gemm.encodeExperts(
                    type: type, weights: .init(try Self.buffer(device, bytes)), rows: rows,
                    columns: columns, experts: experts, expertStride: stride,
                    maxPairsPerExpert: pairs, xDivisor: shared ? slots : 1,
                    x: .init(try Self.buffer(device, x)), y: .init(y), lists: .init(lists),
                    counts: .init(counts), scales: .init(try Self.buffer(device, scales)),
                    indirectSizes: shared ? nil : .init(sizes), encoder: encoder)
            }
            let got = UnsafeBufferPointer(
                start: y.contents().bindMemory(to: Float.self, capacity: pairs * rows),
                count: pairs * rows)
            Self.check(
                got, decoded: decoded, rows: rows, columns: columns, x: x,
                xRow: { shared ? $0 / slots : $0 }, matrixOf: { Int(ids[$0]) },
                scaleOf: { Double(scales[Int(ids[$0])]) }, outputs: pairs,
                label: "\(type.name) experts shared=\(shared)")
        }
    }

    /// GPU time of the dense Q/K/V/gate/rel projection (8,576 x 2,304 Q8_0) and
    /// of one routed layer (8 of 128 NVFP4 experts per token, gate_up and down)
    /// for a chunk of T tokens. Compare with the ANE probe's 1.21-1.50 ms (T=256).
    @Test(.enabled(if: benchmarkEnabled), arguments: [64, 256, 512])
    func prefillChunkBenchmark(tokens: Int) throws {
        let context = try MetalContext()
        let gemm = try GGUFGemmTiled(context: context)
        let device = context.device
        func time(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
            var samples: [Double] = []
            for _ in 0..<7 {
                let commandBuffer = try #require(context.queue.makeCommandBuffer())
                let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
                try body(encoder)
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                samples.append(commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
            }
            return samples.sorted()[3]
        }
        let (rows, columns) = (8_576, 2_304)
        let dense = try Self.buffer(device, Self.matrix(type: .q8_0, rows: rows, columns: columns))
        let x = try Self.buffer(device, Self.activations(tokens * columns, seed: 0))
        let y = try #require(device.makeBuffer(length: tokens * rows * 4))
        let denseSeconds = try time { encoder in
            try gemm.encodeDense(
                type: .q8_0, weights: .init(dense), rows: rows, columns: columns, tokens: tokens,
                x: .init(x), y: .init(y), encoder: encoder)
        }
        let (experts, feedForward, slots) = (128, 1_152, 8)
        let gateUpStride = 2 * feedForward * columns / 64 * 36
        let downStride = columns * feedForward / 64 * 36
        let tile = Self.matrix(type: .nvfp4, rows: 2 * feedForward, columns: columns)
        let gateUp = try #require(device.makeBuffer(length: gateUpStride * experts))
        let down = try #require(device.makeBuffer(length: downStride * experts))
        for buffer in [gateUp, down] {
            tile.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                for offset in stride(from: 0, to: buffer.length, by: raw.count) {
                    buffer.contents().advanced(by: offset).copyMemory(
                        from: base, byteCount: min(raw.count, buffer.length - offset))
                }
            }
        }
        let pairs = tokens * slots
        let sizes = try #require(device.makeBuffer(length: 24))
        let ids = try Self.buffer(
            device, (0..<pairs).map { Int32(($0 * 37 + ($0 / slots) * 11) % experts) })
        let lists = try #require(device.makeBuffer(length: experts * pairs * 4))
        let counts = try #require(device.makeBuffer(length: experts * 4))
        let act = try Self.buffer(device, Self.activations(pairs * feedForward, seed: 1))
        let gateUpOut = try #require(device.makeBuffer(length: pairs * 2 * feedForward * 4))
        let downOut = try #require(device.makeBuffer(length: pairs * columns * 4))
        let moeSeconds = try time { encoder in
            gemm.encodeGrouping(
                ids: .init(ids), pairs: pairs, experts: experts, lists: .init(lists),
                counts: .init(counts), listStride: pairs,
                barrier: { encoder.memoryBarrier(scope: .buffers) }, encoder: encoder)
            encoder.memoryBarrier(scope: .buffers)
            gemm.encodeDispatchSizes(
                counts: .init(counts), experts: experts, rowsOfShapes: [2 * feedForward, columns],
                sizes: .init(sizes), encoder: encoder)
            encoder.memoryBarrier(scope: .buffers)
            try gemm.encodeExperts(
                type: .nvfp4, weights: .init(gateUp), rows: 2 * feedForward, columns: columns,
                experts: experts, expertStride: gateUpStride, maxPairsPerExpert: pairs,
                xDivisor: slots, x: .init(x), y: .init(gateUpOut), lists: .init(lists),
                counts: .init(counts), scales: nil, indirectSizes: .init(sizes), encoder: encoder)
            try gemm.encodeExperts(
                type: .nvfp4, weights: .init(down), rows: columns, columns: feedForward,
                experts: experts, expertStride: downStride, maxPairsPerExpert: pairs,
                xDivisor: 1, x: .init(act), y: .init(downOut), lists: .init(lists),
                counts: .init(counts), scales: nil, indirectSizes: .init(sizes, offset: 12),
                encoder: encoder)
        }
        let written = sizes.contents().bindMemory(to: UInt32.self, capacity: 6)
        let countValues = counts.contents().bindMemory(to: UInt32.self, capacity: experts)
        let largest = (0..<experts).map { countValues[$0] }.max() ?? 0
        FileHandle.standardError.write(
            Data(
                "T=\(tokens) sizes \((0..<6).map { written[$0] }) largest expert \(largest)\n".utf8)
        )
        let denseFlops = 2.0 * Double(tokens * rows * columns)
        let moeFlops = 2.0 * Double(pairs) * Double(3 * feedForward * columns)
        #expect(denseSeconds > 0 && moeSeconds > 0)
        FileHandle.standardError.write(
            Data(
                String(
                    format:
                        "T=%d dense Q8_0 8576x2304: %.3f ms (%.2f TF/s); routed NVFP4 layer: %.3f ms (%.2f TF/s, %.1f us/token)\n",
                    tokens, denseSeconds * 1e3, denseFlops / denseSeconds / 1e12, moeSeconds * 1e3,
                    moeFlops / moeSeconds / 1e12, moeSeconds / Double(tokens) * 1e6
                ).utf8))
    }
}
