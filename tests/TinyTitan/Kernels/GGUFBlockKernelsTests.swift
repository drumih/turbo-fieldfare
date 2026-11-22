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

extension GGUFBlockKernelsTests {
    private static var benchmarkEnabled: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_BENCH"] == "1"
    }

    /// Effective GB/s of the decode block GEMV per type, on one routed layer's
    /// shapes: eight slots over 128 experts of 2,304 x 2,304 (gate_up-sized)
    /// or 2,304 x 1,152 (down),
    /// so the reads are not cache-resident. Compare with the dedicated
    /// NVFP4/Q8_0 kernels' ~87 GB/s (`GGUFGemvIdsTests/routedLayerBenchmark`).
    /// Opt-in: `TINYTITAN_KERNEL_BENCH=1`.
    @Test(
        .enabled(if: benchmarkEnabled),
        arguments: gemvTypes.map { ($0, 2_304) } + [GGUFType.q2_0, .mxfp4].map { ($0, 1_152) })
    func blockGemvBandwidth(type: GGUFType, columns: Int) throws {
        let context = try MetalContext()
        let gemv = try GGUFGemvBlocks(
            context: context, tables: try GGUFBlockTables.buffer(device: context.device))
        let device = context.device
        let (experts, rows, slots) = (128, 2_304, 8)
        let stride = rows * columns / Int(type.blockSize) * Int(type.blockBytes)
        let one = Self.blocks(type: type, rows: rows, columns: columns)
        let weights = try #require(device.makeBuffer(length: stride * experts))
        for expert in 0..<experts {
            one.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                weights.contents().advanced(by: expert * stride).copyMemory(
                    from: base, byteCount: stride)
            }
        }
        let x = try Self.buffer(device, Self.activation(columns, seed: 0))
        let y = try #require(device.makeBuffer(length: slots * rows * 4))
        var samples: [Double] = []
        for round in 0..<9 {
            let ids = (0..<slots).map { Int32(($0 * 13 + round * 37) % experts) }
            let idBuffer = try Self.buffer(device, ids)
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
            for _ in 0..<20 {
                try gemv.encode(
                    .init(
                        type: type, rows: rows, columns: columns, slots: slots,
                        expertStride: stride, xSlotStride: 0),
                    weights: .init(weights), x: .init(x), y: .init(y), ids: .init(idBuffer),
                    scales: nil, encoder: encoder)
            }
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            samples.append((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) / 20)
        }
        let median = samples.sorted()[4]
        let gbps = Double(stride * slots) / median / 1e9
        print(
            String(
                format: "block GEMV %@ %dx%d x%d slots: %.3f ms, %.1f GB/s", type.name, rows,
                columns, slots, median * 1e3, gbps))
        #expect(median > 0)
    }
}

extension GGUFBlockKernelsTests {
    /// GPU ms of one dense projection GEMV (one slot) at Inkling2's shapes, the
    /// block kernels against the dedicated Q8_0 one, as the decode runs them:
    /// 48 projections of one shape spread over distinct weights in one buffer.
    @Test(
        .enabled(if: benchmarkEnabled), arguments: [(3_072, 2_304), (1_024, 2_304), (2_304, 3_072)])
    func denseProjectionBenchmark(rows: Int, columns: Int) throws {
        let context = try MetalContext()
        let blocks = try GGUFGemvBlocks(
            context: context, tables: try GGUFBlockTables.buffer(device: context.device))
        let ids = try GGUFGemvIds(context: context)
        let device = context.device
        let copies = 48
        let zero = try Self.buffer(device, [Int32(0)])
        let x = try Self.buffer(device, Self.activation(columns, seed: 0))
        let y = try #require(device.makeBuffer(length: rows * 4))
        for type in [GGUFType.q8_0, .q6K, .q5K, .q4K, .iq3XXS] {
            let stride = rows * columns / Int(type.blockSize) * Int(type.blockBytes)
            let one =
                type == .q8_0
                ? GGUFGemmTiledTests.matrix(type: .q8_0, rows: rows, columns: columns)
                : Self.blocks(type: type, rows: rows, columns: columns)
            let weights = try #require(device.makeBuffer(length: stride * copies))
            for copy in 0..<copies {
                one.withUnsafeBytes { raw in
                    guard let base = raw.baseAddress else { return }
                    weights.contents().advanced(by: copy * stride).copyMemory(
                        from: base, byteCount: stride)
                }
            }
            var samples: [Double] = []
            for _ in 0..<7 {
                let commandBuffer = try #require(context.queue.makeCommandBuffer())
                let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
                for copy in 0..<copies {
                    let shape = GGUFGemvIds.Shape(
                        type: type, rows: rows, columns: columns, slots: 1, expertStride: 0,
                        xSlotStride: 0)
                    let operand = GGUFGemvIds.Operand(weights, offset: copy * stride)
                    if type == .q8_0 {
                        try ids.encode(
                            shape, weights: operand, x: .init(x), y: .init(y), ids: .init(zero),
                            scales: nil, encoder: encoder)
                    } else {
                        try blocks.encode(
                            shape, weights: operand, x: .init(x), y: .init(y), ids: .init(zero),
                            scales: nil, encoder: encoder)
                    }
                }
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                samples.append(
                    (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) / Double(copies))
            }
            let median = samples.sorted()[3]
            print(
                String(
                    format: "dense %@ %dx%d: %.4f ms, %.1f GB/s", type.name, rows, columns,
                    median * 1e3, Double(stride) / median / 1e9))
            #expect(median > 0)
        }
    }
}

extension GGUFBlockKernelsTests {
    /// GPU ms and GFLOP/s of the small-rows f32 GEMM over a chunk at the
    /// controller (64 x 4,608) and router (128 x 2,304) shapes, through
    /// `Inkling2GGUFOps.gemv`, against the host result.
    @Test(arguments: [(64, 4_608, 512), (128, 2_304, 512), (24, 2_304, 37)])
    func smallRowsChunkGemm(rows: Int, columns: Int, tokens: Int) throws {
        let context = try MetalContext()
        let ops = try Inkling2GGUFOps(context: context)
        let device = context.device
        let w = Self.activation(rows * columns, seed: 5)
        let x = Self.activation(tokens * columns, seed: 9)
        let wb = try Self.buffer(device, w)
        let tensor = Inkling2GGUFTensor(
            operand: .init(wb), type: .f32, rows: rows, columns: columns, byteCount: w.count * 4)
        let xb = try Self.buffer(device, x)
        let y = try #require(device.makeBuffer(length: tokens * rows * 4))
        var samples: [Double] = []
        for _ in 0..<5 {
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
            for _ in 0..<10 {
                try ops.gemv(tensor, x: .init(xb), y: .init(y), tokens: tokens, encoder: encoder)
            }
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            samples.append((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) / 10)
        }
        let got = Self.floats(y, tokens * rows)
        for t in Swift.stride(from: 0, to: tokens, by: 7) {
            for r in 0..<rows {
                var want = 0.0
                for c in 0..<columns {
                    want += Double(w[r * columns + c]) * Double(x[t * columns + c])
                }
                #expect(abs(Double(got[t * rows + r]) - want) <= 1e-4 * max(1, abs(want)))
            }
        }
        let median = samples.sorted()[2]
        print(
            String(
                format: "small-rows GEMM %dx%d x%d tokens: %.3f ms, %.1f GFLOP/s", rows, columns,
                tokens, median * 1e3, 2.0 * Double(rows * columns * tokens) / median / 1e9))
    }
}

extension GGUFBlockKernelsTests {
    /// GPU ms of one routed layer at a chunk (8 distinct of 128 experts per
    /// token; gate_up 2,304 -> 2 x 1,152, down 1,152 -> 2,304) for Q2_K_XL's
    /// and Q4_K_XL's type pairs, with the weight bytes the experts hold, so a
    /// tile that is decode- rather than bandwidth-bound shows as low GB/s.
    @Test(
        .enabled(if: benchmarkEnabled),
        arguments: [
            (GGUFType.nvfp4, GGUFType.nvfp4), (.iq2S, .q2_0), (.iq1S, .q2_0), (.iq2XXS, .nvfp4),
        ], [64, 512])
    func routedChunkBenchmark(types: (GGUFType, GGUFType), tokens: Int) throws {
        let context = try MetalContext()
        let gemm = try GGUFGemmTiled(context: context)
        let device = context.device
        let (experts, hidden, feedForward, slots) = (128, 2_304, 1_152, 8)
        func expertBuffer(_ type: GGUFType, rows: Int, columns: Int) throws -> (MTLBuffer, Int) {
            let stride = rows * columns / Int(type.blockSize) * Int(type.blockBytes)
            let one = Self.blocks(type: type, rows: rows, columns: columns)
            let buffer = try #require(device.makeBuffer(length: stride * experts))
            one.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                for e in 0..<experts {
                    buffer.contents().advanced(by: e * stride).copyMemory(
                        from: base, byteCount: stride)
                }
            }
            return (buffer, stride)
        }
        let (gateUp, gateUpStride) = try expertBuffer(
            types.0, rows: 2 * feedForward, columns: hidden)
        let (down, downStride) = try expertBuffer(types.1, rows: hidden, columns: feedForward)
        let pairs = tokens * slots
        let ids = try Self.buffer(
            device, (0..<pairs).map { Int32(($0 % slots) * 16 + (($0 / slots) * 5) % 16) })
        let tile = GGUFGemmTiled.expertTileTokens(pairs: pairs, experts: experts)
        let lists = try #require(device.makeBuffer(length: experts * pairs * 4))
        let counts = try #require(device.makeBuffer(length: experts * 4))
        let x = try Self.buffer(device, Self.activation(tokens * hidden, seed: 2))
        let act = try Self.buffer(device, Self.activation(pairs * feedForward, seed: 3))
        let gateUpOut = try #require(device.makeBuffer(length: pairs * 2 * feedForward * 4))
        let downOut = try #require(device.makeBuffer(length: pairs * hidden * 4))
        var samples: [Double] = []
        for _ in 0..<7 {
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
            gemm.encodeGrouping(
                ids: .init(ids), pairs: pairs, experts: experts, lists: .init(lists),
                counts: .init(counts), listStride: pairs,
                barrier: { encoder.memoryBarrier(scope: .buffers) }, encoder: encoder)
            encoder.memoryBarrier(scope: .buffers)
            try gemm.encodeExperts(
                type: types.0, weights: .init(gateUp), rows: 2 * feedForward, columns: hidden,
                experts: experts, expertStride: gateUpStride, maxPairsPerExpert: tokens,
                xDivisor: slots, x: .init(x), y: .init(gateUpOut), lists: .init(lists),
                counts: .init(counts), scales: nil, listStride: pairs, tileTokens: tile,
                encoder: encoder)
            encoder.memoryBarrier(scope: .buffers)
            try gemm.encodeExperts(
                type: types.1, weights: .init(down), rows: hidden, columns: feedForward,
                experts: experts, expertStride: downStride, maxPairsPerExpert: tokens,
                xDivisor: 1, x: .init(act), y: .init(downOut), lists: .init(lists),
                counts: .init(counts), scales: nil, listStride: pairs, tileTokens: tile,
                encoder: encoder)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            samples.append(commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
        }
        let median = samples.sorted()[3]
        let bytes = Double((gateUpStride + downStride) * experts)
        print(
            String(
                format:
                    "routed %@/%@ T=%d tile=%d: %.3f ms, %.1f GB/s of expert weights, x48 = %.0f ms",
                types.0.name, types.1.name, tokens, tile, median * 1e3, bytes / median / 1e9,
                median * 48e3))
        #expect(median > 0)
    }
}
