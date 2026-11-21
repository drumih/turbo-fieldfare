import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct GGUFGemvIdsTests {
    private static var benchmarkEnabled: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_BENCH"] == "1"
    }

    /// Deterministic expert matrices: every byte pattern, with realistic
    /// scales written over each block's scale bytes so values stay finite.
    private static func experts(type: GGUFType, experts: Int, rows: Int, columns: Int) -> [UInt8] {
        let blocks = experts * rows * columns / Int(type.blockSize)
        let blockBytes = Int(type.blockBytes)
        var bytes = (0..<(blocks * blockBytes)).map { UInt8(($0 &* 37 &+ 11) & 0xFF) }
        for block in 0..<blocks {
            let base = block * blockBytes
            switch type {
            case .nvfp4:
                // UE4M3 codes 0x28...0x3F: 2^-2 ... ~1.9, one per sub-block.
                for sub in 0..<4 { bytes[base + sub] = UInt8(0x28 + (block * 4 + sub) % 24) }
            case .q8_0:
                let bits = Float16(0.002 + 0.0005 * Float(block % 9)).bitPattern
                bytes[base] = UInt8(bits & 0xFF)
                bytes[base + 1] = UInt8(bits >> 8)
            default:
                break
            }
        }
        return bytes
    }

    private static func activation(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { Float((($0 + seed) * 17) % 29) / 29.0 - 0.5 }
    }

    private static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
        let made = values.withUnsafeBytes { raw -> MTLBuffer? in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base, length: max(raw.count, 16), options: .storageModeShared)
        }
        guard let made else { throw MetalError.bufferAllocationFailed("test operand") }
        return made
    }

    /// Tiles `pattern` across the whole of `buffer`.
    private static func fill(_ buffer: MTLBuffer, with pattern: [UInt8]) {
        pattern.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for offset in stride(from: 0, to: buffer.length, by: raw.count) {
                buffer.contents().advanced(by: offset).copyMemory(
                    from: base, byteCount: min(raw.count, buffer.length - offset))
            }
        }
    }

    @Test(arguments: [GGUFType.nvfp4, .q8_0])
    func matchesHostDecodePerSlot(type: GGUFType) throws {
        let context = try MetalContext()
        let gemv = try GGUFGemvIds(context: context)
        let device = context.device
        let expertCount = 5
        let ids: [Int32] = [3, 0, 4, 3]
        let scales: [Float] = [0.5, 1.25, 2.0, 0.75, 3.0]
        // 37 rows leave partial row tiles; 1,152 and 2,304 columns are the
        // MoE down and gate_up widths (18 and 36 NVFP4 blocks, uneven lanes).
        for (rows, columns) in [(37, 1_152), (6, 2_304)] {
            let shape = GGUFGemvIds.Shape(
                type: type, rows: rows, columns: columns, slots: ids.count,
                expertStride: 0, xSlotStride: 0)
            let stride = shape.matrixBytes
            let bytes = Self.experts(type: type, experts: expertCount, rows: rows, columns: columns)
            #expect(bytes.count == stride * expertCount)
            let decoded = try GGUFDequant.decode(type: type, bytes: bytes)
            for shared in [true, false] {
                let x =
                    shared
                    ? Self.activation(columns, seed: 1)
                    : (0..<ids.count).flatMap { Self.activation(columns, seed: $0 + 2) }
                let yBuffer = try #require(
                    device.makeBuffer(
                        length: ids.count * rows * 4, options: .storageModeShared))
                let call = GGUFGemvIds.Shape(
                    type: type, rows: rows, columns: columns, slots: ids.count,
                    expertStride: stride, xSlotStride: shared ? 0 : columns)
                let commandBuffer = try #require(context.queue.makeCommandBuffer())
                let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
                try gemv.encode(
                    call, weights: .init(try Self.buffer(device, bytes)),
                    x: .init(try Self.buffer(device, x)), y: .init(yBuffer),
                    ids: .init(try Self.buffer(device, ids)),
                    scales: .init(try Self.buffer(device, scales)), encoder: encoder)
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                let got = UnsafeBufferPointer(
                    start: yBuffer.contents().bindMemory(
                        to: Float.self, capacity: ids.count * rows),
                    count: ids.count * rows)
                for (slot, id) in ids.enumerated() {
                    let xs = shared ? 0 : slot * columns
                    for row in 0..<rows {
                        let base = (Int(id) * rows + row) * columns
                        var value = 0.0
                        var magnitude = 0.0
                        for column in 0..<columns {
                            let term = Double(decoded[base + column]) * Double(x[xs + column])
                            value += term
                            magnitude += abs(term)
                        }
                        value *= Double(scales[Int(id)])
                        magnitude *= Double(scales[Int(id)])
                        let error = abs(Double(got[slot * rows + row]) - value)
                        #expect(
                            error <= 1e-5 * magnitude + 1e-6,
                            "\(type.name) \(rows)x\(columns) shared=\(shared) slot \(slot) row \(row): \(got[slot * rows + row]) vs \(value)"
                        )
                    }
                }
            }
        }
    }

    /// GPU time of one decode token's routed-expert work for a layer: eight
    /// gate_up GEMVs (2,304 x 2,304) on a shared input and eight down GEMVs
    /// (2,304 x 1,152) on per-slot inputs, with the weights spread across 128
    /// experts so the reads are not cache-resident. Reports effective GB/s,
    /// the number to compare with the M3's memory bandwidth.
    /// Opt-in: `TINYTITAN_KERNEL_BENCH=1 swift test -c release --filter GGUFGemvIdsTests`.
    @Test(.enabled(if: benchmarkEnabled), arguments: [GGUFType.nvfp4, .q8_0])
    func routedLayerBenchmark(type: GGUFType) throws {
        let context = try MetalContext()
        let gemv = try GGUFGemvIds(context: context)
        let device = context.device
        let experts = 128
        let hidden = 2_304
        let feedForward = 1_152
        let gateUp = GGUFGemvIds.Shape(
            type: type, rows: 2 * feedForward, columns: hidden, slots: 8,
            expertStride: 0, xSlotStride: 0)
        let down = GGUFGemvIds.Shape(
            type: type, rows: hidden, columns: feedForward, slots: 8,
            expertStride: 0, xSlotStride: feedForward)
        let gateUpBytes = gateUp.matrixBytes * experts
        let downBytes = down.matrixBytes * experts
        let weightsGateUp = try #require(
            device.makeBuffer(length: gateUpBytes, options: .storageModeShared))
        let weightsDown = try #require(
            device.makeBuffer(length: downBytes, options: .storageModeShared))
        // Fill with valid blocks so no NaN path skews timing.
        let tile = Self.experts(type: type, experts: 1, rows: 2 * feedForward, columns: hidden)
        Self.fill(weightsGateUp, with: tile)
        Self.fill(weightsDown, with: tile)
        let x = try Self.buffer(device, Self.activation(hidden, seed: 0))
        let act = try Self.buffer(device, Self.activation(8 * feedForward, seed: 3))
        let gateUpOut = try #require(device.makeBuffer(length: 8 * 2 * feedForward * 4))
        let downOut = try #require(device.makeBuffer(length: 8 * hidden * 4))
        let scales = try Self.buffer(device, [Float](repeating: 1, count: experts))
        let layers = 48
        var idSets: [MTLBuffer] = []
        for layer in 0..<layers {
            idSets.append(
                try Self.buffer(device, (0..<8).map { Int32((layer * 37 + $0 * 13) % experts) }))
        }
        var samples: [Double] = []
        for _ in 0..<5 {
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
            for ids in idSets {
                try gemv.encode(
                    GGUFGemvIds.Shape(
                        type: type, rows: gateUp.rows, columns: hidden, slots: 8,
                        expertStride: gateUp.matrixBytes, xSlotStride: 0),
                    weights: .init(weightsGateUp), x: .init(x), y: .init(gateUpOut),
                    ids: .init(ids), scales: .init(scales), encoder: encoder)
                try gemv.encode(
                    GGUFGemvIds.Shape(
                        type: type, rows: hidden, columns: feedForward, slots: 8,
                        expertStride: down.matrixBytes, xSlotStride: feedForward),
                    weights: .init(weightsDown), x: .init(act), y: .init(downOut),
                    ids: .init(ids), scales: .init(scales), encoder: encoder)
            }
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            samples.append(
                (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) / Double(layers))
        }
        samples.sort()
        let median = samples[samples.count / 2]
        let bytesPerLayer = Double(8 * (gateUp.matrixBytes + down.matrixBytes))
        #expect(median > 0)
        FileHandle.standardError.write(
            Data(
                String(
                    format:
                        "%@ routed layer (8 x gate_up + 8 x down): median %.4f ms, %.1f GB/s, x48 = %.2f ms/token\n",
                    type.name, median * 1e3, bytesPerLayer / median / 1e9, median * 48e3
                ).utf8))
    }
}
