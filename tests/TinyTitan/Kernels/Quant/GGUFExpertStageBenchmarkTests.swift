import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The whole routed-expert stage on real weights: every layer's gate/up/down
/// matrix in its own GGUF format, ten experts per layer, 1,440 fused GEMVs for
/// one token. This is the compute the engine's MoE step does per token, measured
/// end to end rather than extrapolated from one matrix.
@Suite struct GGUFExpertStageBenchmarkTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private struct Matrix {
        let buffer: MTLBuffer
        let rows: Int
        let columns: Int
        let type: GGUFType
    }

    private struct Layer {
        let gate: Matrix
        let up: Matrix
        let down: Matrix
    }

    private static func loadMatrix(
        _ context: MetalContext, model: GGUFModel, layer: Int, role: String
    ) throws -> Matrix {
        let weights = try GGUFExpertLayout.weights(layer: layer, expert: 0, in: model)
        let range: GGUFTensorRange
        let rows: Int
        let columns: Int
        switch role {
        case "gate":
            range = weights.gate
            rows = 640
            columns = 2_560
        case "up":
            range = weights.up
            rows = 640
            columns = 2_560
        default:
            range = weights.down
            rows = 2_560
            columns = 640
        }
        let handle = try FileHandle(forReadingFrom: model.shards[range.shardIndex].url)
        defer { try? handle.close() }
        try handle.seek(toOffset: range.fileOffset)
        guard let data = try handle.read(upToCount: Int(range.byteCount)) else {
            throw MetalError.bufferAllocationFailed("expert matrix read")
        }
        guard
            let buffer = context.device.makeBuffer(
                bytes: [UInt8](data), length: data.count, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("expert matrix buffer")
        }
        return Matrix(buffer: buffer, rows: rows, columns: columns, type: range.type)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func measuresOneTokensExpertStage() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let gemv = try GGUFExpertGEMV(context: context)

        var layers: [Layer] = []
        layers.reserveCapacity(48)
        for layer in 0..<48 {
            layers.append(
                Layer(
                    gate: try Self.loadMatrix(context, model: model, layer: layer, role: "gate"),
                    up: try Self.loadMatrix(context, model: model, layer: layer, role: "up"),
                    down: try Self.loadMatrix(context, model: model, layer: layer, role: "down")))
        }

        var rng = SeedTree(0x517A).key("expert-stage")
        let xWideBits = (0..<2_560).map { _ in Float16(rng.uniform(-1, 1)).bitPattern }
        let xDownBits = (0..<640).map { _ in Float16(rng.uniform(-1, 1)).bitPattern }
        func makeBuffer(_ bits: [UInt16], length: Int) throws -> MTLBuffer {
            let buffer: MTLBuffer? = bits.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return nil }
                return context.device.makeBuffer(
                    bytes: base, length: raw.count, options: .storageModeShared)
            }
            guard let buffer else { throw MetalError.bufferAllocationFailed("x") }
            return buffer
        }
        let xWide = try makeBuffer(xWideBits, length: 2_560)
        let xDown = try makeBuffer(xDownBits, length: 640)
        let yGate: MTLBuffer = try {
            guard
                let b = context.device.makeBuffer(
                    length: 640 * MemoryLayout<UInt16>.size, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("y") }
            return b
        }()
        let yDown: MTLBuffer = try {
            guard
                let b = context.device.makeBuffer(
                    length: 2_560 * MemoryLayout<UInt16>.size, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("y") }
            return b
        }()

        // Warm-up: one layer of the same work.
        func encodeToken(commit: Bool) throws -> [MTLCommandBuffer] {
            var buffers: [MTLCommandBuffer] = []
            for layer in layers {
                guard let commandBuffer = context.queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                for _ in 0..<10 {
                    try gemv.encode(
                        commandBuffer: commandBuffer, weights: layer.gate.buffer,
                        x: xWide, y: yGate, rows: layer.gate.rows,
                        columns: layer.gate.columns, type: layer.gate.type)
                    try gemv.encode(
                        commandBuffer: commandBuffer, weights: layer.up.buffer,
                        x: xWide, y: yGate, rows: layer.up.rows,
                        columns: layer.up.columns, type: layer.up.type)
                    try gemv.encode(
                        commandBuffer: commandBuffer, weights: layer.down.buffer,
                        x: xDown, y: yDown, rows: layer.down.rows,
                        columns: layer.down.columns, type: layer.down.type)
                }
                if commit {
                    commandBuffer.commit()
                    buffers.append(commandBuffer)
                }
            }
            return buffers
        }
        for buffer in try encodeToken(commit: true) {
            buffer.waitUntilCompleted()
            if let error = buffer.error {
                throw MetalError.invalidState("expert stage warm-up failed: \(error)")
            }
        }

        let start = DispatchTime.now().uptimeNanoseconds
        let buffers = try encodeToken(commit: true)
        for buffer in buffers {
            buffer.waitUntilCompleted()
            if let error = buffer.error {
                throw MetalError.invalidState("expert stage command failed: \(error)")
            }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let tokPerSecond = 1.0 / seconds
        print(
            "GGUF expert stage: \(String(format: "%.0f", seconds * 1000)) ms/token, "
                + "\(String(format: "%.2f", tokPerSecond)) tok/s expert-compute ceiling "
                + "(1,440 fused GEMVs over 48 layers x 10 experts)")
        #expect(seconds > 0)
    }
}
