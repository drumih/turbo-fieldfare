import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

@Suite struct IQ3XXSGEMVTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private struct Matrix {
        let blocks: [UInt8]
        let rows: Int
        let columns: Int
        let values: [Float]
    }

    /// blk.14's gate is IQ3_XXS in this checkpoint.
    private static func loadGate(_ model: GGUFModel, layer: Int) throws -> Matrix {
        let weights = try GGUFExpertLayout.weights(layer: layer, expert: 0, in: model)
        let range = weights.gate
        let handle = try FileHandle(forReadingFrom: model.shards[range.shardIndex].url)
        defer { try? handle.close() }
        try handle.seek(toOffset: range.fileOffset)
        let data = try #require(try handle.read(upToCount: Int(range.byteCount)))
        let values = try GGUFDequant.decode(type: range.type, bytes: [UInt8](data))
        return Matrix(blocks: [UInt8](data), rows: 640, columns: 2_560, values: values)
    }

    private static func run(
        _ context: MetalContext, matrix: Matrix, x: [Float16], iterations: Int
    ) throws -> (y: [Float16], seconds: Double) {
        let kernel = try IQ3XXSGEMV(context: context)
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard
            let wBuffer = context.device.makeBuffer(
                bytes: matrix.blocks, length: matrix.blocks.count, options: .storageModeShared),
            let xBuffer,
            let yBuffer = context.device.makeBuffer(
                length: matrix.rows * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("IQ3_XXS GEMV buffers")
        }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try kernel.encode(
                commandBuffer: commandBuffer, weights: wBuffer, x: xBuffer, y: yBuffer,
                rows: matrix.rows, columns: matrix.columns)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let pointer = yBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<matrix.rows).map { Float16(bitPattern: pointer[$0]) }, seconds)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func gateGEMVMatchesTheCPUDotProduct() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let matrix = try Self.loadGate(model, layer: 14)
        var rng = SeedTree(0x3C3C).key("iq3xxs-gemv-x")
        let x = (0..<matrix.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        let (y, _) = try Self.run(context, matrix: matrix, x: x, iterations: 1)
        var worstRelative: Float = 0
        for row in 0..<matrix.rows {
            var reference: Float = 0
            let base = row * matrix.columns
            for n in 0..<matrix.columns {
                reference += matrix.values[base + n] * Float(x[n])
            }
            worstRelative = max(
                worstRelative, abs(Float(y[row]) - reference) / max(abs(reference), 1e-3))
        }
        print("IQ3_XXS GEMV worst relative error: \(worstRelative)")
        #expect(worstRelative < 5e-3, "worst relative error \(worstRelative)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func measuresGateGEMV() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let matrix = try Self.loadGate(model, layer: 14)
        var rng = SeedTree(0x4D4D).key("iq3xxs-gemv-bench")
        let x = (0..<matrix.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        _ = try Self.run(context, matrix: matrix, x: x, iterations: 2)
        let iterations = 100
        let (_, seconds) = try Self.run(context, matrix: matrix, x: x, iterations: iterations)
        let perMatrixMs = seconds * 1000 / Double(iterations)
        print(
            "IQ3_XXS GEMV: \(String(format: "%.3f", perMatrixMs)) ms per 640x2560 matrix")
        #expect(perMatrixMs > 0)
    }
}
