import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The down-projection GEMV on real GGUF weights: correctness against a CPU
/// decode-then-dot, then the per-matrix time.
@Suite struct IQ4NLGEMVTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private struct Matrix {
        let blocks: [UInt8]
        let rows: Int
        let columns: Int
        let values: [Float]
    }

    private static func loadDown(_ model: GGUFModel) throws -> Matrix {
        let weights = try GGUFExpertLayout.weights(layer: 0, expert: 0, in: model)
        let range = weights.down
        let handle = try FileHandle(forReadingFrom: model.shards[range.shardIndex].url)
        defer { try? handle.close() }
        try handle.seek(toOffset: range.fileOffset)
        let data = try #require(try handle.read(upToCount: Int(range.byteCount)))
        let values = try GGUFDequant.decode(type: range.type, bytes: [UInt8](data))
        return Matrix(blocks: [UInt8](data), rows: 2_560, columns: 640, values: values)
    }

    private static func run(
        _ context: MetalContext, matrix: Matrix, x: [Float16], iterations: Int
    ) throws -> (y: [Float16], seconds: Double) {
        let kernel = try IQ4NLGEMV(context: context)
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
            throw MetalError.bufferAllocationFailed("IQ4_NL GEMV buffers")
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
    func downGEMVMatchesTheCPUDotProduct() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let matrix = try Self.loadDown(model)
        var rng = SeedTree(0xD0).key("iq4nl-gemv-x")
        let x = (0..<matrix.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        let (y, _) = try Self.run(context, matrix: matrix, x: x, iterations: 1)

        // Determinism across dispatches: the same input must produce the same
        // output. The first run of this suite once reported 0.266 and three
        // consecutive runs then reported 4.8e-4; a data race or a stale shader
        // resource would show up here rather than as a one-off.
        let (again, _) = try Self.run(context, matrix: matrix, x: x, iterations: 1)
        #expect(y == again, "the same dispatch produced different results")

        var worstRelative: Float = 0
        for row in 0..<matrix.rows {
            var reference: Float = 0
            let base = row * matrix.columns
            for n in 0..<matrix.columns {
                reference += matrix.values[base + n] * Float(x[n])
            }
            let scale = max(abs(reference), 1e-3)
            worstRelative = max(worstRelative, abs(Float(y[row]) - reference) / scale)
            if worstRelative > 5e-3 { break }
        }
        print("IQ4_NL GEMV worst relative error: \(worstRelative)")
        #expect(worstRelative < 5e-3, "worst relative error \(worstRelative)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func measuresDownGEMVPerToken() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let matrix = try Self.loadDown(model)
        var rng = SeedTree(0xE0).key("iq4nl-gemv-bench")
        let x = (0..<matrix.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        _ = try Self.run(context, matrix: matrix, x: x, iterations: 2)
        let iterations = 100
        let (_, seconds) = try Self.run(context, matrix: matrix, x: x, iterations: iterations)
        let perMatrixMs = seconds * 1000 / Double(iterations)
        let matricesPerToken = 48.0 * 10.0  // down matrices only
        let computeMs = perMatrixMs * matricesPerToken
        print(
            "IQ4_NL GEMV: \(String(format: "%.3f", perMatrixMs)) ms per 2560x640 matrix; "
                + "down compute \(String(format: "%.0f", computeMs)) ms/token")
        #expect(perMatrixMs > 0)
    }
}
