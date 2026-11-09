import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The PLE gather over real IQ4_NL rows: correctness against the CPU decoder,
/// then the rows/s rate. The table is 28.8 GB and is never materialised here -
/// a window of rows stands in for it, which is exactly what a token reads.
@Suite struct PLEGatherTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private struct Window {
        let bytes: [UInt8]
        let rows: Int
        let values: [Float]
    }

    private static func loadWindow(_ model: GGUFModel, firstRow: Int, rows: Int) throws -> Window {
        let name = "per_layer_token_embd.weight"
        let location = try #require(model.tensor(named: name))
        let shard = model.shards[location.shardIndex]
        let rowBytes = PLEGatherIQ4NL.rowBytes
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(
            toOffset: shard.dataOffset + location.tensor.offset
                + UInt64(firstRow * rowBytes))
        let data = try #require(try handle.read(upToCount: rows * rowBytes))
        let bytes = [UInt8](data)
        return Window(
            bytes: bytes, rows: rows,
            values: try GGUFDequant.decode(type: .iq4NL, bytes: bytes))
    }

    private static func run(
        _ context: MetalContext, window: Window, iterations: Int
    ) throws -> (values: [UInt16], seconds: Double) {
        let kernel = try PLEGatherIQ4NL(context: context)
        let indices = (0..<UInt32(window.rows)).map { $0 }
        let rowBuffer: MTLBuffer? = indices.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard
            let table = context.device.makeBuffer(
                bytes: window.bytes, length: window.bytes.count, options: .storageModeShared),
            let rowBuffer,
            let out = context.device.makeBuffer(
                length: window.rows * PLEGatherIQ4NL.rowElements * MemoryLayout<UInt16>.size,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("PLE gather buffers")
        }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try kernel.encode(
                commandBuffer: commandBuffer, table: table, rows: rowBuffer,
                count: window.rows, out: out)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<(window.rows * 160)).map { pointer[$0] }, seconds)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func gathersIQ4NLRowsLikeTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let window = try Self.loadWindow(model, firstRow: 100_000, rows: 64)
        let context = try MetalContext()
        let (gpu, _) = try Self.run(context, window: window, iterations: 1)
        var mismatches = 0
        for index in 0..<gpu.count
        where gpu[index] != Quantization.bf16Bits(window.values[index]) {
            mismatches += 1
        }
        #expect(mismatches == 0, "\(mismatches) of \(gpu.count) values differ")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func measuresGatherRate() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let window = try Self.loadWindow(model, firstRow: 200_000, rows: 1_024)
        let context = try MetalContext()
        _ = try Self.run(context, window: window, iterations: 2)
        let iterations = 50
        let (_, seconds) = try Self.run(context, window: window, iterations: iterations)
        let rowsPerSecond = Double(window.rows * iterations) / seconds
        // A token asks for (ngramSize - 1) x headsPerNgram = 16 rows on the one
        // PLE layer.
        print(
            "PLE gather: \(String(format: "%.0f", rowsPerSecond)) rows/s; "
                + "16 rows/token = \(String(format: "%.4f", 16 / rowsPerSecond * 1000)) ms/token")
        #expect(rowsPerSecond > 0)
    }
}
