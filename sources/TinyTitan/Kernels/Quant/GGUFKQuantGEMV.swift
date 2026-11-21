import Foundation
import Metal
import TinyTitanCore

/// Fused Q2_K / Q4_K GEMV. Reads GGUF blocks in place; used by the Inkling2
/// GGUF path and microbenchmarked independently of the decoder.
final class GGUFKQuantGEMV {
    /// Weights, activation and output buffers allocated once, so a call or a
    /// benchmark measures the kernel rather than an upload.
    struct Prepared {
        let weights: MTLBuffer
        let x: MTLBuffer
        let y: MTLBuffer
        let rows: Int
        let columns: Int
        let type: GGUFType
    }

    /// The kernel geometries. `rowTiled` is the kernel of record; the one-row
    /// kernel stays as the simple, independently written baseline every
    /// geometry change is A/B'd against (`pairedGeometryBenchmark`).
    enum Kernel: CaseIterable, Sendable {
        /// One SIMD group per row, eight rows per threadgroup
        /// (`gguf_q2k_q4k_gemv.metal`).
        case rowPerSIMDGroup
        /// llama.cpp's geometry: all 32 lanes on a tile of rows, activations
        /// reused across the tile (`gguf_q2k_q4k_tiled.metal`). About 1.5x
        /// the one-row kernel on 2304 x 2304 (Q2_K 0.0299 vs 0.0443 ms,
        /// Q4_K 0.0333 vs 0.0516 ms GPU time per projection, M3 24 GB).
        case rowTiled
    }

    private struct Launch {
        let pipeline: MTLComputePipelineState
        let rowsPerThreadgroup: Int
        let threadsPerThreadgroup: Int
    }

    private let context: MetalContext
    private let pipelines: [String: MTLComputePipelineState]

    init(context: MetalContext) throws {
        self.context = context
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in ["q2k_gemv", "q4k_gemv", "q2k_gemv_tiled", "q4k_gemv_tiled"] {
            pipelines[name] = try context.pipeline(name)
        }
        self.pipelines = pipelines
    }

    private static func expectedBytes(rows: Int, columns: Int, type: GGUFType) throws -> Int {
        switch type {
        case .q2K: return rows * (columns / 256) * 84
        case .q4K: return rows * (columns / 256) * 144
        default: throw GGUFDequantError.unsupportedType(type.name)
        }
    }

    private func launch(_ kernel: Kernel, type: GGUFType) throws -> Launch {
        let prefix = type == .q2K ? "q2k" : "q4k"
        let name: String
        let rows: Int
        let threads: Int
        switch kernel {
        case .rowPerSIMDGroup:
            (name, rows, threads) = ("\(prefix)_gemv", 8, 256)
        case .rowTiled:
            // KQ_TILE_SIMDGROUPS (2) x KQ_TILE_ROWS_Q2K (4) / _Q4K (2).
            (name, rows, threads) = ("\(prefix)_gemv_tiled", type == .q2K ? 8 : 4, 64)
        }
        guard let pipeline = pipelines[name] else {
            throw MetalError.missingFunction(name)
        }
        return Launch(
            pipeline: pipeline, rowsPerThreadgroup: rows, threadsPerThreadgroup: threads)
    }

    func prepare(
        weights: [UInt8], rows: Int, columns: Int, type: GGUFType
    ) throws -> Prepared {
        let expected = try Self.expectedBytes(rows: rows, columns: columns, type: type)
        guard columns > 0, columns % 256 == 0, rows > 0, weights.count == expected,
            let weightBuffer = context.device.makeBuffer(
                bytes: weights, length: weights.count, options: .storageModeShared),
            let xBuffer = context.device.makeBuffer(
                length: columns * 4, options: .storageModeShared),
            let yBuffer = context.device.makeBuffer(
                length: rows * 4, options: .storageModeShared)
        else {
            throw GGUFDequantError.byteCountNotABlockMultiple(
                type: type.name, bytes: weights.count, blockBytes: Int(type.blockBytes))
        }
        return Prepared(
            weights: weightBuffer, x: xBuffer, y: yBuffer,
            rows: rows, columns: columns, type: type)
    }

    /// Encodes one GEMV of `prepared` into `encoder` without committing, so a
    /// caller can batch many projections into one command buffer.
    func encode(
        prepared: Prepared, kernel: Kernel, encoder: MTLComputeCommandEncoder
    ) throws {
        let launch = try launch(kernel, type: prepared.type)
        encoder.setComputePipelineState(launch.pipeline)
        encoder.setBuffer(prepared.weights, offset: 0, index: 0)
        encoder.setBuffer(prepared.x, offset: 0, index: 1)
        encoder.setBuffer(prepared.y, offset: 0, index: 2)
        var rowCount = UInt32(prepared.rows)
        var columnCount = UInt32(prepared.columns)
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&columnCount, length: 4, index: 4)
        let groups =
            (prepared.rows + launch.rowsPerThreadgroup - 1) / launch.rowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: launch.threadsPerThreadgroup, height: 1, depth: 1))
    }

    func gemv(
        prepared: Prepared, x: [Float], kernel: Kernel = .rowTiled
    ) throws -> [Float] {
        guard x.count == prepared.columns else {
            throw GGUFDequantError.byteCountNotABlockMultiple(
                type: prepared.type.name, bytes: x.count, blockBytes: prepared.columns)
        }
        try x.withUnsafeBytes { source in
            guard let base = source.baseAddress else {
                throw MetalError.bufferAllocationFailed("K-quant x copy")
            }
            prepared.x.contents().copyMemory(from: base, byteCount: x.count * 4)
        }
        guard let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        try encode(prepared: prepared, kernel: kernel, encoder: encoder)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = prepared.y.contents().bindMemory(
            to: Float.self, capacity: prepared.rows)
        return Array(UnsafeBufferPointer(start: pointer, count: prepared.rows))
    }

    /// GPU time of `dispatches` back-to-back GEMVs in one command buffer,
    /// divided by `dispatches`. This is the per-projection cost a batched
    /// executor pays; it excludes commit/wait latency, which a one-projection
    /// wall-clock measurement is dominated by at these sizes.
    func gpuSecondsPerDispatch(
        prepared: Prepared, kernel: Kernel, dispatches: Int
    ) throws -> Double {
        guard let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        for _ in 0..<dispatches {
            try encode(prepared: prepared, kernel: kernel, encoder: encoder)
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) / Double(dispatches)
    }

    /// Convenience path that uploads `weights` on every call. Use `prepare`
    /// for anything measured.
    func gemv(
        weights: [UInt8], x: [Float], rows: Int, columns: Int, type: GGUFType,
        kernel: Kernel = .rowTiled
    ) throws -> [Float] {
        guard x.count == columns else {
            throw GGUFDequantError.byteCountNotABlockMultiple(
                type: type.name, bytes: x.count, blockBytes: columns)
        }
        let prepared = try prepare(weights: weights, rows: rows, columns: columns, type: type)
        return try gemv(prepared: prepared, x: x, kernel: kernel)
    }
}
