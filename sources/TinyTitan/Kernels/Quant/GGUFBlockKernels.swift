import Foundation
import Metal
import TinyTitanCore

/// The ggml lookup tables the GPU block decoders read (`gguf_block_decoders.metal`),
/// in one shared buffer laid out at the offsets that file declares.
enum GGUFBlockTables {
    /// Byte offsets, matching `GGUF_TABLE_*` in gguf_block_decoders.metal.
    static let iq2xxs = 0
    static let iq2xs = 2_048
    static let iq2s = 6_144
    static let iq1s = 14_336
    static let iq3xxs = 30_720
    static let ksigns = 31_744
    static let iq4nl = 31_872
    static let byteCount = 31_888

    /// `kvalues_iq4nl` (ggml-common.h).
    static let iq4nlValues: [Int8] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ]

    static func bytes() -> [UInt8] {
        var out = [UInt8](repeating: 0, count: byteCount)
        func put<T: FixedWidthInteger>(_ values: [T], at offset: Int) {
            values.withUnsafeBytes { raw in
                for (index, byte) in raw.enumerated() { out[offset + index] = byte }
            }
        }
        put(GGUFIQTables.iq2xxsGrid, at: iq2xxs)
        put(GGUFIQTables.iq2xsGrid, at: iq2xs)
        put(GGUFIQTables.iq2sGrid, at: iq2s)
        put(GGUFIQTables.iq1sGrid, at: iq1s)
        put(GGUFIQTables.iq3xxsGrid, at: iq3xxs)
        put(GGUFIQTables.ksigns, at: ksigns)
        put(iq4nlValues, at: iq4nl)
        return out
    }

    static func buffer(device: MTLDevice) throws -> MTLBuffer {
        let table = bytes()
        guard
            let buffer = table.withUnsafeBytes({ raw -> MTLBuffer? in
                guard let base = raw.baseAddress else { return nil }
                return device.makeBuffer(
                    bytes: base, length: raw.count, options: .storageModeShared)
            })
        else { throw MetalError.bufferAllocationFailed("GGUF block tables") }
        return buffer
    }
}

/// Expert-indexed GEMVs over the block decoders (`gemv_blocks_*`) for the GGUF
/// types without a dedicated decode kernel: the same contract as
/// `GGUFGemvIds.encode`, `rowsPerSIMDGroup` rows per SIMD group, 16-value windows per lane.
final class GGUFGemvBlocks {
    typealias Operand = GGUFGemvIds.Operand

    private struct Args {
        var rows: UInt32
        var columns: UInt32
        var expertStride: UInt64
        var xSlotStride: UInt32
        var useScale: UInt32
    }

    static let kernels: [GGUFType: String] = [
        .iq2XXS: "gemv_blocks_iq2_xxs", .iq2XS: "gemv_blocks_iq2_xs", .iq2S: "gemv_blocks_iq2_s",
        .iq1S: "gemv_blocks_iq1_s", .iq3XXS: "gemv_blocks_iq3_xxs", .q2_0: "gemv_blocks_q2_0",
        .mxfp4: "gemv_blocks_mxfp4", .q4K: "gemv_blocks_q4_k", .q5K: "gemv_blocks_q5_k",
        .q6K: "gemv_blocks_q6_k", .iq4XS: "gemv_blocks_iq4_xs",
    ]

    /// `GGUF_GEMV_BLOCK_ROWS` in gguf_block_decoders.metal.
    static let rowsPerSIMDGroup = 4

    private var pipelines: [GGUFType: MTLComputePipelineState] = [:]
    let tables: MTLBuffer

    init(context: MetalContext, tables: MTLBuffer) throws {
        for (type, name) in Self.kernels { pipelines[type] = try context.pipeline(name) }
        self.tables = tables
    }

    func encode(
        _ shape: GGUFGemvIds.Shape, weights: Operand, x: Operand, y: Operand, ids: Operand,
        scales: Operand?, encoder: MTLComputeCommandEncoder
    ) throws {
        guard let pipeline = pipelines[shape.type] else {
            throw GGUFDequantError.unsupportedType(shape.type.name)
        }
        guard shape.columns % Int(shape.type.blockSize) == 0, x.offset % 16 == 0,
            shape.xSlotStride % 4 == 0
        else {
            throw MetalError.invalidState(
                "block GEMV: \(shape.rows) x \(shape.columns) \(shape.type.name)")
        }
        encoder.setComputePipelineState(pipeline)
        var args = Args(
            rows: UInt32(shape.rows), columns: UInt32(shape.columns),
            expertStride: UInt64(shape.expertStride), xSlotStride: UInt32(shape.xSlotStride),
            useScale: scales == nil ? 0 : 1)
        encoder.setBuffer(weights.buffer, offset: weights.offset, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(y.buffer, offset: y.offset, index: 2)
        encoder.setBuffer(ids.buffer, offset: ids.offset, index: 3)
        let scaleOperand = scales ?? ids
        encoder.setBuffer(scaleOperand.buffer, offset: scaleOperand.offset, index: 4)
        encoder.setBytes(&args, length: MemoryLayout<Args>.stride, index: 5)
        encoder.setBuffer(tables, offset: 0, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (shape.rows + 2 * Self.rowsPerSIMDGroup - 1) / (2 * Self.rowsPerSIMDGroup)
                    * shape.slots, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
}
