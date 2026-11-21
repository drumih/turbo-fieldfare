import Foundation
import Metal
import TinyTitanCore

/// Expert-indexed GGUF GEMVs (`gguf_gemv_ids.metal`): one dispatch multiplies
/// `slots` matrices, each chosen by an expert id read on the GPU, so a routed
/// MoE layer never returns to the CPU between router and experts. A dense
/// matrix is one slot whose id is 0.
final class GGUFGemvIds {
    /// A buffer plus the byte offset its data starts at.
    struct Operand {
        let buffer: MTLBuffer
        let offset: Int

        init(_ buffer: MTLBuffer, offset: Int = 0) {
            self.buffer = buffer
            self.offset = offset
        }
    }

    /// The geometry of one call; `expertStride` is the byte distance between
    /// consecutive experts' matrices and `xSlotStride` the float distance
    /// between slots' activations (0 shares one activation).
    struct Shape: Equatable {
        let type: GGUFType
        let rows: Int
        let columns: Int
        let slots: Int
        let expertStride: Int
        let xSlotStride: Int

        var matrixBytes: Int { rows * (columns / Int(type.blockSize)) * Int(type.blockBytes) }
    }

    private struct Args {
        var rows: UInt32
        var columns: UInt32
        var expertStride: UInt64
        var xSlotStride: UInt32
        var useScale: UInt32
    }

    static let supported: Set<GGUFType> = [.nvfp4, .q8_0]

    private let nvfp4: MTLComputePipelineState
    private let q8_0: MTLComputePipelineState

    init(context: MetalContext) throws {
        nvfp4 = try context.pipeline("nvfp4_gemv_ids")
        q8_0 = try context.pipeline("q8_0_gemv_ids")
    }

    /// Encodes `y[s] = scale[ids[s]] * W[ids[s]] · x[s]` for every slot.
    /// `scales` is nil when the tensor has no per-expert multipliers.
    func encode(
        _ shape: Shape, weights: Operand, x: Operand, y: Operand, ids: Operand,
        scales: Operand?, encoder: MTLComputeCommandEncoder
    ) throws {
        guard Self.supported.contains(shape.type) else {
            throw GGUFDequantError.unsupportedType(shape.type.name)
        }
        guard shape.rows > 0, shape.slots > 0, shape.columns > 0,
            shape.columns % Int(shape.type.blockSize) == 0
        else {
            throw MetalError.invalidState(
                "GEMV ids: \(shape.rows) x \(shape.columns) is not a \(shape.type.name) shape")
        }
        let rowsPerGroup: Int
        let threads: Int
        switch shape.type {
        case .nvfp4:
            // GEMV_IDS_NVFP4_SIMDGROUPS (2) x GEMV_IDS_NVFP4_ROWS (2); the
            // activation is read as float4, so its offset must be 16-aligned.
            guard x.offset % 16 == 0, shape.xSlotStride % 4 == 0 else {
                throw MetalError.invalidState("NVFP4 GEMV ids: activation is not float4-aligned")
            }
            encoder.setComputePipelineState(nvfp4)
            (rowsPerGroup, threads) = (4, 64)
        default:
            // GEMV_IDS_Q8_0_ROWS (2) shared by GEMV_IDS_Q8_0_SIMDGROUPS (4).
            encoder.setComputePipelineState(q8_0)
            (rowsPerGroup, threads) = (2, 128)
        }
        var args = Args(
            rows: UInt32(shape.rows), columns: UInt32(shape.columns),
            expertStride: UInt64(shape.expertStride), xSlotStride: UInt32(shape.xSlotStride),
            useScale: scales == nil ? 0 : 1)
        encoder.setBuffer(weights.buffer, offset: weights.offset, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(y.buffer, offset: y.offset, index: 2)
        encoder.setBuffer(ids.buffer, offset: ids.offset, index: 3)
        // Bound either way so the argument table is complete; read only when
        // `useScale` is set.
        let scaleOperand = scales ?? ids
        encoder.setBuffer(scaleOperand.buffer, offset: scaleOperand.offset, index: 4)
        encoder.setBytes(&args, length: MemoryLayout<Args>.stride, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (shape.rows + rowsPerGroup - 1) / rowsPerGroup, height: shape.slots,
                depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}
