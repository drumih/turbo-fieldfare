import Foundation
import Metal

/// One dispatch for the same projection of several routed experts.
///
/// A layer's routed experts share a block format per projection (GGUF stores
/// `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps` as one tensor each), and
/// the expert cache keeps every resident expert of a layer in one pool buffer.
/// So the gate (or up, or down) GEMVs of all selected experts can run as one
/// grid of `(rows / 8, matrices)` threadgroups, each reading its matrix at a
/// byte offset into the pool. The kernels call the very row function the
/// single-matrix GEMVs use, so a batched result is bit-identical to running the
/// matrices one at a time; what changes is 10 encoders becoming one.
package final class GGUFExpertBatchGEMV {
    /// The most matrices one dispatch carries; the Metal struct's array size.
    package static let maxMatrices = 32

    private let pipelines: [GGUFType: MTLComputePipelineState]
    private let iq2sGrid: MTLBuffer
    private let iq3xxsGrid: MTLBuffer
    private let ksigns: MTLBuffer

    package init(context: MetalContext) throws {
        self.pipelines = [
            .iq2S: try context.pipeline("iq2s_gemv_batched"),
            .iq3XXS: try context.pipeline("iq3xxs_gemv_batched"),
            .iq4NL: try context.pipeline("iq4nl_gemv_batched"),
            .iq4XS: try context.pipeline("iq4xs_gemv_batched"),
            .q2_0: try context.pipeline("q2_0_gemv_batched"),
        ]
        guard
            let iq2s = Self.makeBuffer(GGUFIQTables.iq2sGrid, device: context.device),
            let iq3 = Self.makeBuffer(GGUFIQTables.iq3xxsGrid, device: context.device),
            let signs = Self.makeBuffer(GGUFIQTables.ksigns, device: context.device)
        else {
            throw MetalError.bufferAllocationFailed("GGUF batched GEMV tables")
        }
        self.iq2sGrid = iq2s
        self.iq3xxsGrid = iq3
        self.ksigns = signs
    }

    /// Whether `matrices` can go through one batched dispatch: a supported
    /// format shared by all, one backing buffer, offsets that fit the kernel's
    /// 32-bit layout, and no more than `maxMatrices`.
    package func supports(_ matrices: [GGUFExpertMatrix]) -> Bool {
        guard let first = matrices.first, matrices.count <= Self.maxMatrices,
            pipelines[first.type] != nil
        else { return false }
        return matrices.allSatisfy { matrix in
            matrix.buffer === first.buffer && matrix.type == first.type
                && matrix.rows == first.rows && matrix.columns == first.columns
                && matrix.offset + matrix.byteCount <= Int(UInt32.max)
        }
    }

    /// Encodes `y[m * yStride + r] = matrices[m] · x[m * xStride ...]` for every
    /// matrix. Strides are in elements. Call `supports(_:)` first.
    package func encode(
        commandBuffer: MTLCommandBuffer,
        matrices: [GGUFExpertMatrix],
        x: MTLBuffer,
        xStride: Int,
        y: MTLBuffer,
        yStride: Int
    ) throws {
        guard supports(matrices), let first = matrices.first,
            let pipeline = pipelines[first.type]
        else {
            throw MetalError.invalidState("batched GEMV given matrices it cannot share a dispatch")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(first.buffer, offset: 0, index: 0)
        var next = 1
        switch first.type {
        case .iq2S:
            encoder.setBuffer(iq2sGrid, offset: 0, index: 1)
            next = 2
        case .iq3XXS:
            encoder.setBuffer(iq3xxsGrid, offset: 0, index: 1)
            encoder.setBuffer(ksigns, offset: 0, index: 2)
            next = 3
        default:
            break
        }
        encoder.setBuffer(x, offset: 0, index: next)
        encoder.setBuffer(y, offset: 0, index: next + 1)
        var rows = UInt32(first.rows)
        var columns = UInt32(first.columns)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: next + 2)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: next + 3)
        var layout = [UInt32](repeating: 0, count: Self.maxMatrices + 2)
        for (index, matrix) in matrices.enumerated() {
            layout[index] = UInt32(matrix.offset)
        }
        layout[Self.maxMatrices] = UInt32(xStride)
        layout[Self.maxMatrices + 1] = UInt32(yStride)
        layout.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            encoder.setBytes(base, length: bytes.count, index: next + 4)
        }
        let rowsPerThreadgroup = 8
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (first.rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                height: matrices.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rowsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func makeBuffer<T>(_ values: [T], device: MTLDevice) -> MTLBuffer? {
        values.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base, length: buffer.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }
}
