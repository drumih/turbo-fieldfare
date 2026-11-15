import Foundation
import Metal
import TinyTitanKernelsC
import TinyTitanCore

/// Metal matvecs for the Nemotron graph.
///
/// The shaders are the same affine GEMV kernels the Qwen decode path uses
/// (`dequant_int4.metal` / `dequant_int8.metal`), called without the
/// constant-folded Qwen shape variants: the Nemotron geometry is different and
/// the unspecialized path is the correct one. The weight files are already
/// mmap'd, so each is wrapped as a no-copy `MTLBuffer`; only the activation and
/// output vectors are staged.
public final class NemotronMetal: NemotronMatvec {
    private let install: NemotronInstall
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let int4: MTLComputePipelineState
    private let int8: MTLComputePipelineState

    private let weightsBuffer: MTLBuffer
    private var layerBuffers: [Int: MTLBuffer] = [:]
    private let xHalf: MTLBuffer
    private let yHalf: MTLBuffer

    /// Pipelines handed in by the caller, so the harness and the engine can
    /// share the same class: the engine takes them from `MetalContext`, the
    /// standalone check compiles the same two shaders itself.
    public init(
        install: NemotronInstall,
        device: MTLDevice,
        int4: MTLComputePipelineState,
        int8: MTLComputePipelineState
    ) throws {
        self.install = install
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw NemotronInstallError.unsupported("no Metal command queue")
        }
        self.queue = queue
        self.int4 = int4
        self.int8 = int8

        let page = Int(getpagesize())
        func rounded(_ value: Int) -> Int { ((value + page - 1) / page) * page }
        let mapping = install.weightsMapping
        guard
            let buffer = device.makeBuffer(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: mapping.base),
                length: rounded(mapping.size),
                options: .storageModeShared, deallocator: nil)
        else {
            throw NemotronInstallError.unsupported("cannot wrap model_weights.bin in Metal")
        }
        weightsBuffer = buffer
        for (layer, _) in install.experts {
            guard let layerMapping = install.layerMapping(layer) else { continue }
            guard
                let layerBuffer = device.makeBuffer(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: layerMapping.base),
                    length: rounded(layerMapping.size),
                    options: .storageModeShared, deallocator: nil)
            else {
                throw NemotronInstallError.unsupported(
                    "cannot wrap packed_experts for layer \(layer) in Metal")
            }
            layerBuffers[layer] = layerBuffer
        }
        let g = install.geometry
        let maximumX = max(
            g.hiddenSize, g.mambaIntermediate, g.moeSharedIntermediate,
            g.moeLatentSize, g.maxMoeIntermediate,
            g.attentionHeads * g.attentionHeadDim, g.topKExperts * g.maxMoeIntermediate)
        let maximumY = max(
            g.vocabSize, g.topKExperts * max(g.maxMoeIntermediate, g.moeLatentSize),
            g.mambaIntermediate + 2 * g.mambaGroups * g.ssmStateSize + g.mambaHeads)
        guard
            let xBuffer = device.makeBuffer(
                length: maximumX * MemoryLayout<Float16>.size,
                options: .storageModeShared),
            let yBuffer = device.makeBuffer(
                length: maximumY * MemoryLayout<Float16>.size,
                options: .storageModeShared)
        else {
            throw NemotronInstallError.unsupported("cannot allocate Metal scratch")
        }
        xHalf = xBuffer
        yHalf = yBuffer
    }

    // MARK: - Protocol

    public func gemv(
        _ tensor: NemotronTensorRef,
        x: UnsafePointer<Float>,
        y: UnsafeMutablePointer<Float>
    ) {
        stage(x, count: tensor.columns, slot: 0)
        guard let command = queue.makeCommandBuffer() else {
            preconditionFailure("Metal refused a command buffer")
        }
        encode(
            command, buffer: weightsBuffer,
            weightOffset: tensor.fileOffset,
            scaleOffset: tensor.scaleOffset,
            biasOffset: tensor.biasOffset,
            bits: tensor.bits ?? 4,
            rows: tensor.rows, columns: tensor.columns,
            xOffset: 0, yOffset: 0)
        command.commit()
        command.waitUntilCompleted()
        read(count: tensor.rows, from: 0, into: y)
    }

    public func gemvExperts(
        layer: Int,
        experts: [Int],
        role: String,
        x: UnsafePointer<Float>,
        xStride: Int,
        y: UnsafeMutablePointer<Float>
    ) {
        guard let buffer = layerBuffers[layer],
            let entry = install.experts[layer],
            let tensor = entry.roles[role],
            let scales = entry.roles["\(role)_scales"],
            let biases = entry.roles["\(role)_biases"],
            !experts.isEmpty
        else {
            return
        }
        let columns = tensor.shape[1]
        let rows = tensor.shape[0]
        for index in experts.indices {
            stage(
                xStride == 0 ? x : x + index * xStride,
                count: columns, slot: xStride == 0 ? 0 : index)
        }
        guard let command = queue.makeCommandBuffer() else {
            preconditionFailure("Metal refused a command buffer")
        }
        for (index, expert) in experts.enumerated() {
            let base = expert * entry.stride
            encode(
                command, buffer: buffer,
                weightOffset: base + tensor.offset,
                scaleOffset: base + scales.offset,
                biasOffset: base + biases.offset,
                bits: tensor.bits ?? 4,
                rows: rows, columns: columns,
                xOffset: (xStride == 0 ? 0 : index * columns) * MemoryLayout<Float16>.size,
                yOffset: index * rows * MemoryLayout<Float16>.size)
        }
        command.commit()
        command.waitUntilCompleted()
        read(count: rows * experts.count, from: 0, into: y)
    }

    // MARK: - Staging

    private func stage(_ x: UnsafePointer<Float>, count: Int, slot: Int) {
        let destination = xHalf.contents()
            .assumingMemoryBound(to: Float16.self)
            .advanced(by: slot * count)
        for index in 0..<count { destination[index] = Float16(x[index]) }
    }

    private func encode(
        _ command: MTLCommandBuffer,
        buffer: MTLBuffer,
        weightOffset: Int, scaleOffset: Int, biasOffset: Int,
        bits: Int, rows: Int, columns: Int,
        xOffset: Int, yOffset: Int
    ) {
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(bits == 4 ? int4 : int8)
        encoder.setBuffer(buffer, offset: weightOffset, index: 0)
        encoder.setBuffer(buffer, offset: scaleOffset, index: 1)
        encoder.setBuffer(buffer, offset: biasOffset, index: 2)
        encoder.setBuffer(xHalf, offset: xOffset, index: 3)
        encoder.setBuffer(yHalf, offset: yOffset, index: 4)
        var m = UInt32(rows)
        var n = UInt32(columns)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * 8, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func read(count: Int, from offset: Int, into y: UnsafeMutablePointer<Float>) {
        let source = yHalf.contents()
            .assumingMemoryBound(to: Float16.self)
            .advanced(by: offset)
        for index in 0..<count { y[index] = Float(source[index]) }
    }
}
