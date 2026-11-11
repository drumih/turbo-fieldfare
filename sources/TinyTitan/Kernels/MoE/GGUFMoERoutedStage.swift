import Foundation
import Metal

/// The as-is routed MoE for one layer: for each selected expert, gate and up
/// through the fused GEMV for their block format, `SiLU(gate)·up`, down through
/// its format, and a weighted sum into the layer's output.
///
/// The checkpoint stays in its blocks end to end: the cache hands over raw
/// bytes, each GEMV decodes as it dots, and no decoded matrix is materialised.
/// Dependencies between the dispatches are carried by the command buffer
/// (Metal tracks buffer hazards within one buffer), so the stage submits in
/// order and waits once.
package final class GGUFMoERoutedStage {
    private let gemv: GGUFExpertGEMV
    private let siluMul: MTLComputePipelineState
    private let accumulate: MTLComputePipelineState
    private let device: MTLDevice

    private let moeWidth: Int
    private let hidden: Int
    private let topK: Int
    private var gateBuffers: [MTLBuffer] = []
    private var upBuffers: [MTLBuffer] = []
    private var actBuffers: [MTLBuffer] = []
    private var downBuffers: [MTLBuffer] = []

    package init(context: MetalContext, arch: ArchConfig) throws {
        self.device = context.device
        self.gemv = try GGUFExpertGEMV(context: context)
        self.siluMul = try context.pipeline("gguf_silu_mul_half")
        self.accumulate = try context.pipeline("gguf_accumulate_scaled_half")
        self.moeWidth = arch.moeIntermediateSize
        self.hidden = arch.hiddenSize
        self.topK = arch.topKExperts
        for _ in 0..<topK {
            guard
                let gate = context.device.makeBuffer(
                    length: moeWidth * 2, options: .storageModePrivate),
                let up = context.device.makeBuffer(
                    length: moeWidth * 2, options: .storageModePrivate),
                let act = context.device.makeBuffer(
                    length: moeWidth * 2, options: .storageModePrivate),
                let down = context.device.makeBuffer(
                    length: hidden * 2, options: .storageModePrivate)
            else {
                throw MetalError.bufferAllocationFailed("GGUF MoE scratch")
            }
            gateBuffers.append(gate)
            upBuffers.append(up)
            actBuffers.append(act)
            downBuffers.append(down)
        }
    }

    /// Encodes the routed contribution for `experts` with `weights` (already
    /// softmaxed and renormalised) into `out`. `out` is accumulated into, so it
    /// must start at zero (or at a residual the caller wants included).
    package func encode(
        commandBuffer: MTLCommandBuffer,
        cache: GGUFExpertBufferCache,
        layer: Int,
        x: MTLBuffer,
        experts: [Int],
        weights: [Float],
        out: MTLBuffer,
        outOffset: Int = 0
    ) throws {
        precondition(experts.count <= topK && experts.count == weights.count)
        try encode(
            commandBuffer: commandBuffer,
            experts: try cache.experts(layer: layer, experts: experts),
            x: x, weights: weights, out: out, outOffset: outOffset)
    }

    /// Encodes the routed contribution of already-fetched `experts` (as
    /// `GGUFExpertBufferCache.experts(layer:experts:)` returned and pinned
    /// them) with their `weights` into `out`, which is accumulated into.
    package func encode(
        commandBuffer: MTLCommandBuffer,
        experts: [GGUFExpertBuffers],
        x: MTLBuffer,
        weights: [Float],
        out: MTLBuffer,
        outOffset: Int = 0
    ) throws {
        precondition(experts.count <= topK && experts.count == weights.count)
        for (index, buffers) in experts.enumerated() {
            try gemv.encode(
                commandBuffer: commandBuffer, weights: buffers.gate.buffer,
                x: x, y: gateBuffers[index],
                rows: moeWidth, columns: hidden, type: buffers.gate.type)
            try gemv.encode(
                commandBuffer: commandBuffer, weights: buffers.up.buffer,
                x: x, y: upBuffers[index],
                rows: moeWidth, columns: hidden, type: buffers.up.type)
            try encodeElementwise(
                commandBuffer: commandBuffer, pipeline: siluMul,
                buffers: [gateBuffers[index], upBuffers[index], actBuffers[index]],
                count: moeWidth)
            try gemv.encode(
                commandBuffer: commandBuffer, weights: buffers.down.buffer,
                x: actBuffers[index], y: downBuffers[index],
                rows: hidden, columns: moeWidth, type: buffers.down.type)
            try encodeAccumulate(
                commandBuffer: commandBuffer, row: downBuffers[index],
                weight: weights[index], into: out, outOffset: outOffset)
        }
    }

    private func encodeElementwise(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        buffers: [MTLBuffer],
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        var n = UInt32(count)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: buffers.count)
        dispatch(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeAccumulate(
        commandBuffer: MTLCommandBuffer,
        row: MTLBuffer,
        weight: Float,
        into out: MTLBuffer,
        outOffset: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(accumulate)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(row, offset: 0, index: 1)
        var w = weight
        encoder.setBytes(&w, length: MemoryLayout<Float>.size, index: 2)
        var n = UInt32(hidden)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch(encoder, pipeline: accumulate, count: hidden)
        encoder.endEncoding()
    }

    private func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        count: Int
    ) {
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256),
                height: 1, depth: 1))
    }
}
