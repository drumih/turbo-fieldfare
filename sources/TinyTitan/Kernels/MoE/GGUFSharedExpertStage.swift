import Foundation
import Metal

/// The as-is shared expert for one layer: gate and up in their block format
/// (Q6_K or Q8_0), SiLU(gate)·up, down at Q8_0, and the whole output scaled by
/// the sigmoid of the per-layer scalar gate.
///
/// The three matrices are small (about 4.4 MB a layer), so they are read once
/// into Metal buffers at construction; the PLE table is the only resident
/// tensor kept on disk.
package final class GGUFSharedExpertStage {
    private struct Layer {
        let gate: MTLBuffer
        let gateType: GGUFType
        let up: MTLBuffer
        let upType: GGUFType
        let down: MTLBuffer
        let scalar: MTLBuffer
    }

    private let dispatcher: GGUFExpertGEMV
    private let bf16: BF16GEMV
    private let siluMul: MTLComputePipelineState
    private let sigmoidAccumulate: MTLComputePipelineState
    private let hidden: Int
    private let moeWidth: Int
    private let layers: [Layer]
    private let gateOut: MTLBuffer
    private let upOut: MTLBuffer
    private let act: MTLBuffer
    private let downOut: MTLBuffer
    private let scalarOut: MTLBuffer
    private let context: MetalContext

    package init(context: MetalContext, arch: ArchConfig, model: GGUFModel) throws {
        self.context = context
        self.dispatcher = try GGUFExpertGEMV(context: context)
        self.bf16 = try BF16GEMV(context: context)
        self.siluMul = try context.pipeline("gguf_silu_mul_half")
        self.sigmoidAccumulate = try context.pipeline("gguf_accumulate_sigmoid_half")
        self.hidden = arch.hiddenSize
        self.moeWidth = arch.moeIntermediateSize

        var built: [Layer] = []
        built.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let gate = try Self.load(model, name: "blk.\(layer).ffn_gate_shexp.weight")
            let up = try Self.load(model, name: "blk.\(layer).ffn_up_shexp.weight")
            let down = try Self.load(model, name: "blk.\(layer).ffn_down_shexp.weight")
            let scalar = try Self.load(
                model, name: "blk.\(layer).ffn_gate_inp_shexp.weight")
            guard
                let gateBuf = context.device.makeBuffer(
                    bytes: gate.bytes, length: gate.bytes.count, options: .storageModeShared),
                let upBuf = context.device.makeBuffer(
                    bytes: up.bytes, length: up.bytes.count, options: .storageModeShared),
                let downBuf = context.device.makeBuffer(
                    bytes: down.bytes, length: down.bytes.count, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("shared expert weights") }
            // The scalar gate is F32 in the file and bf16 for the dot; convert
            // once, here, rather than per token.
            var scalarBits: [UInt16] = []
            let floats = scalar.bytes.withUnsafeBytes { raw -> [Float] in
                let count = raw.count / MemoryLayout<Float>.size
                let pointer = raw.bindMemory(to: Float.self)
                return (0..<count).map { pointer[$0] }
            }
            scalarBits.reserveCapacity(floats.count)
            for value in floats {
                let bits = Quantization.bf16Bits(value)
                scalarBits.append(bits)
            }
            guard
                let scalarBuf = scalarBits.withUnsafeBytes({ raw in
                    guard let base = raw.baseAddress else { return nil as MTLBuffer? }
                    return context.device.makeBuffer(
                        bytes: base, length: raw.count, options: .storageModeShared)
                })
            else { throw MetalError.bufferAllocationFailed("shared scalar gate") }
            built.append(
                Layer(
                    gate: gateBuf, gateType: gate.type, up: upBuf, upType: up.type,
                    down: downBuf, scalar: scalarBuf))
        }
        self.layers = built

        func scratch(_ bytes: Int) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: bytes, options: .storageModePrivate)
            else { throw MetalError.bufferAllocationFailed("shared expert scratch") }
            return buffer
        }
        self.gateOut = try scratch(moeWidth * 2)
        self.upOut = try scratch(moeWidth * 2)
        self.act = try scratch(moeWidth * 2)
        self.downOut = try scratch(hidden * 2)
        self.scalarOut = try scratch(2)
    }

    /// Adds the shared expert's contribution for `layer` into `out`.
    package func encode(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        x: MTLBuffer,
        out: MTLBuffer,
        outOffset: Int = 0
    ) throws {
        let weights = layers[layer]
        try dispatcher.encode(
            commandBuffer: commandBuffer, weights: weights.gate,
            x: x, y: gateOut, rows: moeWidth, columns: hidden, type: weights.gateType)
        try dispatcher.encode(
            commandBuffer: commandBuffer, weights: weights.up,
            x: x, y: upOut, rows: moeWidth, columns: hidden, type: weights.upType)
        try encodeElementwise(
            commandBuffer: commandBuffer, pipeline: siluMul,
            buffers: [gateOut, upOut, act], count: moeWidth)
        try dispatcher.encode(
            commandBuffer: commandBuffer, weights: weights.down,
            x: act, y: downOut, rows: hidden, columns: moeWidth, type: .q8_0)
        try bf16.encode(
            commandBuffer: commandBuffer, weights: weights.scalar, x: x, y: scalarOut,
            m: 1, n: UInt32(hidden))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(sigmoidAccumulate)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(downOut, offset: 0, index: 1)
        encoder.setBuffer(scalarOut, offset: 0, index: 2)
        var n = UInt32(hidden)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: hidden, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(sigmoidAccumulate.maxTotalThreadsPerThreadgroup), 256),
                height: 1, depth: 1))
        encoder.endEncoding()
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
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256),
                height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func load(_ model: GGUFModel, name: String) throws -> (
        bytes: [UInt8], type: GGUFType
    ) {
        guard let location = model.tensor(named: name) else {
            throw MetalError.invalidState("missing \(name)")
        }
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try handle.read(upToCount: Int(location.tensor.byteCount)) ?? Data()
        return ([UInt8](data), location.tensor.type)
    }
}
