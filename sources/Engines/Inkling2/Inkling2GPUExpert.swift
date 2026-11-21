import Foundation
import Metal
import TinyTitanCore

/// GPU GEMV for one packed routed expert. The packed bytes are uploaded as-is;
/// dequantization happens inside the kernel, so no full-precision expert is
/// ever materialized.
/// unchecked-invariant: the Metal device/queue/pipelines are immutable and are
/// used from the single model thread; command buffers are created per call.
public final class Inkling2GPUExpert: @unchecked Sendable {
    let context: MetalContext
    let nvfp4: MTLComputePipelineState
    let mxfp8: MTLComputePipelineState
    private let bf16Float: MTLComputePipelineState
    let bf16FloatX: MTLComputePipelineState
    private let relativeAttention: MTLComputePipelineState
    private let shortConv: MTLComputePipelineState
    private let swiglu: MTLComputePipelineState
    private var logitsWeights: MTLBuffer?
    private var logitsWeightsBytes = 0
    private var logitsX: MTLBuffer?
    private var logitsXBytes = 0
    private var logitsY: MTLBuffer?
    private var logitsYBytes = 0
    private var logitsWeightsLoaded = false
    struct GPUExpertBuffers {
        let pool: MTLBuffer
        let w13Packed: Int
        let w13Scales: Int
        let w13Outer: Int
        let w2Packed: Int
        let w2Scales: Int
        let w2Outer: Int
    }
    private struct LayerPool {
        let buffer: MTLBuffer
        var next: Int
        let capacity: Int
        let slotBytes: Int
    }
    private var expertCache: [Int: [Int: GPUExpertBuffers]] = [:]
    private var expertOrder: [Int: [Int]] = [:]
    private var layerPools: [Int: LayerPool] = [:]
    var projectionCache: [String: MTLBuffer] = [:]
    private var expertX: MTLBuffer?
    private var expertXBytes = 0
    private var expertGateUp: MTLBuffer?
    private var expertGateUpBytes = 0
    private var expertActivated: MTLBuffer?
    private var expertActivatedBytes = 0
    private var expertOut: MTLBuffer?
    private var expertOutBytes = 0
    private let expertSlotsPerLayer = 48
    var packedCache: MTLBuffer?
    var packedBytes = 0
    var scaleCache: MTLBuffer?
    var scaleBytes = 0
    var xCache: MTLBuffer?
    var xBytes = 0
    var yCache: MTLBuffer?
    var yBytes = 0

    public init() throws {
        self.context = try MetalContext()
        self.nvfp4 = try context.pipeline("nvfp4_gemv")
        self.mxfp8 = try context.pipeline("mxfp8_gemv")
        self.bf16Float = try context.pipeline("bf16_gemv_float_simd")
        self.bf16FloatX = try context.pipeline("bf16_gemv_floatx")
        self.relativeAttention = try context.pipeline("relative_attention")
        self.shortConv = try context.pipeline("short_conv4")
        self.swiglu = try context.pipeline("softcapped_swiglu")
    }

    func upload(
        _ data: Data, cache: inout MTLBuffer?, bytes: inout Int, label: String
    ) throws -> MTLBuffer {
        if cache == nil || bytes < data.count {
            guard
                let created = context.device.makeBuffer(
                    length: data.count, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed(label) }
            cache = created
            bytes = data.count
        }
        guard let buffer = cache else {
            throw MetalError.bufferAllocationFailed(label)
        }
        data.copyBytes(
            to: buffer.contents().assumingMemoryBound(to: UInt8.self),
            count: data.count)
        return buffer
    }

    func upload(
        _ values: [Float], cache: inout MTLBuffer?, bytes: inout Int, label: String
    ) throws -> MTLBuffer {
        let data = values.withUnsafeBufferPointer {
            Data(buffer: $0)
        }
        return try upload(data, cache: &cache, bytes: &bytes, label: label)
    }

    func output(
        bytes required: Int, cache: inout MTLBuffer?, bytes: inout Int, label: String
    ) throws -> MTLBuffer {
        if cache == nil || bytes < required {
            guard
                let created = context.device.makeBuffer(
                    length: required, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed(label) }
            cache = created
            bytes = required
        }
        guard let buffer = cache else {
            throw MetalError.bufferAllocationFailed(label)
        }
        return buffer
    }

    public func shortConvolution(
        x: [Float], channels: Int, kernel: [Float], kernelSize: Int,
        tokens: Int, state: [Float]?
    ) throws -> (values: [Float], state: [Float]) {
        func buffer(_ values: [Float], _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    bytes: values, length: values.count * 4,
                    options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed(label) }
            return buffer
        }
        let xBuffer = try buffer(x, "conv x")
        let weightBuffer = try buffer(kernel, "conv weights")
        let stateBuffer = try buffer(
            state ?? [Float](repeating: 0, count: channels * (kernelSize - 1)), "conv state")
        guard
            let outBuffer = context.device.makeBuffer(
                length: x.count * 4, options: .storageModeShared),
            let nextBuffer = context.device.makeBuffer(
                length: channels * (kernelSize - 1) * 4, options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.bufferAllocationFailed("conv output") }
        encoder.setComputePipelineState(shortConv)
        encoder.setBuffer(xBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(stateBuffer, offset: 0, index: 2)
        encoder.setBuffer(outBuffer, offset: 0, index: 3)
        encoder.setBuffer(nextBuffer, offset: 0, index: 4)
        var tokensValue = UInt32(tokens)
        var channelsValue = UInt32(channels)
        var kernelValue = UInt32(kernelSize)
        encoder.setBytes(&tokensValue, length: 4, index: 5)
        encoder.setBytes(&channelsValue, length: 4, index: 6)
        encoder.setBytes(&kernelValue, length: 4, index: 7)
        let total = tokens * channels
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(total, 256), height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let values = Array(
            UnsafeBufferPointer(
                start: outBuffer.contents().bindMemory(to: Float.self, capacity: total),
                count: total))
        let next = Array(
            UnsafeBufferPointer(
                start: nextBuffer.contents().bindMemory(
                    to: Float.self, capacity: channels * (kernelSize - 1)),
                count: channels * (kernelSize - 1)))
        return (values, next)
    }

    public func attention(
        q: [Float], k: [Float], v: [Float], r: [Float], gate: [Float],
        sink: [Float], relative: [Float], tokens: Int, heads: Int, kvHeads: Int,
        dim: Int, relativeDim: Int, extent: Int, sliding: Int,
        pastKeys: Int = 0
    ) throws -> [Float] {
        func buffer(_ values: [Float], _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    bytes: values, length: values.count * 4,
                    options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed(label) }
            return buffer
        }
        let qBuffer = try buffer(q, "q")
        let kBuffer = try buffer(k, "k")
        let vBuffer = try buffer(v, "v")
        let rBuffer = try buffer(r, "r")
        let gateBuffer = try buffer(gate, "gate")
        let sinkBuffer = try buffer(sink, "sink")
        let relBuffer = try buffer(relative, "relative")
        guard
            let outBuffer = context.device.makeBuffer(
                length: tokens * heads * dim * 4, options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.bufferAllocationFailed("attention output") }
        encoder.setComputePipelineState(relativeAttention)
        for (index, buffer) in [
            qBuffer, kBuffer, vBuffer, rBuffer, gateBuffer, sinkBuffer, relBuffer,
        ].enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.setBuffer(outBuffer, offset: 0, index: 7)
        var tokensValue = UInt32(tokens)
        var headsValue = UInt32(heads)
        var kvHeadsValue = UInt32(kvHeads)
        var dimValue = UInt32(dim)
        var relativeValue = UInt32(relativeDim)
        var extentValue = UInt32(extent)
        var slidingValue = UInt32(sliding)
        var pastValue = UInt32(pastKeys)
        encoder.setBytes(&tokensValue, length: 4, index: 8)
        encoder.setBytes(&headsValue, length: 4, index: 9)
        encoder.setBytes(&kvHeadsValue, length: 4, index: 10)
        encoder.setBytes(&dimValue, length: 4, index: 11)
        encoder.setBytes(&relativeValue, length: 4, index: 12)
        encoder.setBytes(&extentValue, length: 4, index: 13)
        encoder.setBytes(&slidingValue, length: 4, index: 14)
        encoder.setBytes(&pastValue, length: 4, index: 15)
        let total = tokens * heads
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(total, 32), height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = outBuffer.contents().bindMemory(
            to: Float.self, capacity: tokens * heads * dim)
        return Array(
            UnsafeBufferPointer(start: pointer, count: tokens * heads * dim))
    }

    public func gemvBF16(
        weights: Data, x: [Float], rows: Int, columns: Int
    ) throws -> [Float] {
        precondition(x.count == columns)
        let weightsBuffer = try upload(
            weights, cache: &packedCache, bytes: &packedBytes, label: "dense weights")
        let xBuffer = try upload(
            x, cache: &xCache, bytes: &xBytes, label: "dense x")
        let yBuffer = try output(
            bytes: rows * 4, cache: &yCache, bytes: &yBytes, label: "dense y")
        guard let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.bufferAllocationFailed("inkling dense gemv") }
        encoder.setComputePipelineState(bf16FloatX)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        var rowsValue = UInt32(rows)
        var columnsValue = UInt32(columns)
        encoder.setBytes(&rowsValue, length: 4, index: 3)
        encoder.setBytes(&columnsValue, length: 4, index: 4)
        let groups = (rows + 7) / 8
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: pointer, count: rows))
    }

    public func logits(weights: Data, hidden: [Float], rows: Int, columns: Int)
        throws -> [Float]
    {
        precondition(hidden.count == columns)
        let halves = hidden.map { Float16($0) }
        let weightsBuffer: MTLBuffer
        if logitsWeightsLoaded, let existing = logitsWeights {
            weightsBuffer = existing
        } else {
            weightsBuffer = try upload(
                weights, cache: &logitsWeights, bytes: &logitsWeightsBytes,
                label: "logits weights")
            logitsWeightsLoaded = true
        }
        let halvesData = halves.withUnsafeBufferPointer { Data(buffer: $0) }
        let xBuffer = try upload(
            halvesData, cache: &logitsX, bytes: &logitsXBytes, label: "logits x")
        let yBuffer = try output(
            bytes: rows * 4, cache: &logitsY, bytes: &logitsYBytes,
            label: "logits y")
        guard let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.bufferAllocationFailed("inkling logits") }
        encoder.setComputePipelineState(bf16Float)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        var rowsValue = UInt32(rows)
        var columnsValue = UInt32(columns)
        encoder.setBytes(&rowsValue, length: 4, index: 3)
        encoder.setBytes(&columnsValue, length: 4, index: 4)
        let groups = (rows + 7) / 8
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: pointer, count: rows))
    }

    func expertBuffers(_ expert: Inkling2PackedExpert) throws -> GPUExpertBuffers {
        let layer = expert.layer
        let index = expert.expert
        if let cached = expertCache[layer]?[index] {
            Inkling2Profiler.record("expertHit", 0)
            if let order = expertOrder[layer], let position = order.firstIndex(of: index) {
                var updated = order
                updated.remove(at: position)
                updated.append(index)
                expertOrder[layer] = updated
            }
            return cached
        }
        Inkling2Profiler.record("expertMiss", 0)
        let w13Bytes = expert.w13.count
        let w13ScaleBytes = expert.w13Scales.count
        let w2Bytes = expert.w2.count
        let w2ScaleBytes = expert.w2Scales.count
        let rawSlot = w13Bytes + w13ScaleBytes + w2Bytes + w2ScaleBytes + 8
        let slotBytes = (rawSlot + 255) & ~255
        if layerPools[layer]?.slotBytes != slotBytes {
            let capacity = expertSlotsPerLayer
            guard
                let buffer = context.device.makeBuffer(
                    length: slotBytes * capacity,
                    options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("expert layer pool") }
            layerPools[layer] = LayerPool(
                buffer: buffer, next: 0, capacity: capacity,
                slotBytes: slotBytes)
            expertCache[layer] = [:]
            expertOrder[layer] = []
        }
        guard var pool = layerPools[layer], let cache = expertCache[layer] else {
            throw MetalError.bufferAllocationFailed("expert layer pool")
        }
        var order = expertOrder[layer] ?? []
        let slot: Int
        if pool.next < pool.capacity {
            slot = pool.next
            pool.next += 1
            layerPools[layer] = pool
        } else {
            guard let oldest = order.first,
                let evicted = expertCache[layer]?.removeValue(forKey: oldest)
            else { throw MetalError.invalidState("expert pool empty") }
            order.removeFirst()
            slot = evicted.w13Packed / slotBytes
        }
        _ = cache
        let base = slot * slotBytes
        let w13Packed = base
        let w13Scales = w13Packed + w13Bytes
        let w2Packed = w13Scales + w13ScaleBytes
        let w2Scales = w2Packed + w2Bytes
        let w13Outer = w2Scales + w2ScaleBytes
        let w2Outer = w13Outer + 4
        let raw = pool.buffer.contents()
        expert.w13.copyBytes(
            to: raw.advanced(by: w13Packed).assumingMemoryBound(to: UInt8.self),
            count: w13Bytes)
        expert.w13Scales.copyBytes(
            to: raw.advanced(by: w13Scales).assumingMemoryBound(to: UInt8.self),
            count: w13ScaleBytes)
        expert.w2.copyBytes(
            to: raw.advanced(by: w2Packed).assumingMemoryBound(to: UInt8.self),
            count: w2Bytes)
        expert.w2Scales.copyBytes(
            to: raw.advanced(by: w2Scales).assumingMemoryBound(to: UInt8.self),
            count: w2ScaleBytes)
        raw.advanced(by: w13Outer).assumingMemoryBound(to: Float.self).pointee =
            expert.w13Outer
        raw.advanced(by: w2Outer).assumingMemoryBound(to: Float.self).pointee =
            expert.w2Outer
        let value = GPUExpertBuffers(
            pool: pool.buffer, w13Packed: w13Packed, w13Scales: w13Scales,
            w13Outer: w13Outer, w2Packed: w2Packed, w2Scales: w2Scales,
            w2Outer: w2Outer)
        expertCache[layer, default: [:]][index] = value
        order.append(index)
        expertOrder[layer] = order
        return value
    }

    /// One command buffer for the whole routed MLP: every expert's w13 GEMV,
    /// the GPU SwiGLU, and its w2 GEMV. Scratch is pooled across layers.
    public func expertForwardBatch(
        _ experts: [Inkling2PackedExpert], x: [Float], hidden: Int,
        intermediate: Int, gateCap: Float, upCap: Float
    ) throws -> [[Float]] {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        let gateUp = try output(
            bytes: 8 * 2 * intermediate * 4, cache: &expertGateUp,
            bytes: &expertGateUpBytes, label: "expert gateUp")
        let activated = try output(
            bytes: 8 * intermediate * 4, cache: &expertActivated,
            bytes: &expertActivatedBytes, label: "expert activated")
        let out = try output(
            bytes: 8 * hidden * 4, cache: &expertOut,
            bytes: &expertOutBytes, label: "expert out")
        let xBuffer = try upload(
            x, cache: &expertX, bytes: &expertXBytes, label: "expert x")
        var outputOffsets: [Int] = []
        for (expertIndex, expert) in experts.enumerated() {
            let buffers = try expertBuffers(expert)
            let gateOffset = expertIndex * 2 * intermediate * 4
            let activationOffset = expertIndex * intermediate * 4
            let outOffset = expertIndex * hidden * 4
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            let pipeline = expert.format == .nvfp4 ? nvfp4 : mxfp8
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffers.pool, offset: buffers.w13Packed, index: 0)
            encoder.setBuffer(buffers.pool, offset: buffers.w13Scales, index: 1)
            var rows = UInt32(2 * intermediate)
            var columns = UInt32(hidden)
            if expert.format == .nvfp4 {
                encoder.setBuffer(buffers.pool, offset: buffers.w13Outer, index: 2)
                encoder.setBuffer(xBuffer, offset: 0, index: 3)
                encoder.setBuffer(gateUp, offset: gateOffset, index: 4)
                encoder.setBytes(&rows, length: 4, index: 5)
                encoder.setBytes(&columns, length: 4, index: 6)
            } else {
                encoder.setBuffer(xBuffer, offset: 0, index: 2)
                encoder.setBuffer(gateUp, offset: gateOffset, index: 3)
                encoder.setBytes(&rows, length: 4, index: 4)
                encoder.setBytes(&columns, length: 4, index: 5)
            }
            encoder.dispatchThreadgroups(
                MTLSize(width: (2 * intermediate + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            guard let activationEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            activationEncoder.setComputePipelineState(swiglu)
            activationEncoder.setBuffer(gateUp, offset: gateOffset, index: 0)
            activationEncoder.setBuffer(activated, offset: activationOffset, index: 1)
            var gate = gateCap
            var up = upCap
            var width = UInt32(intermediate)
            activationEncoder.setBytes(&gate, length: 4, index: 2)
            activationEncoder.setBytes(&up, length: 4, index: 3)
            activationEncoder.setBytes(&width, length: 4, index: 4)
            activationEncoder.dispatchThreads(
                MTLSize(width: intermediate, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(intermediate, 256), height: 1, depth: 1))
            activationEncoder.endEncoding()
            guard let downEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            downEncoder.setComputePipelineState(pipeline)
            downEncoder.setBuffer(buffers.pool, offset: buffers.w2Packed, index: 0)
            downEncoder.setBuffer(buffers.pool, offset: buffers.w2Scales, index: 1)
            var downRows = UInt32(hidden)
            var downColumns = UInt32(intermediate)
            if expert.format == .nvfp4 {
                downEncoder.setBuffer(buffers.pool, offset: buffers.w2Outer, index: 2)
                downEncoder.setBuffer(activated, offset: activationOffset, index: 3)
                downEncoder.setBuffer(out, offset: outOffset, index: 4)
                downEncoder.setBytes(&downRows, length: 4, index: 5)
                downEncoder.setBytes(&downColumns, length: 4, index: 6)
            } else {
                downEncoder.setBuffer(activated, offset: activationOffset, index: 2)
                downEncoder.setBuffer(out, offset: outOffset, index: 3)
                downEncoder.setBytes(&downRows, length: 4, index: 4)
                downEncoder.setBytes(&downColumns, length: 4, index: 5)
            }
            downEncoder.dispatchThreadgroups(
                MTLSize(width: (hidden + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            downEncoder.endEncoding()
            outputOffsets.append(outOffset)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = out.contents().bindMemory(to: Float.self, capacity: 8 * hidden)
        return outputOffsets.map { offset in
            Array(
                UnsafeBufferPointer(
                    start: pointer.advanced(by: offset / 4), count: hidden))
        }
    }

}
