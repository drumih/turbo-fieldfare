import Darwin
import Metal

struct QwenGatedDeltaNetGeometry: Equatable {
    let keyHeads: Int
    let valueHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let convolutionKernel: Int

    var keyDimension: Int { keyHeads * keyHeadDim }
    var valueDimension: Int { valueHeads * valueHeadDim }
    var qkvDimension: Int { keyDimension * 2 + valueDimension }
    var recurrentStateElements: Int {
        valueHeads * keyHeadDim * valueHeadDim
    }

    static let qwen = QwenGatedDeltaNetGeometry(
        keyHeads: 16,
        valueHeads: 32,
        keyHeadDim: 128,
        valueHeadDim: 128,
        convolutionKernel: 4)
}

struct QwenGatedDeltaNetSnapshot {
    let recurrentState: [UInt8]
    let convolutionState: [UInt8]
}

final class QwenGatedDeltaNetState {
    let geometry: QwenGatedDeltaNetGeometry
    let convolutionChannels: Int
    let recurrentBuffer: MTLBuffer
    let convolutionBuffer: MTLBuffer

    init(device: MTLDevice,
         geometry: QwenGatedDeltaNetGeometry,
         convolutionChannels: Int) throws {
        precondition(convolutionChannels > 0)
        self.geometry = geometry
        self.convolutionChannels = convolutionChannels
        let recurrentBytes = geometry.recurrentStateElements * MemoryLayout<Float>.stride
        let convolutionElements = convolutionChannels * (geometry.convolutionKernel - 1)
        guard let recurrent = device.makeBuffer(
            length: recurrentBytes, options: .storageModeShared),
            let convolution = device.makeBuffer(
                length: convolutionElements * MemoryLayout<UInt16>.stride,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.recurrentBuffer = recurrent
        self.convolutionBuffer = convolution
        reset()
    }

    var recurrentStateBytes: Int {
        geometry.recurrentStateElements * MemoryLayout<Float>.stride
    }

    var convolutionStateBytes: Int {
        convolutionChannels * (geometry.convolutionKernel - 1) * MemoryLayout<UInt16>.stride
    }

    func reset() {
        memset(recurrentBuffer.contents(), 0, recurrentStateBytes)
        memset(convolutionBuffer.contents(), 0, convolutionStateBytes)
    }

    func snapshot() -> QwenGatedDeltaNetSnapshot {
        QwenGatedDeltaNetSnapshot(
            recurrentState: bytes(from: recurrentBuffer, count: recurrentStateBytes),
            convolutionState: bytes(from: convolutionBuffer, count: convolutionStateBytes))
    }

    func restore(_ snapshot: QwenGatedDeltaNetSnapshot) {
        precondition(snapshot.recurrentState.count == recurrentStateBytes)
        precondition(snapshot.convolutionState.count == convolutionStateBytes)
        copy(snapshot.recurrentState, to: recurrentBuffer)
        copy(snapshot.convolutionState, to: convolutionBuffer)
    }

    private func bytes(from buffer: MTLBuffer, count: Int) -> [UInt8] {
        let pointer = buffer.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func copy(_ bytes: [UInt8], to buffer: MTLBuffer) {
        _ = bytes.withUnsafeBytes { source in
            memcpy(buffer.contents(), source.baseAddress!, bytes.count)
        }
    }
}

final class QwenGatedDeltaNetStateManager {
    private(set) var states: [QwenGatedDeltaNetState]

    init(context: MetalContext,
         layerCount: Int,
         geometry: QwenGatedDeltaNetGeometry = .qwen,
         convolutionChannels: Int? = nil) throws {
        precondition(layerCount > 0)
        let channels = convolutionChannels ?? geometry.qkvDimension
        self.states = try (0..<layerCount).map { _ in
            try QwenGatedDeltaNetState(
                device: context.device,
                geometry: geometry,
                convolutionChannels: channels)
        }
    }

    func state(layer: Int) -> QwenGatedDeltaNetState {
        states[layer]
    }

    func reset() {
        states.forEach { $0.reset() }
    }

    func reset(layer: Int) {
        states[layer].reset()
    }
}

final class QwenGatedDeltaNet {
    private let convolutionPSO: MTLComputePipelineState
    private let recurrentPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.convolutionPSO = try context.pipeline("qwen_gated_delta_causal_conv")
        self.recurrentPSO = try context.pipeline("qwen_gated_delta_recurrent")
    }

    /// Apply Qwen's depthwise causal convolution and SiLU to one token.
    func encodeCausalConvolution(commandBuffer: MTLCommandBuffer,
                                 input: MTLBuffer,
                                 weights: MTLBuffer,
                                 weightsOffset: Int = 0,
                                 output: MTLBuffer,
                                 state: QwenGatedDeltaNetState) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convolutionPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(state.convolutionBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var channels = UInt32(state.convolutionChannels)
        var kernel = UInt32(state.geometry.convolutionKernel)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 5)
        dispatch(encoder, pipeline: convolutionPSO, count: state.convolutionChannels)
        encoder.endEncoding()
    }

    /// Advance one token of the recurrent gated-delta rule.
    func encodeRecurrent(commandBuffer: MTLCommandBuffer,
                         query: MTLBuffer,
                         key: MTLBuffer,
                         value: MTLBuffer,
                         decay: MTLBuffer,
                         decayOffset: Int = 0,
                         beta: MTLBuffer,
                         betaOffset: Int = 0,
                         output: MTLBuffer,
                         state: QwenGatedDeltaNetState) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(recurrentPSO)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(decay, offset: decayOffset, index: 3)
        encoder.setBuffer(beta, offset: betaOffset, index: 4)
        encoder.setBuffer(state.recurrentBuffer, offset: 0, index: 5)
        encoder.setBuffer(output, offset: 0, index: 6)
        var keyHeads = UInt32(state.geometry.keyHeads)
        var valueHeads = UInt32(state.geometry.valueHeads)
        var keyDim = UInt32(state.geometry.keyHeadDim)
        var valueDim = UInt32(state.geometry.valueHeadDim)
        encoder.setBytes(&keyHeads, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&valueHeads, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.stride, index: 10)
        let width = min(
            state.geometry.valueHeadDim,
            recurrentPSO.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(
                width: state.geometry.valueHeadDim,
                height: state.geometry.valueHeads,
                depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                           pipeline: MTLComputePipelineState,
                           count: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}

final class QwenPrefillDeltaNet {
    private let convolutionPSO: MTLComputePipelineState
    private let splitPSO: MTLComputePipelineState
    private let recurrentPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.convolutionPSO = try context.pipeline("qwen_prefill_gated_delta_causal_conv")
        self.splitPSO = try context.pipeline("qwen_prefill_split_qkv")
        self.recurrentPSO = try context.pipeline("qwen_prefill_gated_delta_recurrent")
    }

    func encodeCausalConvolution(commandBuffer: MTLCommandBuffer,
                                 input: MTLBuffer,
                                 weights: MTLBuffer,
                                 weightsOffset: Int = 0,
                                 output: MTLBuffer,
                                 state: QwenGatedDeltaNetState,
                                 tokenCount: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convolutionPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(state.convolutionBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var channels = UInt32(state.convolutionChannels)
        var kernel = UInt32(state.geometry.convolutionKernel)
        var tokens = tokenCount
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 6)
        dispatch(encoder, pipeline: convolutionPSO, count: state.convolutionChannels)
        encoder.endEncoding()
    }

    func encodeSplitQKV(commandBuffer: MTLCommandBuffer,
                        input: MTLBuffer,
                        query: MTLBuffer,
                        key: MTLBuffer,
                        value: MTLBuffer,
                        tokenCount: UInt32,
                        keyWidth: UInt32,
                        valueWidth: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(splitPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(query, offset: 0, index: 1)
        encoder.setBuffer(key, offset: 0, index: 2)
        encoder.setBuffer(value, offset: 0, index: 3)
        var tokens = tokenCount
        var keys = keyWidth
        var values = valueWidth
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&keys, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&values, length: MemoryLayout<UInt32>.stride, index: 6)
        let width = max(Int(keyWidth), Int(valueWidth))
        encoder.dispatchThreads(
            MTLSize(width: width, height: Int(tokenCount), depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(width, 256), height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRecurrent(commandBuffer: MTLCommandBuffer,
                         query: MTLBuffer,
                         key: MTLBuffer,
                         value: MTLBuffer,
                         decay: MTLBuffer,
                         beta: MTLBuffer,
                         output: MTLBuffer,
                         state: QwenGatedDeltaNetState,
                         tokenCount: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(recurrentPSO)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(decay, offset: 0, index: 3)
        encoder.setBuffer(beta, offset: 0, index: 4)
        encoder.setBuffer(state.recurrentBuffer, offset: 0, index: 5)
        encoder.setBuffer(output, offset: 0, index: 6)
        var tokens = tokenCount
        var keyHeads = UInt32(state.geometry.keyHeads)
        var valueHeads = UInt32(state.geometry.valueHeads)
        var keyDim = UInt32(state.geometry.keyHeadDim)
        var valueDim = UInt32(state.geometry.valueHeadDim)
        var queryStride = UInt32(state.geometry.keyDimension)
        var keyStride = UInt32(state.geometry.keyDimension)
        var valueStride = UInt32(state.geometry.valueDimension)
        var outputStride = UInt32(state.geometry.valueDimension)
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&keyHeads, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&valueHeads, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&queryStride, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&keyStride, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&valueStride, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.stride, index: 15)
        let width = min(
            state.geometry.valueHeadDim,
            recurrentPSO.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(
                width: state.geometry.valueHeadDim,
                height: state.geometry.valueHeads,
                depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                           pipeline: MTLComputePipelineState,
                           count: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}