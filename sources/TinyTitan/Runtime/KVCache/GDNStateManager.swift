import Foundation
import Metal
import TinyTitanCore

/// Fixed-size recurrent state for gated-DeltaNet linear-attention layers
/// (`LayerKind.linear`). Unlike KV rows, this state does not grow with
/// context: each linear layer owns
///
///  - a delta-rule state `S`, FP32 `[numVHeads, valueHeadDim, keyHeadDim]`
///    (2 MiB per layer at Qwen 3.6's 32x128x128), and
///  - a causal-conv tail of the last `convKernelSize - 1` pre-activation
///    `mixed_qkv` rows, FP16 `[convKernelSize - 1, convDim]`.
///
/// Buffers are allocated once in `init`; the decode hot path never allocates.
/// `reset()` zero-fills explicitly — the recurrence and the conv both define
/// the empty-context state as zeros, and zeroing 60-odd MiB per generation
/// start is cheap next to a prefill.
public final class GDNStateManager {
    public let config: ArchConfig

    /// Non-nil only at indices whose layer mask is 2.
    private let stateBuffers: [MTLBuffer?]
    private let convTailBuffers: [MTLBuffer?]
    private let speculativeStateBuffers: [MTLBuffer?]
    private let speculativeConvTailBuffers: [MTLBuffer?]

    public let stateBytesPerLayer: Int
    public let convTailBytesPerLayer: Int

    /// How many independent sequences share this state. Each slot owns a
    /// contiguous region of every layer's state and conv-tail buffer, so a
    /// batched decode step advances one recurrence per sequence. Defaults to
    /// one, and at `slots == 1` every region is at offset 0 exactly as before.
    public let slots: Int

    /// Hard cap, mirroring `KVCacheManager.maximumSlots`, which carries the
    /// reasoning: a sanity bound on the argument, while the memory budget decides
    /// what a machine can really hold.
    public static let maximumSlots = KVCacheManager.maximumSlots

    private static let fp32Size = 4
    private static let fp16Size = 2

    /// Worst-case bytes `slots` sequences' recurrent state and conv tails
    /// occupy. Unlike KV this does not grow with context: it is the same at
    /// load time and at the context limit.
    public static func worstCaseBytes(config: ArchConfig, slots: Int = 1) -> Int {
        precondition(slots > 0, "slots must be positive")
        let linear = config.linearAttention
        let stateBytes =
            linear.numVHeads * linear.valueHeadDim
            * linear.keyHeadDim * fp32Size
        let convTailBytes =
            max(0, linear.convKernelSize - 1)
            * linear.qkvDim * fp16Size
        let linearLayers = (0..<config.numLayers)
            .filter { config.layerIsLinear($0) }
            .count
        return slots * linearLayers * (stateBytes + convTailBytes)
    }

    public init(
        device: MTLDevice, config: ArchConfig,
        slots: Int = 1,
        enableSpeculativeCheckpoint: Bool = false
    ) throws {
        precondition(
            slots > 0 && slots <= Self.maximumSlots,
            "slots must be between 1 and \(Self.maximumSlots)")
        self.config = config
        self.slots = slots
        let la = config.linearAttention
        let stateBytes = la.numVHeads * la.valueHeadDim * la.keyHeadDim * Self.fp32Size
        let convTailBytes = max(0, la.convKernelSize - 1) * la.qkvDim * Self.fp16Size
        self.stateBytesPerLayer = stateBytes
        self.convTailBytesPerLayer = convTailBytes
        let stateBufferBytes = stateBytes * slots
        let convTailBufferBytes = convTailBytes * slots

        var states: [MTLBuffer?] = []
        var tails: [MTLBuffer?] = []
        var speculativeStates: [MTLBuffer?] = []
        var speculativeTails: [MTLBuffer?] = []
        states.reserveCapacity(config.numLayers)
        tails.reserveCapacity(config.numLayers)
        speculativeStates.reserveCapacity(config.numLayers)
        speculativeTails.reserveCapacity(config.numLayers)

        for layer in 0..<config.numLayers {
            guard config.layerIsLinear(layer) else {
                states.append(nil)
                tails.append(nil)
                speculativeStates.append(nil)
                speculativeTails.append(nil)
                continue
            }
            guard stateBytes > 0, convTailBytes > 0 else {
                throw ModelError.internalInconsistency(
                    detail: "linear layer \(layer) present but linearAttention config is empty")
            }
            guard
                let state = device.makeBuffer(
                    length: stateBufferBytes,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            state.label = "gdn.state.layer\(layer)"
            guard
                let tail = device.makeBuffer(
                    length: convTailBufferBytes,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            tail.label = "gdn.convtail.layer\(layer)"
            states.append(state)
            tails.append(tail)
            if enableSpeculativeCheckpoint {
                guard
                    let speculativeState = device.makeBuffer(
                        length: stateBufferBytes,
                        options: .storageModeShared),
                    let speculativeTail = device.makeBuffer(
                        length: convTailBufferBytes,
                        options: .storageModeShared)
                else {
                    throw ModelError.residentBufferWrapFailed
                }
                speculativeState.label = "gdn.speculative-state.layer\(layer)"
                speculativeTail.label = "gdn.speculative-convtail.layer\(layer)"
                speculativeStates.append(speculativeState)
                speculativeTails.append(speculativeTail)
            } else {
                speculativeStates.append(nil)
                speculativeTails.append(nil)
            }
        }
        self.stateBuffers = states
        self.convTailBuffers = tails
        self.speculativeStateBuffers = speculativeStates
        self.speculativeConvTailBuffers = speculativeTails
        zeroAll()
    }

    /// Delta-rule state `S` for a linear layer.
    public func stateBuffer(layer: Int) -> MTLBuffer {
        guard let buffer = stateBuffers[layer] else {
            preconditionFailure("layer \(layer) is not a linear-attention layer")
        }
        return buffer
    }

    /// Rolling window of the last `convKernelSize - 1` mixed_qkv rows.
    public func convTailBuffer(layer: Int) -> MTLBuffer {
        guard let buffer = convTailBuffers[layer] else {
            preconditionFailure("layer \(layer) is not a linear-attention layer")
        }
        return buffer
    }

    func speculativeStateBuffer(layer: Int) -> MTLBuffer {
        guard let buffer = speculativeStateBuffers[layer] else {
            preconditionFailure("layer \(layer) is not a linear-attention layer")
        }
        return buffer
    }

    func speculativeConvTailBuffer(layer: Int) -> MTLBuffer {
        guard let buffer = speculativeConvTailBuffers[layer] else {
            preconditionFailure("layer \(layer) is not a linear-attention layer")
        }
        return buffer
    }

    /// Byte offset of slot `slot`'s delta-rule state in its layer's buffer.
    public func stateOffset(layer: Int, slot: Int) -> Int {
        validateSlot(slot)
        return slot * stateBytesPerLayer
    }

    /// Byte offset of slot `slot`'s conv tail in its layer's buffer.
    public func convTailOffset(layer: Int, slot: Int) -> Int {
        validateSlot(slot)
        return slot * convTailBytesPerLayer
    }

    /// Slot `slot`'s state buffer and the offset the kernel must bind it at.
    public func stateSlot(layer: Int, slot: Int) -> (buffer: MTLBuffer, offset: Int) {
        (stateBuffer(layer: layer), stateOffset(layer: layer, slot: slot))
    }

    /// Slot `slot`'s conv-tail buffer and the offset the kernel must bind it at.
    public func convTailSlot(layer: Int, slot: Int) -> (buffer: MTLBuffer, offset: Int) {
        (convTailBuffer(layer: layer), convTailOffset(layer: layer, slot: slot))
    }

    private func validateSlot(_ slot: Int) {
        precondition(
            slot >= 0 && slot < slots,
            "slot \(slot) is out of range 0..<\(slots)")
    }

    public func isLinear(layer: Int) -> Bool { stateBuffers[layer] != nil }

    /// Reset all recurrent state to the empty-context value (zeros).
    public func reset() {
        zeroAll()
    }

    /// Reset one slot's recurrent state to zeros without touching the others.
    /// The recurrence and the conv both define the empty-context state as
    /// zeros, so a recycled batch slot starts clean.
    public func reset(slot: Int) {
        validateSlot(slot)
        let stateRange = slot * stateBytesPerLayer
        let tailRange = slot * convTailBytesPerLayer
        for layer in 0..<config.numLayers {
            if let state = stateBuffers[layer] {
                memset(state.contents() + stateRange, 0, stateBytesPerLayer)
            }
            if let tail = convTailBuffers[layer] {
                memset(tail.contents() + tailRange, 0, convTailBytesPerLayer)
            }
        }
    }

    func snapshotSegmentLengths() -> [Int] {
        var lengths: [Int] = []
        lengths.reserveCapacity(config.numLayers * 2)
        for layer in 0..<config.numLayers where stateBuffers[layer] != nil {
            lengths.append(stateBytesPerLayer)
            lengths.append(convTailBytesPerLayer)
        }
        return lengths
    }

    func appendSnapshotPayload(
        to payload: inout Data,
        segmentLengths: [Int]
    ) throws {
        guard segmentLengths == snapshotSegmentLengths() else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        var segment = 0
        for layer in 0..<config.numLayers {
            guard let state = stateBuffers[layer],
                let tail = convTailBuffers[layer]
            else { continue }
            let stateLength = segmentLengths[segment]
            payload.append(
                state.contents().assumingMemoryBound(to: UInt8.self),
                count: stateLength)
            segment += 1
            let tailLength = segmentLengths[segment]
            payload.append(
                tail.contents().assumingMemoryBound(to: UInt8.self),
                count: tailLength)
            segment += 1
        }
    }

    /// Restore the on-GPU state captured immediately after the confirmed row
    /// of a two-token target verification batch.
    func encodeSpeculativeRestore(commandBuffer: MTLCommandBuffer) throws {
        for layer in 0..<config.numLayers where stateBuffers[layer] != nil {
            guard speculativeStateBuffers[layer] != nil,
                speculativeConvTailBuffers[layer] != nil
            else {
                throw InferenceStateSnapshotError.invalidLayout
            }
        }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        for layer in 0..<config.numLayers {
            guard let state = stateBuffers[layer],
                let tail = convTailBuffers[layer],
                let speculativeState = speculativeStateBuffers[layer],
                let speculativeTail = speculativeConvTailBuffers[layer]
            else { continue }
            blit.copy(
                from: speculativeState, sourceOffset: 0,
                to: state, destinationOffset: 0,
                size: stateBytesPerLayer)
            blit.copy(
                from: speculativeTail, sourceOffset: 0,
                to: tail, destinationOffset: 0,
                size: convTailBytesPerLayer)
        }
        blit.endEncoding()
    }

    var speculativePayloadBytes: Int {
        speculativeStateBuffers.contains { $0 != nil }
            ? snapshotSegmentLengths().reduce(0, +) : 0
    }

    func restoreSnapshot(
        segmentLengths: [Int],
        bytes: UnsafeRawBufferPointer,
        offset: inout Int
    ) throws {
        guard segmentLengths == snapshotSegmentLengths() else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        reset()
        var segment = 0
        for layer in 0..<config.numLayers {
            guard let state = stateBuffers[layer],
                let tail = convTailBuffers[layer]
            else { continue }
            try copySnapshotSegment(
                bytes: bytes,
                offset: &offset,
                length: segmentLengths[segment],
                destination: state)
            segment += 1
            try copySnapshotSegment(
                bytes: bytes,
                offset: &offset,
                length: segmentLengths[segment],
                destination: tail)
            segment += 1
        }
    }

    private func copySnapshotSegment(
        bytes: UnsafeRawBufferPointer,
        offset: inout Int,
        length: Int,
        destination: MTLBuffer
    ) throws {
        guard length <= destination.length,
            offset >= 0,
            length >= 0,
            offset <= bytes.count - length,
            let source = bytes.baseAddress?.advanced(by: offset)
        else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        memcpy(destination.contents(), source, length)
        offset += length
    }

    private func zeroAll() {
        for buffer in stateBuffers {
            if let buffer { memset(buffer.contents(), 0, buffer.length) }
        }
        for buffer in convTailBuffers {
            if let buffer { memset(buffer.contents(), 0, buffer.length) }
        }
    }
}
