import Foundation
import Darwin
import Metal

/// Which attention variant a layer runs. Gemma 4 interleaves 25 sliding-window
/// layers with 5 full-attention layers. Sourced from
/// `ArchConfig.fullAttentionLayerMask`.
enum LayerKind: Sendable { case swa, full }

enum KVCacheStorageMode: Sendable, Equatable {
    case fp16
    case turboQuant(TurboQuantKVMode)
}

/// Per-layer FP16 K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init` — the
/// decode hot path never allocates. Linear storage sizes every layer for
/// `maxContext`; FP16 ring storage caps SWA layers to their physical capacity
/// while full-attention layers remain linear.
///
/// Full-attention layers share the raw `k_proj` output. K then runs scaled
/// per-head normalization plus RoPE, while V runs no-scale normalization and
/// skips RoPE. The cache buffers must therefore remain separate.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
final class KVCacheManager {
    let config: ArchConfig
    let maxContext: Int
    let storageMode: KVCacheStorageMode
    let fp16RingEnabled: Bool

    private let kBuffers: [MTLBuffer]
    private let vBuffers: [MTLBuffer]
    private let quantizedKBuffers: [MTLBuffer]
    private let quantizedVBuffers: [MTLBuffer]
    private let turboQuantLayouts: [TurboQuantKVLayerLayout]
    private let strides:  [Int]         // bytes per token, per layer
    private let kinds:    [LayerKind]
    private let capacityTokens: [Int]

    private(set) var position: Int = 0

    private static let fp16Size = 2

    init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                storageMode: KVCacheStorageMode = .fp16,
                fp16RingEnabled: Bool = false,
                slidingWindow: Int? = nil,
                maxPrefillChunkTokens: Int = 128,
                fp16RingCapacityOverride: Int? = nil) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        if case .turboQuant(.disabled) = storageMode {
            preconditionFailure("disabled TurboQuant mode cannot be used as a storage mode")
        }
        self.config = config
        self.maxContext = maxContext
        self.storageMode = storageMode
        let ringEnabled = storageMode == .fp16 && fp16RingEnabled
        self.fp16RingEnabled = ringEnabled

        let swaStride  = config.numKVHeads     * config.headDim     * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim  * Self.fp16Size
        let swaCapacity = min(maxContext,
                              max(1, fp16RingCapacityOverride
                                  ?? ((slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens)))

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        var qks: [MTLBuffer] = []
        var qvs: [MTLBuffer] = []
        var qls: [TurboQuantKVLayerLayout] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        qks.reserveCapacity(config.numLayers)
        qvs.reserveCapacity(config.numLayers)
        qls.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)

        for layer in 0..<config.numLayers {
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let stride = isFull ? fullStride : swaStride
            let capacity = ringEnabled && !isFull ? swaCapacity : maxContext
            switch storageMode {
            case .fp16:
                let length = capacity * stride

                guard let kBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                kBuf.label = "kv.K.layer\(layer)"
                ks.append(kBuf)

                guard let vBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                vBuf.label = "kv.V.layer\(layer)"
                vs.append(vBuf)

            case .turboQuant(let mode):
                let layout = TurboQuantKVLayout.layer(mode: mode,
                                                      config: config,
                                                      layer: layer,
                                                      capacity: maxContext)
                guard let kBuf = device.makeBuffer(length: maxContext * layout.key.bytesPerToken,
                                                   options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                kBuf.label = "kv.tq.K.layer\(layer)"
                qks.append(kBuf)

                guard let vBuf = device.makeBuffer(length: maxContext * layout.value.bytesPerToken,
                                                   options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                vBuf.label = "kv.tq.V.layer\(layer)"
                qvs.append(vBuf)
                qls.append(layout)
            }

            st.append(stride)
            kd.append(isFull ? .full : .swa)
            caps.append(capacity)
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.quantizedKBuffers = qks
        self.quantizedVBuffers = qvs
        self.turboQuantLayouts = qls
        self.strides  = st
        self.kinds    = kd
        self.capacityTokens = caps
    }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    func capacity(layer: Int) -> Int { capacityTokens[layer] }

    func ringCapacity(layer: Int) -> Int {
        guard storageMode == .fp16, fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    func turboQuantLayout(layer: Int) -> TurboQuantKVLayerLayout? {
        guard case .turboQuant = storageMode else { return nil }
        return turboQuantLayouts[layer]
    }

    /// Write target for this layer's K projection at `position`.
    func kSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        precondition(storageMode == .fp16, "FP16 K slots are unavailable for TurboQuant storage")
        validateRange(start: position, count: 1)
        return (kBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// Write target for this layer's V projection at `position`. It is distinct
    /// from `kSlot` because full layers normalize K and V differently and apply
    /// RoPE only to K.
    func vSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        precondition(storageMode == .fp16, "FP16 V slots are unavailable for TurboQuant storage")
        validateRange(start: position, count: 1)
        return (vBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    func kRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        precondition(storageMode == .fp16, "FP16 K ranges are unavailable for TurboQuant storage")
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (kBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    func vRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        precondition(storageMode == .fp16, "FP16 V ranges are unavailable for TurboQuant storage")
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (vBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        precondition(storageMode == .fp16, "FP16 K buffers are unavailable for TurboQuant storage")
        validateValidTokenCount(validTokenCount)
        return kBuffers[layer]
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        precondition(storageMode == .fp16, "FP16 V buffers are unavailable for TurboQuant storage")
        validateValidTokenCount(validTokenCount)
        return vBuffers[layer]
    }

    func quantizedKeySlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        let layout = requireTurboQuantLayout(layer: layer)
        validateRange(start: position, count: 1)
        return (quantizedKBuffers[layer], position * layout.key.bytesPerToken)
    }

    func quantizedValueSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        let layout = requireTurboQuantLayout(layer: layer)
        validateRange(start: position, count: 1)
        return (quantizedVBuffers[layer], position * layout.value.bytesPerToken)
    }

    func quantizedKeyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        _ = requireTurboQuantLayout(layer: layer)
        validateValidTokenCount(validTokenCount)
        return quantizedKBuffers[layer]
    }

    func quantizedValueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        _ = requireTurboQuantLayout(layer: layer)
        validateValidTokenCount(validTokenCount)
        return quantizedVBuffers[layer]
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    func advance() { advance(by: 1) }

    func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext, "advance would exceed maxContext")
        position += count
    }

    /// Drop all cached positions and return physical pages to the OS.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. `MADV_DONTNEED` on the page-aligned span
    /// releases resident memory between turns; pages fault back in on next write.
    func reset() {
        position = 0
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            switch storageMode {
            case .fp16:
                advise(kBuffers[layer], pageSize: pageSize, seen: &advised)
                advise(vBuffers[layer], pageSize: pageSize, seen: &advised)
            case .turboQuant:
                advise(quantizedKBuffers[layer], pageSize: pageSize, seen: &advised)
                advise(quantizedVBuffers[layer], pageSize: pageSize, seen: &advised)
            }
        }
    }

    private func requireTurboQuantLayout(layer: Int) -> TurboQuantKVLayerLayout {
        guard case .turboQuant = storageMode else {
            preconditionFailure("TurboQuant KV layout is unavailable for FP16 storage")
        }
        return turboQuantLayouts[layer]
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(count <= maxContext,
                     "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        position % capacityTokens[layer]
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, storageMode == .fp16, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(physicalStart + count <= capacity,
                     "range \(start)..<\(start + count) wraps FP16 KV ring capacity \(capacity)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
