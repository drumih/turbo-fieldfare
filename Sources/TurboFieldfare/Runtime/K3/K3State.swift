import Foundation
import Darwin
import Metal

/// Byte-size breakdown of `K3State` allocations, for the CLI/HUD diagnostics
/// pattern (same role as the KV-cache accounting in the Gemma runner).
public struct K3StateMemoryReport: Sendable, Equatable {
    public var kdaStateBytes: Int
    public var kdaConvBytes: Int
    public var mlaCacheBytes: Int
    public var attnResBytes: Int
    public var totalBytes: Int {
        kdaStateBytes + kdaConvBytes + mlaCacheBytes + attnResBytes
    }

    public init(kdaStateBytes: Int, kdaConvBytes: Int, mlaCacheBytes: Int,
                attnResBytes: Int) {
        self.kdaStateBytes = kdaStateBytes
        self.kdaConvBytes = kdaConvBytes
        self.mlaCacheBytes = mlaCacheBytes
        self.attnResBytes = attnResBytes
    }
}

/// Exact, in-memory checkpoint of one K3 decode cursor. MLA rows are packed
/// only through `position`; the fixed KDA/conv state and small AttnRes scratch
/// are copied in full. Buffers remain in shared Metal storage so capture and
/// restore avoid a second Data representation of the ~434 MB KDA state.
final class K3StateSnapshot: @unchecked Sendable {
    let config: K3ArchConfig
    let maxContext: Int
    let position: Int
    let attnResBlockCount: Int
    let kdaState: MTLBuffer
    let kdaConv: MTLBuffer
    let mlaActive: MTLBuffer
    let mlaActiveBytesPerLayer: Int
    let attnResBlocks: MTLBuffer
    let attnResPrefix: MTLBuffer

    init(config: K3ArchConfig,
         maxContext: Int,
         position: Int,
         attnResBlockCount: Int,
         kdaState: MTLBuffer,
         kdaConv: MTLBuffer,
         mlaActive: MTLBuffer,
         mlaActiveBytesPerLayer: Int,
         attnResBlocks: MTLBuffer,
         attnResPrefix: MTLBuffer) {
        self.config = config
        self.maxContext = maxContext
        self.position = position
        self.attnResBlockCount = attnResBlockCount
        self.kdaState = kdaState
        self.kdaConv = kdaConv
        self.mlaActive = mlaActive
        self.mlaActiveBytesPerLayer = mlaActiveBytesPerLayer
        self.attnResBlocks = attnResBlocks
        self.attnResPrefix = attnResPrefix
    }

    var totalBytes: Int {
        kdaState.length + kdaConv.length
            + mlaLayerBytes
            + attnResBlocks.length + attnResPrefix.length
    }

    private var mlaLayerBytes: Int {
        mlaActiveBytesPerLayer == 0 ? 0 : mlaActive.length
    }
}

/// Decode-state owner for the K3 runtime (the Stage-C2 forward runner binds
/// these buffers into the KDA / MLA / AttnRes kernels).
///
/// Persistent across tokens: KDA recurrent states (fp32, one slab, per-layer
/// offsets), KDA conv states (fp32, one slab), MLA latent KV caches (fp16,
/// one slab, token-major rows of 576 values), and the position cursor.
/// Per-token scratch: the AttnRes block slab (fp16 [maxBlocks][hidden]),
/// the prefix vector, and the block count — rebuilt every token, so they are
/// not part of `reset()`'s persistence contract.
///
/// `reset()` returns physical pages to the OS via `MADV_DONTNEED` (same
/// pattern as `KVCacheManager.reset`): the slabs fault back in zeroed on next
/// write, which is exactly the zero state a fresh generation needs — no
/// explicit zeroing pass.
public final class K3State {
    public let config: K3ArchConfig
    public let maxContext: Int

    /// Tokens written so far (decode advances by 1; prefill by chunk).
    public private(set) var position: Int = 0
    /// AttnRes block-list length for the in-flight token (0 at token start,
    /// +1 at every boundary layer). Scratch, not persisted state.
    public private(set) var attnResBlockCount: Int = 0

    private let kdaStateSlab: MTLBuffer
    private let kdaConvSlab: MTLBuffer
    private let mlaCacheSlab: MTLBuffer
    private let attnResBlockSlab: MTLBuffer
    private let attnResPrefixBuffer: MTLBuffer

    public let kdaLayerCount: Int
    public let mlaLayerCount: Int
    /// fp32 elements per KDA layer in the state slab.
    public let kdaStateStrideElements: Int
    /// fp32 elements per KDA layer in the conv slab.
    public let kdaConvStrideElements: Int
    /// fp16 elements per MLA layer in the cache slab (capacity × row).
    public let mlaCacheStrideElements: Int
    /// fp16 elements per cached token row (latent + rope = 576).
    public var mlaCacheRowElements: Int { config.mlaCacheRowElements }
    /// AttnRes block slab rows (kernel staging maximum).
    public let attnResMaxBlocks: Int

    private static let fp32Size = MemoryLayout<Float>.stride
    private static let fp16Size = MemoryLayout<Float16>.stride

    public init(device: MTLDevice, config: K3ArchConfig, maxContext: Int) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(config.attnResBoundaryCount <= K3AttnRes.maxBlocks,
                     "attnResBoundaryCount \(config.attnResBoundaryCount) exceeds "
                        + "kernel staging (\(K3AttnRes.maxBlocks))")
        self.config = config
        self.maxContext = maxContext
        self.kdaLayerCount = config.kdaLayers0.count
        self.mlaLayerCount = config.mlaLayers0.count
        self.kdaStateStrideElements = config.kdaStateElementsPerLayer
        self.kdaConvStrideElements = config.kdaConvStateElementsPerLayer
        self.mlaCacheStrideElements = maxContext * config.mlaCacheRowElements
        self.attnResMaxBlocks = K3AttnRes.maxBlocks

        func makeSlab(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(length, 1),
                                                 options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        self.kdaStateSlab = try makeSlab(
            kdaLayerCount * config.kdaStateElementsPerLayer * Self.fp32Size,
            "k3.state.kda_recurrent")
        self.kdaConvSlab = try makeSlab(
            kdaLayerCount * config.kdaConvStateElementsPerLayer * Self.fp32Size,
            "k3.state.kda_conv")
        self.mlaCacheSlab = try makeSlab(
            mlaLayerCount * maxContext * config.mlaCacheRowElements * Self.fp16Size,
            "k3.state.mla_latent_kv")
        self.attnResBlockSlab = try makeSlab(
            K3AttnRes.maxBlocks * config.hiddenSize * Self.fp16Size,
            "k3.state.attnres_blocks")
        self.attnResPrefixBuffer = try makeSlab(
            config.hiddenSize * Self.fp16Size,
            "k3.state.attnres_prefix")
    }

    // MARK: - KDA state views

    /// fp32 [heads][headDim][headDim] recurrent state for KDA layer `layer0`.
    public func kdaRecurrentState(layer0: Int) -> (buffer: MTLBuffer, offset: Int) {
        guard let ordinal = config.kdaOrdinal(layer0: layer0) else {
            preconditionFailure("layer \(layer0) is not a KDA layer")
        }
        return (kdaStateSlab, ordinal * kdaStateStrideElements * Self.fp32Size)
    }

    /// fp32 [3][channels][convWidth−1] conv state for KDA layer `layer0`.
    public func kdaConvState(layer0: Int) -> (buffer: MTLBuffer, offset: Int) {
        guard let ordinal = config.kdaOrdinal(layer0: layer0) else {
            preconditionFailure("layer \(layer0) is not a KDA layer")
        }
        return (kdaConvSlab, ordinal * kdaConvStrideElements * Self.fp32Size)
    }

    // MARK: - MLA latent cache views

    /// fp16 [maxContext][latent+rope] latent KV cache for MLA layer `layer0`
    /// (token-major rows). The cache-append kernel takes the token position
    /// separately, so callers bind the layer base.
    public func mlaCache(layer0: Int) -> (buffer: MTLBuffer, offset: Int) {
        guard let ordinal = config.mlaOrdinal(layer0: layer0) else {
            preconditionFailure("layer \(layer0) is not an MLA layer")
        }
        return (mlaCacheSlab, ordinal * mlaCacheStrideElements * Self.fp16Size)
    }

    /// Byte stride of one cached token row (fp16).
    public var mlaCacheRowBytes: Int { mlaCacheRowElements * Self.fp16Size }

    // MARK: - AttnRes scratch views

    /// fp16 [attnResMaxBlocks][hidden] block slab for the in-flight token.
    public var attnResBlocks: MTLBuffer { attnResBlockSlab }
    /// fp16 [hidden] running prefix for the in-flight token.
    public var attnResPrefix: MTLBuffer { attnResPrefixBuffer }

    /// Token start: the block list is rebuilt every token
    /// (docs/K3_DATAFLOW.md — B accumulates within one forward pass only).
    public func beginToken() { attnResBlockCount = 0 }

    /// Record a boundary layer's block append (layers where
    /// `config.isAttnResBoundary(layer0:)` holds).
    public func recordAttnResBoundary() {
        precondition(attnResBlockCount < attnResMaxBlocks,
                     "AttnRes block list exceeds the slab")
        attnResBlockCount += 1
    }

    // MARK: - Position

    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext,
                     "advance would exceed maxContext \(maxContext)")
        position += count
    }

    // MARK: - Prefix snapshot / restore

    func captureSnapshot() throws -> K3StateSnapshot {
        let device = kdaStateSlab.device
        func copyBuffer(_ source: MTLBuffer, length: Int, label: String) throws
            -> MTLBuffer {
            guard let destination = device.makeBuffer(
                    length: max(length, 1), options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            destination.label = label
            if length > 0 {
                memcpy(destination.contents(), source.contents(), length)
            }
            return destination
        }

        let activeMLABytesPerLayer = position * mlaCacheRowBytes
        let activeMLABytes = mlaLayerCount * activeMLABytesPerLayer
        guard let mla = device.makeBuffer(
                length: max(activeMLABytes, 1), options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        mla.label = "k3.snapshot.mla_active"
        if activeMLABytesPerLayer > 0 {
            for ordinal in 0..<mlaLayerCount {
                memcpy(
                    mla.contents().advanced(by: ordinal * activeMLABytesPerLayer),
                    mlaCacheSlab.contents().advanced(
                        by: ordinal * mlaCacheStrideElements * Self.fp16Size),
                    activeMLABytesPerLayer)
            }
        }

        return try K3StateSnapshot(
            config: config,
            maxContext: maxContext,
            position: position,
            attnResBlockCount: attnResBlockCount,
            kdaState: copyBuffer(kdaStateSlab, length: kdaStateSlab.length,
                                 label: "k3.snapshot.kda_recurrent"),
            kdaConv: copyBuffer(kdaConvSlab, length: kdaConvSlab.length,
                                label: "k3.snapshot.kda_conv"),
            mlaActive: mla,
            mlaActiveBytesPerLayer: activeMLABytesPerLayer,
            attnResBlocks: copyBuffer(
                attnResBlockSlab, length: attnResBlockSlab.length,
                label: "k3.snapshot.attnres_blocks"),
            attnResPrefix: copyBuffer(
                attnResPrefixBuffer, length: attnResPrefixBuffer.length,
                label: "k3.snapshot.attnres_prefix"))
    }

    func restoreSnapshot(_ snapshot: K3StateSnapshot) {
        precondition(snapshot.config == config, "snapshot architecture mismatch")
        precondition(snapshot.maxContext == maxContext, "snapshot context mismatch")
        precondition(snapshot.position <= maxContext, "snapshot position out of range")
        precondition(snapshot.kdaState.length == kdaStateSlab.length)
        precondition(snapshot.kdaConv.length == kdaConvSlab.length)
        precondition(snapshot.attnResBlocks.length == attnResBlockSlab.length)
        precondition(snapshot.attnResPrefix.length == attnResPrefixBuffer.length)

        memcpy(kdaStateSlab.contents(), snapshot.kdaState.contents(), kdaStateSlab.length)
        memcpy(kdaConvSlab.contents(), snapshot.kdaConv.contents(), kdaConvSlab.length)
        if snapshot.mlaActiveBytesPerLayer > 0 {
            for ordinal in 0..<mlaLayerCount {
                memcpy(
                    mlaCacheSlab.contents().advanced(
                        by: ordinal * mlaCacheStrideElements * Self.fp16Size),
                    snapshot.mlaActive.contents().advanced(
                        by: ordinal * snapshot.mlaActiveBytesPerLayer),
                    snapshot.mlaActiveBytesPerLayer)
            }
        }
        memcpy(attnResBlockSlab.contents(), snapshot.attnResBlocks.contents(),
               attnResBlockSlab.length)
        memcpy(attnResPrefixBuffer.contents(), snapshot.attnResPrefix.contents(),
               attnResPrefixBuffer.length)
        position = snapshot.position
        attnResBlockCount = snapshot.attnResBlockCount
    }

    // MARK: - Reset / accounting

    /// Drop all persistent state.
    ///
    /// Zeroing is semantic for the KDA slabs only: the delta-rule recurrence
    /// reads S unconditionally, so a fresh generation needs actual zeros.
    /// `MADV_DONTNEED` does NOT zero-fill Metal shared-buffer pages (it is a
    /// best-effort page return here), so those two slabs are memset — the
    /// house `KVCacheManager` trick of relying on fault-in zero pages does
    /// not apply to `MTLBuffer` storage. The MLA cache and AttnRes scratch
    /// follow the house KV-cache rule instead: readers are bounded by
    /// `position` / `attnResBlockCount` (both now 0), so stale bytes are
    /// never consumed; their pages still go back via `MADV_DONTNEED`.
    public func reset() {
        position = 0
        attnResBlockCount = 0
        let pageSize = Int(getpagesize())
        func advise(_ buffer: MTLBuffer) {
            let baseAddr = UInt(bitPattern: buffer.contents())
            let pageMask = UInt(pageSize) - 1
            let alignedStart = (baseAddr + pageMask) & ~pageMask
            let headBytes = min(Int(alignedStart - baseAddr), buffer.length)
            let alignedBytes = (buffer.length - headBytes) / pageSize * pageSize
            // Only fully covered, page-aligned interior pages: a fully
            // covered page holds only this slab's bytes, so neighbours are
            // never discarded.
            if alignedBytes > 0 {
                _ = posix_madvise(buffer.contents().advanced(by: headBytes),
                                  alignedBytes, POSIX_MADV_DONTNEED)
            }
        }
        for buffer in [kdaStateSlab, kdaConvSlab] {
            advise(buffer)
            memset(buffer.contents(), 0, buffer.length)
        }
        advise(mlaCacheSlab)
        advise(attnResBlockSlab)
        advise(attnResPrefixBuffer)
    }

    public func memoryReport() -> K3StateMemoryReport {
        K3StateMemoryReport(
            kdaStateBytes: kdaStateSlab.length,
            kdaConvBytes: kdaConvSlab.length,
            mlaCacheBytes: mlaCacheSlab.length,
            attnResBytes: attnResBlockSlab.length + attnResPrefixBuffer.length)
    }
}
