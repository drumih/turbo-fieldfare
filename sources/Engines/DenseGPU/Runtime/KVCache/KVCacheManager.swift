import Darwin
import Foundation
import Metal
import TinyTitanCore

/// Which attention variant a layer runs. Qwen 3.6 interleaves 30
/// gated-DeltaNet linear-attention layers with 10 full-attention layers;
/// linear layers keep a fixed-size recurrent state (owned by
/// `GDNStateManager`) instead of per-token K/V rows. Sourced from
/// `ArchConfig.fullAttentionLayerMask` (0 = swa, 1 = full, 2 = linear).
public enum LayerKind: Sendable { case swa, full, linear }

/// A read view the attention kernels bind. `offset` stays 0; ring-enabled SWA
/// layers expose the physical start slot for diagnostics while kernels map
/// logical positions with the supplied ring capacity.
/// unchecked-invariant: every stored property is a `let`. The type is only
/// @unchecked because MTLBuffer is not Sendable; the struct itself is a
/// read-only descriptor of a range, and callers that write through it are
/// serialised by whoever owns the buffer.
public struct KVView: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset of logical position 0. Zero for slot 0 -- and so always zero
    /// when there is a single slot -- and the slot's region base otherwise.
    public let offset: Int
    /// Bytes per token, including affine metadata for quantized storage.
    public let stride: Int
    /// Number of valid positions written so far (== `position`), i.e. an
    /// exclusive bound: attention reads `[0, validTokenCount)`.
    ///
    /// Stated as a count because that is what it is and what every kernel
    /// iterates (`p < seqLen`). It is deliberately *not* "the index of the last
    /// valid token", which is `validTokenCount - 1`: the decode path passes
    /// `position + 1` and prefill passes `startPosition + t` precisely so the
    /// just-written token is included. The `keyView(layer:)` /
    /// `valueView(layer:)` convenience overloads default to `position`, which
    /// before `advance()` is one *fewer* than that — correct for a caller reading
    /// everything but the token it is about to write, and not a bound to copy
    /// into a new call site without knowing which of the two it wants.
    public let validTokenCount: Int
    public let precision: KVCachePrecision
    public let valueBytes: Int
    public let groupSize: Int
}

/// Per-layer K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init` — the
/// decode hot path never allocates. Linear storage sizes every layer for
/// `maxContext`; ring storage caps SWA layers to their physical capacity
/// while full-attention layers remain linear. With more than one slot the
/// layer buffer holds `slots` contiguous regions of that same capacity, one per
/// concurrent sequence, and every accessor takes the slot whose region it means.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let config: ArchConfig
    public let maxContext: Int
    public let fp16RingEnabled: Bool
    public let precision: KVCachePrecision
    /// Retained so `reserve` can allocate; init-only allocation was the previous
    /// invariant and growth deliberately relaxes it.
    let device: MTLDevice

    var kBuffers: [MTLBuffer]
    var vBuffers: [MTLBuffer]
    let strides: [Int]  // bytes per token, per layer
    let kinds: [LayerKind]
    var capacityTokens: [Int]
    let valueBytes: [Int]

    /// How many independent sequences share these stores. Each slot owns its own
    /// contiguous region of every layer buffer and its own cursor, so a batched
    /// decode step writes one token per slot without aliasing. Defaults to one:
    /// at `slots == 1` every offset here is exactly what it was before slots
    /// existed, which is what keeps the single-sequence path byte-identical.
    public let slots: Int

    /// Hard cap, so a bad argument cannot ask for an absurd allocation.
    ///
    /// Not the practical limit: what a machine can actually hold is decided per
    /// load by `BatchedMemoryBudget`, which clamps the width to the worst-case
    /// per-slot stores and logs what it built. This is the ceiling that keeps the
    /// multiplication below in a sane range for a number someone typed.
    public static let maximumSlots = 256

    /// Host-KV streaming (`docs/plan-host-kv-streaming.md`).
    ///
    /// When this is non-nil the full-attention layers keep their authoritative
    /// rows in a host arena and `kBuffers`/`vBuffers` become a window of this
    /// many rows per layer and slot. SWA and linear layers are unchanged: an
    /// SWA ring is already bounded, and a linear layer has no rows. A nil window
    /// is the original contiguous layout, byte for byte.
    public let windowTokens: Int?
    public var streamsKVFromHost: Bool { windowTokens != nil }

    var hostK: [UnsafeMutableRawPointer?] = []
    var hostV: [UnsafeMutableRawPointer?] = []
    /// `slots * maxContext * stride` per streamed layer (K and V each).
    var hostArenaLengths: [Int] = []
    /// Per streamed layer: `slots * maxContext` logical -> window row, or
    /// `UInt32.max` when the row is not resident.
    var windowRowMap: [[UInt32]] = []
    /// Per streamed layer: `slots * windowTokens` window row -> logical
    /// position, or `UInt32.max` when free.
    var windowOccupant: [[UInt32]] = []
    /// Per streamed layer and slot: next window row the clock hands out.
    var windowCursor: [[Int]] = []
    /// Per streamed layer: one flag per window row.
    var windowDirty: [[Bool]] = []
    /// Per streamed layer and slot: the logical position being written by the
    /// current decode step. It must survive the selection sweep.
    var windowPinned: [[Int]] = []
    /// Diagnostic counters, read by the acceptance tests.
    public var windowCopyBytes: Int = 0
    public var windowEvictions: Int = 0
    public var windowRowCopies: Int = 0

    /// Prefill writes land here first (quantized or fp16, exactly what the
    /// attention was going to read), then move to the host arena in `advance`
    /// once the chunk's command buffers have completed. A chunk is the unit of
    /// transfer; the window is the unit of read.
    var prefillStagingK: [MTLBuffer?] = []
    var prefillStagingV: [MTLBuffer?] = []
    var prefillStagingCapacity: [Int] = []
    var prefillStagingStart: [Int] = []
    var prefillStagingCount: [Int] = []
    var prefillStagingActive: [Bool] = []
    /// Selection scratch shared by the serial per-layer gather. Sized on demand.
    var prefillIndicesBuf: MTLBuffer?
    var prefillRowsBuf: MTLBuffer?
    var prefillCountsBuf: MTLBuffer?
    var prefillSelectionCapacity: Int = 0
    var windowIndicesBuf: MTLBuffer?
    var windowRowsBuf: MTLBuffer?
    var windowSelectionCapacity: Int = 0
    /// One byte-bound placeholder for the windowed prefill kernel's unused mask
    /// argument.
    var windowEmptyMask: MTLBuffer?
    /// `[maxContext]` bytes reused to mark the union a prefill chunk needs.
    var windowNeeded: [UInt8] = []
    /// Window rows reserved by the gather in progress. An indirection entry
    /// must not be evicted by a later allocation in the same sweep -- that
    /// would leave the kernel reading another key's row.
    var windowProtected: Set<Int> = []

    /// Logical token cursor per slot. `position` is slot 0's -- the
    /// single-sequence case every existing caller uses.
    var positions: [Int]

    /// Slot 0's cursor.
    public var position: Int { positions[0] }

    static let fp16Size = 2
    public static let quantizationGroupSize = 64

    public init(
        device: MTLDevice,
        config: ArchConfig,
        maxContext: Int,
        slots: Int = 1,
        fp16RingEnabled: Bool = false,
        precision: KVCachePrecision = .fp16,
        slidingWindow: Int? = nil,
        maxPrefillChunkTokens: Int = 128,
        windowTokens: Int? = nil
    ) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        precondition(
            slots > 0 && slots <= Self.maximumSlots,
            "slots must be between 1 and \(Self.maximumSlots)")
        if let windowTokens {
            precondition(windowTokens > 0, "windowTokens must be positive")
        }
        self.device = device
        self.config = config
        self.maxContext = maxContext
        self.slots = slots
        self.positions = Array(repeating: 0, count: slots)
        self.precision = precision
        self.windowTokens = windowTokens.map { min($0, maxContext) }
        if let window = self.windowTokens, config.sparseIndexer.enabled {
            let width =
                config.sparseIndexer.budget + config.sparseIndexer.compressRatio - 1
            guard window >= width + 1 else {
                throw ModelError.internalInconsistency(
                    detail:
                        "host-KV window \(window) cannot hold a selection of "
                        + "\(width) keys plus the in-flight token; raise "
                        + "--host-kv-window above \(width)")
            }
        }
        let ringEnabled = fp16RingEnabled
        self.fp16RingEnabled = ringEnabled

        let swaStride = config.numKVHeads * config.headDim * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim * Self.fp16Size
        let swaCapacity = min(
            maxContext,
            max(1, (slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens))

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        var valueByteCounts: [Int] = []
        var hk: [UnsafeMutableRawPointer?] = []
        var hv: [UnsafeMutableRawPointer?] = []
        var hLengths: [Int] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)
        valueByteCounts.reserveCapacity(config.numLayers)
        hk.reserveCapacity(config.numLayers)
        hv.reserveCapacity(config.numLayers)
        hLengths.reserveCapacity(config.numLayers)

        // Linear-attention layers keep no per-token K/V rows; they share one
        // page-sized placeholder so the parallel arrays stay non-optional.
        var linearPlaceholder: MTLBuffer?

        for layer in 0..<config.numLayers {
            let maskValue = config.fullAttentionLayerMask[layer]
            if maskValue == 2 {
                let placeholder: MTLBuffer
                if let existing = linearPlaceholder {
                    placeholder = existing
                } else {
                    guard
                        let made = device.makeBuffer(
                            length: Int(getpagesize()),
                            options: .storageModeShared)
                    else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    made.label = "kv.linear-placeholder"
                    linearPlaceholder = made
                    placeholder = made
                }
                ks.append(placeholder)
                vs.append(placeholder)
                st.append(0)
                kd.append(.linear)
                caps.append(0)
                valueByteCounts.append(0)
                hk.append(nil)
                hv.append(nil)
                hLengths.append(0)
                continue
            }
            let isFull = maskValue != 0
            // The host tier is for full attention only. An SWA layer's ring is
            // already the bounded working set that a window would otherwise
            // provide; putting it on the host would add a copy and buy nothing.
            let streamed = isFull && self.windowTokens != nil
            let fp16Stride = isFull ? fullStride : swaStride
            let elements = fp16Stride / Self.fp16Size
            let layout = Self.rowLayout(elements: elements, precision: precision)
            let stride = layout.stride
            // Linear layers start at `initialCapacityTokens` and grow on demand
            // rather than reserving `maxContext` up front. At 262144 tokens a
            // full reservation is 512 MiB per layer, 20 GiB across 40 -- lazily
            // touched, so it barely shows in RSS, but on a 24 GB machine the
            // mappings alone cost throughput: the same 25-token prompt measured
            // 13.80 s of decode at maxContext 8192 against 22.44 s at 262144.
            // Growing keeps a short conversation at a short conversation's cost
            // while leaving the advertised limit reachable.
            let capacity: Int
            if streamed, let window = self.windowTokens {
                capacity = window
            } else if ringEnabled && !isFull {
                capacity = swaCapacity
            } else {
                capacity = min(maxContext, Self.initialCapacityTokens)
            }
            let length = slots * capacity * stride

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

            if streamed {
                let arenaBytes = slots * maxContext * stride
                guard
                    let hostKPtr = Self.mapHostArena(bytes: arenaBytes),
                    let hostVPtr = Self.mapHostArena(bytes: arenaBytes)
                else {
                    throw ModelError.residentBufferWrapFailed
                }
                hk.append(hostKPtr)
                hv.append(hostVPtr)
                hLengths.append(arenaBytes)
            } else {
                hk.append(nil)
                hv.append(nil)
                hLengths.append(0)
            }

            st.append(stride)
            kd.append(isFull ? .full : .swa)
            caps.append(capacity)
            valueByteCounts.append(layout.valueBytes)
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.strides = st
        self.kinds = kd
        self.capacityTokens = caps
        self.valueBytes = valueByteCounts
        self.hostK = hk
        self.hostV = hv
        self.hostArenaLengths = hLengths
        self.initializeHostStreaming(precision: precision)
    }

    /// Maps a page-aligned anonymous arena. Pages are touched lazily: the
    /// 1M-context arena is 6.3 GB of address space at 4-bit, but only the rows
    /// a conversation wrote are ever resident.
    static func mapHostArena(bytes: Int) -> UnsafeMutableRawPointer? {
        guard bytes > 0 else { return nil }
        let page = Int(getpagesize())
        let rounded = ((bytes + page - 1) / page) * page
        let ptr = mmap(
            nil, rounded, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANON, -1, 0)
        guard let ptr, ptr != MAP_FAILED else { return nil }
        return ptr
    }

    /// Tokens each linear layer is sized for before any growth.
    ///
    /// 8192 because it measured as fast as any smaller reservation and holds an
    /// ordinary conversation without a single grow. Capacity doubles from here.
    public static let initialCapacityTokens = 8_192

    deinit {
        let page = Int(getpagesize())
        for (index, pointer) in hostK.enumerated() {
            guard let pointer, hostArenaLengths[index] > 0 else { continue }
            let rounded = ((hostArenaLengths[index] + page - 1) / page) * page
            _ = munmap(pointer, rounded)
        }
        for (index, pointer) in hostV.enumerated() {
            guard let pointer, hostArenaLengths[index] > 0 else { continue }
            let rounded = ((hostArenaLengths[index] + page - 1) / page) * page
            _ = munmap(pointer, rounded)
        }
    }

}
