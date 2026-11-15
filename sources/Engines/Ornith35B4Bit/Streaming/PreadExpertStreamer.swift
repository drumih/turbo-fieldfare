import Darwin
import Foundation
import TinyTitanFormat
import Metal
import Synchronization
import TinyTitanCore

/// SSD-backed routed-expert streamer with a fixed per-layer slot cache.
/// unchecked-invariant: the expert cache bookkeeping is guarded by `cacheLock`,
/// which is what lets `DispatchQueue.concurrentPerform` fan the misses out
/// across threads. Slot state is published only after all direct reads finish,
/// so concurrent planners never treat partial bytes as resident.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    /// Slot allocations eligible for wiring: one region for the pooled
    /// layout, one per slot otherwise. Recorded at construction; wiring
    /// itself is deferred to `setSlotsPinned`.
    var wireRegions: [(pointer: UnsafeMutableRawPointer, bytes: Int)] = []
    var slotsPinned = false

    /// Wire or release the slot memory. Decode wants it wired; prefill wants
    /// it released.
    ///
    /// Residency only matters during decode, where a reclaimed page costs a
    /// routed-expert SSD read on the critical path. Prefill streams experts
    /// in bulk regardless and needs the headroom: with the cache wired for
    /// the whole session, ANE prefill measured 244.52 s against 175.68 s
    /// released (8-bit, same GPU arm), because Core ML could not place its
    /// arenas. Wiring at the handover instead of at allocation gives decode
    /// its protection without taxing prefill.
    ///
    /// Best-effort in both directions: a refused `mlock` (the wire limit is
    /// finite) must degrade to unpinned behaviour, never fail a request.
    /// `TINYTITAN_NO_PIN=1` disables wiring entirely.
    /// TINYTITAN_NO_PIN=1 leaves the slot cache unwired (measured: decode falls
    /// to 1.8 tok/s on Qwen3.8 4-bit as the budget is reclaimed). Read once:
    /// this is called once per layer per token, and a per-call
    /// `ProcessInfo.environment` rebuild is the same regression the decode
    /// flags had ([[env-reads]]).
    static let pinningDisabled = ProcessInfo.processInfo.environment["TINYTITAN_NO_PIN"] != nil

    /// Whether the last wire attempt covered every region, so a caller can
    /// skip the per-layer walk once the whole cache is wired.
    var isPinned: Bool { slotsPinned }

    func setSlotsPinned(_ wanted: Bool) {
        guard !Self.pinningDisabled else { return }
        guard wanted != slotsPinned else { return }
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var achieved = 0
        var bytes = 0
        for region in wireRegions {
            let rc =
                wanted
                ? mlock(region.pointer, region.bytes)
                : munlock(region.pointer, region.bytes)
            if rc == 0 {
                achieved += 1
                bytes += region.bytes
            } else if Self.wireTraceEnabled {
                FileHandle.standardError.write(
                    Data(
                        "[wire] \(wanted ? "mlock" : "munlock") failed errno=\(errno) bytes=\(region.bytes)\n"
                            .utf8))
            }
        }
        // Treat a partial wire as unpinned so the next call retries rather
        // than believing a half-applied state.
        slotsPinned = wanted && achieved == wireRegions.count
        if Self.wireTraceEnabled {
            let ms = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started) / 1e6
            let verb = wanted ? "mlock" : "munlock"
            FileHandle.standardError.write(
                Data(
                    "[wire] \(verb) \(achieved)/\(wireRegions.count) regions, "
                        .appending("\(bytes / (1024 * 1024)) MiB in ")
                        .appending(String(format: "%.1f", ms))
                        .appending(" ms\n").utf8))
        }
    }

    /// Times the wire/unwire calls (`TINYTITAN_WIRE_TRACE=1`).
    ///
    /// `mlock` on a cache whose pages the OS reclaimed during prefill has to
    /// fault them back before it returns, and the call sits on the critical
    /// path of the first decode token. Whether that is where the post-handover
    /// decode cost actually lives is the question this answers; fixing it
    /// without measuring it first would be guessing.
    static let wireTraceEnabled =
        ProcessInfo.processInfo.environment["TINYTITAN_WIRE_TRACE"] == "1"

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy
    /// Ceiling on the LFU use count used as an eviction *price*; nil is the
    /// shipped uncapped policy. See `expertUseCountCapDefault`.
    public let expertUseCountCap: Int?
    public let ioBackend: ExpertIOBackend
    public let poolSlotStride: Int

    let fd: Int32

    /// Bounded-footprint reader. The default; `.page-cache` (T25's automatic
    /// rule or `TINYTITAN_BOUNDED_IO=0`) opts out.
    ///
    /// Opens its own F_NOCACHE descriptors so expert reads never enter the unified
    /// buffer cache. That makes the slot budget the machine's true footprint,
    /// which is the whole point of streaming a 35B model on 24 GB.
    ///
    /// It is not free. Measured against the page-cache path it costs 15-30% of
    /// decode throughput, because every miss becomes a real device read instead of
    /// a cache hit -- and the cost is worst exactly where the hit rate is lowest
    /// (-40% at the 8-slot floor against -18% at 16 slots).
    ///
    /// It is the default anyway. The page-cache path is faster only by borrowing
    /// memory it never declares: process RSS looks smaller while the OS holds the
    /// difference, so "a 35B model in 1 GB" stops being true. A footprint you can
    /// account for is the product; throughput is what is being traded for it.
    let boundedReader: ParallelExpertReader?
    /// T25's machine-size decision. `.bounded` is the default for a directly
    /// constructed streamer; `Model` passes the resolved one.
    let pageCachePolicy: PageCachePolicy
    /// The prefetch ring's own reader. A reader serialises batches (its batch
    /// lock waits for the current one), so a speculative read issued through the
    /// demand reader blocks the next layer's demand read. This one is only used
    /// by `beginPrefetch`.
    let prefetchReader: ParallelExpertReader?
    /// GPU decode resources when the layer is rANS-coded.
    let ransResources: RANSLayerResources?
    /// Serialises the demand and speculative decodes that share the staging
    /// buffer and descriptors.
    let ransStagingLock = NSLock()
    /// The tail-phase reader. Fronts and tails of one wave load at the same
    /// time on different readers, because a reader serialises its own batches;
    /// this is what lets the front decode overlap the tail read.
    let tailReader: ParallelExpertReader?
    let metalReader: MetalExpertReader?
    let eventCoordinator: ExpertIOEventCoordinator?
    let metalStagingPool: MetalExpertStagingPool?
    let metalIOService: MetalExpertIOService?
    let slotPointers: [UnsafeMutableRawPointer]
    /// Per-expert compressed frames when the layer is stored compressed; `nil`
    /// means raw blobs at `n * expertStride`. A split layer carries two frames
    /// per expert, one per half, so the phase-1 overlap survives compression.
    let compressedFrames: [[ExpertBlobIndex.Blob]]?
    /// Decoded bytes per block in a v3 index; 0 for a legacy whole-phase index.
    let compressedBlockBytes: UInt64
    /// The largest packed form of one expert — all of its frames — which is
    /// how large a staging buffer has to be. The buffers themselves come from
    /// `CompressedScratchPool`, one per concurrent decode, so a plan's misses
    /// decode in parallel instead of queueing behind one lock.
    let compressedScratchBytes: Int
    let slotBuffers: [MTLBuffer]
    let slotBufferOffsets: [UInt64]
    let residencyTable: MTLBuffer

    var nextSlot = 0

    enum SlotState: UInt8 {
        case empty
        case loading
        case resident
    }

    var slotExpert: [Int]
    var slotLastUse: [Int]
    var slotState: [SlotState]
    var slotGeneration: [UInt64]
    var slotPinCount: [Int]
    var expertUseCount: [Int]
    /// The persisted prior this layer was opened with, restored at the
    /// prefill-to-decode reset so history outlives one request. `nil` keeps the
    /// original behaviour: every reset zeroes the score.
    var seededExpertUseCounts: [Int]?
    var expertLoadCount: [Int]
    var useClock = 0
    var statisticsPlans: UInt64 = 0
    var statisticsRequestedExperts: UInt64 = 0
    var statisticsHits: UInt64 = 0
    var statisticsMisses: UInt64 = 0
    var statisticsBytesRead: UInt64 = 0
    var statisticsReadOperations: UInt64 = 0
    var statisticsEvictions: UInt64 = 0
    var statisticsReloads: UInt64 = 0
    var statisticsLoadBatches: UInt64 = 0
    var statisticsTotalLoadNanos: UInt64 = 0
    var statisticsMaximumLoadNanos: UInt64 = 0
    var statisticsLatencyHistogram = [UInt64](repeating: 0, count: 17)
    var statisticsPeakLoadingSlots = 0
    let cacheLock = NSLock()

    public init(
        layout: StreamLayout,
        device: MTLDevice,
        slotCount: Int,
        cachePolicy: ExpertCachePolicy = .lfu,
        expertUseCountCap: Int? = PreadExpertStreamer.expertUseCountCapDefault,
        seedUseCounts: [Int]? = nil,
        eventCoordinator: ExpertIOEventCoordinator? = nil,
        metalStagingPool: MetalExpertStagingPool? = nil,
        metalIOService: MetalExpertIOService? = nil,
        pageCachePolicy: PageCachePolicy = .bounded
    ) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        self.seededExpertUseCounts =
            seedUseCounts?.count == layout.expertsPerLayer ? seedUseCounts : nil
        self.compressedFrames = layout.compressedFrames
        self.compressedBlockBytes = layout.compressedBlockBytes
        if let rans = layout.rans {
            self.ransResources = try RANSLayerResources(
                layout: rans, phases: layout.ransPhases, device: device,
                payloadPath: layout.path)
        } else {
            self.ransResources = nil
        }
        self.cachePolicy = cachePolicy
        self.expertUseCountCap = expertUseCountCap
        self.eventCoordinator = eventCoordinator
        self.metalStagingPool = metalStagingPool
        self.metalIOService = metalIOService
        self.pageCachePolicy = pageCachePolicy
        self.ioBackend = try ExpertIOBackend.environmentValue()
        if let frames = layout.compressedFrames {
            guard frames.count == layout.expertsPerLayer else {
                throw StreamerError.sizeMismatch(
                    expected: UInt64(layout.expertsPerLayer),
                    actual: UInt64(frames.count))
            }
            let largest =
                frames.map { entry in
                    entry.reduce(0) { $0 + Int($1.length) }
                }.max() ?? 0
            guard largest > 0 else {
                throw StreamerError.sizeMismatch(expected: 1, actual: 0)
            }
            self.compressedScratchBytes = largest
        } else {
            self.compressedScratchBytes = 0
        }
        let pageSize = Int(getpagesize())

        let openedFD = open(layout.path, O_RDONLY)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD

        var fileStats = stat()
        // K9: fstat failure must not silently skip size validation — a
        // truncated file would then be read out of bounds by pread.
        guard fstat(openedFD, &fileStats) == 0 else {
            let statErrno = errno
            close(openedFD)
            throw ModelError.posixFailed(call: "fstat(\(layout.path))", errno: statErrno)
        }
        // Checked: `streamOffset` and `streamSize` come from the install's own
        // layout, so a corrupt one must not wrap into a small `required` and pass
        // the size check below -- every expert offset is later computed as
        // `streamOffset + regionOffset` and would then point outside the file.
        let (required, streamRangeOverflow) = layout.streamOffset
            .addingReportingOverflow(layout.streamSize)
        guard !streamRangeOverflow else {
            close(openedFD)
            throw StreamerError.offsetOutOfRange(layout.streamOffset)
        }
        if let phase1 = layout.phase1Bytes, !(0 < phase1 && phase1 < layout.expertStride) {
            close(openedFD)
            throw StreamerError.offsetOutOfRange(phase1)
        }
        if UInt64(fileStats.st_size) < required {
            close(openedFD)
            throw StreamerError.sizeMismatch(
                expected: required,
                actual: UInt64(fileStats.st_size))
        }

        // `Int(...)` traps for a stride above `Int.max`, and the sum in the
        // rounding would trap again near it. Everything upstream bounds the
        // *product* `expertsPerLayer * expertStride` (C69) and pins each expert's
        // range inside a file whose size is verified -- but the stride itself is
        // the only thing that bounds this allocation, so it is converted exactly
        // (which reports) rather than trapping on a value that came out of a
        // layout file.
        guard let stride = Int(exactly: layout.expertStride), stride > 0,
            stride <= Int.max - (pageSize - 1)
        else {
            close(openedFD)
            throw StreamerError.offsetOutOfRange(layout.expertStride)
        }
        let allocationSize = ((stride + pageSize - 1) / pageSize) * pageSize
        // The pool base retains the validated 2 MiB allocation alignment.
        // Individual offsets need only VM-page alignment for pread and Metal;
        // rounding every slot to 2 MiB inflated the 8-bit pool by several GiB.
        self.poolSlotStride = allocationSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        var bufferOffsets: [UInt64] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)
        bufferOffsets.reserveCapacity(slotCount)
        guard
            let residencyTable = device.makeBuffer(
                length: max(1, layout.expertsPerLayer)
                    * MemoryLayout<ExpertResidencyEntry>.stride,
                options: .storageModeShared)
        else {
            close(openedFD)
            throw StreamerError.bufferWrapFailed
        }
        self.residencyTable = residencyTable
        let residencyEntries = residencyTable.contents()
            .bindMemory(
                to: ExpertResidencyEntry.self,
                capacity: max(1, layout.expertsPerLayer))
        for expert in 0..<max(1, layout.expertsPerLayer) {
            residencyEntries[expert] = ExpertResidencyEntry()
        }

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
            close(openedFD)
        }

        for _ in 0..<slotCount {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            wireRegions.append((pointer, allocationSize))
            guard
                let buffer = device.makeBuffer(
                    bytesNoCopy: pointer,
                    length: allocationSize,
                    options: .storageModeShared,
                    deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
            bufferOffsets.append(0)
        }
        // Fail closed when bounded I/O was requested. Falling through to an
        // ordinary descriptor would silently create an unbounded second cache
        // in the macOS page cache and invalidate the declared RAM budget.
        if ioBackend == .metal {
            do {
                if let metalIOService {
                    self.metalReader = try MetalExpertReader(
                        path: layout.path, device: device, service: metalIOService)
                } else {
                    // Direct construction remains useful for focused tests;
                    // Model opens pass the one shared service above.
                    self.metalReader = try MetalExpertReader(
                        path: layout.path, device: device, maximumCommandsInFlight: 4)
                }
            } catch {
                unwind()
                throw error
            }
            self.boundedReader = nil
            self.prefetchReader = nil
            self.tailReader = nil
        } else if pageCachePolicy == .bounded {
            self.metalReader = nil
            do {
                // A compressed layer's wave is 2-5 ranges of 0.6-1.6 MB, so
                // four threads leave the disk under-driven: measured 1.7-2.2
                // GB/s against the raw path's 3.4. Eight threads put every
                // range of a wave in flight at once.
                self.boundedReader = try ParallelExpertReader(
                    path: layout.path,
                    expertStride: Int(layout.expertStride),
                    threads: layout.compressedFrames == nil ? 4 : 8,
                    bypassCache: true)
                self.prefetchReader =
                    layout.compressedFrames == nil
                    ? nil
                    : try? ParallelExpertReader(
                        path: layout.path,
                        expertStride: Int(layout.expertStride),
                        threads: 4,
                        bypassCache: true)
                self.tailReader =
                    layout.compressedFrames == nil
                    ? nil
                    : try? ParallelExpertReader(
                        path: layout.path,
                        expertStride: Int(layout.expertStride),
                        threads: 4,
                        bypassCache: true)
            } catch {
                unwind()
                throw error
            }
        } else {
            self.metalReader = nil
            self.boundedReader = nil
            self.prefetchReader = nil
            self.tailReader = nil
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotBufferOffsets = bufferOffsets
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.slotState = [SlotState](repeating: .empty, count: slotCount)
        self.slotGeneration = [UInt64](repeating: 0, count: slotCount)
        self.slotPinCount = [Int](repeating: 0, count: slotCount)
        self.expertUseCount =
            seededExpertUseCounts
            ?? [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        self.expertLoadCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
    }

    deinit {
        close(fd)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)]
    {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(
        experts: [Int],
        layer: Int = 0,
        avoidingSlots: Set<Int> = [],
        prefetched: [Int: UnsafeMutableRawPointer] = [:]
    ) throws
        -> ExpertCachePlan
    {
        guard
            let plan = makeExpertCachePlan(
                layer: layer,
                experts: experts,
                avoidingSlots: avoidingSlots,
                prefetched: prefetched)
        else {
            // K10: config-triggered placement failure (too few slots for the
            // requested expert set) is recoverable — throw instead of
            // crashing; the runner already handles thrown errors.
            throw ModelError.expertCacheUnplaceable(
                detail:
                    "\(experts.count) experts do not fit in \(slotCount) cache slots (policy \(cachePolicy.rawValue), avoiding \(avoidingSlots.count) slots)"
            )
        }
        return plan
    }

    public func planExpertsCachedIfPossible(
        experts: [Int],
        layer: Int = 0,
        avoidingSlots: Set<Int> = [],
        prefetched: [Int: UnsafeMutableRawPointer] = [:]
    )
        -> ExpertCachePlan?
    {
        makeExpertCachePlan(
            layer: layer, experts: experts, avoidingSlots: avoidingSlots,
            prefetched: prefetched)
    }

}
