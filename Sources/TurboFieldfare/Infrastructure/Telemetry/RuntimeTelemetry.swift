import Darwin
import Foundation

/// Thread-safe runtime telemetry collector for end-to-end inference profiling.
public final class RuntimeTelemetry: @unchecked Sendable {
    public let isEnabled: Bool
    private let lock = NSLock()

    // Generation Timings
    private var prefillStartNanos: UInt64 = 0
    private var prefillEndNanos: UInt64 = 0
    private var decodeStartNanos: UInt64 = 0
    private var decodeEndNanos: UInt64 = 0
    private var ttftNanos: UInt64 = 0
    private var prefillTokensCount: Int = 0
    private var generatedTokensCount: Int = 0

    // CPU Timings
    private var tokenPrepNanos: UInt64 = 0
    private var routerHandoffNanos: UInt64 = 0
    private var schedulingNanos: UInt64 = 0
    private var cacheLookupNanos: UInt64 = 0
    private var ssdPrepNanos: UInt64 = 0
    private var samplingNanos: UInt64 = 0
    private var gpuSyncWaitNanos: UInt64 = 0

    // Expert Cache Stats
    private var cacheSlotsPerLayer: Int = 16
    private var hitsCount: Int = 0
    private var missesCount: Int = 0
    private var evictionsCount: Int = 0
    private var loadsByLayer: [Int] = [Int](repeating: 0, count: 30)
    private var loadsByExpert: [[Int]] = (0..<30).map { _ in [Int](repeating: 0, count: 128) }

    // SSD I/O Stats
    private var ssdReadCount: Int = 0
    private var ssdTotalBytes: UInt64 = 0
    private var ssdReadLatenciesNanos: [UInt64] = []
    private var currentConcurrentReads: Int = 0
    private var peakConcurrentReads: Int = 0

    // GPU / Metal Timings
    private var cb1Nanos: UInt64 = 0
    private var cb2Nanos: UInt64 = 0
    private var sharedExpertNanos: UInt64 = 0
    private var lmHeadNanos: UInt64 = 0

    // Memory Allocations
    private var commonWeightsBytes: UInt64 = 0
    private var expertCacheCapacityBytes: UInt64 = 0
    private var kvCacheBytes: UInt64 = 0
    private var prefillScratchBytes: UInt64 = 0

    public init(isEnabled: Bool = true, cacheSlotsPerLayer: Int = 16) {
        self.isEnabled = isEnabled
        self.cacheSlotsPerLayer = cacheSlotsPerLayer
    }

    @inline(__always)
    public static func currentNanos() -> UInt64 {
        clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    public func reset() {
        guard isEnabled else { return }
        lock.lock()
        defer { lock.unlock() }

        prefillStartNanos = 0
        prefillEndNanos = 0
        decodeStartNanos = 0
        decodeEndNanos = 0
        ttftNanos = 0
        prefillTokensCount = 0
        generatedTokensCount = 0

        tokenPrepNanos = 0
        routerHandoffNanos = 0
        schedulingNanos = 0
        cacheLookupNanos = 0
        ssdPrepNanos = 0
        samplingNanos = 0
        gpuSyncWaitNanos = 0

        hitsCount = 0
        missesCount = 0
        evictionsCount = 0
        loadsByLayer = [Int](repeating: 0, count: 30)
        loadsByExpert = (0..<30).map { _ in [Int](repeating: 0, count: 128) }

        ssdReadCount = 0
        ssdTotalBytes = 0
        ssdReadLatenciesNanos.removeAll(keepingCapacity: true)
        currentConcurrentReads = 0
        peakConcurrentReads = 0

        cb1Nanos = 0
        cb2Nanos = 0
        sharedExpertNanos = 0
        lmHeadNanos = 0

        commonWeightsBytes = 0
        expertCacheCapacityBytes = 0
        kvCacheBytes = 0
        prefillScratchBytes = 0
    }

    // MARK: - Recording Methods

    public func recordPrefillStart(tokens: Int) {
        guard isEnabled else { return }
        lock.lock()
        prefillStartNanos = Self.currentNanos()
        prefillTokensCount = tokens
        lock.unlock()
    }

    public func recordPrefillEnd() {
        guard isEnabled else { return }
        lock.lock()
        prefillEndNanos = Self.currentNanos()
        lock.unlock()
    }

    public func recordDecodeStart() {
        guard isEnabled else { return }
        lock.lock()
        decodeStartNanos = Self.currentNanos()
        lock.unlock()
    }

    public func recordFirstTokenGenerated() {
        guard isEnabled else { return }
        lock.lock()
        if ttftNanos == 0 {
            let now = Self.currentNanos()
            let origin = prefillStartNanos > 0 ? prefillStartNanos : decodeStartNanos
            ttftNanos = now > origin ? now - origin : 0
        }
        lock.unlock()
    }

    public func recordDecodeEnd(tokens: Int) {
        guard isEnabled else { return }
        lock.lock()
        decodeEndNanos = Self.currentNanos()
        generatedTokensCount = tokens
        lock.unlock()
    }

    public func addCPUTime(tokenPrep: UInt64 = 0,
                           routerHandoff: UInt64 = 0,
                           scheduling: UInt64 = 0,
                           cacheLookup: UInt64 = 0,
                           ssdPrep: UInt64 = 0,
                           sampling: UInt64 = 0,
                           gpuSyncWait: UInt64 = 0) {
        guard isEnabled else { return }
        lock.lock()
        tokenPrepNanos &+= tokenPrep
        routerHandoffNanos &+= routerHandoff
        schedulingNanos &+= scheduling
        cacheLookupNanos &+= cacheLookup
        ssdPrepNanos &+= ssdPrep
        samplingNanos &+= sampling
        gpuSyncWaitNanos &+= gpuSyncWait
        lock.unlock()
    }

    public func recordCacheEvent(hits: Int, misses: Int, evictions: Int) {
        guard isEnabled else { return }
        lock.lock()
        hitsCount += hits
        missesCount += misses
        evictionsCount += evictions
        lock.unlock()
    }

    public func recordExpertLoad(layer: Int, expert: Int) {
        guard isEnabled else { return }
        lock.lock()
        if layer >= 0 && layer < loadsByLayer.count {
            loadsByLayer[layer] += 1
            if expert >= 0 && expert < loadsByExpert[layer].count {
                loadsByExpert[layer][expert] += 1
            }
        }
        lock.unlock()
    }

    public func recordSSDReadStart() {
        guard isEnabled else { return }
        lock.lock()
        currentConcurrentReads += 1
        if currentConcurrentReads > peakConcurrentReads {
            peakConcurrentReads = currentConcurrentReads
        }
        lock.unlock()
    }

    public func recordSSDReadEnd(bytes: UInt64, latencyNanos: UInt64) {
        guard isEnabled else { return }
        lock.lock()
        currentConcurrentReads = max(0, currentConcurrentReads - 1)
        ssdReadCount += 1
        ssdTotalBytes += bytes
        ssdReadLatenciesNanos.append(latencyNanos)
        lock.unlock()
    }

    public func addGPUTime(cb1: UInt64 = 0, cb2: UInt64 = 0, sharedExpert: UInt64 = 0, lmHead: UInt64 = 0) {
        guard isEnabled else { return }
        lock.lock()
        cb1Nanos &+= cb1
        cb2Nanos &+= cb2
        sharedExpertNanos &+= sharedExpert
        lmHeadNanos &+= lmHead
        lock.unlock()
    }

    public func setMemoryAllocations(commonWeights: UInt64,
                                     expertCacheCapacity: UInt64,
                                     kvCache: UInt64,
                                     prefillScratch: UInt64) {
        guard isEnabled else { return }
        lock.lock()
        commonWeightsBytes = commonWeights
        expertCacheCapacityBytes = expertCacheCapacity
        kvCacheBytes = kvCache
        prefillScratchBytes = prefillScratch
        lock.unlock()
    }

    public static func queryProcessRSSBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }

    // MARK: - Report Generation

    public func generateReport(modelName: String = "Gemma 4 26B-A4B", contextLength: Int = 4096) -> ProfilingReport {
        lock.lock()
        defer { lock.unlock() }

        let prefillSec = prefillEndNanos > prefillStartNanos
            ? Double(prefillEndNanos - prefillStartNanos) / 1_000_000_000.0 : 0.0
        let decodeSec = decodeEndNanos > decodeStartNanos
            ? Double(decodeEndNanos - decodeStartNanos) / 1_000_000_000.0 : 0.0
        let totalSec = prefillSec + decodeSec

        let prefillTokSec = prefillSec > 0 ? Double(prefillTokensCount) / prefillSec : 0.0
        let decodeTokSec = decodeSec > 0 ? Double(generatedTokensCount) / decodeSec : 0.0

        let genMetrics = ProfilingReport.GenerationMetrics(
            ttftNanos: ttftNanos,
            prefillSeconds: prefillSec,
            decodeSeconds: decodeSec,
            totalSeconds: totalSec,
            prefillTokens: prefillTokensCount,
            generatedTokens: generatedTokensCount,
            prefillTokSec: prefillTokSec,
            decodeTokSec: decodeTokSec)

        let cpuMetrics = ProfilingReport.CPUMetrics(
            tokenPrepNanos: tokenPrepNanos,
            routerHandoffNanos: routerHandoffNanos,
            schedulingNanos: schedulingNanos,
            cacheLookupNanos: cacheLookupNanos,
            ssdPrepNanos: ssdPrepNanos,
            samplingNanos: samplingNanos,
            gpuSyncWaitNanos: gpuSyncWaitNanos)

        let totalLookups = hitsCount + missesCount
        let hitRatePct = totalLookups > 0 ? (Double(hitsCount) / Double(totalLookups)) * 100.0 : 0.0
        let totalLoads = loadsByLayer.reduce(0, +)

        let cacheMetrics = ProfilingReport.ExpertCacheMetrics(
            slotsPerLayer: cacheSlotsPerLayer,
            totalSlots: cacheSlotsPerLayer * 30,
            hits: hitsCount,
            misses: missesCount,
            hitRatePercent: hitRatePct,
            evictions: evictionsCount,
            totalLoads: totalLoads,
            loadsByLayer: loadsByLayer,
            loadsByExpert: loadsByExpert)

        let minLatency = ssdReadLatenciesNanos.min() ?? 0
        let maxLatency = ssdReadLatenciesNanos.max() ?? 0
        let sumLatency = ssdReadLatenciesNanos.reduce(0, +)
        let avgLatency = ssdReadCount > 0 ? sumLatency / UInt64(ssdReadCount) : 0

        let effectiveIOTimeSec = Double(sumLatency) / 1_000_000_000.0
        let effectiveThroughput = effectiveIOTimeSec > 0
            ? (Double(ssdTotalBytes) / (1024.0 * 1024.0)) / effectiveIOTimeSec : 0.0

        let totalTokensForIO = max(1, generatedTokensCount)
        let readsPerToken = Double(ssdReadCount) / Double(totalTokensForIO)
        let bytesPerToken = Double(ssdTotalBytes) / Double(totalTokensForIO)

        let ssdMetrics = ProfilingReport.SSDMetrics(
            totalReads: ssdReadCount,
            totalBytes: ssdTotalBytes,
            readsPerToken: readsPerToken,
            bytesPerToken: bytesPerToken,
            avgReadLatencyNanos: avgLatency,
            minReadLatencyNanos: minLatency,
            maxReadLatencyNanos: maxLatency,
            totalIOSeconds: effectiveIOTimeSec,
            concurrentReadsPeak: peakConcurrentReads,
            effectiveThroughputMBs: effectiveThroughput)

        let gpuMetrics = ProfilingReport.GPUMetrics(
            cb1ExecutionNanos: cb1Nanos,
            cb2ExecutionNanos: cb2Nanos,
            sharedExpertNanos: sharedExpertNanos,
            lmHeadNanos: lmHeadNanos,
            note: "CB1 (Norm/Attn/Router) & CB2 (MoE/Combine) captured via GPU completion callbacks")

        let processRSS = Self.queryProcessRSSBytes()
        let totalMetalAllocations = commonWeightsBytes + expertCacheCapacityBytes + kvCacheBytes + prefillScratchBytes

        let memMetrics = ProfilingReport.MemoryMetrics(
            processResidentBytes: processRSS,
            commonWeightsBytes: commonWeightsBytes,
            expertCacheCapacityBytes: expertCacheCapacityBytes,
            kvCacheBytes: kvCacheBytes,
            prefillScratchBytes: prefillScratchBytes,
            totalMetalAllocationsBytes: totalMetalAllocations)

        return ProfilingReport(
            modelName: modelName,
            contextLength: contextLength,
            generation: genMetrics,
            cpu: cpuMetrics,
            ssd: ssdMetrics,
            expertCache: cacheMetrics,
            gpu: gpuMetrics,
            memory: memMetrics)
    }
}
