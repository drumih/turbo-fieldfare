import Foundation

/// Machine-readable representation of end-to-end runtime profiling metrics.
public struct ProfilingReport: Codable, Sendable, Equatable {
    public struct GenerationMetrics: Codable, Sendable, Equatable {
        public let ttftNanos: UInt64
        public let prefillSeconds: Double
        public let decodeSeconds: Double
        public let totalSeconds: Double
        public let prefillTokens: Int
        public let generatedTokens: Int
        public let prefillTokSec: Double
        public let decodeTokSec: Double

        public init(ttftNanos: UInt64,
                    prefillSeconds: Double,
                    decodeSeconds: Double,
                    totalSeconds: Double,
                    prefillTokens: Int,
                    generatedTokens: Int,
                    prefillTokSec: Double,
                    decodeTokSec: Double) {
            self.ttftNanos = ttftNanos
            self.prefillSeconds = prefillSeconds
            self.decodeSeconds = decodeSeconds
            self.totalSeconds = totalSeconds
            self.prefillTokens = prefillTokens
            self.generatedTokens = generatedTokens
            self.prefillTokSec = prefillTokSec
            self.decodeTokSec = decodeTokSec
        }
    }

    public struct CPUMetrics: Codable, Sendable, Equatable {
        public let tokenPrepNanos: UInt64
        public let routerHandoffNanos: UInt64
        public let schedulingNanos: UInt64
        public let cacheLookupNanos: UInt64
        public let ssdPrepNanos: UInt64
        public let samplingNanos: UInt64
        public let gpuSyncWaitNanos: UInt64

        public init(tokenPrepNanos: UInt64,
                    routerHandoffNanos: UInt64,
                    schedulingNanos: UInt64,
                    cacheLookupNanos: UInt64,
                    ssdPrepNanos: UInt64,
                    samplingNanos: UInt64,
                    gpuSyncWaitNanos: UInt64) {
            self.tokenPrepNanos = tokenPrepNanos
            self.routerHandoffNanos = routerHandoffNanos
            self.schedulingNanos = schedulingNanos
            self.cacheLookupNanos = cacheLookupNanos
            self.ssdPrepNanos = ssdPrepNanos
            self.samplingNanos = samplingNanos
            self.gpuSyncWaitNanos = gpuSyncWaitNanos
        }
    }

    public struct SSDMetrics: Codable, Sendable, Equatable {
        public let totalReads: Int
        public let totalBytes: UInt64
        public let readsPerToken: Double
        public let bytesPerToken: Double
        public let avgReadLatencyNanos: UInt64
        public let minReadLatencyNanos: UInt64
        public let maxReadLatencyNanos: UInt64
        public let totalIOSeconds: Double
        public let concurrentReadsPeak: Int
        public let effectiveThroughputMBs: Double

        public init(totalReads: Int,
                    totalBytes: UInt64,
                    readsPerToken: Double,
                    bytesPerToken: Double,
                    avgReadLatencyNanos: UInt64,
                    minReadLatencyNanos: UInt64,
                    maxReadLatencyNanos: UInt64,
                    totalIOSeconds: Double,
                    concurrentReadsPeak: Int,
                    effectiveThroughputMBs: Double) {
            self.totalReads = totalReads
            self.totalBytes = totalBytes
            self.readsPerToken = readsPerToken
            self.bytesPerToken = bytesPerToken
            self.avgReadLatencyNanos = avgReadLatencyNanos
            self.minReadLatencyNanos = minReadLatencyNanos
            self.maxReadLatencyNanos = maxReadLatencyNanos
            self.totalIOSeconds = totalIOSeconds
            self.concurrentReadsPeak = concurrentReadsPeak
            self.effectiveThroughputMBs = effectiveThroughputMBs
        }
    }

    public struct ExpertCacheMetrics: Codable, Sendable, Equatable {
        public let slotsPerLayer: Int
        public let totalSlots: Int
        public let hits: Int
        public let misses: Int
        public let hitRatePercent: Double
        public let evictions: Int
        public let totalLoads: Int
        public let loadsByLayer: [Int]
        public let loadsByExpert: [[Int]]

        public init(slotsPerLayer: Int,
                    totalSlots: Int,
                    hits: Int,
                    misses: Int,
                    hitRatePercent: Double,
                    evictions: Int,
                    totalLoads: Int,
                    loadsByLayer: [Int],
                    loadsByExpert: [[Int]]) {
            self.slotsPerLayer = slotsPerLayer
            self.totalSlots = totalSlots
            self.hits = hits
            self.misses = misses
            self.hitRatePercent = hitRatePercent
            self.evictions = evictions
            self.totalLoads = totalLoads
            self.loadsByLayer = loadsByLayer
            self.loadsByExpert = loadsByExpert
        }
    }

    public struct GPUMetrics: Codable, Sendable, Equatable {
        public let cb1ExecutionNanos: UInt64
        public let cb2ExecutionNanos: UInt64
        public let sharedExpertNanos: UInt64
        public let lmHeadNanos: UInt64
        public let note: String

        public init(cb1ExecutionNanos: UInt64,
                    cb2ExecutionNanos: UInt64,
                    sharedExpertNanos: UInt64,
                    lmHeadNanos: UInt64,
                    note: String) {
            self.cb1ExecutionNanos = cb1ExecutionNanos
            self.cb2ExecutionNanos = cb2ExecutionNanos
            self.sharedExpertNanos = sharedExpertNanos
            self.lmHeadNanos = lmHeadNanos
            self.note = note
        }
    }

    public struct MemoryMetrics: Codable, Sendable, Equatable {
        public let processResidentBytes: UInt64
        public let commonWeightsBytes: UInt64
        public let expertCacheCapacityBytes: UInt64
        public let kvCacheBytes: UInt64
        public let prefillScratchBytes: UInt64
        public let totalMetalAllocationsBytes: UInt64

        public init(processResidentBytes: UInt64,
                    commonWeightsBytes: UInt64,
                    expertCacheCapacityBytes: UInt64,
                    kvCacheBytes: UInt64,
                    prefillScratchBytes: UInt64,
                    totalMetalAllocationsBytes: UInt64) {
            self.processResidentBytes = processResidentBytes
            self.commonWeightsBytes = commonWeightsBytes
            self.expertCacheCapacityBytes = expertCacheCapacityBytes
            self.kvCacheBytes = kvCacheBytes
            self.prefillScratchBytes = prefillScratchBytes
            self.totalMetalAllocationsBytes = totalMetalAllocationsBytes
        }
    }

    public let modelName: String
    public let contextLength: Int
    public let generation: GenerationMetrics
    public let cpu: CPUMetrics
    public let ssd: SSDMetrics
    public let expertCache: ExpertCacheMetrics
    public let gpu: GPUMetrics
    public let memory: MemoryMetrics

    public init(modelName: String,
                contextLength: Int,
                generation: GenerationMetrics,
                cpu: CPUMetrics,
                ssd: SSDMetrics,
                expertCache: ExpertCacheMetrics,
                gpu: GPUMetrics,
                memory: MemoryMetrics) {
        self.modelName = modelName
        self.contextLength = contextLength
        self.generation = generation
        self.cpu = cpu
        self.ssd = ssd
        self.expertCache = expertCache
        self.gpu = gpu
        self.memory = memory
    }

    public func toJSONString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    public func toFormattedText() -> String {
        func formatNanosMs(_ nanos: UInt64) -> String {
            String(format: "%.2f ms", Double(nanos) / 1_000_000.0)
        }
        func formatMB(_ bytes: UInt64) -> String {
            String(format: "%.2f MB", Double(bytes) / (1024.0 * 1024.0))
        }

        return """
        TurboFieldfare Runtime Profile

        Model: \(modelName)
        Context: \(contextLength)

        Generation
          TTFT:              \(formatNanosMs(generation.ttftNanos))
          Prefill:           \(String(format: "%.1f", generation.prefillTokSec)) tok/s
          Decode:            \(String(format: "%.1f", generation.decodeTokSec)) tok/s
          Generated tokens:  \(generation.generatedTokens)

        CPU
          Token prep:        \(formatNanosMs(cpu.tokenPrepNanos))
          Scheduling:        \(formatNanosMs(cpu.schedulingNanos))
          Cache lookup:      \(formatNanosMs(cpu.cacheLookupNanos))
          Router handoff:    \(formatNanosMs(cpu.routerHandoffNanos))
          SSD prep:          \(formatNanosMs(cpu.ssdPrepNanos))
          Sampling:          \(formatNanosMs(cpu.samplingNanos))
          GPU Sync Wait:     \(formatNanosMs(cpu.gpuSyncWaitNanos))

        Expert Cache
          Capacity:          \(expertCache.slotsPerLayer) slots/layer (\(expertCache.totalSlots) total)
          Hits:              \(expertCache.hits)
          Misses:            \(expertCache.misses)
          Hit rate:          \(String(format: "%.2f", expertCache.hitRatePercent))%
          Evictions:         \(expertCache.evictions)
          Total loads:       \(expertCache.totalLoads)

        SSD
          Reads:             \(ssd.totalReads)
          Bytes:             \(formatMB(ssd.totalBytes))
          Reads/token:       \(String(format: "%.2f", ssd.readsPerToken))
          Bytes/token:       \(formatMB(UInt64(ssd.bytesPerToken)))
          Avg read latency:  \(formatNanosMs(ssd.avgReadLatencyNanos))
          Min read latency:  \(formatNanosMs(ssd.minReadLatencyNanos))
          Max read latency:  \(formatNanosMs(ssd.maxReadLatencyNanos))
          Total I/O time:    \(String(format: "%.3f s", ssd.totalIOSeconds))
          Peak concurrency:  \(ssd.concurrentReadsPeak)
          Throughput:        \(String(format: "%.2f MB/s", ssd.effectiveThroughputMBs))

        GPU (Command Buffer Intervals)
          CB1 (Norm/Attn/Router): \(formatNanosMs(gpu.cb1ExecutionNanos))
          CB2 (MoE/Combine/Tail): \(formatNanosMs(gpu.cb2ExecutionNanos))
          Shared Expert FFN:      \(formatNanosMs(gpu.sharedExpertNanos))
          LM Head:                \(formatNanosMs(gpu.lmHeadNanos))
          Note:                   \(gpu.note)

        Memory
          Process RSS:       \(formatMB(memory.processResidentBytes))
          Common weights:    \(formatMB(memory.commonWeightsBytes))
          Expert cache cap:  \(formatMB(memory.expertCacheCapacityBytes))
          KV cache:          \(formatMB(memory.kvCacheBytes))
          Prefill scratch:   \(formatMB(memory.prefillScratchBytes))
          Total Metal alloc: \(formatMB(memory.totalMetalAllocationsBytes))
        """
    }
}
