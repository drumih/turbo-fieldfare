import Foundation

/// Policy comparison results from offline cache simulation.
public struct CacheSimulationReport: Codable, Sendable, Equatable {
    public struct PolicyResult: Codable, Sendable, Equatable {
        public let policy: String
        public let slotsPerLayer: Int
        public let totalAccesses: Int
        public let hits: Int
        public let misses: Int
        public let hitRatePercent: Double
    }

    public let totalTokens: Int
    public let totalAccesses: Int
    public let results: [PolicyResult]

    public init(totalTokens: Int, totalAccesses: Int, results: [PolicyResult]) {
        self.totalTokens = totalTokens
        self.totalAccesses = totalAccesses
        self.results = results
    }

    public func toFormattedText() -> String {
        var lines: [String] = []
        lines.append("Cache Policy Simulation")
        lines.append("")
        lines.append(String(format: "%-8s | %-6s | %-8s | %-8s | %-8s | %-8s", "Policy", "Slots", "Accesses", "Hits", "Misses", "Hit Rate"))
        lines.append(String(repeating: "-", count: 62))

        for r in results {
            lines.append(String(format: "%-8s | %6d | %8d | %8d | %8d | %7.2f%%",
                                r.policy, r.slotsPerLayer, r.totalAccesses, r.hits, r.misses, r.hitRatePercent))
        }

        return lines.joined(separator: "\n")
    }
}

/// Offline cache simulator evaluating LFU, LRU, and FIFO eviction policies against an ExpertAccessTrace.
public enum ExpertCacheSimulator {

    public enum EvictionPolicy: String, CaseIterable, Sendable {
        case lfu = "LFU"
        case lru = "LRU"
        case fifo = "FIFO"
    }

    public static func simulate(_ trace: ExpertAccessTrace,
                                slotCapacities: [Int] = [4, 8, 16, 32],
                                policies: [EvictionPolicy] = EvictionPolicy.allCases) -> CacheSimulationReport {
        var results: [CacheSimulationReport.PolicyResult] = []

        for capacity in slotCapacities {
            for policy in policies {
                let res = runSimulation(trace: trace, slotsPerLayer: capacity, policy: policy)
                results.append(res)
            }
        }

        return CacheSimulationReport(
            totalTokens: trace.totalTokens,
            totalAccesses: trace.events.count,
            results: results
        )
    }

    private static func runSimulation(trace: ExpertAccessTrace,
                                       slotsPerLayer: Int,
                                       policy: EvictionPolicy) -> CacheSimulationReport.PolicyResult {
        let events = trace.events
        guard !events.isEmpty else {
            return CacheSimulationReport.PolicyResult(
                policy: policy.rawValue,
                slotsPerLayer: slotsPerLayer,
                totalAccesses: 0,
                hits: 0,
                misses: 0,
                hitRatePercent: 0.0
            )
        }

        // Simulate per layer independently
        var layerSimulators: [Int: LayerCacheSim] = [:]
        var hits = 0
        var misses = 0

        for event in events {
            var sim = layerSimulators[event.layer] ?? LayerCacheSim(capacity: slotsPerLayer, policy: policy)
            let isHit = sim.access(expert: event.expert)
            if isHit {
                hits += 1
            } else {
                misses += 1
            }
            layerSimulators[event.layer] = sim
        }

        let total = hits + misses
        let hitRate = total > 0 ? (Double(hits) / Double(total)) * 100.0 : 0.0

        return CacheSimulationReport.PolicyResult(
            policy: policy.rawValue,
            slotsPerLayer: slotsPerLayer,
            totalAccesses: total,
            hits: hits,
            misses: misses,
            hitRatePercent: hitRate
        )
    }

    private struct LayerCacheSim {
        let capacity: Int
        let policy: EvictionPolicy
        private var slots: [Int] = []            // Expert IDs currently resident
        private var lastUseClock: [Int: Int] = [:]// Expert ID -> access clock
        private var useCount: [Int: Int] = [:]   // Expert ID -> access frequency
        private var fifoQueue: [Int] = []        // Insertion order of expert IDs
        private var clock: Int = 0

        init(capacity: Int, policy: EvictionPolicy) {
            self.capacity = capacity
            self.policy = policy
        }

        mutating func access(expert: Int) -> Bool {
            clock += 1
            useCount[expert, default: 0] += 1
            lastUseClock[expert] = clock

            if slots.contains(expert) {
                // Hit!
                return true
            }

            // Miss! Must insert
            if slots.count < capacity {
                slots.append(expert)
                fifoQueue.append(expert)
            } else {
                // Evict
                let evictExpert: Int
                switch policy {
                case .lru:
                    evictExpert = slots.min { (lastUseClock[$0] ?? 0) < (lastUseClock[$1] ?? 0) } ?? slots[0]
                case .lfu:
                    evictExpert = slots.min { (useCount[$0] ?? 0) < (useCount[$1] ?? 0) } ?? slots[0]
                case .fifo:
                    evictExpert = fifoQueue.first ?? slots[0]
                }

                if let idx = slots.firstIndex(of: evictExpert) {
                    slots[idx] = expert
                }
                if let fifoIdx = fifoQueue.firstIndex(of: evictExpert) {
                    fifoQueue.remove(at: fifoIdx)
                }
                fifoQueue.append(expert)
            }

            return false
        }
    }
}
