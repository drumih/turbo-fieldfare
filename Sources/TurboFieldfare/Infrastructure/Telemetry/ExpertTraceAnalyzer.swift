import Foundation

/// Comprehensive analysis results calculated from an ExpertAccessTrace.
public struct ExpertTraceAnalysisReport: Codable, Sendable, Equatable {
    public struct ExpertPopularity: Codable, Sendable, Equatable {
        public let expertID: Int
        public let accessCount: Int
        public let hitCount: Int
        public let missCount: Int
        public let hitRatePercent: Double
        public let ssdReadCount: Int
    }

    public struct LayerLocality: Codable, Sendable, Equatable {
        public let layer: Int
        public let totalAccesses: Int
        public let hits: Int
        public let misses: Int
        public let hitRatePercent: Double
        public let uniqueExpertsCount: Int
    }

    public struct ReuseDistanceStats: Codable, Sendable, Equatable {
        public let averageDistance: Double
        public let minDistance: Int
        public let maxDistance: Int
        public let medianDistance: Double
        public let p90Distance: Double
    }

    public struct RouterScoreStats: Codable, Sendable, Equatable {
        public let meanScore: Float
        public let minScore: Float
        public let maxScore: Float
        public let count: Int
    }

    public let totalTokens: Int
    public let totalAccesses: Int
    public let totalHits: Int
    public let totalMisses: Int
    public let overallHitRatePercent: Double
    public let totalSSDReads: Int
    public let totalSSDBytesRead: UInt64
    public let topExperts: [ExpertPopularity]
    public let perLayerLocality: [LayerLocality]
    public let reuseDistance: ReuseDistanceStats
    public let interArrivalTokenDistance: ReuseDistanceStats
    public let crossTokenReuseRatio: Double
    public let routerScoreStats: RouterScoreStats?

    public init(totalTokens: Int,
                totalAccesses: Int,
                totalHits: Int,
                totalMisses: Int,
                overallHitRatePercent: Double,
                totalSSDReads: Int,
                totalSSDBytesRead: UInt64,
                topExperts: [ExpertPopularity],
                perLayerLocality: [LayerLocality],
                reuseDistance: ReuseDistanceStats,
                interArrivalTokenDistance: ReuseDistanceStats,
                crossTokenReuseRatio: Double,
                routerScoreStats: RouterScoreStats?) {
        self.totalTokens = totalTokens
        self.totalAccesses = totalAccesses
        self.totalHits = totalHits
        self.totalMisses = totalMisses
        self.overallHitRatePercent = overallHitRatePercent
        self.totalSSDReads = totalSSDReads
        self.totalSSDBytesRead = totalSSDBytesRead
        self.topExperts = topExperts
        self.perLayerLocality = perLayerLocality
        self.reuseDistance = reuseDistance
        self.interArrivalTokenDistance = interArrivalTokenDistance
        self.crossTokenReuseRatio = crossTokenReuseRatio
        self.routerScoreStats = routerScoreStats
    }

    public func toFormattedText() -> String {
        func formatMB(_ bytes: UInt64) -> String {
            String(format: "%.2f MB", Double(bytes) / (1024.0 * 1024.0))
        }

        var lines: [String] = []
        lines.append("Expert Access Analysis")
        lines.append("")
        lines.append("Tokens:               \(totalTokens)")
        lines.append("Expert Accesses:      \(totalAccesses)")
        lines.append("Hits:                 \(totalHits)")
        lines.append("Misses:               \(totalMisses)")
        lines.append(String(format: "Overall Hit Rate:     %.2f%%", overallHitRatePercent))
        lines.append("SSD Reads:            \(totalSSDReads) (\(formatMB(totalSSDBytesRead)))")
        lines.append(String(format: "Cross-Token Reuse:    %.2f%%", crossTokenReuseRatio * 100.0))
        lines.append("")
        lines.append("Reuse Distance (Accesses):")
        lines.append(String(format: "  Average:            %.2f", reuseDistance.averageDistance))
        lines.append("  Min / Max:          \(reuseDistance.minDistance) / \(reuseDistance.maxDistance)")
        lines.append(String(format: "  Median / P90:       %.1f / %.1f", reuseDistance.medianDistance, reuseDistance.p90Distance))
        lines.append("")
        lines.append("Inter-Arrival Distance (Tokens):")
        lines.append(String(format: "  Average:            %.2f", interArrivalTokenDistance.averageDistance))
        lines.append("  Min / Max:          \(interArrivalTokenDistance.minDistance) / \(interArrivalTokenDistance.maxDistance)")
        lines.append(String(format: "  Median / P90:       %.1f / %.1f", interArrivalTokenDistance.medianDistance, interArrivalTokenDistance.p90Distance))

        if let scores = routerScoreStats {
            lines.append("")
            lines.append("Router Score Stats:")
            lines.append(String(format: "  Mean Score:         %.4f", scores.meanScore))
            lines.append(String(format: "  Min / Max Score:    %.4f / %.4f", scores.minScore, scores.maxScore))
        }

        lines.append("")
        lines.append("Top Experts (Most Accessed):")
        for (rank, exp) in topExperts.prefix(10).enumerated() {
            lines.append(String(format: "  %2d. Expert %3d: %5d accesses, %5d hits, %5d misses (%.1f%% hit rate, %d SSD reads)",
                                rank + 1, exp.expertID, exp.accessCount, exp.hitCount, exp.missCount, exp.hitRatePercent, exp.ssdReadCount))
        }

        lines.append("")
        lines.append("Per-Layer Locality:")
        for loc in perLayerLocality {
            lines.append(String(format: "  Layer %2d: %5d accesses, %2d unique experts, %5d hits (%.1f%% hit rate)",
                                loc.layer, loc.totalAccesses, loc.uniqueExpertsCount, loc.hits, loc.hitRatePercent))
        }

        return lines.joined(separator: "\n")
    }
}

/// Offline analytical engine for expert access traces.
public enum ExpertTraceAnalyzer {

    public static func analyze(_ trace: ExpertAccessTrace) -> ExpertTraceAnalysisReport {
        let events = trace.events
        let totalAccesses = events.count
        guard totalAccesses > 0 else {
            return ExpertTraceAnalysisReport(
                totalTokens: trace.totalTokens,
                totalAccesses: 0,
                totalHits: 0,
                totalMisses: 0,
                overallHitRatePercent: 0.0,
                totalSSDReads: 0,
                totalSSDBytesRead: 0,
                topExperts: [],
                perLayerLocality: [],
                reuseDistance: .init(averageDistance: 0, minDistance: 0, maxDistance: 0, medianDistance: 0, p90Distance: 0),
                interArrivalTokenDistance: .init(averageDistance: 0, minDistance: 0, maxDistance: 0, medianDistance: 0, p90Distance: 0),
                crossTokenReuseRatio: 0.0,
                routerScoreStats: nil
            )
        }

        let hits = events.filter(\.hit).count
        let misses = totalAccesses - hits
        let hitRatePercent = (Double(hits) / Double(totalAccesses)) * 100.0

        let ssdReads = events.filter(\.ssdRead).count
        let totalBytes = events.reduce(0) { $0 + $1.readSize }

        // Expert Popularity & Top Experts
        var expertStats: [Int: (accesses: Int, hits: Int, misses: Int, ssdReads: Int)] = [:]
        for event in events {
            var stat = expertStats[event.expert] ?? (0, 0, 0, 0)
            stat.accesses += 1
            if event.hit { stat.hits += 1 } else { stat.misses += 1 }
            if event.ssdRead { stat.ssdReads += 1 }
            expertStats[event.expert] = stat
        }

        let topExperts = expertStats.map { (expertID, stat) -> ExpertTraceAnalysisReport.ExpertPopularity in
            let rate = stat.accesses > 0 ? (Double(stat.hits) / Double(stat.accesses)) * 100.0 : 0.0
            return ExpertTraceAnalysisReport.ExpertPopularity(
                expertID: expertID,
                accessCount: stat.accesses,
                hitCount: stat.hits,
                missCount: stat.misses,
                hitRatePercent: rate,
                ssdReadCount: stat.ssdReads
            )
        }.sorted { $0.accessCount > $1.accessCount }

        // Per-Layer Locality
        var layerEvents: [Int: [ExpertAccessEvent]] = [:]
        for event in events {
            layerEvents[event.layer, default: []].append(event)
        }

        let perLayerLocality = (0..<trace.totalLayers).map { layer -> ExpertTraceAnalysisReport.LayerLocality in
            let lEvents = layerEvents[layer] ?? []
            let lTotal = lEvents.count
            let lHits = lEvents.filter(\.hit).count
            let lMisses = lTotal - lHits
            let rate = lTotal > 0 ? (Double(lHits) / Double(lTotal)) * 100.0 : 0.0
            let unique = Set(lEvents.map(\.expert)).count
            return ExpertTraceAnalysisReport.LayerLocality(
                layer: layer,
                totalAccesses: lTotal,
                hits: lHits,
                misses: lMisses,
                hitRatePercent: rate,
                uniqueExpertsCount: unique
            )
        }

        // Reuse Distance (Access step index gaps)
        var lastAccessIndex: [Int: Int] = [:]
        var reuseDistances: [Int] = []

        for (index, event) in events.enumerated() {
            let key = event.layer * 1000 + event.expert
            if let prevIndex = lastAccessIndex[key] {
                reuseDistances.append(index - prevIndex)
            }
            lastAccessIndex[key] = index
        }

        let reuseStats = calculateDistanceStats(reuseDistances)

        // Inter-Arrival Token Distance (Token step gaps)
        var lastTokenStep: [Int: Int] = [:]
        var tokenDistances: [Int] = []

        for event in events {
            let key = event.layer * 1000 + event.expert
            if let prevStep = lastTokenStep[key] {
                let dist = event.tokenStep - prevStep
                if dist > 0 { tokenDistances.append(dist) }
            }
            lastTokenStep[key] = event.tokenStep
        }

        let tokenDistanceStats = calculateDistanceStats(tokenDistances)

        // Cross-Token Reuse Ratio
        var tokenExperts: [Int: Set<Int>] = [:]
        for event in events {
            tokenExperts[event.tokenStep, default: []].insert(event.expert)
        }

        var crossTokenMatches = 0
        var crossTokenEvaluations = 0
        let steps = tokenExperts.keys.sorted()
        for i in 1..<steps.count {
            let prevSet = tokenExperts[steps[i - 1]]!
            let currSet = tokenExperts[steps[i]]!
            let intersection = currSet.intersection(prevSet)
            crossTokenMatches += intersection.count
            crossTokenEvaluations += currSet.count
        }

        let crossTokenRatio = crossTokenEvaluations > 0 ? Double(crossTokenMatches) / Double(crossTokenEvaluations) : 0.0

        // Router Scores Stats
        let scores = events.compactMap(\.routingScore)
        let routerScoreStats: ExpertTraceAnalysisReport.RouterScoreStats?
        if !scores.isEmpty {
            let mean = scores.reduce(0.0, +) / Float(scores.count)
            let minS = scores.min() ?? 0.0
            let maxS = scores.max() ?? 0.0
            routerScoreStats = ExpertTraceAnalysisReport.RouterScoreStats(meanScore: mean, minScore: minS, maxScore: maxS, count: scores.count)
        } else {
            routerScoreStats = nil
        }

        return ExpertTraceAnalysisReport(
            totalTokens: trace.totalTokens,
            totalAccesses: totalAccesses,
            totalHits: hits,
            totalMisses: misses,
            overallHitRatePercent: hitRatePercent,
            totalSSDReads: ssdReads,
            totalSSDBytesRead: totalBytes,
            topExperts: topExperts,
            perLayerLocality: perLayerLocality,
            reuseDistance: reuseStats,
            interArrivalTokenDistance: tokenDistanceStats,
            crossTokenReuseRatio: crossTokenRatio,
            routerScoreStats: routerScoreStats
        )
    }

    private static func calculateDistanceStats(_ distances: [Int]) -> ExpertTraceAnalysisReport.ReuseDistanceStats {
        guard !distances.isEmpty else {
            return ExpertTraceAnalysisReport.ReuseDistanceStats(averageDistance: 0, minDistance: 0, maxDistance: 0, medianDistance: 0, p90Distance: 0)
        }
        let sorted = distances.sorted()
        let sum = sorted.reduce(0, +)
        let avg = Double(sum) / Double(sorted.count)
        let minD = sorted.first!
        let maxD = sorted.last!
        let median = Double(sorted[sorted.count / 2])
        let p90Idx = Int(Double(sorted.count - 1) * 0.90)
        let p90 = Double(sorted[p90Idx])

        return ExpertTraceAnalysisReport.ReuseDistanceStats(
            averageDistance: avg,
            minDistance: minD,
            maxDistance: maxD,
            medianDistance: median,
            p90Distance: p90
        )
    }
}
