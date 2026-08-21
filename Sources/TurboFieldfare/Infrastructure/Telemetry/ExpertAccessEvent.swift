import Foundation

/// Structured record of a single routed MoE expert access event during inference.
public struct ExpertAccessEvent: Codable, Sendable, Equatable {
    /// Token/generation step index (0-indexed).
    public let tokenStep: Int

    /// Transformer layer index (0..29).
    public let layer: Int

    /// Expert ID (0..127).
    public let expert: Int

    /// Routing rank (e.g. 0 for top-1, 1 for top-2).
    public let routingRank: Int?

    /// Router score / weight (probability) assigned to this expert.
    public let routingScore: Float?

    /// Whether this expert was already resident in a slot cache slot (cache hit).
    public let hit: Bool

    /// Whether an SSD read (`pread`) occurred to fetch this expert from disk.
    public let ssdRead: Bool

    /// Number of bytes read from SSD (e.g. 3,358,720 bytes).
    public let readSize: UInt64

    /// Latency of the SSD read in nanoseconds.
    public let readLatencyNanos: UInt64

    /// Whether this access resulted in loading the expert into a cache slot.
    public let cacheInsertion: Bool

    /// Expert ID evicted from the cache slot as a result of this access, if any.
    public let evictedExpert: Int?

    public init(tokenStep: Int,
                layer: Int,
                expert: Int,
                routingRank: Int? = nil,
                routingScore: Float? = nil,
                hit: Bool,
                ssdRead: Bool = false,
                readSize: UInt64 = 0,
                readLatencyNanos: UInt64 = 0,
                cacheInsertion: Bool = false,
                evictedExpert: Int? = nil) {
        self.tokenStep = tokenStep
        self.layer = layer
        self.expert = expert
        self.routingRank = routingRank
        self.routingScore = routingScore
        self.hit = hit
        self.ssdRead = ssdRead
        self.readSize = readSize
        self.readLatencyNanos = readLatencyNanos
        self.cacheInsertion = cacheInsertion
        self.evictedExpert = evictedExpert
    }
}

/// Container struct representing a full recorded sequence of expert access events.
public struct ExpertAccessTrace: Codable, Sendable, Equatable {
    public let modelName: String
    public let timestamp: Date
    public let totalTokens: Int
    public let totalLayers: Int
    public let slotsPerLayer: Int
    public let cachePolicy: String
    public let events: [ExpertAccessEvent]

    public init(modelName: String = "Gemma 4 26B-A4B",
                timestamp: Date = Date(),
                totalTokens: Int,
                totalLayers: Int = 30,
                slotsPerLayer: Int = 16,
                cachePolicy: String = "LFU",
                events: [ExpertAccessEvent]) {
        self.modelName = modelName
        self.timestamp = timestamp
        self.totalTokens = totalTokens
        self.totalLayers = totalLayers
        self.slotsPerLayer = slotsPerLayer
        self.cachePolicy = cachePolicy
        self.events = events
    }
}
