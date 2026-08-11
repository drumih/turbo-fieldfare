import Foundation

/// Thread-safe accumulator for expert access events during inference execution.
public final class ExpertTracer: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [ExpertAccessEvent] = []
    public let isEnabled: Bool
    private var currentTokenStep: Int = 0

    public init(isEnabled: Bool = true) {
        self.isEnabled = isEnabled
    }

    public func setTokenStep(_ step: Int) {
        guard isEnabled else { return }
        lock.lock()
        currentTokenStep = step
        lock.unlock()
    }

    public func currentStep() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return currentTokenStep
    }

    public func recordAccess(tokenStep: Int? = nil,
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
        guard isEnabled else { return }
        lock.lock()
        let step = tokenStep ?? currentTokenStep
        let event = ExpertAccessEvent(
            tokenStep: step,
            layer: layer,
            expert: expert,
            routingRank: routingRank,
            routingScore: routingScore,
            hit: hit,
            ssdRead: ssdRead,
            readSize: readSize,
            readLatencyNanos: readLatencyNanos,
            cacheInsertion: cacheInsertion,
            evictedExpert: evictedExpert
        )
        events.append(event)
        lock.unlock()
    }

    public func getEvents() -> [ExpertAccessEvent] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }

    public func reset() {
        guard isEnabled else { return }
        lock.lock()
        events.removeAll(keepingCapacity: true)
        currentTokenStep = 0
        lock.unlock()
    }

    public func buildTrace(modelName: String = "Gemma 4 26B-A4B",
                           slotsPerLayer: Int = 16,
                           cachePolicy: String = "LFU") -> ExpertAccessTrace {
        lock.lock()
        let currentEvents = events
        let maxStep = currentEvents.map(\.tokenStep).max() ?? 0
        let maxLayer = currentEvents.map(\.layer).max() ?? 0
        lock.unlock()

        return ExpertAccessTrace(
            modelName: modelName,
            timestamp: Date(),
            totalTokens: currentEvents.isEmpty ? 0 : maxStep + 1,
            totalLayers: currentEvents.isEmpty ? 30 : maxLayer + 1,
            slotsPerLayer: slotsPerLayer,
            cachePolicy: cachePolicy,
            events: currentEvents
        )
    }

    public func toJSONData(modelName: String = "Gemma 4 26B-A4B",
                           slotsPerLayer: Int = 16,
                           cachePolicy: String = "LFU") throws -> Data {
        let trace = buildTrace(modelName: modelName, slotsPerLayer: slotsPerLayer, cachePolicy: cachePolicy)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(trace)
    }

    public func toJSONString(modelName: String = "Gemma 4 26B-A4B",
                             slotsPerLayer: Int = 16,
                             cachePolicy: String = "LFU") throws -> String {
        let data = try toJSONData(modelName: modelName, slotsPerLayer: slotsPerLayer, cachePolicy: cachePolicy)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    public func saveJSON(to path: String,
                         modelName: String = "Gemma 4 26B-A4B",
                         slotsPerLayer: Int = 16,
                         cachePolicy: String = "LFU") throws {
        let data = try toJSONData(modelName: modelName, slotsPerLayer: slotsPerLayer, cachePolicy: cachePolicy)
        let url = URL(fileURLWithPath: path)
        try data.write(to: url, options: .atomic)
    }

    public static func loadTrace(from path: String) throws -> ExpertAccessTrace {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(ExpertAccessTrace.self, from: data)
    }
}
