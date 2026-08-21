import Testing
import Foundation
import Metal
@testable import TurboFieldfare

@Suite("Expert Access Tracing & Simulation Tests")
struct ExpertTracingTests {

    @Test("Tracing disabled records no events")
    func testTracingDisabledNoEvents() {
        let tracer = ExpertTracer(isEnabled: false)
        tracer.recordAccess(layer: 0, expert: 5, hit: false)
        #expect(tracer.getEvents().isEmpty)
    }

    @Test("Tracing enabled records access events with accurate properties")
    func testTracingEnabledRecordsEvents() {
        let tracer = ExpertTracer(isEnabled: true)
        tracer.setTokenStep(1)
        tracer.recordAccess(
            layer: 2,
            expert: 14,
            routingRank: 0,
            routingScore: 0.85,
            hit: false,
            ssdRead: true,
            readSize: 3_358_720,
            readLatencyNanos: 150_000,
            cacheInsertion: true,
            evictedExpert: nil
        )

        tracer.recordAccess(
            layer: 2,
            expert: 14,
            routingRank: 0,
            routingScore: 0.90,
            hit: true,
            ssdRead: false,
            readSize: 0,
            readLatencyNanos: 0,
            cacheInsertion: false,
            evictedExpert: nil
        )

        let events = tracer.getEvents()
        #expect(events.count == 2)

        #expect(events[0].tokenStep == 1)
        #expect(events[0].layer == 2)
        #expect(events[0].expert == 14)
        #expect(events[0].routingRank == 0)
        #expect(events[0].routingScore == 0.85)
        #expect(events[0].hit == false)
        #expect(events[0].ssdRead == true)
        #expect(events[0].readSize == 3_358_720)
        #expect(events[0].readLatencyNanos == 150_000)

        #expect(events[1].hit == true)
        #expect(events[1].ssdRead == false)
    }

    @Test("Trace JSON serialization and deserialization roundtrip")
    func testTraceJSONRoundtrip() throws {
        let tracer = ExpertTracer(isEnabled: true)
        tracer.setTokenStep(0)
        tracer.recordAccess(layer: 0, expert: 1, hit: false)
        tracer.recordAccess(layer: 0, expert: 2, hit: false)

        let trace = tracer.buildTrace(modelName: "Gemma 4 Test", slotsPerLayer: 16, cachePolicy: "LFU")
        let jsonString = try tracer.toJSONString(modelName: "Gemma 4 Test", slotsPerLayer: 16, cachePolicy: "LFU")

        #expect(jsonString.contains("\"modelName\" : \"Gemma 4 Test\""))
        #expect(jsonString.contains("\"expert\" : 1"))

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let loadedTrace = try decoder.decode(ExpertAccessTrace.self, from: Data(jsonString.utf8))

        #expect(loadedTrace.events.count == 2)
        #expect(loadedTrace.events[0].expert == 1)
        #expect(loadedTrace.events[1].expert == 2)
    }

    @Test("ExpertTraceAnalyzer calculates statistics accurately")
    func testAnalyzerStatistics() {
        let events = [
            ExpertAccessEvent(tokenStep: 0, layer: 0, expert: 1, routingRank: 0, routingScore: 0.9, hit: false, ssdRead: true, readSize: 3358720),
            ExpertAccessEvent(tokenStep: 0, layer: 0, expert: 2, routingRank: 1, routingScore: 0.1, hit: false, ssdRead: true, readSize: 3358720),
            ExpertAccessEvent(tokenStep: 1, layer: 0, expert: 1, routingRank: 0, routingScore: 0.95, hit: true, ssdRead: false, readSize: 0),
            ExpertAccessEvent(tokenStep: 1, layer: 0, expert: 3, routingRank: 1, routingScore: 0.05, hit: false, ssdRead: true, readSize: 3358720)
        ]

        let trace = ExpertAccessTrace(totalTokens: 2, totalLayers: 1, slotsPerLayer: 16, events: events)
        let report = ExpertTraceAnalyzer.analyze(trace)

        #expect(report.totalAccesses == 4)
        #expect(report.totalHits == 1)
        #expect(report.totalMisses == 3)
        #expect(report.overallHitRatePercent == 25.0)
        #expect(report.totalSSDReads == 3)
        #expect(report.topExperts.first?.expertID == 1)
        #expect(report.topExperts.first?.accessCount == 2)

        let formatted = report.toFormattedText()
        #expect(formatted.contains("Expert Access Analysis"))
        #expect(formatted.contains("Overall Hit Rate:     25.00%"))
    }

    @Test("Offline cache simulator evaluates LFU, LRU, and FIFO hit rates")
    func testCacheSimulator() {
        let events = [
            // Layer 0 access sequence to 2 slots capacity
            ExpertAccessEvent(tokenStep: 0, layer: 0, expert: 1, hit: false),
            ExpertAccessEvent(tokenStep: 0, layer: 0, expert: 2, hit: false),
            ExpertAccessEvent(tokenStep: 1, layer: 0, expert: 1, hit: true), // Hit for 1
            ExpertAccessEvent(tokenStep: 1, layer: 0, expert: 3, hit: false), // Miss, evicts 2 (LFU/LRU)
            ExpertAccessEvent(tokenStep: 2, layer: 0, expert: 1, hit: true)  // Hit for 1
        ]

        let trace = ExpertAccessTrace(totalTokens: 3, totalLayers: 1, slotsPerLayer: 2, events: events)
        let report = ExpertCacheSimulator.simulate(trace, slotCapacities: [2], policies: [.lfu, .lru, .fifo])

        #expect(report.results.count == 3)
        let lfuResult = report.results.first(where: { $0.policy == "LFU" })
        #expect(lfuResult != nil)
        #expect(lfuResult?.hits == 2)
        #expect(lfuResult?.misses == 3)

        let text = report.toFormattedText()
        #expect(text.contains("Cache Policy Simulation"))
        #expect(text.contains("LFU"))
        #expect(text.contains("LRU"))
        #expect(text.contains("FIFO"))
    }

    @Test("RuntimeConfiguration integration for expert tracing")
    func testRuntimeConfigExpertTracing() {
        let configNoTrace = RuntimeConfiguration(expertTracingEnabled: false)
        #expect(configNoTrace.expertTracingEnabled == false)
        #expect(configNoTrace.expertTracer == nil)

        let configWithTrace = RuntimeConfiguration(expertTracingEnabled: true)
        #expect(configWithTrace.expertTracingEnabled == true)
        #expect(configWithTrace.expertTracer != nil)
    }
}
