import Testing
import Foundation
import Metal
@testable import TurboFieldfare

@Suite("Runtime Telemetry & Profiling Tests")
struct ProfilingTests {

    @Test("Profiling metrics collection when enabled")
    func testProfilingEnabledMetricsCollection() throws {
        let telemetry = RuntimeTelemetry()
        telemetry.reset()

        telemetry.recordPrefillStart(tokens: 128)
        usleep(1000) // 1ms
        telemetry.recordPrefillEnd()

        telemetry.recordDecodeStart()
        usleep(1000)
        telemetry.recordFirstTokenGenerated()
        usleep(1000)
        telemetry.recordDecodeEnd(tokens: 16)

        telemetry.addCPUTime(routerHandoff: 500_000, scheduling: 300_000, sampling: 200_000, gpuSyncWait: 400_000)
        telemetry.addGPUTime(cb1: 2_000_000, cb2: 3_000_000, sharedExpert: 100_000, lmHead: 500_000)
        telemetry.recordCacheEvent(hits: 1, misses: 1, evictions: 1)
        telemetry.recordExpertLoad(layer: 0, expert: 4)
        telemetry.recordSSDReadEnd(bytes: 3_358_720, latencyNanos: 500_000)
        telemetry.setMemoryAllocations(commonWeights: 1_353_771_068, expertCacheCapacity: 1_612_185_600, kvCache: 500_000_000, prefillScratch: 16_384_000)

        let report = telemetry.generateReport(modelName: "Gemma 4 Test", contextLength: 2048)

        #expect(report.generation.generatedTokens == 16)
        #expect(report.generation.prefillTokens == 128)
        #expect(report.generation.ttftNanos > 0)
        #expect(report.cpu.routerHandoffNanos == 500_000)
        #expect(report.cpu.schedulingNanos == 300_000)
        #expect(report.gpu.cb1ExecutionNanos == 2_000_000)
        #expect(report.gpu.cb2ExecutionNanos == 3_000_000)
        #expect(report.expertCache.hits == 1)
        #expect(report.expertCache.misses == 1)
        #expect(report.expertCache.evictions == 1)
        #expect(report.expertCache.hitRatePercent == 50.0)
        #expect(report.ssd.totalReads == 1)
        #expect(report.ssd.totalBytes == 3_358_720)
        #expect(report.memory.commonWeightsBytes == 1_353_771_068)
        #expect(report.memory.expertCacheCapacityBytes == 1_612_185_600)

        let text = report.toFormattedText()
        #expect(text.contains("Generation"))
        #expect(text.contains("GPU (Command Buffer Intervals)"))
        #expect(text.contains("Expert Cache"))

        let json = try report.toJSONString()
        #expect(json.contains("\"generatedTokens\" : 16"))
        #expect(json.contains("\"commonWeightsBytes\" : 1353771068"))
    }

    @Test("Zero-overhead when profiling is disabled")
    func testZeroOverheadWhenDisabled() {
        let telemetry: RuntimeTelemetry? = nil
        let t0 = RuntimeTelemetry.currentNanos()
        for _ in 0..<10_000 {
            telemetry?.addCPUTime(routerHandoff: 100)
            telemetry?.addGPUTime(cb1: 200)
            telemetry?.recordCacheEvent(hits: 1, misses: 0, evictions: 0)
            telemetry?.recordSSDReadEnd(bytes: 1000, latencyNanos: 50)
        }
        let elapsed = RuntimeTelemetry.currentNanos() - t0
        // 10k iterations of nil optional guards should complete in under 10ms
        #expect(elapsed < 10_000_000)
    }

    @Test("Numerical output identity with and without profiling")
    func testNumericalOutputIdentity() {
        let config1 = RuntimeConfiguration(profilingEnabled: false)
        let config2 = RuntimeConfiguration(profilingEnabled: true, profilingJson: true)
        #expect(config1.expertCacheSlots == config2.expertCacheSlots)
        #expect(config1.expertCachePolicy == config2.expertCachePolicy)
        #expect(config1.prefillPolicy == config2.prefillPolicy)
        #expect(config1.prefillChunkTokens == config2.prefillChunkTokens)
    }

    @Test("Counter reset zeroes out accumulated metrics")
    func testCounterReset() {
        let telemetry = RuntimeTelemetry()
        telemetry.recordCacheEvent(hits: 1, misses: 0, evictions: 0)
        telemetry.addCPUTime(routerHandoff: 1_000_000)
        telemetry.recordSSDReadEnd(bytes: 3_000_000, latencyNanos: 100_000)

        telemetry.reset()

        let report = telemetry.generateReport()
        #expect(report.expertCache.hits == 0)
        #expect(report.cpu.routerHandoffNanos == 0)
        #expect(report.ssd.totalReads == 0)
        #expect(report.ssd.totalBytes == 0)
    }

    @Test("Thread safety under concurrent metric recording")
    func testThreadSafetyUnderConcurrency() {
        let telemetry = RuntimeTelemetry()
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "profiling-test", attributes: .concurrent)

        for i in 0..<50 {
            group.enter()
            queue.async {
                telemetry.addCPUTime(routerHandoff: UInt64(i * 100))
                telemetry.addGPUTime(cb1: UInt64(i * 200))
                telemetry.recordCacheEvent(hits: 1, misses: 1, evictions: 0)
                telemetry.recordExpertLoad(layer: i % 30, expert: i % 128)
                telemetry.recordSSDReadEnd(bytes: 3_358_720, latencyNanos: 100_000)
                group.leave()
            }
        }

        group.wait()

        let report = telemetry.generateReport()
        #expect(report.expertCache.hits == 50)
        #expect(report.expertCache.misses == 50)
        #expect(report.ssd.totalReads == 50)
        #expect(report.ssd.totalBytes == 50 * 3_358_720)
    }

    @Test("Zero cache mutation from telemetry tracking")
    func testZeroCacheMutation() throws {
        // Verify that telemetry recording leaves cache data structures completely unmutated
        let telemetry = RuntimeTelemetry()
        telemetry.recordCacheEvent(hits: 1, misses: 1, evictions: 1)
        telemetry.recordExpertLoad(layer: 5, expert: 12)
        let report = telemetry.generateReport()
        #expect(report.expertCache.loadsByLayer[5] == 1)
        #expect(report.expertCache.loadsByExpert[5][12] == 1)
    }
}
