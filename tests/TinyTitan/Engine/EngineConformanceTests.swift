import Foundation
import Metal
import Testing
@testable import TinyTitan
@testable import TinyTitanCore

/// TT-097 Phase 1 Task 1: pin the diagnostic counters the server session reads
/// off `runner.` before Phase 1 collapses them behind `EngineStatistics`.
/// Everything in this file is model-free: the runner is built against the
/// in-tree Qwen toy fixture (`QwenToySynthetic`), never a real install.
@Suite struct EngineConformanceTests {

    /// The `runner.total*` counters `ServerModelSession` reads, measured with
    /// `grep -rhoE 'runner\.total[A-Za-z0-9]+' sources/TinyTitanLib/ServerModelSession*.swift`
    /// on 2026-11-16. The handover's "32" under-counted: a letter-only regex
    /// collapsed `totalCb1Nanos`/`totalCb2Nanos` into one name. Two further
    /// runner counters (`totalRDAdviseFailures`, `totalRDAdviseSkipped`) exist
    /// but the session never reads them, so they are outside the contract this
    /// list pins. `totalPrefetchIssued` is the one computed member; the rest
    /// are stored `UInt64` vars on `RealForwardRunner`.
    static let knownDiagnosticCounters: [String] = [
        "totalBodyNanos",
        "totalCachePlanNanos",
        "totalCb1Nanos",
        "totalCb2Nanos",
        "totalEmbedNanos",
        "totalExpertIOHostWaits",
        "totalExpertIOHostWaitsAvoided",
        "totalExposedIoNanos",
        "totalGPUClassifiedHits",
        "totalGPUClassifiedMisses",
        "totalGPUResidencyAllHitLayers",
        "totalGatherNanos",
        "totalHeadFusedNanos",
        "totalHeadNanos",
        "totalHitFixupLayers",
        "totalIOCompletionToFixupSubmitNanos",
        "totalIOQueueNanos",
        "totalIoNanos",
        "totalLoopOtherNanos",
        "totalLoopProgressNanos",
        "totalLoopSampleNanos",
        "totalMissIoNanos",
        "totalPreambleNanos",
        "totalPreamblePinNanos",
        "totalPreambleReleaseNanos",
        "totalPreambleReserveNanos",
        "totalPrefetchAdopted",
        "totalPrefetchIssued",
        "totalRDAdviseBytes",
        "totalRDAdviseCalls",
        "totalRDAdviseNanos",
        "totalRouterReadbackNanos",
        "totalWaitNanos",
    ]

    private func makeRunner() throws -> (directory: URL, runner: RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: 4, qsa: false)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir,
            device: ctx.device,
            expecting: .qwenToy())
        let runtime = try RuntimeConfiguration()
        let runner = try RealForwardRunner(
            model: model,
            context: ctx,
            maxContext: 64,
            slots: 1,
            runtimeConfiguration: runtime)
        return (dir, runner)
    }

    @Test("diagnostic counters are pinned by name and count")
    func diagnosticCountersAreEnumerated() {
        let names = Self.knownDiagnosticCounters
        #expect(!names.isEmpty, "the pinned counter list is empty; fill it from the session grep")
        #expect(names.count == 33, "expected the 33 session-read counters, got \(names.count)")
        #expect(Set(names).count == names.count, "the pinned counter list has duplicates")
        #expect(names == names.sorted(), "keep the pinned list sorted so a diff reads cleanly")
        for name in names {
            #expect(
                name.hasPrefix("total"),
                "pinned counter \(name) does not look like a total* counter")
        }
    }

    @Test("every pinned counter exists on the 6.x runner")
    func pinnedCountersExistOnRunner() throws {
        let (dir, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }

        let storedLabels = Set(
            Mirror(reflecting: runner).children.compactMap(\.label))
        var missing: [String] = []
        for name in Self.knownDiagnosticCounters where name != "totalPrefetchIssued" {
            // `totalPrefetchIssued` is computed, so Mirror (stored properties
            // only) cannot see it; it is asserted by key path below.
            guard storedLabels.contains(name) else {
                missing.append(name)
                continue
            }
        }
        #expect(
            missing.isEmpty,
            "pinned counters missing from the runner: \(missing.joined(separator: ", "))")

        // The computed member, read through its key path so a rename breaks
        // the build instead of the assertion.
        #expect(runner[keyPath: \RealForwardRunner.totalPrefetchIssued] == 0)
        #expect(!Self.knownDiagnosticCounters.contains("totalRDAdviseFailures"))
        #expect(!Self.knownDiagnosticCounters.contains("totalRDAdviseSkipped"))
    }
}
