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

    /// The 33 counter names paired with their `EngineStatistics` and runner
    /// key paths, in the same order as `knownDiagnosticCounters`. A rename on
    /// either side breaks the build here, which is the pin. Computed (not a
    /// stored `let`): key-path tuples do not infer Sendable, and a stored
    /// static of them fails the concurrency-safety check.
    private static var counterKeyPaths:
        [(
            statistics: KeyPath<EngineStatistics, UInt64>,
            runner: KeyPath<RealForwardRunner, UInt64>
        )]
    {
        [
            (\EngineStatistics.totalBodyNanos, \RealForwardRunner.totalBodyNanos),
            (\EngineStatistics.totalCachePlanNanos, \RealForwardRunner.totalCachePlanNanos),
            (\EngineStatistics.totalCb1Nanos, \RealForwardRunner.totalCb1Nanos),
            (\EngineStatistics.totalCb2Nanos, \RealForwardRunner.totalCb2Nanos),
            (\EngineStatistics.totalEmbedNanos, \RealForwardRunner.totalEmbedNanos),
            (\EngineStatistics.totalExpertIOHostWaits, \RealForwardRunner.totalExpertIOHostWaits),
            (
                \EngineStatistics.totalExpertIOHostWaitsAvoided,
                \RealForwardRunner.totalExpertIOHostWaitsAvoided
            ),
            (\EngineStatistics.totalExposedIoNanos, \RealForwardRunner.totalExposedIoNanos),
            (\EngineStatistics.totalGPUClassifiedHits, \RealForwardRunner.totalGPUClassifiedHits),
            (
                \EngineStatistics.totalGPUClassifiedMisses,
                \RealForwardRunner.totalGPUClassifiedMisses
            ),
            (
                \EngineStatistics.totalGPUResidencyAllHitLayers,
                \RealForwardRunner.totalGPUResidencyAllHitLayers
            ),
            (\EngineStatistics.totalGatherNanos, \RealForwardRunner.totalGatherNanos),
            (\EngineStatistics.totalHeadFusedNanos, \RealForwardRunner.totalHeadFusedNanos),
            (\EngineStatistics.totalHeadNanos, \RealForwardRunner.totalHeadNanos),
            (\EngineStatistics.totalHitFixupLayers, \RealForwardRunner.totalHitFixupLayers),
            (
                \EngineStatistics.totalIOCompletionToFixupSubmitNanos,
                \RealForwardRunner.totalIOCompletionToFixupSubmitNanos
            ),
            (\EngineStatistics.totalIOQueueNanos, \RealForwardRunner.totalIOQueueNanos),
            (\EngineStatistics.totalIoNanos, \RealForwardRunner.totalIoNanos),
            (\EngineStatistics.totalLoopOtherNanos, \RealForwardRunner.totalLoopOtherNanos),
            (\EngineStatistics.totalLoopProgressNanos, \RealForwardRunner.totalLoopProgressNanos),
            (\EngineStatistics.totalLoopSampleNanos, \RealForwardRunner.totalLoopSampleNanos),
            (\EngineStatistics.totalMissIoNanos, \RealForwardRunner.totalMissIoNanos),
            (\EngineStatistics.totalPreambleNanos, \RealForwardRunner.totalPreambleNanos),
            (\EngineStatistics.totalPreamblePinNanos, \RealForwardRunner.totalPreamblePinNanos),
            (
                \EngineStatistics.totalPreambleReleaseNanos,
                \RealForwardRunner.totalPreambleReleaseNanos
            ),
            (
                \EngineStatistics.totalPreambleReserveNanos,
                \RealForwardRunner.totalPreambleReserveNanos
            ),
            (\EngineStatistics.totalPrefetchAdopted, \RealForwardRunner.totalPrefetchAdopted),
            (\EngineStatistics.totalPrefetchIssued, \RealForwardRunner.totalPrefetchIssued),
            (\EngineStatistics.totalRDAdviseBytes, \RealForwardRunner.totalRDAdviseBytes),
            (\EngineStatistics.totalRDAdviseCalls, \RealForwardRunner.totalRDAdviseCalls),
            (\EngineStatistics.totalRDAdviseNanos, \RealForwardRunner.totalRDAdviseNanos),
            (
                \EngineStatistics.totalRouterReadbackNanos,
                \RealForwardRunner.totalRouterReadbackNanos
            ),
            (\EngineStatistics.totalWaitNanos, \RealForwardRunner.totalWaitNanos),
        ]
    }

    @Test("statistics forward every pinned counter, and resetStatistics zeroes them")
    func statisticsMatchCounters() throws {
        #expect(Self.counterKeyPaths.count == Self.knownDiagnosticCounters.count)
        let (dir, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let engineRunner: any EngineRunner = runner

        // Seed non-zero values so equality cannot pass on both sides being 0.
        runner.totalWaitNanos = 11
        runner.totalBodyNanos = 22
        runner.totalCb1Nanos = 33
        runner.totalCb2Nanos = 44
        runner.totalLoopSampleNanos = 55
        runner.totalPrefetchAdopted = 66

        let stats = engineRunner.statistics
        for (statisticsPath, runnerPath) in Self.counterKeyPaths {
            #expect(
                stats[keyPath: statisticsPath] == runner[keyPath: runnerPath],
                "\(statisticsPath) does not forward its runner counter")
        }

        runner.resetStatistics()
        #expect(
            runner.statistics == EngineStatistics.zero,
            "resetStatistics must zero every counter, including the ring-issued one")
    }

    @Test("Model forwards its members through EngineModel")
    func engineModelMembersForward() throws {
        let dir = try QwenToySynthetic.write(weightBits: 4, qsa: false)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir,
            device: ctx.device,
            expecting: .qwenToy())
        let engineModel: any EngineModel = model
        #expect(engineModel.config == model.config)
        #expect(engineModel.modelID == model.modelID)
        #expect(engineModel.sourceSnapshotHash == model.sourceSnapshotHash)
        #expect(engineModel.routedExpertWeightBits == model.routedExpertWeightBits)
        #expect(
            engineModel.routedExpertCacheSlotCount() == model.routedExpertCacheSlotCount())
    }
}
