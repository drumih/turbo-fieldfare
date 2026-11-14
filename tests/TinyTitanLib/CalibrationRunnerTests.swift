import Foundation
import Testing

@testable import TinyTitanLib
import TinyTitanCore

@Suite struct CalibrationRunnerTests {
    private func arm(_ name: String, _ knobs: CalibrationProfile.Knobs = .init())
        -> CalibrationArm
    {
        CalibrationArm(name: name, knobs: knobs)
    }

    /// Hands out fixed samples in call order without capturing mutable state
    /// in the @Sendable measurement closure.
    private actor SampleCursor {
        private let baseline: [Double]
        private let arm: [Double]
        private var baselineIndex = 0
        private var armIndex = 0

        init(baseline: [Double], arm: [Double]) {
            self.baseline = baseline
            self.arm = arm
        }

        func next(_ name: String) -> Double {
            if name == "baseline" {
                defer { baselineIndex += 1 }
                return baseline[baselineIndex % baseline.count]
            }
            defer { armIndex += 1 }
            return arm[armIndex % arm.count]
        }
    }

    /// Runs the baseline and one arm with fixed samples per call.
    private func report(
        baseline: [Double], arm samples: [Double], name: String = "arm",
        repeats: Int = 3
    ) async throws -> CalibrationReport {
        let cursor = SampleCursor(baseline: baseline, arm: samples)
        return try await CalibrationRunner(rule: CalibrationKeepRule()).run(
            baseline: arm("baseline"), arms: [arm(name)],
            repeats: repeats
        ) { candidate in
            await cursor.next(candidate.name)
        }
    }

    @Test func winAboveFloorIsKept() async throws {
        let result = try await report(
            baseline: [100, 101, 99], arm: [104, 105, 103])
        let decision = try #require(result.decisions.first)
        #expect(decision.kept)
        #expect(decision.improvement > 0.03)
        #expect(result.keptArms.map(\.name) == ["arm"])
        #expect(decision.reason.contains("kept"))
    }

    @Test func subThreePercentIsRejected() async throws {
        let decision = try #require(
            try await report(baseline: [100, 100, 100], arm: [102, 102, 102])
                .decisions.first)
        #expect(!decision.kept)
        #expect(decision.reason.contains("3% floor"))
    }

    @Test func winInsideNoiseBandIsRejected() async throws {
        // +10% median, but a 10% MAD on both sides: the band is 30%.
        let decision = try #require(
            try await report(
                baseline: [90, 100, 110], arm: [99, 110, 121]
            ).decisions.first)
        #expect(decision.improvement > 0.03)
        #expect(!decision.kept)
        #expect(decision.reason.contains("noise band"))
    }

    @Test func slowerArmIsRejected() async throws {
        let decision = try #require(
            try await report(baseline: [100, 101, 99], arm: [90, 91, 89])
                .decisions.first)
        #expect(!decision.kept)
        #expect(decision.improvement < 0)
    }

    @Test func nonPositiveSamplesAreRejected() async throws {
        let decision = try #require(
            try await report(baseline: [0, 0, 0], arm: [10, 10, 10])
                .decisions.first)
        #expect(!decision.kept)
        #expect(decision.reason.contains("positive"))
    }

    @Test func armsAreInterleavedWithTheBaseline() async throws {
        actor Recorder {
            var order: [String] = []
            func append(_ name: String) { order.append(name) }
        }
        let recorder = Recorder()
        let runner = CalibrationRunner()
        _ = try await runner.run(
            baseline: arm("baseline"), arms: [arm("a"), arm("b")], repeats: 2
        ) { candidate in
            await recorder.append(candidate.name)
            return 10
        }
        let order = await recorder.order
        #expect(
            order == ["baseline", "a", "b", "baseline", "a", "b"])
    }

    @Test func measuredProfileMergesKeptKnobsOnly() async throws {
        let baseline = CalibrationArm(
            name: "baseline",
            knobs: CalibrationProfile.Knobs(prefillChunkTokens: 4_096))
        let kept = CalibrationArm(
            name: "slots",
            knobs: CalibrationProfile.Knobs(expertCacheSlots: 96))
        let rejected = CalibrationArm(
            name: "chunk",
            knobs: CalibrationProfile.Knobs(prefillChunkTokens: 128))
        let runner = CalibrationRunner()
        let measuredAt = Date(timeIntervalSince1970: 1_700_000_000)
        let result = try await runner.run(
            baseline: baseline, arms: [kept, rejected], repeats: 2
        ) { candidate in
            switch candidate.name {
            case "slots": return 120
            case "chunk": return 101
            default: return 100
            }
        }
        #expect(result.keptArms.map(\.name) == ["slots"])
        let profile = result.measuredProfile(
            machine: .init(
                identifier: "Mac15,3", memoryBytes: 24 << 30,
                osVersion: "27.0.1"),
            target: .init(
                modelID: "ornith", family: "qwen36",
                weightBits: 4, contextTokens: 8_192),
            measuredAt: measuredAt)
        #expect(profile.measured)
        #expect(profile.source == "calibrate/1")
        #expect(profile.knobs.expertCacheSlots == 96)
        #expect(profile.knobs.prefillChunkTokens == 4_096)
        #expect(profile.measuredAt != nil)
        try profile.validate()
    }

    @Test func runRefusesTooFewRepetitionsOrNoArms() async throws {
        let runner = CalibrationRunner()
        await #expect(throws: CalibrationRunner.RunError.self) {
            _ = try await runner.run(
                baseline: self.arm("baseline"), arms: [self.arm("a")],
                repeats: 1
            ) { _ in 10 }
        }
        await #expect(throws: CalibrationRunner.RunError.self) {
            _ = try await runner.run(
                baseline: self.arm("baseline"), arms: [], repeats: 2
            ) { _ in 10 }
        }
    }
}
