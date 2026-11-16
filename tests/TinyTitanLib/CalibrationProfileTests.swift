import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanCore

@Suite struct CalibrationProfileTests {
    private func profile(
        schemaVersion: Int = CalibrationProfile.currentSchemaVersion,
        machine: CalibrationProfile.Machine = .init(
            identifier: "Mac15,3", memoryBytes: 24 << 30,
            osVersion: "27.0.1"),
        target: CalibrationProfile.Target = .init(
            modelID: "ornith-1.5-35b-a3b", family: "qwen36",
            weightBits: 4, contextTokens: 8192),
        knobs: CalibrationProfile.Knobs = .init(),
        measured: Bool = false,
        measuredAt: String? = nil,
        source: String = "unmeasured"
    ) -> CalibrationProfile {
        CalibrationProfile(
            machine: machine, target: target, knobs: knobs,
            measured: measured, measuredAt: measuredAt, source: source,
            schemaVersion: schemaVersion)
    }

    @Test func schemaRoundTrips() throws {
        let original = profile(
            knobs: .init(
                prefillChunkTokens: 128, expertCacheSlots: 64,
                q8Attention: true, readerThreads: 8, boundedIO: true,
                prefillTileDepth: 4),
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let data = try original.encodedJSON()
        let decoded = try CalibrationProfile.decode(from: data)
        #expect(decoded == original)
        try decoded.validate()
    }

    @Test func unmeasuredDraftHasNoKnobs() throws {
        let draft = profile()
        #expect(draft.measured == false)
        #expect(draft.knobs.prefillChunkTokens == nil)
        #expect(draft.knobs.expertCacheSlots == nil)
        try draft.validate()
    }

    @Test func unsupportedSchemaIsRefused() {
        #expect(throws: CalibrationProfileError.self) {
            try profile(schemaVersion: 2).validate()
        }
    }

    @Test func emptyModelIDIsRefused() {
        #expect(throws: CalibrationProfileError.self) {
            try profile(
                target: .init(
                    modelID: "", family: "qwen36", weightBits: 4,
                    contextTokens: 8192)
            ).validate()
        }
    }

    @Test func unsupportedContextIsRefused() {
        #expect(throws: CalibrationProfileError.self) {
            try profile(
                target: .init(
                    modelID: "x", family: "qwen36", weightBits: 4,
                    contextTokens: 3_000)
            ).validate()
        }
    }

    @Test func unsupportedWeightBitsIsRefused() {
        #expect(throws: CalibrationProfileError.self) {
            try profile(
                target: .init(
                    modelID: "x", family: "qwen36", weightBits: 6,
                    contextTokens: 8192)
            ).validate()
        }
    }

    @Test func badKnobsAreRefused() {
        for knobs in [
            CalibrationProfile.Knobs(prefillChunkTokens: 100),
            CalibrationProfile.Knobs(expertCacheSlots: 12),
            CalibrationProfile.Knobs(readerThreads: 0),
            CalibrationProfile.Knobs(readerThreads: 65),
            CalibrationProfile.Knobs(prefillTileDepth: 0),
        ] {
            #expect(throws: CalibrationProfileError.self) {
                try profile(knobs: knobs).validate()
            }
        }
    }

    @Test func badTimestampIsRefused() {
        #expect(throws: CalibrationProfileError.self) {
            try profile(measured: true, measuredAt: "yesterday").validate()
        }
    }

    @Test func malformedJSONIsRefused() {
        let data = Data(#"{"schemaVersion":1,"machine":{}}"#.utf8)
        #expect(throws: CalibrationProfileError.self) {
            try CalibrationProfile.decode(from: data)
        }
    }

    @Test func currentMachineIsIdentified() {
        let machine = CalibrationProfile.currentMachine()
        #expect(!machine.identifier.isEmpty)
        #expect(machine.memoryBytes > 0)
        #expect(!machine.osVersion.isEmpty)
    }
}
