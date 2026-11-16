import Foundation
import Testing
import TinyTitanLib

@testable import TinyTitanServerCore
@testable import TinyTitanCore

@Suite struct CalibrationProfileArgumentsTests {
    /// A manifest the real `ManifestReader.peekIdentity` accepts, so the
    /// profile-application tests exercise the identity binding against an
    /// install rather than a path that does not exist.
    private static func toyInstall(modelID: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-calibration-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        let arch: [String: Any] = [
            "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 128,
            "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 32,
            "fullHeadDim": 32, "vocabSize": 1024, "slidingWindow": 1024,
            "finalLogitSoftcap": 0.0, "ropeTheta": 10_000_000.0,
            "fullRopeTheta": 10_000_000.0, "partialRotaryFactor": 0.25,
            "numLayers": 4, "numExperts": 8, "topKExperts": 8,
            "tieWordEmbeddings": false, "attentionKEqV": false,
            "hiddenActivation": "silu", "fullAttentionLayerMask": [2, 1, 2, 1],
            "family": "qwen36",
        ]
        let slot: [String: Any] = [
            "weightBits": 4, "scheme": "affine", "scaleType": "bf16",
            "biasType": "bf16", "groupSize": 64,
        ]
        try JSONSerialization.data(
            withJSONObject: [
                "magic": "SSDAI", "versionMajor": 1, "versionMinor": 0,
                "flags": ["streamingPresent": true], "modelID": modelID,
                "arch": arch,
                "files": [
                    "model_weights.bin": [
                        "size": 1024,
                        "sha256": String(repeating: "0", count: 64),
                    ]
                ] as [String: Any],
                "expertsPerLayer": 8, "numLayers": 4, "expertStride": 16384,
                "quant": [
                    "embedding": slot, "attention": slot, "router": slot,
                    "sharedExpert": slot, "routedExpert": slot,
                ] as [String: Any],
            ] as [String: Any],
            options: [.sortedKeys]
        ).write(to: url.appendingPathComponent("manifest.json"))
        return url
    }

    private func profile(
        knobs: CalibrationProfile.Knobs,
        modelID: String = "ornith",
        family: String = "qwen36",
        weightBits: Int = 4
    ) throws -> URL {
        let profile = CalibrationProfile(
            machine: CalibrationProfile.currentMachine(),
            target: .init(
                modelID: modelID, family: family, weightBits: weightBits,
                contextTokens: 8192),
            knobs: knobs,
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-calibration-\(UUID().uuidString).json")
        try profile.encodedJSON().write(to: url)
        return url
    }

    @Test func profileFillsKnobsTheFlagsLeftUnset() throws {
        let install = try Self.toyInstall(modelID: "ornith")
        defer { try? FileManager.default.removeItem(at: install) }
        let url = try profile(
            knobs: .init(prefillChunkTokens: 128, expertCacheSlots: 96))
        defer { try? FileManager.default.removeItem(at: url) }
        let arguments = try ServerArguments.parse([
            "--model", install.path, "--max-context", "8192",
            "--calibration-profile", url.path,
        ])
        #expect(arguments.prefillChunkTokens == 128)
        #expect(arguments.expertCacheSlots == 96)
        #expect(arguments.calibrationProfile == url.path)
        #expect(arguments.calibrationLogLine?.contains("prefillChunkTokens=128") == true)
        #expect(arguments.calibrationLogLine?.contains("expertCacheSlots=96") == true)
    }

    @Test func explicitFlagsWinOverTheProfile() throws {
        let install = try Self.toyInstall(modelID: "ornith")
        defer { try? FileManager.default.removeItem(at: install) }
        let url = try profile(
            knobs: .init(prefillChunkTokens: 128, expertCacheSlots: 96))
        defer { try? FileManager.default.removeItem(at: url) }
        let arguments = try ServerArguments.parse([
            "--model", install.path, "--max-context", "8192",
            "--calibration-profile", url.path,
            "--prefill-chunk", "256", "--expert-cache-slots", "32",
        ])
        #expect(arguments.prefillChunkTokens == 256)
        #expect(arguments.expertCacheSlots == 32)
    }

    @Test func switchlessKnobsAreReportedNotApplied() throws {
        let install = try Self.toyInstall(modelID: "ornith")
        defer { try? FileManager.default.removeItem(at: install) }
        let url = try profile(knobs: .init(boundedIO: false))
        defer { try? FileManager.default.removeItem(at: url) }
        let arguments = try ServerArguments.parse([
            "--model", install.path, "--max-context", "8192",
            "--calibration-profile", url.path,
        ])
        #expect(arguments.prefillChunkTokens == nil)
        #expect(arguments.expertCacheSlots == nil)
        #expect(arguments.calibrationLogLine?.contains("not applied: boundedIO") == true)
    }

    @Test func foreignMachineProfileIsRefused() throws {
        let install = try Self.toyInstall(modelID: "ornith")
        defer { try? FileManager.default.removeItem(at: install) }
        var machine = CalibrationProfile.currentMachine()
        machine = .init(
            identifier: "Mac99,9", memoryBytes: machine.memoryBytes,
            osVersion: machine.osVersion)
        let profile = CalibrationProfile(
            machine: machine,
            target: .init(
                modelID: "ornith", family: "qwen36", weightBits: 4,
                contextTokens: 8192),
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-calibration-foreign-\(UUID().uuidString).json")
        try profile.encodedJSON().write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", install.path, "--max-context", "8192",
                "--calibration-profile", url.path,
            ])
        }
    }

    /// A profile measured on another macOS release is not evidence about this
    /// one, and a profile for a different checkpoint is not evidence about
    /// this install: both are refused rather than applied.
    @Test func foreignOSAndModelProfilesAreRefused() throws {
        let install = try Self.toyInstall(modelID: "ornith")
        defer { try? FileManager.default.removeItem(at: install) }
        var otherOS = CalibrationProfile.currentMachine()
        otherOS = .init(
            identifier: otherOS.identifier, memoryBytes: otherOS.memoryBytes,
            osVersion: "0.0.1")
        let osProfile = CalibrationProfile(
            machine: otherOS,
            target: .init(
                modelID: "ornith", family: "qwen36", weightBits: 4,
                contextTokens: 8192),
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let osURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-calibration-os-\(UUID().uuidString).json")
        try osProfile.encodedJSON().write(to: osURL)
        defer { try? FileManager.default.removeItem(at: osURL) }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", install.path, "--max-context", "8192",
                "--calibration-profile", osURL.path,
            ])
        }
        for (modelID, family, bits) in [
            ("other", "qwen36", 4), ("ornith", "qwen38", 4),
            ("ornith", "qwen36", 8),
        ] {
            let url = try profile(
                knobs: .init(), modelID: modelID, family: family,
                weightBits: bits)
            defer { try? FileManager.default.removeItem(at: url) }
            #expect(throws: ServerArgumentError.self) {
                try ServerArguments.parse([
                    "--model", install.path, "--max-context", "8192",
                    "--calibration-profile", url.path,
                ])
            }
        }
    }

    @Test func noProfileLeavesTheFlagsAlone() throws {
        let arguments = try ServerArguments.parse([
            "--model", "/tmp/model", "--max-context", "8192",
        ])
        #expect(arguments.calibrationProfile == nil)
        #expect(arguments.calibrationLogLine == nil)
        #expect(arguments.prefillChunkTokens == nil)
        #expect(arguments.expertCacheSlots == nil)
    }

    @Test func corruptProfileIsRefused() throws {
        let install = try Self.toyInstall(modelID: "ornith")
        defer { try? FileManager.default.removeItem(at: install) }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-calibration-bad-\(UUID().uuidString).json")
        try Data(#"{"schemaVersion":99}"#.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", install.path, "--calibration-profile", url.path,
            ])
        }
    }
}
