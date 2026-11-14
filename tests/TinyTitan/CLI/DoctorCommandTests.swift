import Foundation
import Testing
import TinyTitanLib
import TinyTitanCore

@testable import TinyTitanCLICore
@testable import TinyTitan

/// K2's output contract. Every input is synthetic: capabilities, the models
/// tree, free space and the process list. Nothing here reads this host's chip,
/// memory or running processes.
@Suite struct DoctorCommandTests {
    private func capabilities(
        apple10: Bool,
        ram: UInt64 = 24 << 30
    ) -> HardwareCapabilities {
        HardwareCapabilities(
            chipModel: "MacTest,1",
            gpuName: "Synthetic GPU",
            gpuFamily: apple10 ? .apple10 : .apple9,
            supportsApple10TensorOps: apple10,
            metal4Available: true,
            anePresent: true,
            physicalMemoryBytes: ram,
            osVersion: "26.0.0")
    }

    /// A models tree with one 1 GiB routed-expert install and round geometry.
    private func makeModelsDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("doctor-\(UUID().uuidString)")
        let models = root.appendingPathComponent("models")
        let install = models.appendingPathComponent("qwen-doctor-toy")
        let experts = install.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(
            at: experts, withIntermediateDirectories: true)
        let weights = install.appendingPathComponent("model_weights.bin")
        guard FileManager.default.createFile(atPath: weights.path, contents: nil)
        else {
            throw CocoaError(.fileWriteUnknown)
        }
        let handle = try FileHandle(forWritingTo: weights)
        try handle.truncate(atOffset: UInt64(1) << 30)
        try handle.close()
        try Data(repeating: 0, count: 128).write(
            to: experts.appendingPathComponent("layout.json"))
        let manifest: [String: Any] = [
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": [String: Bool](),
            "modelID": "qwen-doctor-toy",
            "arch": [
                "hiddenSize": 256,
                "ffnIntermediate": 512,
                "moeIntermediateSize": 128,
                "numHeads": 8,
                "numKVHeads": 4,
                "numFullKVHeads": 4,
                "headDim": 256,
                "fullHeadDim": 256,
                "vocabSize": 1_024,
                "slidingWindow": 512,
                "finalLogitSoftcap": 0.0,
                "ropeTheta": 10_000.0,
                "fullRopeTheta": 10_000.0,
                "partialRotaryFactor": 0.25,
                "numLayers": 2,
                "numExperts": 4,
                "topKExperts": 2,
                "tieWordEmbeddings": false,
                "attentionKEqV": true,
                "hiddenActivation": "silu",
                "fullAttentionLayerMask": [1, 0],
                "family": "qwen36",
            ] as [String: Any],
            "files": [
                "model_weights.bin": [
                    "size": 1_073_741_824,
                    "sha256": String(repeating: "0", count: 64),
                ],
                "packed_experts/layout.json": [
                    "size": 128,
                    "sha256": String(repeating: "0", count: 64),
                ],
            ] as [String: Any],
            "expertsPerLayer": 4,
            "numLayers": 2,
            "expertStride": 1_048_576,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: manifest, options: [.sortedKeys])
        try data.write(to: install.appendingPathComponent("manifest.json"))
        return models
    }

    private func environment(
        models: URL,
        capabilities: HardwareCapabilities,
        processes: [DoctorReport.Process] = [],
        environment variables: [String: String] = [:],
        calibration: DoctorReport.Calibration = .notConfigured,
        maxContext: Int = 4_096
    ) -> DoctorEnvironment {
        DoctorEnvironment(
            capabilities: capabilities,
            modelsDirectory: models,
            environment: variables,
            maxContextTokens: maxContext,
            disk: DoctorReport.Disk(
                path: models.path, freeBytes: 100 << 30,
                capacityBytes: 494 << 30),
            otherProcesses: processes,
            calibration: calibration)
    }

    @Test func humanFormatIsGoldenForInjectedMachineAndModelsTree() throws {
        let models = try self.makeModelsDirectory()
        defer { try? FileManager.default.removeItem(at: models.deletingLastPathComponent()) }
        let report = DoctorReport.build(
            self.environment(
                models: models, capabilities: self.capabilities(apple10: false)))
        let expected = """
            TinyTitan doctor
            machine: MacTest,1 (Synthetic GPU)  macOS 26.0.0  RAM 24.00 GiB
            models: \(models.path)  (1 installed, 1.00 GiB on disk)
            disk: 100.00 GiB free of 494.00 GiB at \(models.path)
            gpu: apple9  apple10=no  metal4=yes  ane=yes
            tensor-ops: fallback (device does not support Apple10 MPP tensor operations)
            calibration: none
            other TinyTitan model processes: none

            models:
              qwen-doctor-toy  (qwen36, 4-bit)
                dir: qwen-doctor-toy
                size on disk: 1.00 GiB
                weights: 1.00 GiB  kv@4096 (8-bit): 17.00 MiB  cache: 512.00 MiB (floor 16.00 MiB)  total: 1.52 GiB
                fit: yes (6.3% of 24.00 GiB)
                expert slots: 256  prefill chunk: 4096
                page cache: page-cache — install is 1073741952 bytes and RAM is at least 2x it
            """
        #expect(report.renderedHuman() == expected)
    }

    @Test func jsonParsesAndHasStableKeys() throws {
        let models = try self.makeModelsDirectory()
        defer { try? FileManager.default.removeItem(at: models.deletingLastPathComponent()) }
        let report = DoctorReport.build(
            self.environment(
                models: models, capabilities: self.capabilities(apple10: false)))
        let data = try report.jsonData()
        let root = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(
            Set(root.keys) == [
                "schema", "machine", "models_dir", "models_dir_exists", "disk",
                "tensor_ops", "models", "skipped", "calibration",
                "other_model_processes", "messages",
            ])
        let machine = try #require(root["machine"] as? [String: Any])
        #expect(
            Set(machine.keys) == [
                "chip", "gpu_name", "gpu_family", "apple10_tensor_ops",
                "metal4", "ane", "ram_bytes", "os_version",
            ])
        let tensorOps = try #require(root["tensor_ops"] as? [String: Any])
        #expect(tensorOps["mode"] as? String == "auto")
        #expect(tensorOps["path"] as? String == "fallback")
        #expect(tensorOps["capability_present"] as? Bool == false)
        let modelsArray = try #require(root["models"] as? [[String: Any]])
        #expect(modelsArray.count == 1)
        let model = try #require(modelsArray.first)
        #expect(model["id"] as? String == "qwen-doctor-toy")
        #expect(model["expert_slots"] as? Int == 256)
        #expect(model["prefill_chunk_tokens"] as? Int == 4_096)
        let fit = try #require(model["fit"] as? [String: Any])
        #expect(fit["total_bytes"] as? Int == 1_628_438_528)
        #expect(fit["fits"] as? Bool == true)
        let pageCache = try #require(model["page_cache"] as? [String: Any])
        #expect(pageCache["policy"] as? String == "page-cache")
    }

    @Test func absentModelsDirectoryIsAnExplicitMessage() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("doctor-missing-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let models = root.appendingPathComponent("models")
        let report = DoctorReport.build(
            self.environment(
                models: models, capabilities: self.capabilities(apple10: false)))
        #expect(!report.modelsDirExists)
        #expect(report.models.isEmpty)
        #expect(
            report.messages.contains(
                "models directory not found: \(models.path)"))
        let human = report.renderedHuman()
        #expect(human.contains("models: \(models.path) — not found"))
        #expect(
            human.contains("notice: models directory not found: \(models.path)"))
        let data = try report.jsonData()
        let rootObject = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(rootObject["models_dir_exists"] as? Bool == false)
    }

    @Test func apple10CapabilitySelectsTheTensorPathInTheReport() throws {
        let models = try self.makeModelsDirectory()
        defer { try? FileManager.default.removeItem(at: models.deletingLastPathComponent()) }
        let report = DoctorReport.build(
            self.environment(
                models: models, capabilities: self.capabilities(apple10: true)))
        #expect(report.tensorOps.path == "tensor-ops")
        #expect(report.tensorOps.capabilityPresent)
    }

    @Test func offSwitchIsReportedAndOnIsAttempted() throws {
        let models = try self.makeModelsDirectory()
        defer { try? FileManager.default.removeItem(at: models.deletingLastPathComponent()) }
        let off = DoctorReport.build(
            self.environment(
                models: models, capabilities: self.capabilities(apple10: true),
                environment: ["TINYTITAN_TENSOROPS": "off"]))
        #expect(off.tensorOps.mode == "off")
        #expect(off.tensorOps.path == "fallback")
        let on = DoctorReport.build(
            self.environment(
                models: models, capabilities: self.capabilities(apple10: false),
                environment: ["TINYTITAN_TENSOROPS": "on"]))
        #expect(on.tensorOps.mode == "on")
        #expect(on.tensorOps.path == "tensor-ops")
    }

    @Test func calibrationStatusIsReportedWithoutReadingTheHost() throws {
        let models = try self.makeModelsDirectory()
        defer { try? FileManager.default.removeItem(at: models.deletingLastPathComponent()) }
        let valid = DoctorReport.Calibration(
            configured: true, path: "/tmp/profile.json", status: "valid",
            detail: nil, measured: true, source: "calibrate/1",
            modelID: "qwen-doctor-toy", contextTokens: 4_096,
            machineMatches: true)
        let report = DoctorReport.build(
            self.environment(
                models: models, capabilities: self.capabilities(apple10: false),
                calibration: valid))
        let human = report.renderedHuman()
        #expect(human.contains("calibration: valid (measured: yes"))
        #expect(human.contains("machine matches: yes"))
        let invalid = DoctorReport.Calibration(
            configured: true, path: "/tmp/broken.json", status: "invalid",
            detail: "malformed calibration profile", measured: nil,
            source: nil, modelID: nil, contextTokens: nil,
            machineMatches: nil)
        let failed = DoctorReport.build(
            self.environment(
                models: models, capabilities: self.capabilities(apple10: false),
                calibration: invalid))
        #expect(
            failed.renderedHuman().contains(
                "calibration: invalid (malformed calibration profile)"))
    }

    @Test func optionsParseAndRefuseBadValues() throws {
        let options = try DoctorOptions.parse([
            "--json", "--models-dir", "/tmp/models", "--max-context", "8192",
            "--calibration-profile", "/tmp/profile.json",
        ])
        #expect(options.json)
        #expect(options.modelsDirectory == "/tmp/models")
        #expect(options.maxContext == 8_192)
        #expect(options.calibrationProfile == "/tmp/profile.json")
        #expect(throws: DoctorCommandError.self) {
            try DoctorOptions.parse(["--max-context", "123"])
        }
        #expect(throws: DoctorCommandError.self) {
            try DoctorOptions.parse(["--nope"])
        }
        #expect(throws: DoctorCommandError.self) {
            try DoctorOptions.parse(["--models-dir"])
        }
    }

    @Test func unsupportedFamiliesAreRefusedWithAReason() {
        // A memory fit is not a capability claim: K2's install validates but
        // its graph does not exist, and doctor must say that rather than list
        // it as a model this machine can run.
        let k2 = DoctorModelScanner.unsupportedInstallReason(.k2Horizon)
        #expect(k2?.contains("server's probe backend") == true)
        #expect(
            DoctorModelScanner.unsupportedInstallReason(.qwen36MTP)?
                .contains("MTP draft head") == true)
        #expect(DoctorModelScanner.unsupportedInstallReason(.qwen36) == nil)
        #expect(DoctorModelScanner.unsupportedInstallReason(.qwen38flash) == nil)
        #expect(DoctorModelScanner.unsupportedInstallReason(.qwen35Dense) == nil)
    }
}
