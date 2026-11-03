import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// Builds a models directory by hand: the shapes the catalog must recognise,
/// and the ones it must skip without failing.
private enum CatalogFixture {
    static func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-catalog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func write(_ object: Any, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }

    static func directory(_ root: URL, _ name: String) throws -> URL {
        let url = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func slot(_ bits: Int) -> [String: Any] {
        [
            "weightBits": bits, "scheme": "affine", "scaleType": "bf16", "biasType": "bf16",
            "groupSize": 64,
        ]
    }

    /// A manifest the real reader accepts, with the family and width under test.
    static func install(
        _ root: URL, _ name: String, modelID: String, family: String,
        bits: Int, tokenizer: Bool = true
    ) throws {
        let url = try directory(root, name)
        let arch: [String: Any] = [
            "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 128,
            "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 32,
            "fullHeadDim": 32, "vocabSize": 1024, "slidingWindow": 1024,
            "finalLogitSoftcap": 0.0, "ropeTheta": 10_000_000.0, "fullRopeTheta": 10_000_000.0,
            "partialRotaryFactor": 0.25, "numLayers": 4, "numExperts": 8, "topKExperts": 8,
            "tieWordEmbeddings": false, "attentionKEqV": false, "hiddenActivation": "silu",
            "fullAttentionLayerMask": [2, 1, 2, 1], "family": family,
        ]
        try write(
            [
                "magic": "SSDAI", "versionMajor": 1, "versionMinor": 0,
                "flags": ["streamingPresent": true], "modelID": modelID, "arch": arch,
                "files": [
                    "model_weights.bin": [
                        "size": 1024, "sha256": String(repeating: "0", count: 64),
                    ]
                ],
                "expertsPerLayer": 8, "numLayers": 4, "expertStride": 16384,
                "quant": [
                    "embedding": slot(8), "attention": slot(8), "router": slot(8),
                    "sharedExpert": slot(bits), "routedExpert": slot(bits),
                ],
            ] as [String: Any], to: url.appendingPathComponent("manifest.json"))
        if tokenizer {
            let folder = try directory(url, "tokenizer")
            try Data("{}".utf8).write(to: folder.appendingPathComponent("tokenizer.json"))
        }
    }

    static func snapshot(
        _ root: URL, _ name: String, config: [String: Any],
        complete: Bool = true
    ) throws {
        let url = try directory(root, name)
        try write(config, to: url.appendingPathComponent("config.json"))
        guard complete else { return }
        try write(
            ["weight_map": [String: String]()],
            to: url.appendingPathComponent("model.safetensors.index.json"))
        try Data("{}".utf8).write(to: url.appendingPathComponent("tokenizer.json"))
    }

    static func cpuConfig(
        bits: Int = 4, modelType: String = "qwen3_5_dense",
        id: String? = "qwen3.5-2b_4-Bit", name: String? = "Qwen 3.5 2B",
        quantized: Bool = true
    ) -> [String: Any] {
        var config: [String: Any] = ["model_type": modelType, "max_position_embeddings": 262_144]
        if quantized {
            config["quantization"] = ["bits": bits, "group_size": 64, "mode": "affine"]
        }
        if let id { config["model_id"] = id }
        if let name { config["display_name"] = name }
        return config
    }
}

@Suite("Model catalog")
struct ModelCatalogTests {
    @Test func recognisesInstallsAndSnapshotsAndSkipsTheRest() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.install(
            root, "qwen3.6_35B_A3B_8Bit", modelID: "qwen3.6-35b-a3b",
            family: "qwen36", bits: 8)
        try CatalogFixture.install(
            root, "qwen3.6_35B_A3B_MTP_4Bit", modelID: "qwen3.6-35b-a3b-mtp-4bit",
            family: "qwen36_mtp", bits: 4)
        try CatalogFixture.install(
            root, "no-tokenizer", modelID: "other", family: "qwen36",
            bits: 4, tokenizer: false)
        try CatalogFixture.snapshot(root, "qwen3.5_2B_4Bit", config: CatalogFixture.cpuConfig())
        try CatalogFixture.snapshot(
            root, "unnamed_2B_8Bit",
            config: CatalogFixture.cpuConfig(bits: 8, id: nil, name: nil))
        try CatalogFixture.snapshot(
            root, "llama", config: CatalogFixture.cpuConfig(modelType: "llama"))
        try CatalogFixture.snapshot(
            root, "unquantised", config: CatalogFixture.cpuConfig(quantized: false))
        try CatalogFixture.snapshot(
            root, "half-written", config: CatalogFixture.cpuConfig(), complete: false)
        let broken = try CatalogFixture.directory(root, "broken")
        try Data("not json".utf8).write(to: broken.appendingPathComponent("manifest.json"))
        _ = try CatalogFixture.directory(root, "gguf")
        _ = try CatalogFixture.directory(root, ".cache")
        try Data().write(to: root.appendingPathComponent("qwen3.6_35B_A3B_8Bit.install.lock"))

        let catalog = ModelCatalog.scan(directory: root)

        #expect(
            catalog.entries.map(\.id) == [
                "qwen3.6-35b-a3b_8-Bit", "qwen3.5-2b_4-Bit", "unnamed_2B_8Bit",
            ])
        let gpu = try #require(catalog.entry(id: "qwen3.6-35b-a3b_8-Bit"))
        #expect(gpu.name == "Qwen 3.6 35B-A3B")
        #expect(gpu.kind == .gpu(.qwen36))
        #expect(gpu.quant == 8)
        #expect(gpu.sampling == GenerationDefaults.house)
        #expect(gpu.sizeBytes > 0)
        let cpu = try #require(catalog.entry(id: "qwen3.5-2b_4-Bit"))
        #expect(cpu.name == "Qwen 3.5 2B")
        #expect(cpu.kind == .cpu(.qwen35Dense))
        #expect(cpu.quant == 4)
        #expect(cpu.sampling == CPUModelFamily.qwen35Dense.samplingDefaults)
        #expect(cpu.contextLimit == 262_144)
        // No model_id or display_name: the directory names it.
        #expect(catalog.entry(id: "unnamed_2B_8Bit")?.name == "unnamed_2B_8Bit")

        let skipped = Dictionary(
            uniqueKeysWithValues: catalog.skipped.map {
                ($0.path.lastPathComponent, $0.reason)
            })
        #expect(
            Set(skipped.keys) == [
                "qwen3.6_35B_A3B_MTP_4Bit", "no-tokenizer", "llama",
                "unquantised", "half-written", "broken", "gguf",
            ])
        #expect(skipped["qwen3.6_35B_A3B_MTP_4Bit"]?.contains("draft head") == true)
        #expect(skipped["llama"]?.contains("does not implement") == true)
        #expect(skipped["half-written"]?.contains("incomplete") == true)
    }

    /// K2's install is listed like any other: the router picks the probe
    /// backend by family, and the context is the probe's cap rather than the
    /// checkpoint's 524,288.
    @Test func k2InstallIsListedWithTheProbeCeiling() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.install(
            root, "k2-horizon-4bit", modelID: "k2-horizon-4bit",
            family: "k2_horizon", bits: 4)
        let catalog = ModelCatalog.scan(directory: root)
        #expect(catalog.skipped.isEmpty)
        let entry = try #require(catalog.entries.first)
        #expect(entry.kind == .gpu(.k2Horizon))
        #expect(entry.backend == .gpu)
        #expect(entry.contextLimit == K2ProbeBackend.contextCeiling)
        #expect(entry.sampling.temperature == 1.0)
        #expect(entry.sampling.topP == 0.95)
    }

    @Test func skippedDirectoriesAreReportedOnceInOneNotice() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.snapshot(root, "good", config: CatalogFixture.cpuConfig())
        _ = try CatalogFixture.directory(root, "gguf")
        let catalog = ModelCatalog.scan(directory: root)
        let pipe = Pipe()
        catalog.reportSkipped(to: pipe.fileHandleForWriting)
        try pipe.fileHandleForWriting.close()
        let text = pipe.fileHandleForReading.readDataToEndOfFile().lossyUTF8String
        #expect(text.hasPrefix("catalog: skipped 1 of 2 directories"))
        #expect(text.contains("gguf: neither manifest.json"))
    }

    @Test func aModelLinkedInFromElsewhereIsServed() throws {
        let root = try CatalogFixture.root()
        let elsewhere = try CatalogFixture.root()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: elsewhere)
        }
        try CatalogFixture.snapshot(
            elsewhere, "qwen3.5_2B_4Bit", config: CatalogFixture.cpuConfig())
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("linked"),
            withDestinationURL: elsewhere.appendingPathComponent("qwen3.5_2B_4Bit"))
        let catalog = ModelCatalog.scan(directory: root)
        #expect(catalog.entries.map(\.id) == ["qwen3.5-2b_4-Bit"])
        #expect(catalog.skipped.isEmpty)
    }

    @Test func aMissingDirectoryIsReportedNotFatal() {
        let catalog = ModelCatalog.scan(
            directory: URL(fileURLWithPath: "/nonexistent/models-\(UUID())"))
        #expect(catalog.entries.isEmpty)
        #expect(catalog.skipped.count == 1)
    }

    @Test func aDuplicateIdIsServedOnce() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.snapshot(root, "a", config: CatalogFixture.cpuConfig())
        try CatalogFixture.snapshot(root, "b", config: CatalogFixture.cpuConfig())
        let catalog = ModelCatalog.scan(directory: root)
        #expect(catalog.entries.map(\.id) == ["qwen3.5-2b_4-Bit"])
        #expect(catalog.skipped.first?.reason.hasPrefix("duplicate id") == true)
    }

    @Test func theModelFlagFindsAnEntryByIDOrPath() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.snapshot(root, "qwen3.5_2B_4Bit", config: CatalogFixture.cpuConfig())
        let catalog = ModelCatalog.scan(directory: root)
        #expect(catalog.entry(idOrPath: "qwen3.5-2b_4-Bit")?.id == "qwen3.5-2b_4-Bit")
        let path = root.appendingPathComponent("qwen3.5_2B_4Bit").path + "/"
        #expect(catalog.entry(idOrPath: path)?.id == "qwen3.5-2b_4-Bit")
        #expect(catalog.entry(idOrPath: "nothing") == nil)
    }

    /// The exact document a launcher script parses, field for field.
    @Test func theCatalogJSONIsTheLaunchersShape() throws {
        let catalog = ModelCatalog(
            directory: URL(fileURLWithPath: "/abs"),
            entries: [
                ModelCatalog.Entry(
                    id: "qwen3.6-35b-a3b_8-Bit", name: "Qwen 3.6 35B-A3B",
                    kind: .gpu(.qwen36), quant: 8, path: URL(fileURLWithPath: "/abs/path"),
                    sampling: GenerationDefaults.house, sizeBytes: 36_200_000_000)
            ])
        let text = try catalog.jsonData().lossyUTF8String
        #expect(
            text
                == #"{"models":[{"id":"qwen3.6-35b-a3b_8-Bit","name":"Qwen 3.6 35B-A3B","family":"qwen36","quant":8,"backend":"gpu","engines":"gpu","path":"/abs/path","thinking":["off","on"],"sampling":{"temperature":0.6,"top_p":0.95,"top_k":20},"size_gb":36.2}]}"#
        )
    }

    @Test func aCPUModelAndAnEffortModelReportWhatTheyWillUse() throws {
        let catalog = ModelCatalog(
            directory: URL(fileURLWithPath: "/abs"),
            entries: [
                ModelCatalog.Entry(
                    id: "qwen3.8-flash-next_4-Bit", name: "Qwen 3.8 Flash Next 125B-A6B",
                    kind: .gpu(.qwen38flash), quant: 4, path: URL(fileURLWithPath: "/f"),
                    sampling: ModelProfile.resolve(
                        modelID: "qwen3.8-flash-next",
                        family: .qwen38flash, weightBits: 4,
                        environment: [:]
                    ).sampling),
                ModelCatalog.Entry(
                    id: "qwen3.5-2b_8-Bit", name: "Qwen 3.5 2B", kind: .cpu(.qwen35Dense),
                    quant: 8, path: URL(fileURLWithPath: "/c"),
                    sampling: CPUModelFamily.qwen35Dense.samplingDefaults),
            ])
        let document = try #require(
            JSONSerialization.jsonObject(with: try catalog.jsonData()) as? [String: Any])
        let models = try #require(document["models"] as? [[String: Any]])
        #expect(models[0]["thinking"] as? [String] == ["off", "low", "medium", "xhigh"])
        #expect((models[0]["sampling"] as? [String: Any])?["temperature"] as? Double == 1.0)
        #expect(models[1]["backend"] as? String == "cpu")
        #expect(models[1]["family"] as? String == "qwen3_5_dense")
        #expect(models[1]["thinking"] as? [String] == ["off", "on"])
        #expect((models[1]["sampling"] as? [String: Any])?["top_p"] as? Double == 0.95)
    }
    /// An install declares which engines can serve it, and only the dense family
    /// declares two: its `.ssdai` payload is the same file for the CPU and the
    /// GPU engine, so the engine is a request-level choice rather than a
    /// property of the model.
    @Test func onlyTheDenseFamilyIsServedByBothEngines() {
        let dense = ModelCatalog.Entry(
            id: "dense_4-Bit", name: "Dense", kind: .gpu(.qwen35Dense), quant: 4,
            path: URL(fileURLWithPath: "/models/dense_4Bit"),
            sampling: GenerationDefaults.house, engines: [.gpu, .cpu])
        #expect(dense.backend == .gpu)
        #expect(dense.engines == [.gpu, .cpu])
        // The alias keeps the install and switches the engine.
        let cpu = dense.served(by: .cpu, id: "dense_4-Bit@cpu")
        #expect(cpu?.kind == .cpu(.qwen35Dense))
        #expect(cpu?.path == dense.path)
        #expect(cpu?.quant == dense.quant)
        #expect(cpu?.engines == [.gpu, .cpu])
        // And the other direction, for a request that names `@gpu` explicitly.
        #expect(dense.served(by: .gpu, id: "dense_4-Bit@gpu")?.kind == .gpu(.qwen35Dense))

        // Every other family has one engine, and asking for the other is
        // refused rather than silently served by the wrong one.
        let moe = ModelCatalog.Entry(
            id: "moe_4-Bit", name: "MoE", kind: .gpu(.qwen36), quant: 4,
            path: URL(fileURLWithPath: "/models/moe_4Bit"),
            sampling: GenerationDefaults.house)
        #expect(moe.engines == [.gpu])
        #expect(moe.served(by: .cpu, id: "moe_4-Bit@cpu") == nil)
        let snapshot = ModelCatalog.Entry(
            id: "snap", name: "Snap", kind: .cpu(.qwen35Dense), quant: 8,
            path: URL(fileURLWithPath: "/models/snap"),
            sampling: CPUModelFamily.qwen35Dense.samplingDefaults)
        // A converted snapshot is CPU-only whatever the family: the GPU path
        // reads `.ssdai` installs, so there is no GPU engine to name.
        #expect(snapshot.engines == [.cpu])
        #expect(snapshot.served(by: .gpu, id: "snap@gpu") == nil)
        #expect(snapshot.served(by: .cpu, id: "snap@cpu")?.kind == .cpu(.qwen35Dense))
    }

    /// A manifest carries an id, not a name, so an install missing from this
    /// table is listed as its raw id in `/v1/models`, the launcher and the app.
    ///
    /// The list is the shipped set, and the count is asserted so that adding a
    /// model is a deliberate edit here rather than a silent fallback to the id.
    /// The KAT-Coder row is what the count guards: it was added with the model,
    /// before the model had an install.
    @Test func everyShippedModelHasADisplayName() {
        let shipped = [
            "qwen3.6-35b-a3b", "ornith-1.5-35b-a3b", "ornith-1.5-35b-a3b-bf16dense",
            "qwen-agentworld",
            "kat-coder-v2.5", "qwen3.8-flash-next", "qwen4-omni-120b-think",
            "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
        ]
        for id in shipped {
            let name = ModelCatalog.displayNames[id]
            #expect(name?.isEmpty == false, "no display name for \(id)")
        }
        let names = Array(ModelCatalog.displayNames.values)
        #expect(Set(names).count == names.count, "two models share a display name")
        #expect(
            ModelCatalog.displayNames.count == shipped.count,
            "a display name was added or removed without updating the shipped list")
    }

    // MARK: - The size bound on a scanned config

    /// The probe reads `config.json` from every directory the scan finds, and
    /// `Data(contentsOf:)` grows as it reads, so an unbounded read is an
    /// unbounded allocation over a directory nobody asked the server to load.
    /// The bound is the manifest's, shared rather than invented.
    @Test func aScannedConfigOverTheBoundIsRefusedAndSaysSo() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.write(
            CatalogFixture.cpuConfig(),
            to: root.appendingPathComponent("config.json"))
        let size = try Data(
            contentsOf: root.appendingPathComponent("config.json")
        ).count
        #expect(size > 8, "the fixture is too small to cross an 8-byte bound")

        var described = ""
        do {
            _ = try ModelCatalog.readSnapshotConfig(root, maxBytes: 8)
        } catch {
            described = String(describing: error)
        }
        #expect(
            described.contains("\(size)") && described.contains("over the 8-byte metadata bound"),
            "the refusal named neither the size nor the cap: \(described)")

        // The same file inside the bound reads, so the refusal above is the
        // bound and not the path or the decoder.
        #expect(try ModelCatalog.readSnapshotConfig(root).count == size)
    }

    /// The family probe goes through the same helper, so it must still resolve a
    /// normal snapshot -- otherwise the bound above would have quietly turned
    /// `--cpu` into a refusal for every real model.
    @Test func theFamilyProbeStillReadsAnOrdinaryConfig() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.write(
            CatalogFixture.cpuConfig(),
            to: root.appendingPathComponent("config.json"))
        #expect(try ModelCatalog.snapshotFamily(root) == .qwen35Dense)
    }
}
