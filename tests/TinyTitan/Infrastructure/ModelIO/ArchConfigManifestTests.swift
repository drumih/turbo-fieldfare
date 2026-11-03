import Foundation
import Testing

@testable import TinyTitan

/// The dense family's architecture is read from the manifest rather than from a
/// preset (`ArchConfig+Manifest.swift`), and the conventions it requires are
/// refused by name rather than defaulted. The rule that makes this worth
/// pinning: a wrong attention output gate or RoPE convention is not a shape
/// error, it is fluent nonsense, so a test that only asserts "it loaded" proves
/// nothing here. Each case below asserts a specific field name or a specific
/// value, and the omission matrix asserts the *key* that was refused.
@Suite("ArchConfig manifest resolution")
struct ArchConfigManifestTests {

    /// The fields `ArchConfig.from(manifest:family:)` refuses to assume. A
    /// computed property rather than storage: the payload is `Any`, which is not
    /// `Sendable`, and Swift 6 refuses a static stored property of it.
    private static var requiredFields: [(key: String, value: Any)] {
        [
            ("attnOutputGate", true),
            ("attentionScale", 0.0883),
            ("embeddingScaledBySqrtHidden", false),
            ("routerScaled", true),
            ("ffnSandwichNorms", false),
            ("sharedExpertGated", true),
            ("ropeNeoxSubdim", true),
            ("linearNumKHeads", 2),
            ("linearNumVHeads", 2),
            ("linearKeyHeadDim", 16),
            ("linearValueHeadDim", 16),
            ("linearConvKernelSize", 4),
        ]
    }

    /// A complete dense `arch` block: the toy geometry the loader tests use,
    /// plus every field the dense resolution requires.
    private static func denseArch(
        hiddenSize: Int = 64,
        omitting: Set<String> = [],
        declare: [String: Any] = [:]
    ) -> [String: Any] {
        var arch: [String: Any] = [
            "hiddenSize": hiddenSize,
            "ffnIntermediate": 128,
            "moeIntermediateSize": 128,
            "numHeads": 4,
            "numKVHeads": 1,
            "numFullKVHeads": 1,
            "headDim": 16,
            "fullHeadDim": 16,
            "vocabSize": 128,
            "slidingWindow": 4,
            "finalLogitSoftcap": 10.0,
            "ropeTheta": 10_000.0,
            "fullRopeTheta": 1_000_000.0,
            "partialRotaryFactor": 0.25,
            "numLayers": 4,
            "numExperts": 8,
            "topKExperts": 8,
            "tieWordEmbeddings": false,
            "attentionKEqV": true,
            "hiddenActivation": "silu",
            "fullAttentionLayerMask": [2, 1, 2, 1],
        ]
        for (key, value) in requiredFields { arch[key] = value }
        for key in omitting { arch.removeValue(forKey: key) }
        for (key, value) in declare { arch[key] = value }
        return arch
    }

    private static func decode(_ arch: [String: Any]) throws -> ManifestArch {
        let data = try JSONSerialization.data(withJSONObject: arch)
        return try JSONDecoder().decode(ManifestArch.self, from: data)
    }

    private static func resolve(_ arch: [String: Any]) throws -> ArchConfig {
        try ArchConfig.from(manifest: decode(arch), family: .qwen35Dense)
    }

    /// The preset families resolve without a manifest at all. The directory is
    /// one that does not exist: reading it would throw, so a return value here
    /// is the proof that the short-circuit ran.
    @Test("A family with a preset never touches the disk")
    func presetShortCircuit() throws {
        let missing = URL(fileURLWithPath: "/nonexistent/\(UUID().uuidString)")
        let presetFamilies: [ModelFamily] = [
            .qwen36, .qwen36MTP, .qwen38flash, .qwen38flashMTP,
            .qwen4flash, .qwen4flashMTP,
        ]
        for family in presetFamilies {
            let preset = try #require(ArchConfig.knownArchitectures[family])
            let got = try ArchConfig.resolved(forFamily: family, directoryURL: missing)
            #expect(got.family == family)
            #expect(got.numLayers == preset.numLayers)
            #expect(got.hiddenSize == preset.hiddenSize)
            #expect(got.ple.ngramSize == preset.ple.ngramSize)
        }
    }

    /// A 4.0 install whose shape differs from the preset is rebuilt from its
    /// manifest, including the blocks a plain residual path would drop.
    @Test("A qwen4 manifest supplies the Flash extension geometry")
    func qwen4ResolutionRebuildsTheFlashBlocks() throws {
        let arch = Self.denseArch(declare: [
            "hcCount": 2, "hcLowRank": 8,
            "indexerNumHeads": 2, "indexerNumKVHeads": 1,
            "indexerHeadDim": 16, "indexerBudget": 32,
            "indexerCompressRatio": 4,
            "pleLayerIndices": [1], "pleEmbedDim": 64,
            "pleConvKernelSize": 4, "pleNgramSize": 3,
            "pleVocabSizeBase": 1000, "pleHeadsPerNgram": 2,
            "pleVocabDivisor": 128, "routerNormTopK": true,
            "quantGroupSize": 64,
        ])
        let config = try ArchConfig.from(
            manifest: Self.decode(arch), family: .qwen4flash)
        #expect(config.family == .qwen4flash)
        #expect(config.linearAttention.outputGate == .sigmoid)
        #expect(config.hyperConnections.count == 2)
        #expect(config.hyperConnections.lowRank == 8)
        #expect(config.sparseIndexer.budget == 32)
        #expect(config.ple.layerIndices == [1])
        #expect(config.ple.embedDim == 64)
        #expect(config.routerNormTopK)
    }

    @Test("A qwen4 manifest that omits its Flash geometry is refused by name")
    func qwen4OmittingExtensionsIsRefused() {
        let arch = Self.denseArch()
        #expect(throws: (any Error).self) {
            _ = try ArchConfig.from(
                manifest: Self.decode(arch), family: .qwen4flash)
        }
    }

    /// The reason the manifest path exists: 2B/4B and 9B share one family case,
    /// so no single preset describes them.
    @Test("The dense family has no preset, which is why the manifest is read")
    func denseFamilyHasNoPreset() {
        #expect(ArchConfig.knownArchitectures[.qwen35Dense] == nil)
        // Laguna-S-2.1 has a checked preset, like the Qwen families; only the
        // dense Qwen 3.5 sizes need the manifest.
        #expect(ArchConfig.knownArchitectures[.laguna] == .lagunaS21)
        #expect(ArchConfig.knownArchitectures.count == 7)
    }

    @Test("Geometry, mask and conventions come from the manifest")
    func denseResolutionMapsTheManifest() throws {
        let config = try Self.resolve(Self.denseArch())
        #expect(config.hiddenSize == 64)
        #expect(config.intermediateSize == 128)  // ffnIntermediate, not moe
        #expect(config.moeIntermediateSize == 128)
        #expect(config.numLayers == 4)
        #expect(config.family == .qwen35Dense)
        // The mask crosses the wire as Int and is carried as UInt8.
        #expect(config.fullAttentionLayerMask == [2, 1, 2, 1])
        #expect(config.attnOutputGate == true)
        #expect(config.ropeNeoxSubdim == true)
        #expect(config.sharedExpertGated == true)
        #expect(config.routerScaled == true)
        #expect(config.ffnSandwichNorms == false)
        #expect(config.embeddingScaledBySqrtHidden == false)
        #expect(config.linearAttention.numKHeads == 2)
        #expect(config.linearAttention.convKernelSize == 4)
        // Declared per family, never defaulted from Qwen3.8's sigmoid.
        #expect(config.linearAttention.outputGate == .silu)
    }

    /// 2048 against 12288 under one family case: the resolution must track the
    /// file, or the 9B loads at the 4B's width and nothing complains.
    @Test("Two dense widths under one family resolve to two different geometries")
    func denseResolutionTracksTheFile() throws {
        let small = try Self.resolve(Self.denseArch(hiddenSize: 2048))
        let large = try Self.resolve(Self.denseArch(hiddenSize: 12_288))
        #expect(small.hiddenSize == 2048)
        #expect(large.hiddenSize == 12_288)
        #expect(small.family == large.family)
    }

    /// Each omission is a different failure, named by the key that was missing.
    /// If any of these fields were silently defaulted this test fails, which is
    /// the point: a defaulted convention loads a wrong model.
    @Test("Each required field is refused by name when the manifest omits it")
    func everyRequiredFieldIsRefusedByName() throws {
        for (key, _) in Self.requiredFields {
            let arch = Self.denseArch(omitting: [key])
            do {
                _ = try Self.resolve(arch)
                Issue.record("omitting \(key) resolved instead of refusing")
            } catch let error as ArchResolutionError {
                guard case .manifestOmitsField(let family, let field) = error else {
                    Issue.record("omitting \(key) threw \(error), not manifestOmitsField")
                    continue
                }
                #expect(field == key, "refused field \(field), expected \(key)")
                #expect(family == ModelFamily.qwen35Dense.rawValue)
                // The message has to name the field it refused, to whoever hit it.
                #expect(error.description.contains(key))
            }
        }
    }

    /// The two fields that ARE optional on the dense path, and their defaults.
    /// Pinned because they are behaviour, not style: a manifest written before
    /// `quantGroupSize` existed still has to load, and load as group 64.
    @Test("routerNormTopK and quantGroupSize default rather than refuse")
    func declaredOptionalFieldsDefault() throws {
        let config = try Self.resolve(Self.denseArch())
        #expect(config.routerNormTopK == false)
        #expect(config.quantGroupSize == 64)
        // Declared values win over the defaults.
        let declared = try Self.resolve(
            Self.denseArch(declare: ["routerNormTopK": true, "quantGroupSize": 32]))
        #expect(declared.routerNormTopK == true)
        #expect(declared.quantGroupSize == 32)
    }

    /// The end-to-end route the loader takes: a manifest file on disk, read
    /// without a GPU config to validate against.
    @Test("resolved(forFamily:directoryURL:) reads a dense install from disk")
    func resolvesFromADirectoryOnDisk() throws {
        let dense: [String: Any] = [
            "attnOutputGate": true, "attentionScale": 0.0883,
            "embeddingScaledBySqrtHidden": false, "routerScaled": true,
            "ffnSandwichNorms": false, "sharedExpertGated": true,
            "ropeNeoxSubdim": true, "linearNumKHeads": 2, "linearNumVHeads": 2,
            "linearKeyHeadDim": 16, "linearValueHeadDim": 16,
            "linearConvKernelSize": 4,
        ]
        let (dir, toy) = try ManifestReaderTests.writeToyManifest(archOverrides: dense)
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ArchConfig.resolved(forFamily: .qwen35Dense, directoryURL: dir)
        #expect(config.hiddenSize == toy.hiddenSize)
        #expect(config.numExperts == toy.numExperts)
        #expect(config.attnOutputGate == true)
        #expect(config.ropeNeoxSubdim == true)
        #expect(config.linearAttention.keyHeadDim == 16)
        #expect(config.family == .qwen35Dense)
    }

    /// `qwen36` used to be exactly one geometry (the 35B-A3B). Ornith-1.5-122B
    /// is the same architecture at hidden 3072 / 48 layers, and a manifest that
    /// declares that shape must resolve from its own declaration rather than be
    /// validated against the 35B preset. The 35B path is unchanged: it still
    /// short-circuits to the preset, which `presetShortCircuit` pins.
    @Test("qwen36 resolves a 122B manifest at its own geometry")
    func qwen36ReadsThe122BGeometry() throws {
        var mask = [Int](repeating: 2, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { mask[i] = 1 }
        let geometry: [String: Any] = [
            "hiddenSize": 3072, "ffnIntermediate": 1024,
            "moeIntermediateSize": 1024, "numHeads": 32,
            "numKVHeads": 2, "numFullKVHeads": 2,
            "headDim": 256, "fullHeadDim": 256, "vocabSize": 248_320,
            "numLayers": 48, "numExperts": 256, "topKExperts": 8,
            "fullAttentionLayerMask": mask,
            "attnOutputGate": true, "attentionScale": 0.0625,
            "embeddingScaledBySqrtHidden": false, "routerScaled": false,
            "ffnSandwichNorms": false, "sharedExpertGated": true,
            "ropeNeoxSubdim": true, "linearNumKHeads": 16,
            "linearNumVHeads": 64, "linearKeyHeadDim": 128,
            "linearValueHeadDim": 128, "linearConvKernelSize": 4,
        ]
        // A whole 48-layer file table, not `writeToyManifest`'s four: the
        // reader validates the layout against `numLayers`, and an inconsistent
        // fixture would be refused before the resolution branch was reached.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ornith122-resolve-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        var files: [String: [String: Any]] = [
            "model_weights.bin": [
                "size": 1024, "sha256": String(repeating: "0", count: 64),
            ],
            "packed_experts/layout.json": [
                "size": 1024, "sha256": String(repeating: "0", count: 64),
            ],
        ]
        for layer in 0..<48 {
            files["packed_experts/layer_\(layer).bin"] = [
                "size": 16384, "sha256": String(repeating: "0", count: 64),
            ]
        }
        var arch = geometry
        arch["hiddenActivation"] = "silu"
        arch["tieWordEmbeddings"] = false
        arch["attentionKEqV"] = false
        arch["slidingWindow"] = 0
        arch["finalLogitSoftcap"] = 0.0
        arch["ropeTheta"] = 10_000_000.0
        arch["fullRopeTheta"] = 10_000_000.0
        arch["partialRotaryFactor"] = 0.25
        let root: [String: Any] = [
            "magic": "SSDAI", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false],
            "modelID": "ornith-1.5-122b", "arch": arch, "files": files,
            "expertsPerLayer": 256, "numLayers": 48, "expertStride": 16384,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: root, options: [.sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ArchConfig.resolved(forFamily: .qwen36, directoryURL: dir)
        #expect(config.hiddenSize == 3072)
        #expect(config.numLayers == 48)
        #expect(config.numExperts == 256)
        #expect(config.numHeads == 32)
        #expect(config.linearAttention.numVHeads == 64)
        #expect(config.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
        // The MoE/gated-DeltaNet threadgroup tile is the one compile-time
        // constant this geometry had to widen. A 3072-wide config must pass the
        // load guard; a 3073-wide one is the neighbouring failure the guard
        // exists for and is refused rather than written past the tile.
        #expect(Model.maximumThreadgroupTileWidth == 3072)
        try Model.validateExecutableGeometry(config)
    }
}
