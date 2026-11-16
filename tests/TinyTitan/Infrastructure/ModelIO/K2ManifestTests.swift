import Foundation
import Testing
@testable import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanFormat

/// K2-Horizon's identity, architecture resolution and the mixed dense/sparse
/// install shape.
///
/// The rule these tests pin: a K2 install is identified as K2 and read from its
/// own manifest fields -- not from the Qwen geometry it happens to resemble --
/// and its three dense layers are not treated as missing packed-expert files.
/// Both failures are silent in production until a model is loaded, and the
/// second one refused every K2 install at validation while `--verify-install`
/// (which reads per-layer counts) passed it.
@Suite("K2-Horizon manifest and mixed layout")
struct K2ManifestTests {

    /// A 4-layer K2-shaped manifest: layers 0-2 dense, layer 3 with two routed
    /// experts. Tiny on purpose -- the shape is what is under test, not the
    /// 36B constants (the repacker's tests pin those against the real config).
    private static func manifest(
        archOverrides: [String: Any] = [:],
        filesOverride: [String: [String: Any]]? = nil
    ) -> [String: Any] {
        var arch: [String: Any] = [
            "hiddenSize": 64,
            "ffnIntermediate": 128,
            "moeIntermediateSize": 128,
            "numHeads": 4,
            "numKVHeads": 1,
            "numFullKVHeads": 1,
            "headDim": 16,
            "fullHeadDim": 16,
            "vocabSize": 128,
            "slidingWindow": 0,
            "finalLogitSoftcap": 0.0,
            "ropeTheta": 10_000_000.0,
            "fullRopeTheta": 10_000_000.0,
            "partialRotaryFactor": 1.0,
            "numLayers": 4,
            "numExperts": 2,
            "topKExperts": 1,
            "tieWordEmbeddings": false,
            "attentionKEqV": false,
            "hiddenActivation": "silu",
            "fullAttentionLayerMask": [1, 1, 1, 1],
            "family": "k2_horizon",
            "attnOutputGate": true,
            "attentionScale": 0.25,
            "embeddingScaledBySqrtHidden": false,
            "routerScaled": false,
            "ffnSandwichNorms": false,
            "sharedExpertGated": true,
            "ropeNeoxSubdim": true,
            "linearNumKHeads": 0,
            "linearNumVHeads": 0,
            "linearKeyHeadDim": 0,
            "linearValueHeadDim": 0,
            "linearConvKernelSize": 0,
            "normGroups": 2,
            "attentionGate": "softplus",
            "routerHasBias": true,
            "routerScaling": 2.5,
            "movaNumExperts": 4,
            "movaTopKExperts": 2,
            "movaRouterScaling": 2.5,
            "movaRouterHasBias": true,
            "movaActivation": "silu",
            "denseLayerIndices": [0, 1, 2],
        ]
        for (key, value) in archOverrides { arch[key] = value }

        func fileEntry(size: UInt64) -> [String: Any] {
            ["size": size, "sha256": String(repeating: "0", count: 64)]
        }
        var files: [String: [String: Any]] = [
            "model_weights.bin": fileEntry(size: 4096),
            "packed_experts/layout.json": fileEntry(size: 1024),
            // Only the sparse layer carries a packed-expert file.
            "packed_experts/layer_3.bin": fileEntry(size: 32768),
        ]
        if let filesOverride { files = filesOverride }

        let slot: [String: Any] = [
            "weightBits": 4, "scheme": "affine", "scaleType": "bf16",
            "biasType": "bf16", "groupSize": Quantization.groupSize,
        ]
        let quant: [String: Any] = [
            "embedding": slot, "attention": slot, "router": slot,
            "sharedExpert": slot, "routedExpert": slot,
        ]
        return [
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": [
                "streamingPresent": true, "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            "modelID": "k2-horizon-toy",
            "arch": arch,
            "quant": quant,
            "files": files,
            "expertsPerLayer": 2,
            "numLayers": 4,
            "expertStride": 16384,
        ]
    }

    private static func write(_ root: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("k2-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        let data = try JSONSerialization.data(
            withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    @Test("A K2 install is identified as K2, not inferred as qwen36")
    func identityIsDeclaredNotInferred() throws {
        let dir = try Self.write(Self.manifest())
        defer { try? FileManager.default.removeItem(at: dir) }
        let identity = try ManifestReader.peekIdentity(directoryURL: dir)
        #expect(identity.family == .k2Horizon)
        #expect(identity.modelID == "k2-horizon-toy")
        #expect(identity.weightBits == 4)
    }

    @Test("The architecture is read from the manifest's K2 fields")
    func architectureComesFromTheManifest() throws {
        let dir = try Self.write(Self.manifest())
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ArchConfig.resolved(forFamily: .k2Horizon, directoryURL: dir)
        #expect(config.modelFamily == .k2Horizon)
        #expect(config.normGroups == 2)
        #expect(config.attentionGate == .softplus)
        #expect(config.routerHasBias)
        #expect(config.routerScaling == 2.5)
        #expect(config.denseLayerIndices == [0, 1, 2])
        #expect(config.mova.numExperts == 4)
        #expect(config.mova.topKExperts == 2)
        #expect(config.mova.routerScaling == 2.5)
        #expect(config.mova.routerHasBias)
        #expect(config.mova.activation == "silu")
        #expect(config.mova.enabled)
        // The manifest here omits `routerNormTopK`, as the real 4-bit install
        // does. K2's config normalizes the selected scores, so the family's
        // fallback is true; defaulting it to the Qwen `false` would hand the
        // router unnormalized weights.
        #expect(config.routerNormTopK)
    }

    /// The dense layers have no `packed_experts/layer_0*.bin`; a validator that
    /// demanded one refused every K2 install even though `--verify-install`,
    /// which reads the per-layer count, had passed it.
    @Test("Dense layers need no packed-expert file; a sparse layer still does")
    func mixedInstallValidatesPerLayer() throws {
        let dir = try Self.write(Self.manifest())
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ArchConfig.resolved(forFamily: .k2Horizon, directoryURL: dir)
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.expertsPerLayer == 2)

        // The same install with the one sparse layer's file removed must fail,
        // and name that layer rather than a dense one.
        var files = Self.manifest()["files"] as? [String: [String: Any]] ?? [:]
        files.removeValue(forKey: "packed_experts/layer_3.bin")
        let brokenDir = try Self.write(Self.manifest(filesOverride: files))
        defer { try? FileManager.default.removeItem(at: brokenDir) }
        let brokenConfig = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: brokenDir)
        var missingName: String?
        do {
            _ = try ManifestReader.load(directoryURL: brokenDir, expecting: brokenConfig)
        } catch ModelError.missingFile(let name) {
            missingName = name
        }
        #expect(missingName == "packed_experts/layer_3.bin")
    }

    /// The layout's per-layer count is the authority: cross-validation must
    /// skip a layer that declares no experts instead of demanding a manifest
    /// entry for a file the writer never created.
    @Test("Layout cross-validation accepts empty layers and sizes only packed ones")
    func crossValidationSkipsEmptyLayers() throws {
        func layer(_ index: Int, experts: [SSDAIExpertV1], count: Int?) -> SSDAILayerV1 {
            SSDAILayerV1(
                layer: index,
                file: String(format: "layer_%02d.bin", index),
                experts: experts, expertsPerLayer: count)
        }
        let packed = (0..<2).map { index in
            SSDAIExpertV1(
                expert: index, physicalRank: index,
                offset: UInt64(index) * 16384, size: 16384, tensors: [:])
        }
        let layout = SSDAIPackedExpertsLayoutV1(
            expertStride: 16384, numLayers: 4, expertsPerLayer: 2,
            layers: [
                layer(0, experts: [], count: 0),
                layer(1, experts: [], count: 0),
                layer(2, experts: [], count: 0),
                layer(3, experts: packed, count: nil),
            ])
        let sizes: [String: UInt64] = [
            "packed_experts/layer_03.bin": 32768
        ]
        try SSDAIV1StructuralValidator.crossValidate(
            manifestNumLayers: 4, manifestExpertsPerLayer: 2,
            manifestExpertStride: 16384, manifestFileSizes: sizes,
            layout: layout)

        // Removing the packed layer's entry is still an error.
        var error: TinyTitanFormatError?
        do {
            try SSDAIV1StructuralValidator.crossValidate(
                manifestNumLayers: 4, manifestExpertsPerLayer: 2,
                manifestExpertStride: 16384, manifestFileSizes: [:],
                layout: layout)
        } catch let thrown as TinyTitanFormatError {
            error = thrown
        }
        #expect(error != nil)
    }

    @Test("An unknown attention gate is refused by value, not defaulted")
    func unknownGateIsRefused() throws {
        let dir = try Self.write(
            Self.manifest(archOverrides: ["attentionGate": "gelu"]))
        defer { try? FileManager.default.removeItem(at: dir) }
        var failure: ArchResolutionError?
        do {
            _ = try ArchConfig.resolved(forFamily: .k2Horizon, directoryURL: dir)
        } catch let thrown as ArchResolutionError {
            failure = thrown
        }
        guard case .manifestDeclaresInvalidValue(_, let field, let value) = failure else {
            Issue.record("expected a refused manifest value, got \(String(describing: failure))")
            return
        }
        #expect(field == "attentionGate")
        #expect(value == "gelu")
    }

    /// The other half of the gate's contract: an absent K2 field is not a
    /// reason to fall back to the Qwen default, it is a refused manifest.
    @Test("A K2 manifest that omits a shape field is refused by name")
    func omittedFieldIsRefused() throws {
        var root = Self.manifest()
        var arch = try #require(root["arch"] as? [String: Any])
        arch.removeValue(forKey: "attentionGate")
        root["arch"] = arch
        let dir = try Self.write(root)
        defer { try? FileManager.default.removeItem(at: dir) }
        var failure: ArchResolutionError?
        do {
            _ = try ArchConfig.resolved(forFamily: .k2Horizon, directoryURL: dir)
        } catch let thrown as ArchResolutionError {
            failure = thrown
        }
        guard case .manifestOmitsField(_, let field) = failure else {
            Issue.record("expected a refused missing field, got \(String(describing: failure))")
            return
        }
        #expect(field == "attentionGate")
    }
}
