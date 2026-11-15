import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanRepackCore

@Suite struct ManifestReaderTests {

    /// The runtime loads `manifest.json` with its own bound and the repacker
    /// validates the same file with another. While the runtime's was 4 MiB and
    /// the repacker's 64 MiB, an install passed `--verify-install` and then
    /// refused to load -- which is exactly what happened to KAT-Coder-V2.5-Dev,
    /// whose manifest is 6.25 MB. They are asserted equal rather than merely
    /// adequate: a manifest one accepts must be one the other can read.
    @Test func runtimeAndRepackerAgreeOnTheManifestCeiling() {
        #expect(ManifestReader.defaultMaxBytes == VerifiedInstallTool.metadataMaxBytes)
        #expect(
            VerifiedInstallReceiptReader.defaultMaxBytes == VerifiedInstallTool.metadataMaxBytes)
    }

    /// The ceiling has to apply to the *peek* as well as to the load, and that
    /// is the half the equality above cannot see: `peekFamily` read its own
    /// literal 4 MiB while the load read the shared 64 MiB, so a KAT-Coder
    /// manifest (6.25 MB, the very file that motivated the raise) was refused
    /// before the ceiling that accommodates it was ever consulted. It is not a
    /// cosmetic divergence — `peekFamily` is the first thing `Engine.load`
    /// (`sources/TinyTitanLib/Engine.swift:66`) and the server's session
    /// (`ServerModelSession+Loading.swift:98`) do, and `Engine` maps any error
    /// from it to `TinyTitanError.notAnInstall`, so a real install was
    /// reported as not being one.
    @Test func theFamilyPeekUsesTheManifestCeilingTheLoadUses() throws {
        let (dir, _) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("manifest.json")
        let object = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any])
        var files = try #require(object["files"] as? [String: Any])
        // Pad the tensor table the way a real one grows: many small per-tensor
        // entries, not one big blob, so the file is a plausible manifest.
        for entry in 1...60_000 {
            files["packed_experts/pad_\(entry).bin"] = [
                "size": entry, "sha256": String(repeating: "0", count: 64),
            ]
        }
        var padded = object
        padded["files"] = files
        let data = try JSONSerialization.data(withJSONObject: padded)
        try data.write(to: url)
        let ceiling = Int(ManifestReader.defaultMaxBytes)
        #expect(
            data.count > 4 * 1024 * 1024 && data.count < ceiling,
            "padding is \(data.count) bytes, which is not between 4 MiB and \(ceiling)")
        var peekError: Error?
        do {
            _ = try ManifestReader.peekFamily(directoryURL: dir)
        } catch {
            peekError = error
        }
        let bytes = data.count
        let reason = String(describing: peekError)
        #expect(peekError == nil, "peekFamily refused a \(bytes)-byte manifest: \(reason)")
    }

    /// Build a manifest dictionary for a 2-layer toy ArchConfig and write it
    /// into a temp directory. Returns the directory URL and the toy config.
    static func writeToyManifest(
        _ overrides: [String: Any] = [:],
        flags: [String: Bool] = [
            "streamingPresent": true,
            "turboQuantKV": false,
            "aneSharedExpert": false,
        ],
        archOverrides: [String: Any] = [:],
        filesOverride: [String: [String: Any]]? = nil,
        config: ArchConfig = .qwenToy()
    ) throws
        -> (URL, ArchConfig)
    {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let toy = config
        var archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads,
            "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim,
            "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize,
            "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta,
            "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers,
            "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        for (k, v) in archOverrides { archDict[k] = v }

        var files: [String: [String: Any]]
        if let f = filesOverride {
            files = f
        } else {
            files = [
                "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": [
                    "size": 1024, "sha256": String(repeating: "0", count: 64),
                ],
            ]
            for L in 0..<toy.numLayers {
                files["packed_experts/layer_\(L).bin"] = [
                    "size": 16384, "sha256": String(repeating: "0", count: 64),
                ]
            }
        }

        var root: [String: Any] = [
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": flags,
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": 16384,
        ]
        for (k, v) in overrides { root[k] = v }

        let data = try JSONSerialization.data(
            withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return (dir, toy)
    }

    static func quant(
        sharedExpertBits: Int = 4,
        routerBits: Int = 8
    ) -> [String: Any] {
        func slot(_ bits: Int) -> [String: Any] {
            [
                "weightBits": bits,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": Quantization.groupSize,
            ]
        }
        return [
            "embedding": slot(4),
            "attention": slot(4),
            "router": slot(routerBits),
            "sharedExpert": slot(sharedExpertBits),
            "routedExpert": slot(4),
        ]
    }

    @Test func loadsValidManifest() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.magic == "SSDAI")
        #expect(m.numLayers == toy.numLayers)
        #expect(m.expertStride == 16384)
    }

    /// One quant entry per tensor, not only the five slots.
    ///
    /// The runtime resolves a tensor's width through its override before the
    /// slot's. That is how a dense install keeps `mlp.*` at 4 bits inside an
    /// 8-bit slot, and how the hyper-connection gates, the PLE key projection
    /// and the indexer's keys would be promoted for ~10 MB rather than taking
    /// the whole attention block — 61% of the active parameters — to 8 bits.
    @Test func aPerTensorQuantEntryBecomesAnOverride() throws {
        let stem = "language_model.model.layers.0.self_attn.indexer.index_q_proj"
        var quant = Self.quant()
        quant[stem] = Self.quantSlot(8)
        let (dir, toy) = try Self.writeToyManifest(["quant": quant])
        defer { try? FileManager.default.removeItem(at: dir) }

        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.quantOverrides == [stem: 8])
        // The five slots are unchanged: an override says one tensor differs,
        // not that the build did.
        #expect(m.quant?.attention.weightBits == 4)
    }

    /// The guard between an override's width and a kernel's arithmetic.
    ///
    /// Every reader that takes a width as arithmetic -- `32 / bits` lanes, the
    /// packed-row extents `requireAffine` cross-checks -- would otherwise unpack
    /// the same bytes *wrongly* rather than fail: a 6-bit tensor gets five lanes
    /// per word and answers fluently and wrongly, and the GPU GEMVs assert
    /// `[4, 8]` and abort the process instead. `SSDAIManifestQuantV1.init(from:)`
    /// refuses such a width while decoding, which is the difference between a
    /// load error and either of those. The message has to name the tensor: the
    /// overrides are keyed by stem, so "the quant block is wrong" sends nobody
    /// anywhere.
    @Test func anOverrideAtAWidthNoKernelImplementsIsRefusedAtDecode() throws {
        let stem = "language_model.model.layers.0.mlp.gate_proj"
        for bits in [6, 3, 16, 2] {
            var quant = Self.quant()
            quant[stem] = Self.quantSlot(bits)
            let (dir, toy) = try Self.writeToyManifest(["quant": quant])
            defer { try? FileManager.default.removeItem(at: dir) }
            do {
                _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
                Issue.record("a \(bits)-bit override must not decode")
            } catch let error as ModelError {
                let text = "\(error)"
                #expect(text.contains(stem), "the refusal must name \(stem), got \(text)")
                #expect(
                    text.contains("\(bits) bits"),
                    "the refusal must name \(bits) bits, got \(text)")
            } catch {
                Issue.record("unexpected error \(error)")
            }
        }
    }

    /// A promoted tensor is a *name*, not a width.
    ///
    /// `weightBits: 16` cannot be the spelling: every reader that takes a
    /// width as arithmetic (`32 / bits` lanes, packed-row extents) would unpack
    /// the same bytes wrongly, and `anOverrideAtAWidthNoKernelImplementsIsRefusedAtDecode`
    /// pins that refusal. The list says which tensors the slot describes but
    /// does not pack; the slots keep describing the packing.
    @Test func aPromotedTensorIsAListNotAWidth() throws {
        let stem = "language_model.model.layers.0.mlp.shared_expert.gate_proj"
        var quant = Self.quant()
        quant["promoted"] = [stem]
        let (dir, toy) = try Self.writeToyManifest(["quant": quant])
        defer { try? FileManager.default.removeItem(at: dir) }

        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.quantPromoted == [stem])
        // The five slots are unchanged, and nothing became an override.
        #expect(m.quantOverrides.isEmpty)
        #expect(m.quant?.sharedExpert.weightBits == 4)
        #expect(m.quant?.attention.weightBits == 4)
    }

    /// A promoted role resolves to 16 for kernel selection, and only then.
    ///
    /// 16 is the runtime's name for "unquantized"; an affine dispatcher is
    /// never built for it, and the fused kernels refuse it, which is what keeps
    /// a bf16 tensor out of the nibble readers.
    @Test func aPromotedRoleResolvesToSixteen() {
        let stem = "language_model.model.layers.0.mlp.shared_expert.gate_proj"
        #expect(
            ManifestQuant.roleWeightBits(
                roleSuffix: ".mlp.shared_expert.gate_proj",
                overrides: [:], fallback: 8, promoted: [stem]) == 16)
        // Same role, no marker: the slot's width, as before.
        #expect(
            ManifestQuant.roleWeightBits(
                roleSuffix: ".mlp.shared_expert.gate_proj",
                overrides: [:], fallback: 8, promoted: []) == 8)
        // A different role is untouched by someone else's promotion.
        #expect(
            ManifestQuant.roleWeightBits(
                roleSuffix: ".self_attn.q_proj",
                overrides: [:], fallback: 4, promoted: [stem]) == 4)
    }

    @Test func aTensorResolvesThroughItsOverrideThenItsSlot() throws {
        let quant = try Self.decodedQuant()
        let stem = "language_model.model.layers.0.self_attn.indexer.index_q_proj"

        #expect(
            quant.slot(
                forTensorNamed: "\(stem).weight",
                overrides: [stem: 8],
                fallback: quant.attention
            ).weightBits == 8)
        #expect(
            quant.slot(
                forTensorNamed: "\(stem).weight",
                overrides: [:],
                fallback: quant.attention
            ).weightBits == 4)
        // Restating the slot's width is not an override.
        #expect(
            quant.slot(
                forTensorNamed: "\(stem).weight",
                overrides: [stem: 4],
                fallback: quant.attention
            ).weightBits == 4)
    }

    /// The three families whose weights read the attention slot until a
    /// manifest overrides them, resolved the way `Model` resolves them.
    @Test func aRoleResolvesBySuffixForTheSlotReadingFamilies() throws {
        let quant = try Self.decodedQuant()
        let overrides = [
            "language_model.model.layers.0.attn_hyper_connection.block_inject_weight": 8,
            "language_model.model.layers.0.ple.key_proj": 8,
            "language_model.model.layers.0.self_attn.indexer.index_q_proj": 8,
        ]
        let fallback = quant.attention.weightBits
        #expect(
            ManifestQuant.roleWeightBits(
                roleSuffix: "hyper_connection.block_inject_weight",
                overrides: overrides, fallback: fallback) == 8)
        #expect(
            ManifestQuant.roleWeightBits(
                roleSuffix: ".ple.key_proj", overrides: overrides, fallback: fallback) == 8)
        #expect(
            ManifestQuant.roleWeightBits(
                roleSuffix: ".self_attn.indexer.index_q_proj",
                overrides: overrides, fallback: fallback) == 8)
        // A family with no override keeps the attention slot.
        #expect(
            ManifestQuant.roleWeightBits(
                roleSuffix: ".mlp.gate_proj", overrides: overrides, fallback: fallback) == 4)
    }

    static func quantSlot(_ bits: Int) -> [String: Any] {
        [
            "weightBits": bits, "scheme": "affine", "scaleType": "bf16",
            "biasType": "bf16", "groupSize": Quantization.groupSize,
        ]
    }

    private static func decodedQuant() throws -> ManifestQuant {
        let data = try JSONSerialization.data(withJSONObject: quant())
        return try JSONDecoder().decode(ManifestQuant.self, from: data)
    }

    /// No `arch.family` is declared here, so this pins the shape-inference fallback: a
    /// Qwen 3.6 toy manifest must read back as `.qwen36`.
    @Test func peekFamilyInfersFullQwenWhenNoFamilyIsDeclared() throws {
        let arch = ArchConfig.qwen36_35B_A3B
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 2, "sha256": String(repeating: "0", count: 64)],
        ]
        for layer in 0..<arch.numLayers {
            files[String(format: "packed_experts/layer_%02d.bin", layer)] = [
                "size": 1,
                "sha256": String(repeating: "0", count: 64),
            ]
        }
        let (dir, _) = try Self.writeToyManifest(
            [:],
            archOverrides: [
                "hiddenSize": arch.hiddenSize,
                "ffnIntermediate": arch.intermediateSize,
                "moeIntermediateSize": arch.moeIntermediateSize,
                "numHeads": arch.numHeads,
                "numKVHeads": arch.numKVHeads,
                "numFullKVHeads": arch.numFullKVHeads,
                "headDim": arch.headDim,
                "fullHeadDim": arch.fullHeadDim,
                "vocabSize": arch.vocabSize,
                "slidingWindow": arch.slidingWindow,
                "finalLogitSoftcap": arch.finalLogitSoftcap,
                "ropeTheta": arch.ropeTheta,
                "fullRopeTheta": arch.fullRopeTheta,
                "partialRotaryFactor": arch.partialRotaryFactor,
                "numLayers": arch.numLayers,
                "numExperts": arch.numExperts,
                "topKExperts": arch.topKExperts,
                "tieWordEmbeddings": arch.tieWordEmbeddings,
                "attentionKEqV": arch.attentionKEqV,
                "hiddenActivation": arch.hiddenActivation,
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
            ],
            filesOverride: files,
            config: arch)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try ManifestReader.peekFamily(directoryURL: dir) == .qwen36)
    }

    @Test func peekIdentityPreservesCompatibleModelID() throws {
        let modelID = "ornith-1.5-35b-a3b-4bit"
        let (dir, _) = try Self.writeToyManifest(["modelID": modelID])
        defer { try? FileManager.default.removeItem(at: dir) }

        let identity = try ManifestReader.peekIdentity(directoryURL: dir)
        #expect(identity.modelID == modelID)
        #expect(identity.family == .qwen36)
    }

    @Test func peekIdentityRejectsEmptyModelID() throws {
        let (dir, _) = try Self.writeToyManifest(["modelID": ""])
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect {
            _ = try ManifestReader.peekIdentity(directoryURL: dir)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("modelID is empty")
        }
    }

    @Test func peekFamilyDetectsMTPByArchitectureShape() throws {
        let arch = ArchConfig.qwen36MTP
        let (dir, _) = try Self.writeToyManifest(
            [:],
            archOverrides: [
                "hiddenSize": arch.hiddenSize,
                "ffnIntermediate": arch.intermediateSize,
                "moeIntermediateSize": arch.moeIntermediateSize,
                "numHeads": arch.numHeads,
                "numKVHeads": arch.numKVHeads,
                "numFullKVHeads": arch.numFullKVHeads,
                "headDim": arch.headDim,
                "fullHeadDim": arch.fullHeadDim,
                "vocabSize": arch.vocabSize,
                "slidingWindow": arch.slidingWindow,
                "finalLogitSoftcap": arch.finalLogitSoftcap,
                "ropeTheta": arch.ropeTheta,
                "fullRopeTheta": arch.fullRopeTheta,
                "partialRotaryFactor": arch.partialRotaryFactor,
                "numLayers": arch.numLayers,
                "numExperts": arch.numExperts,
                "topKExperts": arch.topKExperts,
                "tieWordEmbeddings": arch.tieWordEmbeddings,
                "attentionKEqV": arch.attentionKEqV,
                "hiddenActivation": arch.hiddenActivation,
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
            ],
            filesOverride: [
                "model_weights.bin": ["size": 1, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": [
                    "size": 2, "sha256": String(repeating: "0", count: 64),
                ],
                "packed_experts/layer_0.bin": [
                    "size": 1, "sha256": String(repeating: "0", count: 64),
                ],
            ],
            config: arch)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try ManifestReader.peekFamily(directoryURL: dir) == .qwen36MTP)
    }

    @Test func missingManifestThrowsPartialInstall() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .qwenToy())
        } throws: { error in
            if case ModelError.partialInstall = error { return true }
            return false
        }
    }

    @Test func oversizedManifestRejectsBeforeDecode() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try Data(repeating: 0x20, count: 64).write(to: manifestURL)

        #expect {
            _ = try ManifestReader.load(
                directoryURL: dir,
                expecting: toy,
                maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }

    @Test func wrongMagicThrowsNotASSDAIDirectory() throws {
        let (dir, toy) = try Self.writeToyManifest(["magic": "NOT_SSDAI"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.notASSDAIDirectory) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    /// The rename's compatibility contract: every install built before 5.15
    /// carries `GTURBO`, and it must keep loading.
    ///
    /// Rewriting the field instead would invalidate each install's receipt —
    /// `verified-install.json` binds the manifest's digest and the directory
    /// path — so a legacy read is what makes this a rename rather than a repack
    /// of every install in existence.
    @Test func legacyGTURBOMagicStillLoads() throws {
        let (dir, toy) = try Self.writeToyManifest(["magic": "GTURBO"])
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.magic == "GTURBO", "the legacy magic is read as written")
        #expect(m.numLayers == toy.numLayers)
    }

    @Test func versionTwoThrowsUnsupportedVersion() throws {
        let (dir, toy) = try Self.writeToyManifest(["versionMajor": 2])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unsupportedVersion(let maj, _) = error { return maj == 2 }
            return false
        }
    }

    @Test func unknownFlagThrowsUnknownFlag() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: [
            "streamingPresent": true,
            "newFangledOption": true,
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unknownFlag(let n) = error { return n == "newFangledOption" }
            return false
        }
    }

    @Test func removedTurboQuantFlagIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: [
            "streamingPresent": true,
            "turboQuantKV": true,
            "aneSharedExpert": false,
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("removed TurboQuant KV")
        }
    }

    @Test func productionManifestRequiresQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(config: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.quant is required")
        }
    }

    @Test func productionManifestAcceptsInt4SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 4)],
            config: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 4)
    }

    @Test func productionManifestAcceptsHistoricalInt8SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 8)],
            config: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 8)
    }

    @Test func productionManifestRejectsUnsupportedQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 3, routerBits: 4)],
            config: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    @Test func archMismatchThrowsArchMismatch() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["hiddenSize": 4096])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.archMismatch(let field, _, _) = error else { return false }
            return field == "hiddenSize"
        }
    }

    @Test func nonPageAlignedExpertStrideThrows() throws {
        let (dir, toy) = try Self.writeToyManifest(["expertStride": 1024])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.expertStrideNotPageAligned = error { return true }
            return false
        }
    }

    @Test func missingLayerFileThrowsMissingFile() throws {
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": [
                "size": 1024, "sha256": String(repeating: "0", count: 64),
            ],
            // intentionally do not list layer_0.bin or layer_1.bin
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.missingFile = error { return true }
            return false
        }
    }

    @Test func acceptsZeroPaddedLayerFilenames() throws {
        // Writer emits packed_experts/layer_%02d.bin; loader should accept either form.
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": [
                "size": 1024, "sha256": String(repeating: "0", count: 64),
            ],
            "packed_experts/layer_00.bin": [
                "size": 16384, "sha256": String(repeating: "0", count: 64),
            ],
            "packed_experts/layer_01.bin": [
                "size": 16384, "sha256": String(repeating: "0", count: 64),
            ],
            "packed_experts/layer_02.bin": [
                "size": 16384, "sha256": String(repeating: "0", count: 64),
            ],
            "packed_experts/layer_03.bin": [
                "size": 16384, "sha256": String(repeating: "0", count: 64),
            ],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.numLayers == toy.numLayers)
    }
}

extension ArchConfig {
    /// Tiny baseline used across the loader tests. 4 layers alternating
    /// gated-DeltaNet linear (mask 2) and full attention (mask 1), 8 experts
    /// (top-8 for the fixed-top-k decode kernels), gated shared expert,
    /// untied lm_head. Numbers are intentionally toy but respect every kernel
    /// divisibility constraint (D % 64, keyHeadDim % 32, even rotaryDim).
    static func qwenToy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 128,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 32,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 1024,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 4,
            numExperts: 8,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 1, 2, 1],
            hiddenActivation: "silu",
            family: "qwen36",
            attnOutputGate: true,
            attentionScale: 0.125,  // 32^-0.5
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 4,
                keyHeadDim: 32, valueHeadDim: 32,
                convKernelSize: 4)
        )
    }
}

/// Validation of the family-extension geometry (hyper-connections, the QSA
/// indexer, PLE). These fields are optional so older manifests still load, but
/// when a manifest declares them they must match the architecture the runtime
/// would actually execute -- otherwise a checkpoint with, say, a different
/// hyper-connection rank runs silently against the wrong constants and
/// produces plausible nonsense.
@Suite("Manifest extension geometry")
struct ManifestExtensionGeometryTests {
    private static func toyWithExtensions() -> ArchConfig {
        var base = ArchConfig.qwenToy()
        base = ArchConfig(
            hiddenSize: base.hiddenSize, intermediateSize: base.intermediateSize,
            moeIntermediateSize: base.moeIntermediateSize, numHeads: base.numHeads,
            numKVHeads: base.numKVHeads, numFullKVHeads: base.numFullKVHeads,
            headDim: base.headDim, fullHeadDim: base.fullHeadDim,
            vocabSize: base.vocabSize, slidingWindow: base.slidingWindow,
            finalLogitSoftcap: base.finalLogitSoftcap, ropeTheta: base.ropeTheta,
            fullRopeTheta: base.fullRopeTheta,
            partialRotaryFactor: base.partialRotaryFactor,
            numLayers: base.numLayers, numExperts: base.numExperts,
            topKExperts: base.topKExperts,
            tieWordEmbeddings: base.tieWordEmbeddings,
            attentionKEqV: base.attentionKEqV,
            fullAttentionLayerMask: base.fullAttentionLayerMask,
            hiddenActivation: base.hiddenActivation, family: base.family,
            attnOutputGate: base.attnOutputGate,
            attentionScale: base.attentionScale,
            embeddingScaledBySqrtHidden: base.embeddingScaledBySqrtHidden,
            routerScaled: base.routerScaled,
            ffnSandwichNorms: base.ffnSandwichNorms,
            sharedExpertGated: base.sharedExpertGated,
            ropeNeoxSubdim: base.ropeNeoxSubdim,
            linearAttention: base.linearAttention,
            hyperConnections: HyperConnectionConfig(count: 4, lowRank: 320),
            sparseIndexer: SparseIndexerConfig(
                numHeads: 4, numKVHeads: 1,
                headDim: 128, budget: 2048,
                compressRatio: 4),
            ple: PLEConfig(
                layerIndices: [1], embedDim: 2560,
                convKernelSize: 4, ngramSize: 3,
                vocabSizeBase: 20_000_000, headsPerNgram: 8,
                vocabDivisor: 128, seed: 1234),
            routerNormTopK: true, quantGroupSize: 64)
        return base
    }

    @Test("A manifest that omits the extension fields still validates")
    func absentFieldsAreAccepted() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
    }

    @Test("Matching extension fields validate")
    func matchingFieldsAccepted() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            archOverrides: [
                "hcCount": 4, "hcLowRank": 320,
                "indexerBudget": 2048, "quantGroupSize": 64,
                "pleLayerIndices": [1],
            ],
            config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
    }

    @Test("A disagreeing hyper-connection rank is rejected, not ignored")
    func mismatchedRankRejected() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            archOverrides: ["hcLowRank": 256], config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
        }
    }

    @Test("A disagreeing quantization group size is rejected")
    func mismatchedGroupSizeRejected() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            archOverrides: ["quantGroupSize": 32], config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        // Repacking at the wrong group size would corrupt every weight, so
        // this is the single most important field in this set.
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
        }
    }

    @Test("A disagreeing PLE layer set is rejected")
    func mismatchedPLERejected() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            archOverrides: ["pleLayerIndices": [2]], config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
        }
    }

    /// The 1.0 rule was "validate what is declared", which left the guard
    /// biting only on a manifest that happened to carry the keys: a writer that
    /// emitted no block was validating nothing at all, and a family sharing the
    /// target's hyper-connections was exactly that case. From 1.1, an
    /// architecture that has the geometry refuses a manifest without it.
    @Test("A 1.1 manifest for a family with the geometry must declare it")
    func extensionGeometryIsMandatoryFromVersion11() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            ["versionMinor": 1], config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        var field = ""
        var actual = ""
        do {
            _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
            Issue.record("a 1.1 manifest with no extension block loaded")
        } catch let error as ModelError {
            // Naming the field, not merely failing: an operator whose install is
            // refused has to be able to tell a stale manifest from a wrong one.
            if case ModelError.archMismatch(let f, _, let a) = error {
                field = f
                actual = a
            }
        }
        #expect(field == "hcCount")
        #expect(actual == "not declared")
    }

    /// The other half of the rule: 1.0 installs are still read, because the
    /// manifest is hash-bound into `verified-install.json` and is never
    /// rewritten in place. `absentFieldsAreAccepted` is this case at minor 0.
    @Test("A complete 1.1 block validates")
    func completeBlockAcceptedAtVersion11() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            ["versionMinor": 1],
            archOverrides: [
                "hcCount": 4, "hcLowRank": 320,
                "indexerNumHeads": 4, "indexerNumKVHeads": 1,
                "indexerHeadDim": 128, "indexerBudget": 2048,
                "indexerCompressRatio": 4,
                "pleLayerIndices": [1], "pleEmbedDim": 2560,
                "pleConvKernelSize": 4, "pleNgramSize": 3,
                "pleVocabSizeBase": 20_000_000, "pleHeadsPerNgram": 8,
                "pleVocabDivisor": 128,
                "routerNormTopK": true, "quantGroupSize": 64,
            ],
            config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
    }

    @Test("A 1.1 manifest for a family without the geometry needs no block")
    func plainFamilyNeedsNoBlockAtVersion11() throws {
        let (dir, _) = try ManifestReaderTests.writeToyManifest(["versionMinor": 1])
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try ManifestReader.load(directoryURL: dir, expecting: .qwenToy())
    }
}
