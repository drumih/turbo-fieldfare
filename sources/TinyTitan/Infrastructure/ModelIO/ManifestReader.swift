import Foundation
import TinyTitanFormat

public enum ManifestReader {
    /// Weight widths this build accepts. 6-bit was withdrawn: non-power-of-two
    /// packing measured 46.8 GB/s against 60 for 4-bit and 8-bit, and a 26 GB
    /// model does not fit a 24 GB machine.
    public static let supportedWeightBits: Set<Int> = [4, 8]

    /// Bound on `manifest.json`, which scales with the install's tensor count:
    /// its `quant` table carries one entry per tensor whose width differs from
    /// the install's base. A routed-expert model ships that table per expert
    /// role, so KAT-Coder-V2.5-Dev's manifest is 6.25 MB against the ~43 KB of
    /// the packed-expert installs, and a 4 MiB bound here refused to load an
    /// install `--verify-install` had just validated against its own 64 MiB cap
    /// (`VerifiedInstallTool.metadataMaxBytes`). 64 MiB is that same ceiling,
    /// which leaves the manifest room to grow about tenfold. The bound still
    /// exists: it caps the allocation before the JSON decoder makes its copy,
    /// so a corrupt or hostile file cannot allocate without limit.
    public static let defaultMaxBytes: UInt64 = 64 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = SSDAIFormatV1.knownFlags

    /// Fixed required entries. Packed-layer filenames come from layout.json and
    /// are cross-validated only after that document is decoded.
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(
        directoryURL: URL,
        expecting: ArchConfig,
        maxBytes: UInt64 = defaultMaxBytes
    ) throws -> Manifest {
        let directory = try SSDAIModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        return try decode(data: data, expecting: expecting)
    }

    package static func decode(
        data: Data,
        expecting: ArchConfig
    ) throws -> Manifest {
        let manifest: Manifest
        do {
            let wire = try SSDAIManifestCodec.decodeUnchecked(data)
            guard SSDAIFormatV1.isSupportedMagic(wire.magic) else {
                throw ModelError.notASSDAIDirectory
            }
            guard wire.versionMajor == SSDAIFormatV1.versionMajor,
                wire.versionMinor >= 0
            else {
                throw ModelError.unsupportedVersion(
                    major: wire.versionMajor,
                    minor: wire.versionMinor)
            }
            for key in wire.flags.keys where !SSDAIFormatV1.knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
            if wire.expertStride % SSDAIFormatV1.alignmentBytes != 0 {
                throw ModelError.expertStrideNotPageAligned(
                    stride: wire.expertStride,
                    pageSize: Int(SSDAIFormatV1.alignmentBytes))
            }
            try SSDAIManifestCodec.validate(wire)
            manifest = Manifest(wire: wire)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting)
        return manifest
    }

    /// Read a manifest without validating it against a GPU `ArchConfig`.
    ///
    /// `load` cross-checks the manifest's architecture against the config the
    /// runtime is about to build, which is right for a GPU install and wrong
    /// for a dense one: there is no GPU config for that family, and the CPU
    /// engine's reader takes the manifest's own facts instead. Everything the
    /// format itself guarantees -- magic, version, known flags, page-aligned
    /// expert stride, the wire-level codec rules -- is still checked here.
    public static func read(
        directoryURL: URL,
        maxBytes: UInt64 = defaultMaxBytes
    ) throws -> Manifest {
        let directory = try SSDAIModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let manifest: Manifest
        do {
            let wire = try SSDAIManifestCodec.decodeUnchecked(data)
            guard SSDAIFormatV1.isSupportedMagic(wire.magic) else {
                throw ModelError.notASSDAIDirectory
            }
            guard wire.versionMajor == SSDAIFormatV1.versionMajor,
                wire.versionMinor >= 0
            else {
                throw ModelError.unsupportedVersion(
                    major: wire.versionMajor,
                    minor: wire.versionMinor)
            }
            for key in wire.flags.keys where !SSDAIFormatV1.knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
            if wire.expertStride % SSDAIFormatV1.alignmentBytes != 0 {
                throw ModelError.expertStrideNotPageAligned(
                    stride: wire.expertStride,
                    pageSize: Int(SSDAIFormatV1.alignmentBytes))
            }
            try SSDAIManifestCodec.validate(wire)
            manifest = Manifest(wire: wire)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        return manifest
    }

    /// Extract the architecture dimensions from the manifest without full
    /// cross-validation so it can be used to auto-select the expected
    /// configuration (e.g. by the installation probe).
    public static func peekFamily(directoryURL: URL) throws -> ModelFamily {
        try peekIdentity(directoryURL: directoryURL).family
    }

    /// Read the manifest's model identity and compatible runtime family without
    /// mapping weights or creating a Metal device.
    public static func peekIdentity(directoryURL: URL) throws -> ManifestIdentity {
        let directory = try SSDAIModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata(
            "manifest.json", maxBytes: Self.defaultMaxBytes)
        let wire = try JSONDecoder().decode(SSDAIManifestV1.self, from: data)
        guard !wire.modelID.isEmpty else {
            throw ModelError.indexCorrupt(detail: "manifest modelID is empty")
        }
        let bits = wire.quant?.routedExpert.weightBits ?? 4
        switch wire.arch.hiddenActivation {
        case "silu":
            break
        default:
            throw ModelError.unsupportedArchitecture(
                detail: "hiddenActivation=\(wire.arch.hiddenActivation)")
        }
        // Prefer what the manifest declares. Inferring family from layer shape
        // only worked while the families differed in shape; it silently
        // reports qwen36 for anything it does not recognise, which is how a
        // Qwen3.8-Flash-Next payload gets loaded against the wrong
        // architecture and fails on a hidden-size mismatch rather than being
        // identified.
        if let declared = wire.arch.family,
            let family = ModelFamily(rawValue: declared)
        {
            return ManifestIdentity(
                modelID: wire.modelID, family: family,
                weightBits: bits)
        }
        let mtp = ArchConfig.qwen36MTP
        if wire.arch.numLayers == mtp.numLayers,
            wire.arch.slidingWindow == mtp.slidingWindow,
            wire.arch.fullAttentionLayerMask == mtp.fullAttentionLayerMask.map(Int.init)
        {
            return ManifestIdentity(
                modelID: wire.modelID, family: .qwen36MTP,
                weightBits: bits)
        }
        return ManifestIdentity(
            modelID: wire.modelID, family: .qwen36,
            weightBits: bits)
    }

    static func validate(
        _ m: Manifest,
        against expected: ArchConfig
    ) throws {
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, minor: m.versionMinor, expected: expected)
        if let quant = m.quant {
            try validateQuant(quant, family: expected.family)
        } else if expected.numLayers == ArchConfig.qwen36_35B_A3B.numLayers,
            expected.hiddenSize == ArchConfig.qwen36_35B_A3B.hiddenSize
        {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required for the production architecture")
        }
        for f in requiredFiles where m.files[f] == nil {
            throw ModelError.missingFile(name: f)
        }
        // Validate that all expected layer files are listed in the manifest.
        // Accept both `layer_0.bin` and `layer_00.bin` naming conventions.
        //
        // Only when the install packs experts at all. A dense model has
        // `expertsPerLayer: 0` and an empty layout -- there is nothing streamed
        // per layer to name -- and requiring the files anyway refused every
        // dense install with a message about a file the format never promised
        // to write. A mixed install (K2's three dense layers followed by 45
        // sparse ones) is the same case per layer: its dense layers carry no
        // expert file, and `arch.denseLayerIndices` is where the install says
        // which ones. The layout cross-check reads the per-layer counts and is
        // what bounds every file that does exist.
        if m.expertsPerLayer > 0 {
            let dense = Set(m.arch.denseLayerIndices ?? [])
            for L in 0..<m.numLayers where !dense.contains(L) {
                let layerFileShort = String(format: "packed_experts/layer_%d.bin", L)
                let layerFilePadded = String(format: "packed_experts/layer_%02d.bin", L)
                if m.files[layerFileShort] == nil && m.files[layerFilePadded] == nil {
                    throw ModelError.missingFile(name: layerFileShort)
                }
            }
        }
    }

    private static func validateQuant(
        _ quant: ManifestQuant,
        family: ModelFamily
    ) throws {
        let allowedRouterBits: Set<Int>
        switch family {
        case .qwen36MTP:
            allowedRouterBits = [4, 8]
        case .qwen36:
            allowedRouterBits = [8]
        case .qwen38flash, .qwen38flashMTP:
            // The community MLX checkpoints quantize the router at the model's
            // uniform width (4 or 8 bits, group 32). The draft head is
            // quantized with the target, so it inherits the same rule.
            allowedRouterBits = [4, 8]
        case .qwen4flash, .qwen4flashMTP:
            // The 4.0 Flash path inherits the 3.8 preview's uniform router
            // width until its checkpoint declares otherwise.
            allowedRouterBits = [4, 8]
        case .qwen35Dense:
            // A dense model has no router tensor at all, so there is no width
            // to constrain. The slot still has to satisfy `validateQuant`'s
            // table, so allow both rather than inventing a rule.
            allowedRouterBits = [4, 8]
        case .k2Horizon, .laguna:
            // These routers stay bf16 in the checkpoint; the install's slot
            // still has to name a width the format accepts, and the runtime
            // reads the tensor's actual dtype.
            allowedRouterBits = [4, 8]
        }
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, Self.supportedWeightBits),
            ("attention", quant.attention, Self.supportedWeightBits),
            ("router", quant.router, allowedRouterBits),
            ("sharedExpert", quant.sharedExpert, Self.supportedWeightBits),
            ("routedExpert", quant.routedExpert, Self.supportedWeightBits),
        ]
        for (name, slot, allowedBits) in slots {
            // 6-bit was withdrawn rather than deprecated, so say so instead of
            // letting a previously working model fail as "unsupported
            // quantization" with no route forward.
            if slot.weightBits == 6 {
                // Not `indexCorrupt`: the payload is intact and the user would
                // otherwise be told to re-download a file that is fine.
                throw ModelError.unsupportedArchitecture(
                    detail: """
                        6-bit models are no longer supported (\(name) is 6-bit). \
                        Its packing is not a power of two, which measured 46.8 GB/s \
                        against 60 for both 4-bit and 8-bit, and it does not fit \
                        24 GB. Install the 4-bit or 8-bit build instead.
                        """)
            }
            guard allowedBits.contains(slot.weightBits),
                slot.scheme.lowercased() == "affine",
                slot.scaleType.lowercased() == "bf16",
                slot.biasType.lowercased() == "bf16",
                slot.groupSize == Quantization.groupSize
            else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
    }

    private static func validateArch(
        _ a: ManifestArch,
        minor: Int,
        expected e: ArchConfig
    ) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T
        ) throws {
            if actual != expected {
                throw ModelError.archMismatch(
                    field: field,
                    expected: "\(expected)",
                    actual: "\(actual)")
            }
        }
        try check("hiddenSize", a.hiddenSize, e.hiddenSize)
        try check("ffnIntermediate", a.ffnIntermediate, e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads", a.numHeads, e.numHeads)
        try check("numKVHeads", a.numKVHeads, e.numKVHeads)
        try check("numFullKVHeads", a.numFullKVHeads, e.numFullKVHeads)
        try check("headDim", a.headDim, e.headDim)
        try check("fullHeadDim", a.fullHeadDim, e.fullHeadDim)
        try check("vocabSize", a.vocabSize, e.vocabSize)
        try check("slidingWindow", a.slidingWindow, e.slidingWindow)
        try check("finalLogitSoftcap", a.finalLogitSoftcap, e.finalLogitSoftcap)
        try check("ropeTheta", a.ropeTheta, e.ropeTheta)
        try check("fullRopeTheta", a.fullRopeTheta, e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers", a.numLayers, e.numLayers)
        try check("numExperts", a.numExperts, e.numExperts)
        try check("topKExperts", a.topKExperts, e.topKExperts)
        try check("tieWordEmbeddings", a.tieWordEmbeddings, e.tieWordEmbeddings)
        try check("attentionKEqV", a.attentionKEqV, e.attentionKEqV)
        try check("hiddenActivation", a.hiddenActivation, e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check(
            "fullAttentionLayerMask",
            actualMask.description,
            e.fullAttentionLayerMask.description)

        // Extension geometry. A 1.0 manifest may legitimately omit it, because
        // installs written before these families existed carry no such keys and
        // no receipt may be rewritten in place; from
        // `extensionGeometryMandatoryFromMinor` a manifest for an architecture
        // that *has* the geometry has to declare it. Either way, what is
        // declared is compared: a checkpoint whose hyper-connection, indexer or
        // n-gram geometry differs from this runtime's is not the architecture
        // being executed, and silence here is a wrong answer rather than a
        // refusal.
        let declares = e.hyperConnections.enabled || e.sparseIndexer.enabled || e.ple.enabled
        let require = declares && minor >= SSDAIFormatV1.extensionGeometryMandatoryFromMinor
        func checkOptional<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T?, _ expected: T
        ) throws {
            guard let actual else {
                if require {
                    throw ModelError.archMismatch(
                        field: field,
                        expected: "\(expected)",
                        actual: "not declared")
                }
                return
            }
            try check(field, actual, expected)
        }
        try checkOptional("hcCount", a.hcCount, e.hyperConnections.count)
        try checkOptional("hcLowRank", a.hcLowRank, e.hyperConnections.lowRank)
        try checkOptional(
            "indexerNumHeads", a.indexerNumHeads,
            e.sparseIndexer.numHeads)
        try checkOptional(
            "indexerNumKVHeads", a.indexerNumKVHeads,
            e.sparseIndexer.numKVHeads)
        try checkOptional(
            "indexerHeadDim", a.indexerHeadDim,
            e.sparseIndexer.headDim)
        try checkOptional(
            "indexerBudget", a.indexerBudget,
            e.sparseIndexer.budget)
        try checkOptional(
            "indexerCompressRatio", a.indexerCompressRatio,
            e.sparseIndexer.compressRatio)
        try checkOptional(
            "pleLayerIndices", a.pleLayerIndices?.description,
            e.ple.layerIndices.description)
        try checkOptional("pleEmbedDim", a.pleEmbedDim, e.ple.embedDim)
        try checkOptional(
            "pleConvKernelSize", a.pleConvKernelSize,
            e.ple.convKernelSize)
        try checkOptional("pleNgramSize", a.pleNgramSize, e.ple.ngramSize)
        try checkOptional(
            "pleVocabSizeBase", a.pleVocabSizeBase,
            e.ple.vocabSizeBase)
        try checkOptional(
            "pleHeadsPerNgram", a.pleHeadsPerNgram,
            e.ple.headsPerNgram)
        try checkOptional(
            "pleVocabDivisor", a.pleVocabDivisor,
            e.ple.vocabDivisor)
        try checkOptional("routerNormTopK", a.routerNormTopK, e.routerNormTopK)
        try checkOptional("quantGroupSize", a.quantGroupSize, e.quantGroupSize)

        try validateK2Extensions(a, expected: e)
    }

    /// K2's layer shape is required, not optional: a manifest that omits the
    /// softplus gate, the 2.5 router scale, the grouped norm or the MoVA expert
    /// count would otherwise be executed under a Qwen-shaped default, which
    /// answers fluently and wrongly. The writer emits every key for this family,
    /// so absence is a corrupt or foreign manifest.
    ///
    /// Split out of `validateArch` when the ten comparisons took that body past
    /// the 120-line gate; the extension block is what grows with each family.
    private static func validateK2Extensions(
        _ a: ManifestArch,
        expected e: ArchConfig
    ) throws {
        guard e.family == .k2Horizon else { return }
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T?, _ expected: T
        ) throws {
            guard let actual else {
                throw ModelError.archMismatch(
                    field: field,
                    expected: "\(expected)",
                    actual: "not declared")
            }
            if actual != expected {
                throw ModelError.archMismatch(
                    field: field,
                    expected: "\(expected)",
                    actual: "\(actual)")
            }
        }
        try check("normGroups", a.normGroups, e.normGroups)
        try check("attentionGate", a.attentionGate, e.attentionGate.rawValue)
        try check("routerHasBias", a.routerHasBias, e.routerHasBias)
        try check("routerScaling", a.routerScaling, e.routerScaling)
        try check("movaNumExperts", a.movaNumExperts, e.mova.numExperts)
        try check("movaTopKExperts", a.movaTopKExperts, e.mova.topKExperts)
        try check(
            "movaRouterScaling", a.movaRouterScaling, e.mova.routerScaling)
        try check(
            "movaRouterHasBias", a.movaRouterHasBias, e.mova.routerHasBias)
        try check("movaActivation", a.movaActivation, e.mova.activation)
        try check(
            "denseLayerIndices", a.denseLayerIndices?.description,
            e.denseLayerIndices.description)
    }
}
