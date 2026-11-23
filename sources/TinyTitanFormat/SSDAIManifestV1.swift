import Foundation

package struct SSDAIManifestFileV1: Codable, Equatable, Sendable {
    package let size: UInt64
    package let sha256: String

    package init(size: UInt64, sha256: String) {
        self.size = size
        self.sha256 = sha256
    }
}

package struct SSDAIManifestArchV1: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let ffnIntermediate: Int
    package let moeIntermediateSize: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let numFullKVHeads: Int
    package let headDim: Int
    package let fullHeadDim: Int
    package let vocabSize: Int
    package let slidingWindow: Int
    package let finalLogitSoftcap: Double
    package let ropeTheta: Double
    package let fullRopeTheta: Double
    package let partialRotaryFactor: Double
    package let numLayers: Int
    package let numExperts: Int
    package let topKExperts: Int
    package let tieWordEmbeddings: Bool
    package let attentionKEqV: Bool
    package let hiddenActivation: String
    package let fullAttentionLayerMask: [Int]

    /// The family the payload was repacked for. Absent in the earliest
    /// manifests, which predate more than one family; the reader falls back to
    /// inferring from layer shape when it is missing.
    package let family: String?

    // Family extension geometry; absent in manifests written before these
    // families existed, validated by the reader whenever present.
    package let hcCount: Int?
    package let hcLowRank: Int?
    package let indexerNumHeads: Int?
    package let indexerNumKVHeads: Int?
    package let indexerHeadDim: Int?
    package let indexerBudget: Int?
    package let indexerCompressRatio: Int?
    package let pleLayerIndices: [Int]?
    package let pleEmbedDim: Int?
    package let pleConvKernelSize: Int?
    package let pleNgramSize: Int?
    package let pleVocabSizeBase: Int?
    package let pleHeadsPerNgram: Int?
    package let pleVocabDivisor: Int?
    package let routerNormTopK: Bool?
    package let quantGroupSize: Int?

    /// Gated-DeltaNet geometry. Optional for the same reason as the block
    /// above: manifests written before a reader needed them do not carry them.
    /// The writer has always emitted these keys, so they were present on disk
    /// and simply not decoded.
    package let linearNumKHeads: Int?
    package let linearNumVHeads: Int?
    package let linearKeyHeadDim: Int?
    package let linearValueHeadDim: Int?
    package let linearConvKernelSize: Int?

    /// The layer conventions, which the writer has always emitted and no reader
    /// decoded until a family without an architecture preset needed them (the
    /// dense Qwen 3.5 models). Optional so every earlier manifest still
    /// decodes.
    package let attnOutputGate: Bool?
    package let attentionScale: Double?
    package let embeddingScaledBySqrtHidden: Bool?
    package let routerScaled: Bool?
    package let ffnSandwichNorms: Bool?
    package let sharedExpertGated: Bool?
    package let ropeNeoxSubdim: Bool?

    /// K2-Horizon's layer shape. Written only for that family, but optional
    /// here so every earlier manifest still decodes; a reader that needs them
    /// refuses a K2 manifest that omits one by name rather than assuming.
    package let normGroups: Int?
    package let attentionGate: String?
    package let routerHasBias: Bool?
    package let routerScaling: Double?
    package let movaNumExperts: Int?
    package let movaTopKExperts: Int?
    package let movaRouterScaling: Double?
    package let movaRouterHasBias: Bool?
    package let movaActivation: String?
    package let denseLayerIndices: [Int]?
    /// A variant's own geometry (`SSDAIVariantGeometryV1`); absent for every
    /// family that predates it.
    package let variantGeometry: SSDAIVariantGeometryV1?

    package init(
        hiddenSize: Int, ffnIntermediate: Int, moeIntermediateSize: Int,
        numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
        headDim: Int, fullHeadDim: Int, vocabSize: Int,
        slidingWindow: Int, finalLogitSoftcap: Double,
        ropeTheta: Double, fullRopeTheta: Double,
        partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
        topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
        hiddenActivation: String, fullAttentionLayerMask: [Int],
        family: String? = nil,
        hcCount: Int? = nil,
        hcLowRank: Int? = nil,
        indexerNumHeads: Int? = nil,
        indexerNumKVHeads: Int? = nil,
        indexerHeadDim: Int? = nil,
        indexerBudget: Int? = nil,
        indexerCompressRatio: Int? = nil,
        pleLayerIndices: [Int]? = nil,
        pleEmbedDim: Int? = nil,
        pleConvKernelSize: Int? = nil,
        pleNgramSize: Int? = nil,
        pleVocabSizeBase: Int? = nil,
        pleHeadsPerNgram: Int? = nil,
        pleVocabDivisor: Int? = nil,
        routerNormTopK: Bool? = nil,
        quantGroupSize: Int? = nil,
        linearNumKHeads: Int? = nil,
        linearNumVHeads: Int? = nil,
        linearKeyHeadDim: Int? = nil,
        linearValueHeadDim: Int? = nil,
        linearConvKernelSize: Int? = nil,
        attnOutputGate: Bool? = nil,
        attentionScale: Double? = nil,
        embeddingScaledBySqrtHidden: Bool? = nil,
        routerScaled: Bool? = nil,
        ffnSandwichNorms: Bool? = nil,
        sharedExpertGated: Bool? = nil,
        ropeNeoxSubdim: Bool? = nil,
        normGroups: Int? = nil,
        attentionGate: String? = nil,
        routerHasBias: Bool? = nil,
        routerScaling: Double? = nil,
        movaNumExperts: Int? = nil,
        movaTopKExperts: Int? = nil,
        movaRouterScaling: Double? = nil,
        movaRouterHasBias: Bool? = nil,
        movaActivation: String? = nil,
        denseLayerIndices: [Int]? = nil,
        variantGeometry: SSDAIVariantGeometryV1? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.ffnIntermediate = ffnIntermediate
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.hiddenActivation = hiddenActivation
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.family = family
        self.hcCount = hcCount
        self.hcLowRank = hcLowRank
        self.indexerNumHeads = indexerNumHeads
        self.indexerNumKVHeads = indexerNumKVHeads
        self.indexerHeadDim = indexerHeadDim
        self.indexerBudget = indexerBudget
        self.indexerCompressRatio = indexerCompressRatio
        self.pleLayerIndices = pleLayerIndices
        self.pleEmbedDim = pleEmbedDim
        self.pleConvKernelSize = pleConvKernelSize
        self.pleNgramSize = pleNgramSize
        self.pleVocabSizeBase = pleVocabSizeBase
        self.pleHeadsPerNgram = pleHeadsPerNgram
        self.pleVocabDivisor = pleVocabDivisor
        self.routerNormTopK = routerNormTopK
        self.quantGroupSize = quantGroupSize
        self.linearNumKHeads = linearNumKHeads
        self.linearNumVHeads = linearNumVHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelSize = linearConvKernelSize
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
        self.normGroups = normGroups
        self.attentionGate = attentionGate
        self.routerHasBias = routerHasBias
        self.routerScaling = routerScaling
        self.movaNumExperts = movaNumExperts
        self.movaTopKExperts = movaTopKExperts
        self.movaRouterScaling = movaRouterScaling
        self.movaRouterHasBias = movaRouterHasBias
        self.movaActivation = movaActivation
        self.denseLayerIndices = denseLayerIndices
        self.variantGeometry = variantGeometry
    }
}

package struct SSDAIManifestV1: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: SSDAIManifestArchV1
    package let quant: SSDAIManifestQuantV1?
    package let files: [String: SSDAIManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64

    package init(
        magic: String = SSDAIFormatV1.magic,
        versionMajor: Int = SSDAIFormatV1.versionMajor,
        versionMinor: Int = SSDAIFormatV1.versionMinor,
        flags: [String: Bool], modelID: String,
        sourceSnapshotHash: String?, arch: SSDAIManifestArchV1,
        quant: SSDAIManifestQuantV1?,
        files: [String: SSDAIManifestFileV1],
        expertsPerLayer: Int, numLayers: Int, expertStride: UInt64
    ) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.flags = flags
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.arch = arch
        self.quant = quant
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
    }
}

package enum SSDAIManifestCodec {
    package static func decode(_ data: Data) throws -> SSDAIManifestV1 {
        let manifest = try decodeUnchecked(data)
        try validate(manifest)
        return manifest
    }

    package static func decodeUnchecked(_ data: Data) throws -> SSDAIManifestV1 {
        let manifest: SSDAIManifestV1
        do { manifest = try JSONDecoder().decode(SSDAIManifestV1.self, from: data) } catch {
            throw TinyTitanFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
        return manifest
    }

    package static func validate(_ manifest: SSDAIManifestV1) throws {
        guard SSDAIFormatV1.isSupportedMagic(manifest.magic) else {
            throw TinyTitanFormatError.invalid(
                field: "manifest.magic", reason: "expected SSDAI (or the legacy GTURBO)")
        }
        guard manifest.versionMajor == SSDAIFormatV1.versionMajor,
            manifest.versionMinor >= 0
        else {
            throw TinyTitanFormatError.invalid(
                field: "manifest.version", reason: "unsupported version")
        }
        for flag in manifest.flags.keys where !SSDAIFormatV1.knownFlags.contains(flag) {
            throw TinyTitanFormatError.invalid(
                field: "manifest.flags.\(flag)", reason: "unknown v1 flag")
        }
        // A dense payload has no routed experts at all -- the planner writes
        // expertsPerLayer 0 and expertStride 0 for one -- so "greater than
        // zero" is the wrong test for it. Every other family still needs both.
        let isDense = manifest.arch.family == "qwen3_5_dense"
        guard !manifest.modelID.isEmpty,
            manifest.numLayers > 0,
            isDense
                ? (manifest.expertsPerLayer == 0 && manifest.expertStride == 0)
                : (manifest.expertsPerLayer > 0 && manifest.expertStride > 0),
            manifest.expertStride % SSDAIFormatV1.alignmentBytes == 0
        else {
            throw TinyTitanFormatError.invalid(
                field: "manifest", reason: "invalid dimensions or stride")
        }
        guard manifest.arch.numLayers == manifest.numLayers,
            manifest.arch.numExperts == manifest.expertsPerLayer
        else {
            throw TinyTitanFormatError.invalid(
                field: "manifest.arch", reason: "dimensions disagree with streaming metadata")
        }
        let arch = manifest.arch
        // `moeIntermediateSize` is 0 for a dense model, which has no
        // per-expert FFN; its FFN width is `ffnIntermediate`.
        guard arch.hiddenSize > 0, arch.ffnIntermediate > 0,
            isDense || arch.moeIntermediateSize > 0, arch.numHeads > 0,
            arch.numKVHeads > 0, arch.numFullKVHeads > 0,
            arch.headDim > 0, arch.fullHeadDim > 0,
            arch.vocabSize > 0, arch.slidingWindow >= 0,
            isDense || (arch.topKExperts > 0 && arch.topKExperts <= arch.numExperts),
            arch.finalLogitSoftcap.isFinite,
            arch.ropeTheta.isFinite, arch.ropeTheta > 0,
            arch.fullRopeTheta.isFinite, arch.fullRopeTheta > 0,
            arch.partialRotaryFactor.isFinite,
            arch.partialRotaryFactor >= 0, arch.partialRotaryFactor <= 1,
            !arch.hiddenActivation.isEmpty,
            arch.fullAttentionLayerMask.count == arch.numLayers,
            arch.fullAttentionLayerMask.allSatisfy({ $0 == 0 || $0 == 1 || $0 == 2 })
        else {
            throw TinyTitanFormatError.invalid(
                field: "manifest.arch", reason: "invalid architecture values")
        }
        try arch.variantGeometry?.validate(field: "manifest.arch.variantGeometry")
        if let quant = manifest.quant {
            for (name, slot) in [
                ("embedding", quant.embedding),
                ("attention", quant.attention),
                ("router", quant.router),
                ("sharedExpert", quant.sharedExpert),
                ("routedExpert", quant.routedExpert),
            ] {
                guard slot.weightBits > 0, slot.weightBits <= 32,
                    slot.groupSize > 0,
                    !slot.scheme.isEmpty, !slot.scaleType.isEmpty,
                    !slot.biasType.isEmpty
                else {
                    throw TinyTitanFormatError.invalid(
                        field: "manifest.quant.\(name)", reason: "invalid quantization values")
                }
            }
        }
        let reservedFiles = SSDAIInstallFileNames.reserved
        let filePaths = manifest.files.keys.sorted()
        var canonicalPaths: [String: String] = [:]
        for path in filePaths {
            try SSDAIPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = SSDAIPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                !reservedFiles.contains(key),
                !reservedFiles.contains(where: { key.hasPrefix("\($0)/") })
            else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
            guard let entry = manifest.files[path] else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "missing entry")
            }
            guard entry.sha256.count == 64,
                entry.sha256.unicodeScalars.allSatisfy({ scalar in
                    ("0"..."9").contains(Character(String(scalar)))
                        || ("a"..."f").contains(Character(String(scalar)))
                        || ("A"..."F").contains(Character(String(scalar)))
                })
            else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.files.\(path).sha256",
                    reason: "expected 64 hexadecimal characters")
            }
        }
        for (key, path) in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                let ancestor = components.joined(separator: "/")
                if canonicalPaths[ancestor] != nil {
                    throw TinyTitanFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "file path collides with a directory prefix")
                }
            }
        }
    }

}
