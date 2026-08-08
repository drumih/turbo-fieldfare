import Foundation

package struct GTurboManifestFileV2: Codable, Equatable, Sendable {
    package let size: UInt64
    package let sha256: String

    package init(size: UInt64, sha256: String) {
        self.size = size
        self.sha256 = sha256
    }
}

package struct GTurboManifestKDAV2: Codable, Equatable, Sendable {
    package let numHeads: Int
    package let headDim: Int
    package let convWidth: Int
    package let decayLowRankSize: Int
    package let decayProjectionSize: Int
    package let gateLowerBound: Double
    package let fullRankOutputGate: Bool

    package init(numHeads: Int, headDim: Int, convWidth: Int,
                 decayLowRankSize: Int, decayProjectionSize: Int,
                 gateLowerBound: Double, fullRankOutputGate: Bool) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.convWidth = convWidth
        self.decayLowRankSize = decayLowRankSize
        self.decayProjectionSize = decayProjectionSize
        self.gateLowerBound = gateLowerBound
        self.fullRankOutputGate = fullRankOutputGate
    }
}

package struct GTurboManifestMLAV2: Codable, Equatable, Sendable {
    package let numHeads: Int
    package let qLoraRank: Int
    package let kvLoraRank: Int
    package let qkNopeHeadDim: Int
    package let qkRopeHeadDim: Int
    package let vHeadDim: Int
    package let outputGate: Bool

    package init(numHeads: Int, qLoraRank: Int, kvLoraRank: Int,
                 qkNopeHeadDim: Int, qkRopeHeadDim: Int, vHeadDim: Int,
                 outputGate: Bool) {
        self.numHeads = numHeads
        self.qLoraRank = qLoraRank
        self.kvLoraRank = kvLoraRank
        self.qkNopeHeadDim = qkNopeHeadDim
        self.qkRopeHeadDim = qkRopeHeadDim
        self.vHeadDim = vHeadDim
        self.outputGate = outputGate
    }
}

package struct GTurboManifestAttnResV2: Codable, Equatable, Sendable {
    package let blockSize: Int

    package init(blockSize: Int) {
        self.blockSize = blockSize
    }
}

package struct GTurboManifestMoEV2: Codable, Equatable, Sendable {
    package let numExperts: Int
    package let topKExperts: Int
    package let latentBottleneckSize: Int
    package let expertIntermediateSize: Int
    package let numSharedExperts: Int
    package let sharedExpertIntermediateSize: Int
    package let situGLUGateBeta: Double
    package let situGLUUpBeta: Double
    package let routerRenormalize: Bool
    package let routerCorrectionBias: Bool

    package init(numExperts: Int, topKExperts: Int,
                 latentBottleneckSize: Int, expertIntermediateSize: Int,
                 numSharedExperts: Int, sharedExpertIntermediateSize: Int,
                 situGLUGateBeta: Double, situGLUUpBeta: Double,
                 routerRenormalize: Bool, routerCorrectionBias: Bool) {
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.latentBottleneckSize = latentBottleneckSize
        self.expertIntermediateSize = expertIntermediateSize
        self.numSharedExperts = numSharedExperts
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.situGLUGateBeta = situGLUGateBeta
        self.situGLUUpBeta = situGLUUpBeta
        self.routerRenormalize = routerRenormalize
        self.routerCorrectionBias = routerCorrectionBias
    }
}

// Refuse-defaults rule: the synthesized Decodable conformance is deliberate.
// Every arch key is required on the wire; a missing key fails decoding instead
// of falling back to a default (a defaulted `fullAttnLayers` would silently
// turn MLA layers into KDA). Do not add `init(from:)` fallbacks here.
package struct GTurboManifestArchV2: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let vocabSize: Int
    package let numLayers: Int
    package let denseMLPIntermediateSize: Int
    package let rmsNormEpsilon: Double
    package let tieWordEmbeddings: Bool
    package let hiddenActivation: String
    package let bosTokenID: Int
    package let eosTokenID: Int
    // Explicit 1-based layer lists copied from the source config, never
    // inferred. `denseLayers` marks dense-FFN layers and overlaps the
    // attention lists; kdaLayers/fullAttnLayers are disjoint and together
    // cover every layer.
    package let denseLayers: [Int]
    package let kdaLayers: [Int]
    package let fullAttnLayers: [Int]
    package let kda: GTurboManifestKDAV2
    package let mla: GTurboManifestMLAV2
    package let attnRes: GTurboManifestAttnResV2
    package let moe: GTurboManifestMoEV2

    package init(hiddenSize: Int, vocabSize: Int, numLayers: Int,
                 denseMLPIntermediateSize: Int, rmsNormEpsilon: Double,
                 tieWordEmbeddings: Bool, hiddenActivation: String,
                 bosTokenID: Int, eosTokenID: Int,
                 denseLayers: [Int], kdaLayers: [Int], fullAttnLayers: [Int],
                 kda: GTurboManifestKDAV2, mla: GTurboManifestMLAV2,
                 attnRes: GTurboManifestAttnResV2, moe: GTurboManifestMoEV2) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.denseMLPIntermediateSize = denseMLPIntermediateSize
        self.rmsNormEpsilon = rmsNormEpsilon
        self.tieWordEmbeddings = tieWordEmbeddings
        self.hiddenActivation = hiddenActivation
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID
        self.denseLayers = denseLayers
        self.kdaLayers = kdaLayers
        self.fullAttnLayers = fullAttnLayers
        self.kda = kda
        self.mla = mla
        self.attnRes = attnRes
        self.moe = moe
    }
}

package struct GTurboManifestQuantSlotV2: Codable, Equatable, Sendable {
    package let weightBits: Int
    package let scheme: String
    package let scaleType: String
    package let biasType: String
    package let groupSize: Int

    package init(weightBits: Int, scheme: String, scaleType: String,
                 biasType: String, groupSize: Int) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

package struct GTurboManifestQuantV2: Codable, Equatable, Sendable {
    package let embedding: GTurboManifestQuantSlotV2
    package let attention: GTurboManifestQuantSlotV2
    package let router: GTurboManifestQuantSlotV2
    package let sharedExpert: GTurboManifestQuantSlotV2
    package let latentProjection: GTurboManifestQuantSlotV2
    package let denseMLP: GTurboManifestQuantSlotV2
    package let routedExpert: GTurboManifestQuantSlotV2

    package init(embedding: GTurboManifestQuantSlotV2,
                 attention: GTurboManifestQuantSlotV2,
                 router: GTurboManifestQuantSlotV2,
                 sharedExpert: GTurboManifestQuantSlotV2,
                 latentProjection: GTurboManifestQuantSlotV2,
                 denseMLP: GTurboManifestQuantSlotV2,
                 routedExpert: GTurboManifestQuantSlotV2) {
        self.embedding = embedding
        self.attention = attention
        self.router = router
        self.sharedExpert = sharedExpert
        self.latentProjection = latentProjection
        self.denseMLP = denseMLP
        self.routedExpert = routedExpert
    }
}

package struct GTurboManifestV2: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: GTurboManifestArchV2
    package let quant: GTurboManifestQuantV2
    package let files: [String: GTurboManifestFileV2]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64

    package init(magic: String = GTurboFormatV2.magic,
                 versionMajor: Int = GTurboFormatV2.versionMajor,
                 versionMinor: Int = GTurboFormatV2.versionMinor,
                 flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: GTurboManifestArchV2,
                 quant: GTurboManifestQuantV2,
                 files: [String: GTurboManifestFileV2],
                 expertsPerLayer: Int, numLayers: Int, expertStride: UInt64) {
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

package enum GTurboManifestCodecV2 {
    package static func decode(_ data: Data) throws -> GTurboManifestV2 {
        let manifest = try decodeUnchecked(data)
        try validate(manifest)
        return manifest
    }

    package static func decodeUnchecked(_ data: Data) throws -> GTurboManifestV2 {
        let manifest: GTurboManifestV2
        do { manifest = try JSONDecoder().decode(GTurboManifestV2.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)") }
        return manifest
    }

    package static func encode(_ manifest: GTurboManifestV2) throws -> Data {
        try validate(manifest)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(manifest))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
    }

    package static func validate(_ manifest: GTurboManifestV2) throws {
        guard manifest.magic == GTurboFormatV2.magic else {
            throw GTurboFormatError.invalid(field: "manifest.magic", reason: "expected GTURBO")
        }
        guard manifest.versionMajor == GTurboFormatV2.versionMajor,
              manifest.versionMinor >= 0 else {
            throw GTurboFormatError.invalid(field: "manifest.version", reason: "unsupported version")
        }
        for flag in manifest.flags.keys where !GTurboFormatV2.knownFlags.contains(flag) {
            throw GTurboFormatError.invalid(field: "manifest.flags.\(flag)", reason: "unknown v2 flag")
        }
        guard !manifest.modelID.isEmpty,
              manifest.numLayers > 0, manifest.expertsPerLayer > 0,
              manifest.expertStride > 0,
              manifest.expertStride % GTurboFormatV2.alignmentBytes == 0 else {
            throw GTurboFormatError.invalid(field: "manifest", reason: "invalid dimensions or stride")
        }
        guard manifest.arch.numLayers == manifest.numLayers,
              manifest.arch.moe.numExperts == manifest.expertsPerLayer else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch", reason: "dimensions disagree with streaming metadata")
        }
        let arch = manifest.arch
        guard arch.hiddenSize > 0, arch.vocabSize > 0,
              arch.denseMLPIntermediateSize > 0,
              arch.rmsNormEpsilon.isFinite, arch.rmsNormEpsilon > 0,
              !arch.hiddenActivation.isEmpty,
              arch.bosTokenID >= 0, arch.eosTokenID >= 0,
              arch.bosTokenID != arch.eosTokenID else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch", reason: "invalid architecture values")
        }
        guard arch.kda.numHeads > 0, arch.kda.headDim > 0,
              arch.kda.convWidth > 0, arch.kda.decayLowRankSize > 0,
              arch.kda.decayProjectionSize > 0,
              arch.kda.gateLowerBound.isFinite, arch.kda.gateLowerBound < 0 else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch.kda", reason: "invalid KDA values")
        }
        guard arch.mla.numHeads > 0, arch.mla.qLoraRank > 0,
              arch.mla.kvLoraRank > 0, arch.mla.qkNopeHeadDim > 0,
              arch.mla.qkRopeHeadDim > 0, arch.mla.vHeadDim > 0 else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch.mla", reason: "invalid MLA values")
        }
        guard arch.attnRes.blockSize > 0 else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch.attnRes", reason: "invalid AttnRes block size")
        }
        guard arch.moe.numExperts > 0, arch.moe.topKExperts > 0,
              arch.moe.topKExperts <= arch.moe.numExperts,
              arch.moe.latentBottleneckSize > 0, arch.moe.expertIntermediateSize > 0,
              arch.moe.numSharedExperts > 0, arch.moe.sharedExpertIntermediateSize > 0,
              arch.moe.situGLUGateBeta.isFinite, arch.moe.situGLUGateBeta > 0,
              arch.moe.situGLUUpBeta.isFinite, arch.moe.situGLUUpBeta > 0 else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch.moe", reason: "invalid MoE values")
        }
        for (name, list) in [
            ("denseLayers", arch.denseLayers),
            ("kdaLayers", arch.kdaLayers),
            ("fullAttnLayers", arch.fullAttnLayers),
        ] {
            guard list.allSatisfy({ $0 >= 1 && $0 <= arch.numLayers }),
                  Set(list).count == list.count,
                  list == list.sorted() else {
                throw GTurboFormatError.invalid(
                    field: "manifest.arch.\(name)",
                    reason: "expected sorted unique 1-based layer ids")
            }
        }
        guard Set(arch.kdaLayers).isDisjoint(with: Set(arch.fullAttnLayers)) else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch", reason: "kdaLayers and fullAttnLayers overlap")
        }
        let layerUnion = Set(arch.denseLayers)
            .union(arch.kdaLayers)
            .union(arch.fullAttnLayers)
        guard layerUnion == Set(1...arch.numLayers) else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch",
                reason: "layer lists must cover exactly 1...numLayers")
        }
        for (name, slot) in [
            ("embedding", manifest.quant.embedding),
            ("attention", manifest.quant.attention),
            ("router", manifest.quant.router),
            ("sharedExpert", manifest.quant.sharedExpert),
            ("latentProjection", manifest.quant.latentProjection),
            ("denseMLP", manifest.quant.denseMLP),
            ("routedExpert", manifest.quant.routedExpert),
        ] {
            guard slot.weightBits > 0, slot.weightBits <= 32,
                  slot.groupSize > 0,
                  !slot.scheme.isEmpty, !slot.scaleType.isEmpty,
                  !slot.biasType.isEmpty else {
                throw GTurboFormatError.invalid(
                    field: "manifest.quant.\(name)", reason: "invalid quantization values")
            }
        }
        let reservedFiles: Set<String> = ["manifest.json", "verified-install.json"]
        let filePaths = manifest.files.keys.sorted()
        var canonicalPaths: [String: String] = [:]
        for path in filePaths {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = GTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                  !reservedFiles.contains(key),
                  !reservedFiles.contains(where: { key.hasPrefix("\($0)/") }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
            let entry = manifest.files[path]!
            guard entry.sha256.count == 64,
                  entry.sha256.unicodeScalars.allSatisfy({ scalar in
                      ("0"..."9").contains(Character(String(scalar)))
                          || ("a"..."f").contains(Character(String(scalar)))
                          || ("A"..."F").contains(Character(String(scalar)))
                  }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path).sha256", reason: "expected 64 hexadecimal characters")
            }
        }
        for (key, path) in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                let ancestor = components.joined(separator: "/")
                if canonicalPaths[ancestor] != nil {
                    throw GTurboFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "file path collides with a directory prefix")
                }
            }
        }
    }
}
