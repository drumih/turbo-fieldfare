import Foundation

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]

    // Family extensions. Optional so legacy Gemma manifests decode unchanged;
    // absent values validate against the Gemma defaults in `ArchConfig`.
    public let family: String?
    public let attnOutputGate: Bool?
    public let attentionScale: Double?
    public let embeddingScaledBySqrtHidden: Bool?
    public let routerScaled: Bool?
    public let ffnSandwichNorms: Bool?
    public let sharedExpertGated: Bool?
    public let ropeNeoxSubdim: Bool?
    public let linearNumKHeads: Int?
    public let linearNumVHeads: Int?
    public let linearKeyHeadDim: Int?
    public let linearValueHeadDim: Int?
    public let linearConvKernelSize: Int?
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = [
        "streamingPresent", "turboQuantKV", "aneSharedExpert"
    ]

    /// Required file entries (relative to `model.gturbo/`). Layer files
    /// `packed_experts/layer_<L>.bin` for L in 0..<numLayers are checked
    /// after decode against `numLayers` (with the zero-padded "layer_%02d"
    /// naming the writer produces; falling back to plain "layer_<L>" when
    /// only the unpadded form is present, for toy synthetics).
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting,
                     directoryURL: directoryURL)
        return manifest
    }

    private static func metadataFileSize(_ url: URL,
                                         fileName: String) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attrs[.size] as? NSNumber else {
            throw ModelError.indexCorrupt(detail: "\(fileName): file size unavailable")
        }
        return number.uint64Value
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig,
                         directoryURL: URL) throws {
        guard m.magic == "GTURBO" else { throw ModelError.notAGTurboDirectory }
        guard m.versionMajor == 1 else {
            throw ModelError.unsupportedVersion(major: m.versionMajor, minor: m.versionMinor)
        }
        for key in m.flags.keys {
            if !knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
        }
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let quant = m.quant {
            try validateQuant(quant)
        } else if isProductionArch(expected) {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        let pageSize = UInt64(getpagesize())
        guard m.expertStride % pageSize == 0 else {
            throw ModelError.expertStrideNotPageAligned(stride: m.expertStride,
                                                        pageSize: Int(pageSize))
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
        for L in 0..<m.numLayers {
            let padded = String(format: "packed_experts/layer_%02d.bin", L)
            let plain  = "packed_experts/layer_\(L).bin"
            if m.files[padded] == nil && m.files[plain] == nil {
                throw ModelError.missingFile(name: padded)
            }
        }
    }

    /// A manifest matching one of the shipped production baselines must carry
    /// quantization metadata; toy/synthetic manifests may omit it.
    private static func isProductionArch(_ expected: ArchConfig) -> Bool {
        for baseline in ArchConfig.knownArchitectures.values {
            if expected.numLayers == baseline.numLayers,
               expected.hiddenSize == baseline.hiddenSize {
                return true
            }
        }
        return false
    }

    private static func validateQuant(_ quant: ManifestQuant) throws {
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, [4]),
            ("attention", quant.attention, [4]),
            ("router", quant.router, [8]),
            ("sharedExpert", quant.sharedExpert, [4, 8]),
            ("routedExpert", quant.routedExpert, [4]),
        ]
        for (name, slot, allowedBits) in slots {
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  slot.groupSize == Quantization.groupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)

        // Family extensions: absent fields mean the Gemma defaults.
        let gemmaDefaults = ArchConfig.gemma4_26B_A4B
        try check("family",
                  a.family ?? ModelFamily.gemma4.rawValue,
                  e.family.rawValue)
        try check("attnOutputGate",
                  a.attnOutputGate ?? gemmaDefaults.attnOutputGate,
                  e.attnOutputGate)
        try check("attentionScale",
                  a.attentionScale ?? gemmaDefaults.attentionScale,
                  e.attentionScale)
        try check("embeddingScaledBySqrtHidden",
                  a.embeddingScaledBySqrtHidden ?? gemmaDefaults.embeddingScaledBySqrtHidden,
                  e.embeddingScaledBySqrtHidden)
        try check("routerScaled",
                  a.routerScaled ?? gemmaDefaults.routerScaled,
                  e.routerScaled)
        try check("ffnSandwichNorms",
                  a.ffnSandwichNorms ?? gemmaDefaults.ffnSandwichNorms,
                  e.ffnSandwichNorms)
        try check("sharedExpertGated",
                  a.sharedExpertGated ?? gemmaDefaults.sharedExpertGated,
                  e.sharedExpertGated)
        try check("ropeNeoxSubdim",
                  a.ropeNeoxSubdim ?? gemmaDefaults.ropeNeoxSubdim,
                  e.ropeNeoxSubdim)
        try check("linearNumKHeads",
                  a.linearNumKHeads ?? 0, e.linearAttention.numKHeads)
        try check("linearNumVHeads",
                  a.linearNumVHeads ?? 0, e.linearAttention.numVHeads)
        try check("linearKeyHeadDim",
                  a.linearKeyHeadDim ?? 0, e.linearAttention.keyHeadDim)
        try check("linearValueHeadDim",
                  a.linearValueHeadDim ?? 0, e.linearAttention.valueHeadDim)
        try check("linearConvKernelSize",
                  a.linearConvKernelSize ?? 0, e.linearAttention.convKernelSize)
    }

    /// Decode just enough of `manifest.json` to identify the model family,
    /// without arch validation. Used by `Model.load` auto-detection.
    public static func peekFamily(directoryURL: URL,
                                  maxBytes: UInt64 = defaultMaxBytes) throws -> ModelFamily {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        guard let raw = manifest.arch.family else { return .gemma4 }
        guard let family = ModelFamily(rawValue: raw) else {
            throw ModelError.indexCorrupt(detail: "unknown arch.family \"\(raw)\"")
        }
        return family
    }
}
