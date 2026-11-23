import Foundation
import TinyTitanFormat

// The manifest value types: file entries, the arch block, quant slots and the
// decoded document the reader returns.
//
// Split out of `ManifestReader.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion.
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

    // Family extension geometry. Optional because manifests written before
    // these families existed do not carry them; when a manifest DOES declare
    // them they are validated, so a checkpoint whose hyper-connection, QSA
    // indexer or PLE geometry differs from the runtime's cannot be run
    // silently against the wrong constants.
    public let hcCount: Int?
    public let hcLowRank: Int?
    public let indexerNumHeads: Int?
    public let indexerNumKVHeads: Int?
    public let indexerHeadDim: Int?
    public let indexerBudget: Int?
    public let indexerCompressRatio: Int?
    public let pleLayerIndices: [Int]?
    public let pleEmbedDim: Int?
    public let pleConvKernelSize: Int?
    public let pleNgramSize: Int?
    public let pleVocabSizeBase: Int?
    public let pleHeadsPerNgram: Int?
    public let pleVocabDivisor: Int?
    public let routerNormTopK: Bool?
    public let quantGroupSize: Int?
    // The layer conventions, carried by every manifest this repacker writes and
    // optional for the same reason as the block above. The architecture presets
    // state them per family; a family without a preset (the dense 3.5-lineage
    // models) must read them here rather than assume them, because picking the
    // wrong one produces confident nonsense instead of an error -- a wrong
    // attention output gate or RoPE convention is not a shape mismatch.
    public let attnOutputGate: Bool?
    public let attentionScale: Double?
    public let embeddingScaledBySqrtHidden: Bool?
    public let routerScaled: Bool?
    public let ffnSandwichNorms: Bool?
    public let sharedExpertGated: Bool?
    public let ropeNeoxSubdim: Bool?
    /// K2-Horizon's layer shape. Optional for the same reason as the block
    /// above; the K2 arch resolver requires every one of them by name, because
    /// reading a softplus gate as a sigmoid or a 2.5 router scale as 1 produces
    /// fluent nonsense rather than a shape error.
    public let normGroups: Int?
    public let attentionGate: String?
    public let routerHasBias: Bool?
    public let routerScaling: Double?
    public let movaNumExperts: Int?
    public let movaTopKExperts: Int?
    public let movaRouterScaling: Double?
    public let movaRouterHasBias: Bool?
    public let movaActivation: String?
    public let denseLayerIndices: [Int]?
    /// Gated-DeltaNet geometry. Optional for the same reason as the block
    /// above: manifests written before a reader needed them do not carry
    /// them, and decoding an absent key as nil is what keeps those installs
    /// loadable. The dense CPU engine is the reader that needs them.
    public let linearNumKHeads: Int?
    public let linearNumVHeads: Int?
    public let linearKeyHeadDim: Int?
    public let linearValueHeadDim: Int?
    public let linearConvKernelSize: Int?
    /// A variant's own geometry, read only by that variant (TT-121).
    public let variantGeometry: ManifestVariantGeometry?
}

/// Named scalars and integer lists a variant stores in its manifest and alone
/// interprets; the Core reader carries them without knowing what they mean.
public struct ManifestVariantGeometry: Decodable, Equatable, Sendable {
    public let scalars: [String: Double]
    public let lists: [String: [Int]]

    public init(scalars: [String: Double], lists: [String: [Int]]) {
        self.scalars = scalars
        self.lists = lists
    }
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int

    package init(
        weightBits: Int, scheme: String, scaleType: String, biasType: String, groupSize: Int
    ) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot

    package init(
        embedding: ManifestQuantSlot, attention: ManifestQuantSlot, router: ManifestQuantSlot,
        sharedExpert: ManifestQuantSlot, routedExpert: ManifestQuantSlot
    ) {
        self.embedding = embedding
        self.attention = attention
        self.router = router
        self.sharedExpert = sharedExpert
        self.routedExpert = routedExpert
    }

    /// The slot a tensor is actually stored in.
    ///
    /// The five slots above are the build's *defaults*; a tensor whose width
    /// differs from every slot's carries a per-tensor override instead, keyed
    /// by tensor stem (the name without `.weight`). A dense 3.5-lineage install is
    /// exactly that case: its `mlp.*` projections are 4-bit and its
    /// full-attention `k_proj`/`v_proj` 8-bit, while the `sharedExpert` and
    /// `attention` slots say 8 and 4. Reading the slot where the override
    /// applies is how a kernel comes to read the wrong number of bytes -- no
    /// error, just a wrong model -- so every check and every binding that can
    /// see a tensor name resolves through here.
    public func slot(
        forTensorNamed name: String, overrides: [String: Int],
        fallback: ManifestQuantSlot
    ) -> ManifestQuantSlot {
        let stem = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        guard let bits = overrides[stem], bits != fallback.weightBits else { return fallback }
        return ManifestQuantSlot(
            weightBits: bits, scheme: fallback.scheme,
            scaleType: fallback.scaleType, biasType: fallback.biasType,
            groupSize: fallback.groupSize)
    }

    /// The width a *role's* tensors are stored at: the override the manifest
    /// declares for that role, else the fallback.
    ///
    /// Keyed by a name suffix rather than by one tensor, because the runtime
    /// builds one kernel per role and a manifest's overrides name real tensors:
    /// asking for `layers.0.self_attn.k_proj` would ask about a tensor a
    /// Gated-DeltaNet layer does not have, and "no override" would read an
    /// 8-bit k_proj as 4-bit nibbles.
    ///
    /// Sorted, so the answer is deterministic. Uniformity across a role is
    /// enforced by the loader (`Model.validateRoleUniformity`), so there is
    /// never more than one value to find; sorting means that if one ever slips
    /// past, the width does not change between runs.
    public static func roleWeightBits(
        roleSuffix: String,
        overrides: [String: Int],
        fallback: Int,
        promoted: Set<String> = []
    ) -> Int {
        if let override = overrides.sorted(by: { $0.key < $1.key })
            .first(where: { $0.key.hasSuffix(roleSuffix) })?.value
        {
            return override
        }
        // A promoted role is unquantized: 16 tells the caller no affine kernel
        // may read it. Only reached when the slot would otherwise have claimed
        // it, so norms and other always-bf16 tensors never come through here.
        if promoted.contains(where: { $0.hasSuffix(roleSuffix) }) { return 16 }
        return fallback
    }
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
    /// Per-tensor width overrides, keyed by tensor stem ("language_model
    /// .model.layers.3.self_attn.k_proj" -> 8). Absent when the build has no
    /// overrides, and in manifests written before this existed.
    public let quantOverrides: [String: Int]
    /// Tensor stems stored at the checkpoint's own bf16 inside a quantized
    /// slot. Their width is 16 when a kernel has to be chosen and absent from
    /// the packing arithmetic, which is why this is a set of names rather than
    /// a `weightBits` value.
    public let quantPromoted: Set<String>
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}
