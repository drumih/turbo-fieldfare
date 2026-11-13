import Darwin
import Foundation
import Metal
import TinyTitanFormat

public struct ModelLoadStats: Sendable {
    public var manifestSha256Nanos: UInt64
    public var receiptValidationNanos: UInt64
    public var eagerSha256Nanos: UInt64

    public init(
        manifestSha256Nanos: UInt64 = 0,
        receiptValidationNanos: UInt64 = 0,
        eagerSha256Nanos: UInt64 = 0
    ) {
        self.manifestSha256Nanos = manifestSha256Nanos
        self.receiptValidationNanos = receiptValidationNanos
        self.eagerSha256Nanos = eagerSha256Nanos
    }
}

/// Bounded routed-expert cache configuration.
public enum ExpertStreamingMode: Sendable {
    /// Read each expert into one of `slotCount` 2 MB-aligned cache slots.
    case pread(slotCount: Int)
}

/// Loaded `.ssdai/` model. Resident weights live behind one mmap'd
/// `MTLBuffer`; routed expert weights live behind per-layer streaming
/// backends opened lazily on first touch.
public struct Model {
    /// unchecked-invariant: all `let`, holding two read-only TensorViews and
    /// their bit widths. @unchecked only because TensorView is.
    struct SharedTargetWeights: @unchecked Sendable {
        let embedding: TensorView
        let lmHead: TensorView
        let embeddingBits: Int
        let lmHeadBits: Int
    }
    public let device: MTLDevice
    public let config: ArchConfig
    /// bf16 views of fp32 tensors the kernels read as bf16, promoted at load.
    /// Empty for every family whose checkpoints already store them as bf16.
    let promotedBF16: [String: TensorView]
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    public let integrityPolicy: ModelIntegrityPolicy
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }
    /// 16 when the table is kept at the checkpoint's bf16. Read from the
    /// tensor, like `effectiveRouterWeightBits`: a bf16-rest install keeps the
    /// embedding unquantized inside a slot that still says 4 bits, and the
    /// lookup has to match the bytes, not the slot.
    public var embeddingWeightBits: Int {
        if let sharedTargetWeights { return sharedTargetWeights.embeddingBits }
        if (try? embedding())?.dtype == 1 { return 16 }
        return manifest.quant?.embedding.weightBits ?? 4
    }
    public var lmHeadWeightBits: Int {
        // The head is a role of its own: a manifest can declare it separately
        // from the attention slot (and the dense family's 9B does), so it is
        // resolved by name first. The embedding slot is the fallback because
        // the repacker quantizes a separate lm_head with the embedding's layout
        // (padded to the same vocab rows), which `validateRuntimeSchema`
        // checks. A head kept at bf16 is 16 whatever the manifest says, for the
        // same reason as `embeddingWeightBits`.
        if let sharedTargetWeights { return sharedTargetWeights.lmHeadBits }
        if (try? lmHead())?.dtype == 1 { return 16 }
        return roleWeightBits(
            roleSuffix: ".lm_head",
            fallback: manifest.quant?.embedding.weightBits ?? 4)
    }
    public var attentionWeightBits: Int { manifest.quant?.attention.weightBits ?? 4 }
    public var routerWeightBits: Int { manifest.quant?.router.weightBits ?? 8 }

    /// True when the GDN `in_proj_a` / `in_proj_b` pair was promoted to the
    /// checkpoint's bf16. They travel together -- both are `numVHeads` rows of
    /// the same projection -- so one probe decides the pair.
    public var gdnABIsBF16: Bool {
        guard let view = try? linearInProjA(layer: 0) else { return false }
        return view.dtype == 1
    }

    /// The width the router GEMV must actually be built for.
    ///
    /// The manifest slot says how the slot is stored; a family can be promoted
    /// to the checkpoint's own bf16 inside it, and then the tensor's dtype is
    /// what the kernel has to match. 16 means unquantized -- the shader reads
    /// bfloat directly and ignores the scale and bias companions, which a
    /// promoted tensor does not have.
    ///
    /// Read from the tensor rather than the slot because that is the thing
    /// that can differ: getting it from the slot is exactly the mistake the
    /// INT4-only kernels made.
    public var effectiveRouterWeightBits: Int {
        guard let view = try? router(layer: 0) else { return routerWeightBits }
        return view.dtype == 1 ? 16 : routerWeightBits
    }
    public var sharedExpertWeightBits: Int { manifest.quant?.sharedExpert.weightBits ?? 8 }

    /// A tensor's own declared width, when the manifest overrides its slot.
    ///
    /// The five slots are the build's defaults; a tensor whose width differs
    /// from every slot's carries a per-tensor override keyed by stem (the name
    /// without `.weight`). The dense Qwen 3.5 installs are built that way.
    func weightBits(forTensorNamed name: String) -> Int? {
        let stem = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        if manifest.quantPromoted.contains(stem) { return 16 }
        return manifest.quantOverrides[stem]
    }

    /// The width a *role's* tensors are stored at: the per-tensor override the
    /// manifest declares for that role, else the slot's width.
    ///
    /// Keyed by the role's name suffix rather than by one layer's tensor,
    /// because a manifest's overrides name real tensors: asking for
    /// `layers.0.self_attn.k_proj` asks about a tensor a Gated-DeltaNet layer
    /// does not have, and the answer "no override" would silently read an 8-bit
    /// k_proj as 4-bit nibbles. The dense installs declare their deviations
    /// exactly that way -- `k_proj`/`v_proj` at 8 bits on the full-attention
    /// layers only -- so the lookup scans the role.
    ///
    /// Uniformity across the role is required, because the runner builds one
    /// dispatcher per role; `validateRuntimeSchema` refuses an install that
    /// declares otherwise rather than reading some layers at the wrong width.
    func roleWeightBits(roleSuffix: String, fallback: Int) -> Int {
        ManifestQuant.roleWeightBits(
            roleSuffix: roleSuffix,
            overrides: manifest.quantOverrides,
            fallback: fallback,
            promoted: manifest.quantPromoted)
    }

    /// The width the `q_proj`/`o_proj` pair is stored at.
    public var qoProjectionWeightBits: Int {
        roleWeightBits(roleSuffix: ".self_attn.q_proj", fallback: attentionWeightBits)
    }

    /// The width the `k_proj`/`v_proj` pair is stored at (one role: they share
    /// a shape and the KV cache treats them together).
    public var kvProjectionWeightBits: Int {
        roleWeightBits(roleSuffix: ".self_attn.k_proj", fallback: attentionWeightBits)
    }

    /// The width the feed-forward projections are stored at. For a dense model
    /// these are its `mlp.*` tensors, declared per tensor at 4 bits while the
    /// `sharedExpert` slot says 8; for a MoE family it is the slot, unchanged.
    public var ffnWeightBits: Int {
        roleWeightBits(roleSuffix: ".mlp.gate_proj", fallback: sharedExpertWeightBits)
    }

    /// The width the Gated-DeltaNet projections are stored at.
    public var gdnProjectionWeightBits: Int {
        roleWeightBits(roleSuffix: ".linear_attn.in_proj_qkv", fallback: attentionWeightBits)
    }

    /// The width the hyper-connection write gates are stored at.
    ///
    /// These are the tensors a per-tensor width exists for. They read the
    /// attention slot until a manifest overrides them, and promoting the whole
    /// slot instead is the measured-but-expensive route: the attention block is
    /// 61% of the active parameters, so taking it from 4 to 8 bits costs
    /// +2.10 GB resident against the ~10 MB `tools/precision_probe.py` says
    /// actually needs the precision. One kernel instance serves every layer, so
    /// the gates have to be uniform and `validateRoleUniformity` refuses a
    /// manifest that declares otherwise.
    public var hyperConnectionWeightBits: Int {
        // No leading dot: the stem is `attn_hyper_connection.…` or
        // `mlp_hyper_connection.…`, and `.hyper_connection…` matches neither.
        roleWeightBits(
            roleSuffix: "hyper_connection.block_inject_weight",
            fallback: attentionWeightBits)
    }

    /// The width the PLE key projection is stored at, resolved the same way.
    public var pleKeyWeightBits: Int {
        roleWeightBits(roleSuffix: ".ple.key_proj", fallback: attentionWeightBits)
    }

    /// The width the sparse indexer's key projections are stored at.
    public var qsaIndexerWeightBits: Int {
        roleWeightBits(
            roleSuffix: ".self_attn.indexer.index_q_proj",
            fallback: attentionWeightBits)
    }

    public var routedExpertWeightBits: Int { manifest.quant?.routedExpert.weightBits ?? 4 }

    /// The width one routed expert is actually stored at.
    ///
    /// A tiered install keeps hot experts at the manifest width and cold ones
    /// at 2-bit in the same layer file; `layout.json` records that per tensor,
    /// so the dispatch can pick a pipeline per expert instead of assuming the
    /// install-wide value. Returns the manifest width when the layout carries no
    /// override, which is every install that existed before tiering.
    @available(
        *, deprecated,
        message: "No caller (TT-050): the tiered install this serves has no converter yet."
    )
    public func routedExpertBits(layer: Int, expert: Int) -> Int {
        guard
            let entry = try? packedExpertsLayout.expert(layer: layer, expert: expert),
            let bits = entry.subTensors["gate"]?.bits
        else {
            return routedExpertWeightBits
        }
        return bits
    }
    /// The manifest's recorded digest of `model_weights.bin`. The manifest is
    /// itself bound by the install receipt, so this is a trustworthy identity
    /// for anything derived from these weights — the ANE prefill sidecar uses
    /// it to refuse a sidecar exported from a different model.
    public var weightsDigestFromManifest: String? {
        manifest.files["model_weights.bin"]?.sha256
    }
    var mtpResidentTensorBytes: Int { residentBuffer.buffer.length }
    var mtpExpertStrideBytes: Int { Int(packedExpertsLayout.expertStride) }

    let residentBuffer: ResidentBuffer
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
    /// Total bytes the install declares. The page-cache rule compares this
    /// with physical memory; an install whose manifest lists nothing is
    /// treated as unknown and stays on the bounded reader.
    let installSizeBytes: UInt64
    /// T25's machine-size decision, made once at load so every layer streamer
    /// uses the same answer.
    let pageCachePolicy: PageCachePolicy
    let directoryURL: URL
    let modelDirectory: SSDAIModelDirectory
    let sharedTargetWeights: SharedTargetWeights?
    /// The routing history this install left behind, if any. Seeds each
    /// layer's LFU score at open and is updated after every request.
    let expertUsageProfile: ExpertUsageProfile?
    let expertProfileURL: URL?

    /// Lazy state. Held inside a reference box so `Model` can stay a struct
    /// while still letting accessors mutate layer state via a serial queue.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue
    /// Serialises profile writes so two requests cannot interleave a rewrite.
    let profileWriteQueue: DispatchQueue
    /// One completion timeline per slot, in the mode this model was loaded with
    /// (see `ExpertIOEventCoordinator.makeTimelines`).
    let expertIOTimeline: RuntimeExpertIOTimeline
    let expertIOTimelines: [ExpertIOEventCoordinator]?

    /// unchecked-invariant: every access goes through `streamersQueue`, the
    /// serial queue on the owning Model. The box exists so Model can stay a
    /// struct while still mutating per-layer streamer state. Every member,
    /// including the wiring flags and the pin diagnostics added in 5.0.3, is
    /// read and written only inside `streamersQueue.sync` / `.async` blocks;
    /// the queue is the lock.
    final class StreamersBox: @unchecked Sendable {
        var streamers: [PreadExpertStreamer?]
        var layerVerified: [Bool]
        /// True once every opened streamer's slots are wired (see
        /// `setExpertCachePinned`); cleared by any unpin or partial wire.
        var pinnedComplete = false
        /// Wire each layer as it opens (`profile.keepExpertCacheWired`, the
        /// row's own measured value).
        var keepWired = false
        /// Diagnostic: time spent waiting to enter the serial queue in
        /// `setExpertCachePinned`.
        var pinQueueWaitNanos: UInt64 = 0
        /// One staging ring is shared by every lazy layer streamer. Allocating
        /// one per layer would turn a small event bridge into hundreds of MiB
        /// of undeclared working set.
        var metalStagingPool: MetalExpertStagingPool?
        /// Layer files need separate handles, but not separate MTLIO queues.
        /// One queue prevents prefill from exhausting Metal-I/O worker threads.
        var metalIOService: MetalExpertIOService?
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    init(
        device: MTLDevice,
        config: ArchConfig,
        streamingMode: ExpertStreamingMode,
        expertCachePolicy: ExpertCachePolicy,
        expertIOTimeline: RuntimeExpertIOTimeline = .shared,
        integrityPolicy: ModelIntegrityPolicy,
        residentBuffer: ResidentBuffer,
        residentIndex: ResidentIndex,
        packedExpertsLayout: PackedExpertsLayout,
        manifest: Manifest,
        directoryURL: URL,
        modelDirectory: SSDAIModelDirectory,
        sharedTargetWeights: SharedTargetWeights? = nil,
        promotedBF16: [String: TensorView] = [:]
    ) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.residentBuffer = residentBuffer
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        let installBytes = PageCachePolicyResolver.installSize(
            fileSizes: manifest.files.values.map(\.size))
        self.installSizeBytes = installBytes
        self.pageCachePolicy =
            PageCachePolicyResolver.resolve(
                physicalMemory: ProcessInfo.processInfo.physicalMemory,
                installBytes: installBytes
            ).policy
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.profileWriteQueue = DispatchQueue(label: "TinyTitan.expert-profile")
        let profileURL = Self.expertProfileURL(directoryURL: directoryURL)
        self.expertProfileURL = profileURL
        self.expertUsageProfile = profileURL.flatMap {
            ExpertUsageProfile.load(
                url: $0,
                layers: packedExpertsLayout.numLayers,
                expertsPerLayer: packedExpertsLayout.expertsPerLayer)
        }
        self.promotedBF16 = promotedBF16
        self.sharedTargetWeights = sharedTargetWeights
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "TinyTitan.expert-streamers")
        self.expertIOTimeline = expertIOTimeline
        self.expertIOTimelines = ExpertIOEventCoordinator.makeTimelines(
            device: device,
            numLayers: packedExpertsLayout.numLayers,
            mode: expertIOTimeline)
    }

    /// The Qwen3.8-Flash-Next draft head's own tensors.
    ///
    /// It has no embedding, no head and no n-gram block: it borrows the first
    /// two from the target it drafts for and does not have the third. What is
    /// its own is the single decoder layer and the fusion pair, so that is
    /// what gets checked.
    static func validateQwen38DraftSchema(
        checks: RuntimeSchemaChecks,
        quant: ManifestQuant,
        config: ArchConfig
    ) throws {
        let hcDim = config.hiddenSize * config.hyperConnections.count
        try checks.requireBF16(
            "model.language_model.hyper_connection_mixer.hc_norm", count: hcDim)
        try checks.requireBF16(
            "model.language_model.layers.0.attn_hyper_connection.hc_norm",
            count: hcDim)
        try checks.requireBF16(
            "model.language_model.layers.0.mlp_hyper_connection.hc_norm",
            count: hcDim)
        // Without both projections this is not a draft head, it is a
        // detached layer.
        try checks.requireBF16("pre_fc_norm_hidden", count: hcDim)
        try checks.requireBF16("pre_fc_norm_embedding", count: config.hiddenSize)
        // [D, D], not [D, hc_dim]: the wide residual is mean-collapsed to one
        // stream before this projection sees it. The reference's prose says
        // otherwise and its own code works out that the prose cannot be right;
        // the tensor shape settles it. Either may be kept at bf16
        // (`encodeDraftFusionProjection` dispatches on the dtype).
        try checks.requireAffineOrBF16(
            "fc_hidden.weight",
            rows: config.hiddenSize,
            columns: config.hiddenSize,
            slot: quant.attention)
        try checks.requireAffineOrBF16(
            "fc_embedding.weight",
            rows: config.hiddenSize,
            columns: config.hiddenSize,
            slot: quant.attention)
    }

    /// The Qwen 4.0 Flash draft head's own tensors.
    ///
    /// Copied from the 3.8 draft check. The draft has no embedding, no head and
    /// no n-gram block: it borrows the first two from the target it drafts for.
    /// What is its own is the single decoder layer and the fusion pair.
    static func validateQwen4DraftSchema(
        checks: RuntimeSchemaChecks,
        quant: ManifestQuant,
        config: ArchConfig
    ) throws {
        let hcDim = config.hiddenSize * config.hyperConnections.count
        try checks.requireBF16(
            TensorSchema.schema(for: .qwen4flashMTP).finalNorm, count: hcDim)
        try checks.requireBF16(
            "model.language_model.layers.0.attn_hyper_connection.hc_norm",
            count: hcDim)
        try checks.requireBF16(
            "model.language_model.layers.0.mlp_hyper_connection.hc_norm",
            count: hcDim)
        try checks.requireBF16("pre_fc_norm_hidden", count: hcDim)
        try checks.requireBF16("pre_fc_norm_embedding", count: config.hiddenSize)
        // Both projections are [D, D]: the wide residual is mean-collapsed to
        // one stream before the hidden branch sees it. The tensor shape settles
        // the reference prose's ambiguity, exactly as it did for the 3.8 draft.
        try checks.requireAffine(
            "fc_hidden.weight",
            rows: config.hiddenSize,
            columns: config.hiddenSize,
            slot: quant.attention)
        try checks.requireAffine(
            "fc_embedding.weight",
            rows: config.hiddenSize,
            columns: config.hiddenSize,
            slot: quant.attention)
    }

    /// Attach a native MTP sidecar to a target without copying either large
    /// tensor. The returned model retains the target's Metal buffers and uses
    /// its actual 4/6/8-bit head kernels.
    public func sharingTargetWeights(from target: Model) throws -> Model {
        let pairing = (config.family, target.config.family)
        let familiesMatch =
            pairing == (.qwen36MTP, .qwen36)
            || pairing == (.qwen38flashMTP, .qwen38flash)
            || pairing == (.qwen4flashMTP, .qwen4flash)
        guard familiesMatch,
            config.hiddenSize == target.config.hiddenSize,
            config.vocabSize == target.config.vocabSize,
            Self.mtpLineagesAreCompatible(
                sidecarID: modelID,
                targetID: target.modelID)
        else {
            throw ModelError.indexCorrupt(
                detail: "MTP sidecar is incompatible with the target model")
        }
        return Model(
            device: device,
            config: config,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            expertIOTimeline: expertIOTimeline,
            integrityPolicy: integrityPolicy,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            packedExpertsLayout: packedExpertsLayout,
            manifest: manifest,
            directoryURL: directoryURL,
            modelDirectory: modelDirectory,
            sharedTargetWeights: SharedTargetWeights(
                embedding: try target.embedding(),
                lmHead: try target.lmHead(),
                embeddingBits: target.embeddingWeightBits,
                lmHeadBits: target.lmHeadWeightBits))
    }

    /// The Qwen3.5-MoE tensor contract is shared by Qwen 3.6 and Ornith 1.5,
    /// but their trained embeddings and heads are not interchangeable. Keep
    /// synthetic and privately named compatible checkpoints usable while
    /// rejecting a known cross-model pairing before any generation begins.
    static func mtpLineagesAreCompatible(
        sidecarID: String,
        targetID: String
    ) -> Bool {
        func lineage(_ modelID: String) -> String? {
            let normalized = modelID.lowercased()
            if normalized.contains("ornith-1.5") { return "ornith-1.5" }
            // Checked before the generic qwen3.6 prefix so the flash lineage
            // is not swallowed by it.
            if normalized.contains("qwen3.8-flash-next") { return "qwen3.8-flash-next" }
            if normalized.contains("qwen3.6") || normalized.hasPrefix("qwen-") {
                return "qwen3.6"
            }
            return nil
        }
        guard let sidecar = lineage(sidecarID),
            let target = lineage(targetID)
        else {
            return true
        }
        return sidecar == target
    }

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

}
