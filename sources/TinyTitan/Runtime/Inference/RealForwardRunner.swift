import Foundation
import Metal

/// Compatible Qwen3.5-MoE real-forward decode pass, composing the production
/// kernels against the `.ssdai` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<40:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = v_proj(a)
///     per-head q/k_norm (bf16w)
///     NeoX RoPE on Q + K (full-attention layers; linear layers use
///     Gated-DeltaNet recurrent state instead of K/V slots)
///     write K and V into the cache slots
///     attn = attention(scale, full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)  // sigmoid-gated by shared_expert_gate
///     xr = rmsnorm_no_scale(h)
///     idx, w = router_topk(xr, effective_scale[L], per_expert_scale[L])
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=h1, routedBlobs=fetch(idx), w)
///     h = h + h2
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), lm_head^T)  // softmax: Sampler
///
/// Direct against `Model`; this is the only production decode forward path.
/// unchecked-invariant: exclusively owned by one caller for its lifetime and
/// never shared. In the server it is a `private let` on the `ServerModelSession`
/// actor, so every entry point is already actor-isolated; the CLI and each
/// server model session drive one runner from a single task. Its ~19 mutable
/// properties are decode cursors and scratch handles with no internal locking,
/// so two concurrent callers would corrupt them -- the ownership is the whole
/// safety argument, not an implementation detail.
public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting,
    ContinuableLogitProducer, @unchecked Sendable
{
    struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Qwen3.5-MoE [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    let model: Model
    let ctx: MetalContext
    let kv: KVCacheManager?
    let cfg: ArchConfig
    /// How many independent sequences this runner's KV and GDN stores hold.
    /// One unless the server batches; `produce(…slot:)` selects the sequence.
    public let slots: Int
    /// Serializes forward steps so concurrent slots never share the runner's
    /// scratch. Held around each `produce`/`prefillChunked`, not across a whole
    /// generation.
    let forwardStepGate = ForwardStepGate()

    // Kernels
    let embedInt4: EmbedLookupInt4
    let affineEmbed: AffineQuantEmbeddingLookup?
    let bf16Rest: BF16RestKernels  // a bf16 embedding / lm_head (+BF16Rest.swift)
    let rms: RMSNorm
    let int4: DequantInt4GEMV
    let affine: AffineQuantGEMV?
    /// The k/v role's dispatcher when its width differs from the attention slot's.
    let affineKV: AffineQuantGEMV?
    /// Every affine dispatcher this model's roles need, by width. `int4` covers
    /// 4, which needs no entry here.
    let affineByWidth: [Int: AffineQuantGEMV]
    /// Vocabulary head GEMV, keyed off `lmHeadWeightBits` rather than the
    /// attention slot. Nil when the head is 4-bit.
    let affineHead: AffineQuantGEMV?
    let attention: Attention
    let kvQuantizer: KVCacheQuantizer?
    let shared: SharedExpertRuntime
    let moe: MoE
    let fusionHead: LMHeadChainInt4
    let fusedQKVGEMV: FusedQKVGEMV
    let fusedQKVEpilogue: FusedQKVEpilogue

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    let elementwise: Elementwise?

    /// The Gated Residual, for families that carry one. Owns its own scratch,
    /// so a family without hyper-connections allocates nothing.
    /// Set by `TINYTITAN_ACT_DUMP`; nil disables every dump call site.
    let activationDumpDirectory: URL?
    /// Dumps recorded mid-layer, performed once the layer's work is awaited.
    /// The runner is single-flight per generation, so this needs no lock; it is
    /// only ever touched from the encoding task.
    var pendingDumps: [PendingActivationDump] = []
    /// Dumped positions are `[first, first + count)`; a deep probe moves them.
    package var activationDumpFirstPosition = 0
    package var activationDumpPositionCount = RealForwardRunner.dumpPositionCount
    /// Row 0's position during a wide verify pass, for its per-row stage dumps.
    var wideVerifyDumpStart: Int?
    /// True during a wide verify pass: multi-row stages take decode's kernels.
    var decodeOrderVerify = false
    let hyperConnection: HyperConnection?
    /// The n-gram (PLE) block, its row addressing, and the table it reads.
    /// All three or none: a family without PLE layers leaves them nil.
    /// The sparse-attention indexer, for families that have one. Nil leaves
    /// the runtime on dense attention, which is exact only inside
    /// `QSAExactness`'s window.
    let qsaIndexer: QSAIndexer?
    let pleBlock: PLEBlock?
    let pleHash: PLEHash?
    let ngramTable: NgramTableReader?
    /// The current token and its predecessors, nearest first, as `PLEHash`
    /// wants them. Only `ngramSize` entries are ever consulted.
    var pleContext: [Int32] = []
    let gdn: GDN?
    let gdnState: GDNStateManager?
    let rope: RoPE?
    let int8ScalarGate: DequantInt8GEMV?
    /// Used instead of `int8ScalarGate` when the scalar gate was promoted to
    /// the checkpoint's bf16. Kept as a separate member rather than folded
    /// into SlotGEMV because the quantized one is built with decode-specific
    /// shapes that the generic wrapper does not carry.
    let bf16ScalarGate: BF16GEMV?
    /// Used for any resident projection whose tensor was promoted to the
    /// checkpoint's bf16. Held unconditionally: promotion is a property of the
    /// install, not of the family, and the pipeline costs nothing unused.
    let bf16Projection: BF16GEMV

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    let prefillEmbed: PrefillEmbedLookupInt4
    let prefillRMS: PrefillRMSNorm
    let prefillQMM: PrefillInt4QMM
    let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    let prefillQKVEpilogue: PrefillQKVEpilogue
    let prefillAttention: PrefillAttention
    let prefillRouter: PrefillRouter
    let prefillSharedExpert: PrefillSharedExpert
    let prefillGroupedMoE: PrefillGroupedRoutedMoE
    let prefillMoE: PrefillMoE
    let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    let hidden: MTLBuffer  // [D] FP16
    let normed: MTLBuffer  // [D] FP16
    let attnOut: MTLBuffer  // [N_HEADS * head_dim] FP16
    let qScratch: MTLBuffer  // [N_HEADS * head_dim] FP16
    let kStage: MTLBuffer  // [max KV heads * head_dim] FP16, current token
    let vStage: MTLBuffer  // [max KV heads * head_dim] FP16, current token
    let oOut: MTLBuffer  // [D] FP16
    let h1Buf: MTLBuffer  // [D] FP16 (dense MLP output)
    let h2Buf: MTLBuffer  // [D] FP16 (routed output)
    let routedX: MTLBuffer  // [D] FP16 (pre_feedforward_layernorm_2 output)
    let denseX: MTLBuffer  // [D] FP16 (pre_feedforward_layernorm output)
    let denseScratchGate: MTLBuffer  // [F=2112] FP16
    let denseScratchUp: MTLBuffer  // [F=2112] FP16
    let denseScratchAct: MTLBuffer  // [F=2112] FP16
    let routerInput: MTLBuffer  // [D] FP16 (rmsnorm_no_scale(h))
    let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    let outIndices: MTLBuffer  // [topK] UInt32
    let outWeights: MTLBuffer  // [topK] FP16
    /// Trace-only next-layer router result. It is never read by inference.
    let prefetchPredictionIndices: MTLBuffer
    let prefetchPredictionWeights: MTLBuffer
    /// TINYTITAN_PROBE2_TRACE=1: a second probe scores layer L+2's router on the
    /// current residual, trace-only, to measure whether a two-layer-ahead
    /// prediction is accurate enough to widen the prefetch window.
    static let probe2TraceEnabled =
        ProcessInfo.processInfo.environment["TINYTITAN_PROBE2_TRACE"] == "1"
    let prefetchPrediction2Indices: MTLBuffer
    let prefetchPrediction2Weights: MTLBuffer
    var lastPredictedNext2Layer: [Int] = []
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    let moeActs: MTLBuffer  // [topK * FmoE] FP16
    /// Width-2 MTP verify scratch (B2 pair schedule): per-row activation and
    /// output buffers plus two persistent routed argument buffers, created on
    /// first verify. Per-row buffers are deliberately *separate allocations*,
    /// not offsets into one: Metal hazard tracking is whole-buffer, so a
    /// shared acts buffer would falsely serialize row 1's phase 1 behind
    /// row 0's phase 2 and cost real GPU concurrency. The rewrite-per-layer
    /// hazard on the argument buffers is safe because the pair schedule waits
    /// on each layer's routed command before the next layer re-encodes them.
    var verifyPairActs: [MTLBuffer] = []
    var verifyPairY: [MTLBuffer] = []
    var verifyPairArgBuffers: [MTLBuffer] = []
    let moeHitActiveSlots: MTLBuffer  // [topK] UInt32
    let moeMissActiveSlots: MTLBuffer  // [topK] UInt32
    let residencyHitCount: MTLBuffer
    let residencyHitPositions: MTLBuffer
    let residencyMissCount: MTLBuffer
    let residencyMissPositions: MTLBuffer
    let residencyMissExperts: MTLBuffer
    let residencyResolvedSlots: MTLBuffer
    let residencyResolvedGenerations: MTLBuffer
    let greedyTokenBuf: MTLBuffer  // 4 B UInt32 fused-head output
    let verificationHidden: MTLBuffer  // [2, D] FP16 shared readback
    let verificationLogits: MTLBuffer  // [2, vocab] FP16 shared readback
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    let qPackedScratch: MTLBuffer?  // [2 * N_HEADS * head_dim] packed [q ; gate]
    let attnGateScratch: MTLBuffer?  // [N_HEADS * head_dim]
    let gdnQKVRaw: MTLBuffer?  // [qkvDim] raw in_proj_qkv output
    let gdnConvOut: MTLBuffer?  // [qkvDim] conv + SiLU output
    let gdnZ: MTLBuffer?  // [valueDim]
    let gdnA: MTLBuffer?  // [numVHeads]
    let gdnB: MTLBuffer?  // [numVHeads]
    let gdnY: MTLBuffer?  // [valueDim] delta-rule output
    let gdnOut: MTLBuffer?  // [valueDim] gated-norm output
    let sharedScalarGateBuf: MTLBuffer?  // [1] shared-expert gate logit
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    let onesPerExpertScale: MTLBuffer?
    var prefillChunkState = PrefillChunkCommitState()
    var prefillScratch: PrefillChunkScratchBuffers?
    // Batched head scratch: `rows` final-normed rows and `rows` logits rows.
    var batchNormedBuffer: MTLBuffer?
    var batchLogitsBuffer: MTLBuffer?
    var batchScratchRows = 0
    var batchedDecodeScratch: BatchedDecodeScratch?
    var fusedDecodeState = FusedDecodeState()
    var headFourRowLaunches = 0  // test-visible: width-4 head launches
    static let mtpChunkCapacity = 32
    let mtpTokenBlock: MTLBuffer?
    let mtpEmbeddingBlock: MTLBuffer?
    let mtpNormalizedEmbeddingBlock: MTLBuffer?
    let mtpNormalizedHiddenBlock: MTLBuffer?
    let mtpConcatBlock: MTLBuffer?
    let mtpProjectedBlock: MTLBuffer?
    let mtpTargetHiddenBlock: MTLBuffer?
    var mtpPrefillReadback: MTLBuffer?
    /// Reusable UInt32 token-ID buffer for chunked prefill (R23): sized to the
    /// largest chunk seen so far and grown on demand, so the prefill hot path
    /// never allocates an MTLBuffer per chunk.
    var prefillTokenBuffer: MTLBuffer?

    /// Host scratch reused across prefill chunks (R38) and decode layers (R16).
    /// The runner is single-flight per generation (guarded by
    /// `prefillChunkState` and the callers' serial decode loop), so these
    /// never alias concurrent work.
    var routeIDScratch: [UInt32] = []
    var routeWeightScratch: [Float16] = []
    var decodeExpertsScratch: [Int] = []
    var decodeHitSlotsScratch: [UInt32] = []
    var decodeMissSlotsScratch: [UInt32] = []
    var decodeHitSplitRoutedBufsScratch: [MTLBuffer] = []
    var decodeHitSplitRoutedOffsetsScratch: [Int] = []
    var decodeRoutedBufsScratch: [MTLBuffer] = []
    var decodeRoutedOffsetsScratch: [Int] = []

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    let effectiveScaleBuffers: [MTLBuffer]
    /// The (model, width) tuning this runner was built with.
    let profile: ModelProfile
    let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    let useFusedGreedyHead: Bool
    let prefillAttentionPath: RuntimePrefillAttentionPath
    let decodeExpertExecution: RuntimeDecodeExpertExecution
    let expertIOSynchronization: RuntimeExpertIOSynchronization
    let expertIOSubmission: RuntimeExpertIOSubmission
    let expertIOBackend: ExpertIOBackend
    let predictivePrefetch: ExpertPrefetchRing?
    let anePrefill: ANEPrefillAttention?
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    var rdadviseSkipUntilPosition: Int = -1
    var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    var rdadviseAdaptivePosition: Int = -1
    var rdadviseAdaptivePositionBytes: UInt64 = 0
    public convenience init(
        model: Model, context: MetalContext, maxContext: Int,
        slots: Int = 1,
        runtimeConfiguration: RuntimeConfiguration = .production,
        enableSpeculativeGDN: Bool = false
    ) throws {
        try self.init(
            model: model, context: context, maxContext: maxContext, slots: slots,
            runtimeConfiguration: runtimeConfiguration,
            enableSpeculativeGDN: enableSpeculativeGDN,
            capabilities: nil)
    }

    /// The designated initializer: every stored property comes from the builder, and an
    /// unset field is a construction bug rather than a silently nil runner.
    init(_ bp: Builder) {
        self.model = Self.staged(bp.model, "model")
        self.ctx = Self.staged(bp.ctx, "ctx")
        self.cfg = Self.staged(bp.cfg, "cfg")
        self.maxContext = Self.staged(bp.maxContext, "maxContext")
        self.slots = Self.staged(bp.slots, "slots")
        self.useFusedGreedyHead = Self.staged(bp.useFusedGreedyHead, "useFusedGreedyHead")
        self.prefillAttentionPath = Self.staged(bp.prefillAttentionPath, "prefillAttentionPath")
        self.profile = Self.staged(bp.profile, "profile")
        self.decodeExpertExecution = Self.staged(bp.decodeExpertExecution, "decodeExpertExecution")
        self.expertIOSynchronization =
            Self.staged(bp.expertIOSynchronization, "expertIOSynchronization")
        self.expertIOSubmission = Self.staged(bp.expertIOSubmission, "expertIOSubmission")
        self.expertIOBackend = Self.staged(bp.expertIOBackend, "expertIOBackend")
        self.predictivePrefetch = Self.staged(bp.predictivePrefetch, "predictivePrefetch")
        self.anePrefill = Self.staged(bp.anePrefill, "anePrefill")
        self.rdadvisePolicyMode = Self.staged(bp.rdadvisePolicyMode, "rdadvisePolicyMode")
        self.rdadviseAdaptiveState = Self.staged(bp.rdadviseAdaptiveState, "rdadviseAdaptiveState")
        self.rdadviseEnabled = Self.staged(bp.rdadviseEnabled, "rdadviseEnabled")
        self.kv = Self.staged(bp.kv, "kv")
        self.embedInt4 = Self.staged(bp.embedInt4, "embedInt4")
        self.affineEmbed = Self.staged(bp.affineEmbed, "affineEmbed")
        self.bf16Rest = Self.staged(bp.bf16Rest, "bf16Rest")
        self.rms = Self.staged(bp.rms, "rms")
        self.int4 = Self.staged(bp.int4, "int4")
        self.affineByWidth = Self.staged(bp.affineByWidth, "affineByWidth")
        self.affine = Self.staged(bp.affine, "affine")
        self.affineKV = Self.staged(bp.affineKV, "affineKV")
        self.affineHead = Self.staged(bp.affineHead, "affineHead")
        self.attention = Self.staged(bp.attention, "attention")
        self.kvQuantizer = Self.staged(bp.kvQuantizer, "kvQuantizer")
        self.shared = Self.staged(bp.shared, "shared")
        self.moe = Self.staged(bp.moe, "moe")
        self.fusionHead = Self.staged(bp.fusionHead, "fusionHead")
        self.fusedQKVGEMV = Self.staged(bp.fusedQKVGEMV, "fusedQKVGEMV")
        self.fusedQKVEpilogue = Self.staged(bp.fusedQKVEpilogue, "fusedQKVEpilogue")
        self.prefillEmbed = Self.staged(bp.prefillEmbed, "prefillEmbed")
        self.prefillRMS = Self.staged(bp.prefillRMS, "prefillRMS")
        self.prefillQMM = Self.staged(bp.prefillQMM, "prefillQMM")
        self.prefillMPPAffineInt4 = Self.staged(bp.prefillMPPAffineInt4, "prefillMPPAffineInt4")
        self.prefillQKVEpilogue = Self.staged(bp.prefillQKVEpilogue, "prefillQKVEpilogue")
        self.prefillAttention = Self.staged(bp.prefillAttention, "prefillAttention")
        self.prefillRouter = Self.staged(bp.prefillRouter, "prefillRouter")
        self.prefillSharedExpert = Self.staged(bp.prefillSharedExpert, "prefillSharedExpert")
        self.prefillGroupedMoE = Self.staged(bp.prefillGroupedMoE, "prefillGroupedMoE")
        self.prefillMoE = Self.staged(bp.prefillMoE, "prefillMoE")
        self.prefillFinalRowHead = Self.staged(bp.prefillFinalRowHead, "prefillFinalRowHead")
        self.elementwise = Self.staged(bp.elementwise, "elementwise")
        self.activationDumpDirectory =
            Self.staged(bp.activationDumpDirectory, "activationDumpDirectory")
        self.hyperConnection = Self.staged(bp.hyperConnection, "hyperConnection")
        self.qsaIndexer = Self.staged(bp.qsaIndexer, "qsaIndexer")
        self.pleHash = Self.staged(bp.pleHash, "pleHash")
        self.ngramTable = Self.staged(bp.ngramTable, "ngramTable")
        self.pleBlock = Self.staged(bp.pleBlock, "pleBlock")
        self.gdn = Self.staged(bp.gdn, "gdn")
        self.gdnState = Self.staged(bp.gdnState, "gdnState")
        self.rope = Self.staged(bp.rope, "rope")
        self.int8ScalarGate = Self.staged(bp.int8ScalarGate, "int8ScalarGate")
        self.bf16ScalarGate = Self.staged(bp.bf16ScalarGate, "bf16ScalarGate")
        self.bf16Projection = Self.staged(bp.bf16Projection, "bf16Projection")
        self.hidden = Self.staged(bp.hidden, "hidden")
        self.normed = Self.staged(bp.normed, "normed")
        self.attnOut = Self.staged(bp.attnOut, "attnOut")
        self.qScratch = Self.staged(bp.qScratch, "qScratch")
        self.kStage = Self.staged(bp.kStage, "kStage")
        self.vStage = Self.staged(bp.vStage, "vStage")
        self.oOut = Self.staged(bp.oOut, "oOut")
        self.h1Buf = Self.staged(bp.h1Buf, "h1Buf")
        self.h2Buf = Self.staged(bp.h2Buf, "h2Buf")
        self.routedX = Self.staged(bp.routedX, "routedX")
        self.denseX = Self.staged(bp.denseX, "denseX")
        self.denseScratchGate = Self.staged(bp.denseScratchGate, "denseScratchGate")
        self.denseScratchUp = Self.staged(bp.denseScratchUp, "denseScratchUp")
        self.denseScratchAct = Self.staged(bp.denseScratchAct, "denseScratchAct")
        self.routerInput = Self.staged(bp.routerInput, "routerInput")
        self.zeroResidual = Self.staged(bp.zeroResidual, "zeroResidual")
        self.outIndices = Self.staged(bp.outIndices, "outIndices")
        self.outWeights = Self.staged(bp.outWeights, "outWeights")
        self.prefetchPredictionIndices =
            Self.staged(bp.prefetchPredictionIndices, "prefetchPredictionIndices")
        self.prefetchPrediction2Indices =
            Self.staged(bp.prefetchPrediction2Indices, "prefetchPrediction2Indices")
        self.prefetchPrediction2Weights =
            Self.staged(bp.prefetchPrediction2Weights, "prefetchPrediction2Weights")
        self.prefetchPredictionWeights =
            Self.staged(bp.prefetchPredictionWeights, "prefetchPredictionWeights")
        self.moeActs = Self.staged(bp.moeActs, "moeActs")
        self.moeHitActiveSlots = Self.staged(bp.moeHitActiveSlots, "moeHitActiveSlots")
        self.moeMissActiveSlots = Self.staged(bp.moeMissActiveSlots, "moeMissActiveSlots")
        self.residencyHitCount = Self.staged(bp.residencyHitCount, "residencyHitCount")
        self.residencyHitPositions = Self.staged(bp.residencyHitPositions, "residencyHitPositions")
        self.residencyMissCount = Self.staged(bp.residencyMissCount, "residencyMissCount")
        self.residencyMissPositions =
            Self.staged(bp.residencyMissPositions, "residencyMissPositions")
        self.residencyMissExperts = Self.staged(bp.residencyMissExperts, "residencyMissExperts")
        self.residencyResolvedSlots =
            Self.staged(bp.residencyResolvedSlots, "residencyResolvedSlots")
        self.residencyResolvedGenerations =
            Self.staged(bp.residencyResolvedGenerations, "residencyResolvedGenerations")
        self.greedyTokenBuf = Self.staged(bp.greedyTokenBuf, "greedyTokenBuf")
        self.verificationHidden = Self.staged(bp.verificationHidden, "verificationHidden")
        self.verificationLogits = Self.staged(bp.verificationLogits, "verificationLogits")
        self.qPackedScratch = Self.staged(bp.qPackedScratch, "qPackedScratch")
        self.attnGateScratch = Self.staged(bp.attnGateScratch, "attnGateScratch")
        self.gdnQKVRaw = Self.staged(bp.gdnQKVRaw, "gdnQKVRaw")
        self.gdnConvOut = Self.staged(bp.gdnConvOut, "gdnConvOut")
        self.gdnZ = Self.staged(bp.gdnZ, "gdnZ")
        self.gdnA = Self.staged(bp.gdnA, "gdnA")
        self.gdnB = Self.staged(bp.gdnB, "gdnB")
        self.gdnY = Self.staged(bp.gdnY, "gdnY")
        self.gdnOut = Self.staged(bp.gdnOut, "gdnOut")
        self.sharedScalarGateBuf = Self.staged(bp.sharedScalarGateBuf, "sharedScalarGateBuf")
        self.mtpTokenBlock = Self.staged(bp.mtpTokenBlock, "mtpTokenBlock")
        self.mtpEmbeddingBlock = Self.staged(bp.mtpEmbeddingBlock, "mtpEmbeddingBlock")
        self.mtpNormalizedEmbeddingBlock =
            Self.staged(bp.mtpNormalizedEmbeddingBlock, "mtpNormalizedEmbeddingBlock")
        self.mtpNormalizedHiddenBlock =
            Self.staged(bp.mtpNormalizedHiddenBlock, "mtpNormalizedHiddenBlock")
        self.mtpConcatBlock = Self.staged(bp.mtpConcatBlock, "mtpConcatBlock")
        self.mtpProjectedBlock = Self.staged(bp.mtpProjectedBlock, "mtpProjectedBlock")
        self.mtpTargetHiddenBlock = Self.staged(bp.mtpTargetHiddenBlock, "mtpTargetHiddenBlock")
        self.mtpPrefillReadback = Self.staged(bp.mtpPrefillReadback, "mtpPrefillReadback")
        self.sharedExpertProjections =
            Self.staged(bp.sharedExpertProjections, "sharedExpertProjections")
        self.effectiveScaleBuffers = Self.staged(bp.effectiveScaleBuffers, "effectiveScaleBuffers")
        self.onesPerExpertScale = Self.staged(bp.onesPerExpertScale, "onesPerExpertScale")
    }

    var decodeIOBaseline: ExpertStreamingStatistics?

    public internal(set) var totalIoNanos: UInt64 = 0
    public internal(set) var totalCb1Nanos: UInt64 = 0
    public internal(set) var totalCb2Nanos: UInt64 = 0
    public internal(set) var totalHeadNanos: UInt64 = 0
    public internal(set) var totalHeadFusedNanos: UInt64 = 0
    // Overlap-analysis counters (TINYTITAN_RUNNER_STATS): the per-layer wall spent
    // waiting on the attention+router command buffer (covers the previous
    // layer's routed CB plus this layer's cb1 on the GPU) and the per-layer
    // loop-body wall. body = cb1 + wait + readback/plan + rdadvise + io + cb2.
    public internal(set) var totalWaitNanos: UInt64 = 0
    public internal(set) var totalBodyNanos: UInt64 = 0
    // Between-token segments (TINYTITAN_RUNNER_STATS): what a decode step spends
    // before its layer loop starts -- the produce preamble (cache pin, KV
    // reserve, QSA check), the embed dispatches, the n-gram gather -- and,
    // filled in by the completion loop, the sampler wait, the progress
    // callback and the loop's own remainder. Together with `body` and `head`
    // these account for the whole token, so a GPU gap can be placed.
    public internal(set) var totalPrefetchAdopted: UInt64 = 0
    public internal(set) var totalPreambleNanos: UInt64 = 0
    public internal(set) var totalPreambleReleaseNanos: UInt64 = 0
    public internal(set) var totalPreamblePinNanos: UInt64 = 0
    public internal(set) var totalPreambleReserveNanos: UInt64 = 0
    public internal(set) var totalEmbedNanos: UInt64 = 0
    public internal(set) var totalGatherNanos: UInt64 = 0
    public var totalLoopSampleNanos: UInt64 = 0
    public var totalLoopProgressNanos: UInt64 = 0
    public var totalLoopOtherNanos: UInt64 = 0
    public internal(set) var totalMissIoNanos: UInt64 = 0
    public internal(set) var totalExposedIoNanos: UInt64 = 0
    public internal(set) var totalHitFixupLayers: UInt64 = 0
    public internal(set) var totalRouterReadbackNanos: UInt64 = 0
    public internal(set) var totalCachePlanNanos: UInt64 = 0
    public internal(set) var totalIOQueueNanos: UInt64 = 0
    public internal(set) var totalIOCompletionToFixupSubmitNanos: UInt64 = 0
    public internal(set) var totalExpertIOHostWaits: UInt64 = 0
    public internal(set) var totalExpertIOHostWaitsAvoided: UInt64 = 0
    public internal(set) var totalGPUClassifiedHits: UInt64 = 0
    public internal(set) var totalGPUClassifiedMisses: UInt64 = 0
    public internal(set) var totalGPUResidencyAllHitLayers: UInt64 = 0
    public internal(set) var lastGreedyToken: UInt32 = 0
    public internal(set) var totalRDAdviseNanos: UInt64 = 0
    public internal(set) var totalRDAdviseCalls: UInt64 = 0
    public internal(set) var totalRDAdviseBytes: UInt64 = 0
    public internal(set) var totalRDAdviseFailures: UInt64 = 0
    public internal(set) var totalRDAdviseSkipped: UInt64 = 0
    var kernelGPUTimings: [KernelGPUTiming] = []
    /// Command buffers split out per kernel under TINYTITAN_KERNEL_SPLIT, awaiting
    /// their GPU timestamps; drained into `kernelGPUTimings` after the layer's
    /// tail wait.
    var splitTimedBuffers: [(role: String, cb: MTLCommandBuffer)] = []
    let kernelGPUTimingsEnabled =
        ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_STATS"] != nil
    let runnerStatsEnabled =
        ProcessInfo.processInfo.environment["TINYTITAN_RUNNER_STATS"] != nil

    /// Open file descriptor for TINYTITAN_ROUTE_TRACE, or -1. Opened once and
    /// never closed: the runner lives as long as the process, and a decode
    /// loop is the wrong place to manage a diagnostic file's lifetime.
    let routeTraceFD: Int32 = {
        guard let path = ProcessInfo.processInfo.environment["TINYTITAN_ROUTE_TRACE"],
            !path.isEmpty
        else { return -1 }
        return RealForwardRunner.openTraceFile(path: path)
    }()

    /// JSONL trace for the v4.3 predictive-prefetch qualification probe.
    /// It deliberately records only exact routing and authoritative cache
    /// residency before planning; enabling it cannot submit I/O or alter cache
    /// decisions. Kept separate from TINYTITAN_ROUTE_TRACE for compatibility.
    let prefetchTraceFD: Int32 = {
        guard let path = ProcessInfo.processInfo.environment["TINYTITAN_PREFETCH_TRACE"],
            !path.isEmpty
        else { return -1 }
        return RealForwardRunner.openTraceFile(path: path)
    }()

    // MARK: - Chunked prefill helpers

    // MARK: - Decode routed-expert helpers

}
