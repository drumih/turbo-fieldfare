import Foundation
import Metal

// The staging object for RealForwardRunner's two-phase initializer. Every field
// is optional (a double optional where the property itself is optional) so the
// builder can be filled in construction order, and the class's designated
// initializer refuses an unset field. Split out of RealForwardRunner.swift
// (2026-09-28) under the 500-line-per-file rule; construction order is unchanged.

extension RealForwardRunner {
    /// Values gathered by `buildCore` and `buildScratch` before the runner exists.
    final class Builder {
        var model: Model?
        var ctx: MetalContext?
        var cfg: ArchConfig?
        var maxContext: Int?
        var slots: Int?
        var useFusedGreedyHead: Bool?
        var prefillAttentionPath: RuntimePrefillAttentionPath?
        var profile: ModelProfile?
        var decodeExpertExecution: RuntimeDecodeExpertExecution?
        var expertIOSynchronization: RuntimeExpertIOSynchronization?
        var expertIOSubmission: RuntimeExpertIOSubmission?
        var expertIOBackend: ExpertIOBackend?
        var predictivePrefetch: ExpertPrefetchRing??
        var anePrefill: ANEPrefillAttention??
        var rdadvisePolicyMode: RDAdvicePolicyMode?
        var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState?
        var rdadviseEnabled: Bool?
        var kv: KVCacheManager??
        var embedInt4: EmbedLookupInt4?
        var affineEmbed: AffineQuantEmbeddingLookup??
        var bf16Rest: BF16RestKernels?
        var rms: RMSNorm?
        var int4: DequantInt4GEMV?
        var affineByWidth: [Int: AffineQuantGEMV]?
        var affine: AffineQuantGEMV??
        var affineKV: AffineQuantGEMV??
        var affineHead: AffineQuantGEMV??
        var attention: Attention?
        var kvQuantizer: KVCacheQuantizer??
        var shared: SharedExpertRuntime?
        var moe: MoE?
        var fusionHead: LMHeadChainInt4?
        var fusedQKVGEMV: FusedQKVGEMV?
        var fusedQKVEpilogue: FusedQKVEpilogue?
        var prefillEmbed: PrefillEmbedLookupInt4?
        var prefillRMS: PrefillRMSNorm?
        var prefillQMM: PrefillInt4QMM?
        var prefillMPPAffineInt4: MPPPrefillInt4QMM??
        var prefillQKVEpilogue: PrefillQKVEpilogue?
        var prefillAttention: PrefillAttention?
        var prefillRouter: PrefillRouter?
        var prefillSharedExpert: PrefillSharedExpert?
        var prefillGroupedMoE: PrefillGroupedRoutedMoE?
        var prefillMoE: PrefillMoE?
        var prefillFinalRowHead: PrefillFinalRowHeadInt4?
        var elementwise: Elementwise??
        var activationDumpDirectory: URL??
        var hyperConnection: HyperConnection??
        var qsaIndexer: QSAIndexer??
        var pleHash: PLEHash??
        var ngramTable: NgramTableReader??
        var pleBlock: PLEBlock??
        var gdn: GDN??
        var gdnState: GDNStateManager??
        var rope: RoPE??
        var int8ScalarGate: DequantInt8GEMV??
        var bf16ScalarGate: BF16GEMV??
        var bf16Projection: BF16GEMV?
        var hidden: MTLBuffer?
        var normed: MTLBuffer?
        var attnOut: MTLBuffer?
        var qScratch: MTLBuffer?
        var kStage: MTLBuffer?
        var vStage: MTLBuffer?
        var oOut: MTLBuffer?
        var h1Buf: MTLBuffer?
        var h2Buf: MTLBuffer?
        var routedX: MTLBuffer?
        var denseX: MTLBuffer?
        var denseScratchGate: MTLBuffer?
        var denseScratchUp: MTLBuffer?
        var denseScratchAct: MTLBuffer?
        var routerInput: MTLBuffer?
        var zeroResidual: MTLBuffer?
        var outIndices: MTLBuffer?
        var outWeights: MTLBuffer?
        var prefetchPredictionIndices: MTLBuffer?
        var prefetchPrediction2Indices: MTLBuffer?
        var prefetchPrediction2Weights: MTLBuffer?
        var prefetchPredictionWeights: MTLBuffer?
        var moeActs: MTLBuffer?
        var moeHitActiveSlots: MTLBuffer?
        var moeMissActiveSlots: MTLBuffer?
        var residencyHitCount: MTLBuffer?
        var residencyHitPositions: MTLBuffer?
        var residencyMissCount: MTLBuffer?
        var residencyMissPositions: MTLBuffer?
        var residencyMissExperts: MTLBuffer?
        var residencyResolvedSlots: MTLBuffer?
        var residencyResolvedGenerations: MTLBuffer?
        var greedyTokenBuf: MTLBuffer?
        var verificationHidden: MTLBuffer?
        var verificationLogits: MTLBuffer?
        var qPackedScratch: MTLBuffer??
        var attnGateScratch: MTLBuffer??
        var gdnQKVRaw: MTLBuffer??
        var gdnConvOut: MTLBuffer??
        var gdnZ: MTLBuffer??
        var gdnA: MTLBuffer??
        var gdnB: MTLBuffer??
        var gdnY: MTLBuffer??
        var gdnOut: MTLBuffer??
        var sharedScalarGateBuf: MTLBuffer??
        var mtpTokenBlock: MTLBuffer??
        var mtpEmbeddingBlock: MTLBuffer??
        var mtpNormalizedEmbeddingBlock: MTLBuffer??
        var mtpNormalizedHiddenBlock: MTLBuffer??
        var mtpConcatBlock: MTLBuffer??
        var mtpProjectedBlock: MTLBuffer??
        var mtpTargetHiddenBlock: MTLBuffer??
        var mtpPrefillReadback: MTLBuffer??
        var sharedExpertProjections: [LayerSharedExpertProjections]?
        var effectiveScaleBuffers: [MTLBuffer]?
        var onesPerExpertScale: MTLBuffer??
    }
}

extension RealForwardRunner {
    /// Returns a builder field once the two build phases have set it. An unset field is a
    /// construction bug, so it traps here rather than handing back a silently nil runner.
    /// Internal rather than private so the designated initializer in
    /// RealForwardRunner.swift can call it.
    static func staged<T>(_ value: T?, _ field: String) -> T {
        guard let value else {
            preconditionFailure("the runner builder did not set \(field)")
        }
        return value
    }
}
