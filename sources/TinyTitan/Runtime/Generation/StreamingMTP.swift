import Foundation
import Metal

/// Target-verified greedy native-MTP session. The target always verifies the draft; a
/// rejected draft is rolled back to the GPU checkpoint captured immediately
/// after the confirmed boundary row.
public final class StreamingMTPDecoder: LogitProducer, ContextWindowReporting,
    /// unchecked-invariant: owns two RealForwardRunners and is driven by one
    /// task at a time, inheriting their exclusive-ownership rule.
    @unchecked Sendable
{
    public let target: RealForwardRunner
    let draft: RealForwardRunner
    public let memoryPlan: StreamingMTPMemoryPlan
    public private(set) var statistics = MTPStatistics()
    public let maxContext: Int
    public let draftMaxContext: Int
    private let targetConfig: ArchConfig
    private var boundaryHidden: Data?

    public init(
        targetModel: Model,
        mtpSidecar: Model,
        context: MetalContext,
        maxContext: Int,
        memoryBudgetMiB: Int = StreamingMTPMemoryPlan.defaultBudgetMiB,
        runtimeConfiguration: RuntimeConfiguration = .production
    ) throws {
        guard
            targetModel.config.family == .qwen36
                || targetModel.config.family == .qwen38flash
                || targetModel.config.family == .qwen4flash
        else {
            throw StreamingMTPError.targetMustBeQwen36
        }
        guard
            mtpSidecar.config.family == .qwen36MTP
                || mtpSidecar.config.family == .qwen38flashMTP
                || mtpSidecar.config.family == .qwen4flashMTP
        else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        guard runtimeConfiguration.ropeScalingMode == .none else {
            throw StreamingMTPError.yaRNUnsupported
        }
        // Validate MTP tensors exist before attempting weight sharing. The
        // errors (missing tensor, wrong layout) propagate as-is (R19) — a
        // broken sidecar must fail loudly at session construction.
        //
        // The two families fuse differently -- one projection over a
        // concatenation against two projections that are summed -- so each
        // checks for its own tensors. Asking for the other's would fail here
        // on a sidecar that is perfectly sound.
        if mtpSidecar.config.family == .qwen38flashMTP
            || mtpSidecar.config.family == .qwen4flashMTP
        {
            _ = try mtpSidecar.mtpWideNorm()
            _ = try mtpSidecar.mtpTokenNorm()
            _ = try mtpSidecar.mtpHiddenProjection()
            _ = try mtpSidecar.mtpEmbeddingProjection()
        } else {
            _ = try mtpSidecar.mtpProjection()
            _ = try mtpSidecar.mtpEmbeddingNorm()
            _ = try mtpSidecar.mtpHiddenNorm()
        }
        let boundDraft = try mtpSidecar.sharingTargetWeights(from: targetModel)
        let targetRunner = try RealForwardRunner(
            model: targetModel,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtimeConfiguration,
            enableSpeculativeGDN: true)
        let draftContext = min(
            maxContext,
            StreamingMTPMemoryPlan.defaultDraftKVTokens)
        let draftRuntime = try RuntimeConfiguration(
            expertCacheSlots: StreamingMTPMemoryPlan.expertSlots,
            expertCachePolicy: runtimeConfiguration.expertCachePolicy,
            rdadvisePolicy: runtimeConfiguration.rdadvisePolicy,
            prefillEnabled: true,
            prefillChunkTokens: 32,
            prefillAttentionPath: runtimeConfiguration.prefillAttentionPath,
            forceLogitsHead: boundDraft.lmHeadWeightBits != 4,
            kvCachePrecision: runtimeConfiguration.kvCachePrecision)
        let draftRunner = try RealForwardRunner(
            model: boundDraft,
            context: context,
            maxContext: draftContext,
            runtimeConfiguration: draftRuntime)
        let draftConfig = boundDraft.config
        let scratch =
            PrefillChunkScratchLayout(
                config: draftConfig,
                chunkTokens: 32
            ).totalPersistentBytes
            + 2 * draftConfig.vocabSize * MemoryLayout<Float16>.stride
            + 32 * draftConfig.hiddenSize * 7 * MemoryLayout<Float16>.stride
        self.memoryPlan = try StreamingMTPMemoryPlan(
            budgetMiB: memoryBudgetMiB,
            residentTensorBytes: mtpSidecar.mtpResidentTensorBytes,
            expertStrideBytes: mtpSidecar.mtpExpertStrideBytes,
            draftKVTokens: draftContext,
            kvCachePrecision: runtimeConfiguration.kvCachePrecision,
            targetRollbackBytes: targetRunner.speculativeRollbackBytes,
            scratchBytes: scratch)
        self.target = targetRunner
        self.draft = draftRunner
        self.maxContext = maxContext
        self.draftMaxContext = draftContext
        self.targetConfig = targetModel.config
    }

    /// The residual width the target carries, which is what the draft is
    /// handed: `hc_count * hidden` for a hyper-connection family.
    private var targetResidualWidth: Int {
        targetConfig.hyperConnections.enabled
            ? targetConfig.hiddenSize * targetConfig.hyperConnections.count
            : targetConfig.hiddenSize
    }

    /// Per-step draft/verify decisions on stderr (`TINYTITAN_MTP_TRACE=1`).
    /// Speculative decoding is supposed to be exact, so when its output
    /// differs from the scalar path the question is always which of these
    /// four numbers is wrong, and the text cannot answer that.
    static let traceEnabled =
        ProcessInfo.processInfo.environment["TINYTITAN_MTP_TRACE"] == "1"

    public func reset() {
        target.reset()
        draft.reset()
        boundaryHidden = nil
        statistics = MTPStatistics()
    }

    /// Required only for protocol compatibility. Callers should use
    /// `prepare`/`advance`; silently taking the scalar path would make an MTP
    /// session's state ambiguous.
    public func produce(
        token: Int32, position: Int,
        into logits: MTLBuffer
    ) async throws {
        throw StreamingMTPError.draftNotReady
    }

    func prepare(
        promptIds: [Int32],
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig,
        logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> Int32 {
        guard config.isPureGreedy else { throw StreamingMTPError.greedyOnly }
        guard promptIds.count + config.maxNewTokens <= maxContext,
            promptIds.count + config.maxNewTokens <= draft.maxContext
        else {
            throw GeneratorError.contextOverflow(
                prompt: promptIds.count,
                maxNew: config.maxNewTokens,
                maxContext: min(maxContext, draft.maxContext))
        }
        reset()
        let result = try await target.prefillChunkedWithMTP(
            tokens: promptIds[...],
            config: prefillConfig,
            into: logits,
            mtp: draft,
            onProgress: onProgress)
        boundaryHidden = result.lastTargetHidden
        switch result.target.seed {
        case .greedyToken(let token):
            return Int32(bitPattern: token)
        case .logitsWritten:
            return Self.argmax(logits, count: targetConfig.vocabSize)
        }
    }

    func advance(boundaryToken: Int32) async throws -> MTPDecodeBatch {
        guard let boundaryHidden else { throw StreamingMTPError.draftNotReady }
        let tProposal = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let draftToken = try await draft.advanceMTPForFamily(
            tokens: [boundaryToken][...],
            targetHiddenRows: boundaryHidden,
            startPosition: draft.continuationPosition,
            predictNext: true)
        // `advanceMTP` with `predictNext: true` always returns a token unless
        // a logic error regressed the prediction path; a distinct error keeps
        // that debuggable instead of masquerading as draft-not-ready (R24).
        guard let draftToken else {
            throw StreamingMTPError.internalInconsistency(
                "draft advance returned no prediction token")
        }

        let tCheckpoint = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let checkpoint = try target.captureSpeculativeCheckpoint(
            maximumBytes: memoryPlan.targetRollbackBytes)
        let tVerify = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let verification = try await target.verifyGreedyPair(
            [boundaryToken, draftToken],
            startPosition: checkpoint.position)
        let tAfterVerify = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let rowBytes = targetResidualWidth * MemoryLayout<Float16>.stride
        let hiddenAfterBoundary = verification.hiddenRows.subdata(in: 0..<rowBytes)
        let accepted = verification.predictionAfterFirst == draftToken
        if Self.traceEnabled {
            let line =
                "[mtp] pos=\(draft.continuationPosition)"
                + " boundary=\(boundaryToken) draft=\(draftToken)"
                + " afterFirst=\(verification.predictionAfterFirst)"
                + " afterSecond=\(verification.predictionAfterSecond)"
                + " accepted=\(accepted)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        if accepted {
            let tCommit = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            _ = try await draft.advanceMTPForFamily(
                tokens: [draftToken][...],
                targetHiddenRows: hiddenAfterBoundary,
                startPosition: draft.continuationPosition,
                predictNext: false)
            self.boundaryHidden = verification.hiddenRows.subdata(in: rowBytes..<(2 * rowBytes))
            statistics.record(accepted: true, emitted: 2, targetPasses: 1)
            statistics.recordPhases(
                proposal: tCheckpoint &- tProposal,
                checkpoint: tVerify &- tCheckpoint,
                verify: tAfterVerify &- tVerify,
                verifyBackbone: verification.backboneNanos,
                verifyHead: verification.headNanos,
                verifyArgmax: verification.argmaxNanos,
                commit: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tCommit,
                rollback: 0)
            return MTPDecodeBatch(
                tokenIDs: [draftToken, verification.predictionAfterSecond],
                backedPrefixCount: 1,
                acceptedDraft: true,
                statistics: statistics)
        }

        // The proposal pass appended one draft KV row. It represented a
        // prediction that the target rejected, so align with Qwen's reference
        // loop by trimming it before the verified replacement is processed.
        let tRollback = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try draft.rewindMTP(to: draft.continuationPosition - 1)
        try target.rollbackSpeculativeCheckpoint(checkpoint)
        self.boundaryHidden = hiddenAfterBoundary
        statistics.record(accepted: false, emitted: 1, targetPasses: 1)
        statistics.recordPhases(
            proposal: tCheckpoint &- tProposal,
            checkpoint: tVerify &- tCheckpoint,
            verify: tAfterVerify &- tVerify,
            verifyBackbone: verification.backboneNanos,
            verifyHead: verification.headNanos,
            verifyArgmax: verification.argmaxNanos,
            commit: 0,
            rollback: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tRollback)
        return MTPDecodeBatch(
            tokenIDs: [verification.predictionAfterFirst],
            backedPrefixCount: 0,
            acceptedDraft: false,
            statistics: statistics)
    }

    private static func argmax(_ logits: MTLBuffer, count: Int) -> Int32 {
        let values = logits.contents().assumingMemoryBound(to: Float16.self)
        var best = 0
        var bestValue = Float(values[0])
        for index in 1..<count {
            let value = Float(values[index])
            if value > bestValue {
                best = index
                bestValue = value
            }
        }
        return Int32(best)
    }

    var targetPosition: Int { target.continuationPosition }
}
