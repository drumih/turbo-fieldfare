import Foundation
import Metal

/// Prefill strategy for `K3Engine.generate`.
public enum K3PrefillMode: Sendable, Equatable {
    /// Replay every prompt token through the single-token decode path.
    /// Correct and deliberately naive — kept as the reference/fallback the
    /// chunked path is validated against.
    case serialReplay
    /// Stage-E1 chunked prefill: `[chunkTokens x hidden]` rows per pass,
    /// NAX tensor-op QMM where available (`K3PrefillGEMM`).
    case chunked(chunkTokens: Int = 32)

    /// Chunk sizes the prefiller supports (scratch geometry and the NAX
    /// tile-M padding are pinned to these).
    public static let allowedChunkTokens = [32, 64, 128, 256]

    public static func isAllowedChunkTokens(_ value: Int) -> Bool {
        allowedChunkTokens.contains(value)
    }
}

/// Exact continuation point for one `K3Engine` instance. The snapshot owns
/// the recurrent KDA/conv state, active MLA cache rows, last-token logits,
/// and routing-predictor history needed to resume without replaying the
/// cached prefix. It is deliberately engine-bound: resident model weights
/// and runtime geometry are not duplicated or treated as interchangeable.
public final class K3PrefixSnapshot: @unchecked Sendable {
    public let tokenIDs: [Int32]
    public let memoryBytes: Int

    fileprivate let engineID: UUID
    fileprivate let state: K3StateSnapshot
    fileprivate let logits: MTLBuffer
    fileprivate let predictions: [Int: [Int]]

    fileprivate init(engineID: UUID,
                     tokenIDs: [Int32],
                     state: K3StateSnapshot,
                     logits: MTLBuffer,
                     predictions: [Int: [Int]]) {
        self.engineID = engineID
        self.tokenIDs = tokenIDs
        self.state = state
        self.logits = logits
        self.predictions = predictions
        self.memoryBytes = state.totalBytes + logits.length
    }
}

/// Timing/statistics footer for one `generate` call, mirroring the house CLI
/// footer fields (`stop`, `prefill`, `new`, `decode`, `tok/s`) plus the K3
/// extras: time to first token, the forward runner's phase counters, and the
/// expert-streaming counters.
public struct K3GenerateStats: Sendable, Equatable {
    public var reason: StopReason
    public var prefillTokens: Int
    /// Prompt tokens restored from an exact K3 state snapshot.
    public var cachedPromptTokens: Int
    /// Prompt tokens actually evaluated during this call.
    public var computedPrefillTokens: Int
    public var newTokens: Int
    public var prefillSeconds: Double
    public var decodeSeconds: Double
    /// Wall time from generate() entry to the first sampled token.
    public var timeToFirstTokenSeconds: Double
    public var tokensPerSecond: Double
    public var position: Int
    public var forward: K3ForwardTimings
    public var expertStreaming: K3ExpertStreamingStats
    /// Per-chunk wall times (ms) when the chunked prefiller ran; empty for
    /// serial replay.
    public var prefillChunkMillis: [Double]
    /// Exact token prefix represented by the engine state at return.
    /// The final sampled stop/max token has not yet been forwarded and is
    /// therefore reported separately in `uncommittedBoundaryTokenIDs`.
    public var stateBackedTokenIDs: [Int32]
    public var uncommittedBoundaryTokenIDs: [Int32]

    public init(reason: StopReason, prefillTokens: Int, newTokens: Int,
                cachedPromptTokens: Int = 0,
                computedPrefillTokens: Int? = nil,
                prefillSeconds: Double, decodeSeconds: Double,
                timeToFirstTokenSeconds: Double, tokensPerSecond: Double,
                position: Int, forward: K3ForwardTimings,
                expertStreaming: K3ExpertStreamingStats,
                prefillChunkMillis: [Double] = [],
                stateBackedTokenIDs: [Int32] = [],
                uncommittedBoundaryTokenIDs: [Int32] = []) {
        self.reason = reason
        self.prefillTokens = prefillTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.computedPrefillTokens = computedPrefillTokens
            ?? max(prefillTokens - cachedPromptTokens, 0)
        self.newTokens = newTokens
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.tokensPerSecond = tokensPerSecond
        self.position = position
        self.forward = forward
        self.expertStreaming = expertStreaming
        self.prefillChunkMillis = prefillChunkMillis
        self.stateBackedTokenIDs = stateBackedTokenIDs
        self.uncommittedBoundaryTokenIDs = uncommittedBoundaryTokenIDs
    }

    /// House-style one-line footer.
    public var footerLine: String {
        let ioBytes = expertStreaming.bytesRead
        let msForward = Double(forward.totalNanos) / 1e6
        let chunks = prefillChunkMillis.isEmpty
            ? ""
            : " chunks=\(prefillChunkMillis.count)x"
                + String(format: "%.1f", prefillChunkMillis.reduce(0, +)
                            / Double(prefillChunkMillis.count)) + "ms"
        let cache = cachedPromptTokens > 0 ? " cached=\(cachedPromptTokens)tok" : ""
        return "[stop=\(reason) prefill=\(prefillTokens)tok\(cache) new=\(newTokens)tok "
            + "decode=\(String(format: "%.2f", decodeSeconds))s "
            + "tok/s=\(String(format: "%.3f", tokensPerSecond)) "
            + "ttft=\(String(format: "%.3f", timeToFirstTokenSeconds))s "
            + "fwd=\(String(format: "%.1f", msForward))ms "
            + "experts=\(expertStreaming.demandHits)/\(expertStreaming.demandTotal)hit "
            + "io=\(ioBytes / (1024 * 1024))MB\(chunks)]"
    }
}

/// Public decode engine for Kimi K3 `.gturbo` v2 bundles: model + state +
/// expert streaming + forward runner + sampler, with the house-style
/// generate loop (raw completion over token ids; chat formatting lives with
/// the caller). A caller may capture and restore an exact engine-bound prefix
/// snapshot; unmatched generations still start from a fresh reset.
///
/// One generation at a time per engine instance (the runner's scratch
/// buffers and the streaming banks are single-in-flight, same contract as
/// the house `RealForwardRunner`). `generate` always starts from a fresh
/// `reset()` unless an exact `K3PrefixSnapshot` is supplied.
public final class K3Engine {
    public let model: K3Model
    public let context: MetalContext
    public let maxContext: Int

    private let state: K3State
    private let streaming: K3ExpertStreaming
    private let runner: K3ForwardRunner
    private let sampler: K3Sampler
    private let probs: MTLBuffer
    private let outToken: MTLBuffer
    /// Stage-E1 prefiller instances, one per requested chunk size (they own
    /// chunk-sized scratch plus the expert tile pool, so they are built on
    /// first use and reused across generate calls).
    private var prefillerCache: [Int: K3ChunkedPrefiller] = [:]
    /// Forces the portable (non-NAX) prefill GEMM path; tests use it to
    /// exercise the fallback on NAX-capable silicon.
    private let prefillForceFallback: Bool
    private let snapshotEngineID = UUID()
    private var captureActivationTrace = false

    /// Open a K3 bundle and build the full decode stack. `device` must be
    /// the system default Metal device (the shared `MetalContext` compiles
    /// the house + K3 shader libraries against it); pass nil to use it
    /// implicitly.
    public static func load(bundleURL: URL,
                            device: MTLDevice? = nil,
                            maxContext: Int,
                            expecting: K3ArchConfig = .kimiK3,
                            prefetchPolicy: K3ExpertPrefetchPolicy = .predict,
                            slotsPerBank: Int = K3ExpertStreaming.defaultSlotsPerBank,
                            ioSplits: Int = K3ExpertStreaming.defaultIOSplits,
                            ioWorkers: K3ExpertIOWorkers = .adaptive,
                            ioCachePolicy: K3ExpertIOCachePolicy = .automatic,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            prefillForceFallback: Bool = false) throws -> K3Engine {
        let context = try MetalContext()
        let dev = device ?? context.device
        let model = try K3Model.load(bundleURL: bundleURL,
                                     device: dev,
                                     expecting: expecting,
                                     integrityPolicy: integrityPolicy)
        let state = try K3State(device: dev, config: model.config,
                                maxContext: maxContext)
        let streaming = try K3ExpertStreaming(model: model,
                                              policy: prefetchPolicy,
                                              slotsPerBank: slotsPerBank,
                                              ioSplits: ioSplits,
                                              ioWorkers: ioWorkers,
                                              ioCachePolicy: ioCachePolicy)
        let runner = try K3ForwardRunner(model: model, context: context,
                                         state: state, streaming: streaming)
        return try K3Engine(model: model, context: context, maxContext: maxContext,
                            state: state, streaming: streaming, runner: runner,
                            prefillForceFallback: prefillForceFallback)
    }

    private init(model: K3Model, context: MetalContext, maxContext: Int,
                 state: K3State, streaming: K3ExpertStreaming,
                 runner: K3ForwardRunner, prefillForceFallback: Bool) throws {
        self.model = model
        self.context = context
        self.maxContext = maxContext
        self.state = state
        self.streaming = streaming
        self.runner = runner
        self.sampler = try K3Sampler(context: context,
                                     vocab: model.config.vocabSize)
        guard let probs = context.device.makeBuffer(
                length: model.config.vocabSize * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let outToken = context.device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.probs = probs
        self.outToken = outToken
        self.prefillForceFallback = prefillForceFallback
    }

    /// Lazily build the chunked prefiller for a chunk size (it owns
    /// chunk-sized scratch plus the expert tile pool — 2 × slotsPerBank ×
    /// expertStride bytes, ~560 MB at the canonical stride).
    private func prefiller(chunkTokens: Int) throws -> K3ChunkedPrefiller {
        if let cached = prefillerCache[chunkTokens] { return cached }
        let prefiller = try K3ChunkedPrefiller(
            model: model, context: context, state: state,
            shared: runner.prefillShared, streaming: streaming,
            chunkTokens: chunkTokens,
            forceFallback: prefillForceFallback ? true : nil)
        if captureActivationTrace { prefiller.enableActivationTrace() }
        prefillerCache[chunkTokens] = prefiller
        return prefiller
    }

    /// Current decode cursor (tokens written to the state slabs).
    public var position: Int { state.position }

    /// Drop all decode state: KDA/conv/MLA slabs, AttnRes scratch, the
    /// position cursor, and the streaming engine's routing predictions.
    /// Resident weights and bank bytes are untouched (they are re-read on
    /// demand either way).
    public func reset() {
        state.reset()
        streaming.resetPrediction()
    }

    /// Zero the timing/streaming counters (diagnostics between runs).
    public func resetStatistics() {
        runner.resetTimings()
        streaming.resetStats()
    }

    public func forwardTimings() -> K3ForwardTimings { runner.timings }

    public func expertStreamingStats() -> K3ExpertStreamingStats {
        streaming.stats()
    }

    /// fp32 logits of the most recent head-emitting `produce` (the last
    /// prefill token or the last decoded token). Test/diagnostic hook.
    public func lastLogits() -> [Float] {
        let count = model.config.vocabSize
        let ptr = logitsPointer()
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Compare the first real activation boundary for one token against a
    /// scalar CPU reconstruction of this exact bundle's resident weights.
    /// This does not reset or otherwise mutate the engine state.
    public func activationDiagnostics(token: Int32) throws -> K3ActivationDiagnostics {
        captureActivationTrace = true
        for prefiller in prefillerCache.values { prefiller.enableActivationTrace() }
        return try runner.activationProbe(token: token)
    }

    public func routerActivationDiagnostics() -> K3RouterActivationDiagnostics? {
        for prefiller in prefillerCache.values {
            if let result = prefiller.lastRouterDiagnostics { return result }
        }
        return runner.lastRouterDiagnostics
    }

    /// Validate selected lm-head rows against scalar real-weight GEMVs. The
    /// rows should include the reported top logits and sampled output token.
    public func headActivationDiagnostics(tokenIDs: [Int]) throws
        -> K3HeadActivationDiagnostics? {
        let headInput = prefillerCache.values.compactMap(\.lastDiagnosticHeadInput).first
            ?? runner.lastDiagnosticHeadInput
        guard let headInput else { return nil }
        let rows = Array(Set(tokenIDs.filter { (0..<model.config.vocabSize).contains($0) }))
            .sorted()
        guard !rows.isEmpty else { return nil }
        let reference = try K3ActivationReference.affineGEMVSamples(
            model.lmHead, x: headInput, rows: rows, roundToFP16: false)
        let logits = lastLogits()
        let actual = rows.map { logits[$0] }
        return K3HeadActivationDiagnostics(
            logits: K3ActivationReference.compare(
                name: "lm_head.logits", actual: actual, reference: reference),
            tokenIDs: rows)
    }

    /// Capture the exact state represented by `tokenIDs`. The caller must
    /// supply the same token sequence used to reach the current cursor; the
    /// count check prevents publishing partial or uncommitted boundaries.
    public func capturePrefixSnapshot(tokenIDs: [Int32]) throws -> K3PrefixSnapshot {
        guard !tokenIDs.isEmpty, tokenIDs.count == state.position else {
            throw GeneratorError.invalidGenerationConfig(
                "prefix snapshot token count must equal engine position")
        }
        let stateSnapshot = try state.captureSnapshot()
        guard let logits = context.device.makeBuffer(
                length: runner.logitsBuffer.length, options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        logits.label = "k3.snapshot.last_logits"
        memcpy(logits.contents(), runner.logitsBuffer.contents(), logits.length)
        return K3PrefixSnapshot(
            engineID: snapshotEngineID,
            tokenIDs: tokenIDs,
            state: stateSnapshot,
            logits: logits,
            predictions: streaming.capturePredictions())
    }

    private func logitsPointer() -> UnsafePointer<Float> {
        UnsafePointer(runner.logitsBuffer.contents()
            .bindMemory(to: Float.self, capacity: model.config.vocabSize))
    }

    /// Prefill `promptTokens`, then decode up to `maxNew` tokens, stopping at
    /// the model eos (`config.eosTokenID`) or any `config.extraStopTokens`.
    /// `onToken` fires per emitted non-stop token.
    ///
    /// Prefill modes: `.serialReplay` replays each prompt token through the
    /// single-token decode path (reference); `.chunked(chunkTokens:)` runs the
    /// Stage-E1 batched prefiller (NAX tensor-op QMM where available). Both
    /// leave the state slabs and the last position's logits in the same
    /// shape; the chunked path is validated to fp16-chained parity against
    /// serial replay.
    ///
    /// `config.maxNewTokens` is ignored in favor of the explicit `maxNew`;
    /// the sampling knobs (temperature/topK/topP/repetitionPenalty/seed)
    /// come from `config`. `config.stopStrings` need a detokenizer and are
    /// therefore not evaluated at this layer.
    @discardableResult
    public func generate(promptTokens: [Int32],
                         config: GenerationConfig = GenerationConfig(),
                         maxNew: Int,
                         prefillMode: K3PrefillMode = .serialReplay,
                         prefixSnapshot: K3PrefixSnapshot? = nil,
                         onToken: ((Int32) -> Void)? = nil) throws -> K3GenerateStats {
        guard !promptTokens.isEmpty else { throw GeneratorError.emptyPrompt }
        guard maxNew > 0 else {
            throw GeneratorError.invalidGenerationConfig("maxNew must be greater than zero")
        }
        // The worst-case produce position is prompt + maxNew - 2 (the last
        // sampled token is never re-produced), and prefill itself needs
        // prompt <= maxContext: the exact bound is prompt + maxNew - 1.
        guard promptTokens.count + maxNew - 1 <= maxContext else {
            throw GeneratorError.contextOverflow(prompt: promptTokens.count,
                                                 maxNew: maxNew,
                                                 maxContext: maxContext)
        }
        var sampling = config
        sampling.maxNewTokens = maxNew
        try sampling.validate()

        let startNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let cachedPromptTokens: Int
        if let snapshot = prefixSnapshot {
            guard snapshot.engineID == snapshotEngineID else {
                throw GeneratorError.invalidGenerationConfig(
                    "prefix snapshot belongs to a different K3 engine")
            }
            guard snapshot.tokenIDs.count <= promptTokens.count,
                  promptTokens.prefix(snapshot.tokenIDs.count)
                    .elementsEqual(snapshot.tokenIDs) else {
                throw GeneratorError.invalidGenerationConfig(
                    "prefix snapshot tokens do not match the prompt")
            }
            state.restoreSnapshot(snapshot.state)
            memcpy(runner.logitsBuffer.contents(), snapshot.logits.contents(),
                   runner.logitsBuffer.length)
            streaming.restorePredictions(snapshot.predictions)
            cachedPromptTokens = snapshot.tokenIDs.count
        } else {
            reset()
            cachedPromptTokens = 0
        }
        let promptSuffix = Array(promptTokens.dropFirst(cachedPromptTokens))

        // Evaluate only the uncached suffix. If it is empty, the snapshot's
        // saved logits already produce the first sample.
        var history = promptTokens
        history.reserveCapacity(promptTokens.count + maxNew)
        var chunkMillis: [Double] = []
        if !promptSuffix.isEmpty {
            switch prefillMode {
            case .serialReplay:
                for (offset, token) in promptSuffix.enumerated() {
                    try runner.produce(
                        token: token,
                        position: cachedPromptTokens + offset,
                        emitHead: offset == promptSuffix.count - 1)
                }
            case .chunked(let chunkTokens):
                let prefiller = try prefiller(chunkTokens: chunkTokens)
                try prefiller.prefill(tokens: promptSuffix)
                chunkMillis = prefiller.lastChunkMillis
            }
        }
        let decodeStartNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let prefillSeconds = Double(decodeStartNanos - startNanos) / 1e9

        var generated = 0
        var reason: StopReason = .maxTokens
        var ttftSeconds = 0.0
        var position = state.position
        let eos = Int32(model.config.eosTokenID)
        var boundaryToken: Int32?

        while true {
            let token = sample(config: sampling, history: history,
                               position: generated)
            if generated == 0 {
                ttftSeconds = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - startNanos) / 1e9
            }
            generated += 1
            boundaryToken = token

            if token == eos || sampling.extraStopTokens.contains(token) {
                reason = .eos
                break
            }
            onToken?(token)
            if generated >= maxNew {
                reason = .maxTokens
                break
            }
            history.append(token)
            try runner.produce(token: token, position: position, emitHead: true)
            position += 1
        }

        let decodeSeconds = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                                    - decodeStartNanos) / 1e9
        return K3GenerateStats(
            reason: reason,
            prefillTokens: promptTokens.count,
            newTokens: generated,
            cachedPromptTokens: cachedPromptTokens,
            computedPrefillTokens: promptSuffix.count,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            timeToFirstTokenSeconds: ttftSeconds,
            tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
            position: state.position,
            forward: runner.timings,
            expertStreaming: streaming.stats(),
            prefillChunkMillis: chunkMillis,
            stateBackedTokenIDs: history,
            uncommittedBoundaryTokenIDs: boundaryToken.map { [$0] } ?? [])
    }

    /// One sampling step over the runner's logits (house `sampleOnce`
    /// shape: repetition penalty on the host, then the GPU kernels).
    private func sample(config: GenerationConfig, history: [Int32],
                        position: Int) -> Int32 {
        guard let cb = context.queue.makeCommandBuffer() else { return 0 }
        sampler.sample(commandBuffer: cb,
                       logits: runner.logitsBuffer,
                       probs: probs,
                       history: history,
                       config: config,
                       position: position,
                       outToken: outToken)
        cb.commit()
        cb.waitUntilCompleted()
        return Int32(bitPattern: outToken.contents().load(as: UInt32.self))
    }
}
