import Foundation
import Metal
import TurboFieldfare
import Synchronization

final class GenerationTaskRegistry: Sendable {
    private struct Entry: Sendable {
        let id: UUID
        var task: Task<Void, Never>?
    }

    private let state = Mutex<Entry?>(nil)

    func reserve(_ id: UUID) -> Bool {
        state.withLock { entry in
            guard entry == nil else { return false }
            entry = Entry(id: id, task: nil)
            return true
        }
    }

    func attach(_ task: Task<Void, Never>, to id: UUID) {
        let shouldCancel = state.withLock { entry -> Bool in
            guard entry?.id == id else { return true }
            entry?.task = task
            return false
        }
        if shouldCancel { task.cancel() }
    }

    func take(_ id: UUID) -> Task<Void, Never>? {
        state.withLock { entry in
            guard entry?.id == id else { return nil }
            defer { entry = nil }
            return entry?.task
        }
    }

    func takeCurrent() -> Task<Void, Never>? {
        state.withLock { entry in
            defer { entry = nil }
            return entry?.task
        }
    }

    func clear(_ id: UUID) {
        state.withLock { entry in
            if entry?.id == id { entry = nil }
        }
    }

}

/// Real-model inference client for the Mac app. Wraps the same raw-completion
/// loop the CLI uses (`runRawCompletion`, BOS + verbatim encode, no chat
/// template) behind the `AppInferenceClient` event stream, with an explicit
/// load lifecycle so the resident weights stay warm across generations.
public final class RealInferenceClient: AppModelLifecycleClient, @unchecked Sendable {
    private let session: RealInferenceSession
    private let memorySampler: AppMemorySampler
    private let generationTasks = GenerationTaskRegistry()

    public init(memorySampler: AppMemorySampler = AppMemorySampler()) {
        self.memorySampler = memorySampler
        self.session = RealInferenceSession()
    }

    public func ensureLoaded(modelDirectory: URL,
                             maxContextTokens: Int,
                             options: AppRuntimeOptions,
                             forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        try await session.ensureLoaded(
            key: SessionLoadKey(directory: modelDirectory.standardizedFileURL,
                                maxContext: maxContextTokens,
                                options: options,
                                forceLogitsHead: forceLogitsHead),
            onState: onState)
    }

    public func unload() async {
        await session.unload()
    }

    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let generationID = UUID()
            guard generationTasks.reserve(generationID) else {
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task { [self] in
                await session.run(request: request,
                                  memorySampler: memorySampler,
                                  continuation: continuation)
                generationTasks.clear(generationID)
            }
            generationTasks.attach(task, to: generationID)

            continuation.onTermination = { [generationTasks] _ in
                generationTasks.take(generationID)?.cancel()
            }
        }
    }

    public func cancel() {
        generationTasks.takeCurrent()?.cancel()
    }

}

struct SessionLoadKey: Equatable, Sendable {
    var directory: URL
    var maxContext: Int
    var options: AppRuntimeOptions
    var forceLogitsHead: Bool

    init(directory: URL,
         maxContext: Int,
         options: AppRuntimeOptions,
         forceLogitsHead: Bool = false) {
        self.directory = directory.standardizedFileURL
        self.maxContext = maxContext
        self.options = options
        self.forceLogitsHead = forceLogitsHead
    }
}

struct TokenizerDirectoryCache: Equatable, Sendable {
    private(set) var directory: URL?

    func shouldReload(for modelDirectory: URL) -> Bool {
        directory != modelDirectory.standardizedFileURL
    }

    mutating func markLoaded(for modelDirectory: URL) {
        directory = modelDirectory.standardizedFileURL
    }

    mutating func clear() {
        directory = nil
    }
}

/// Owns the loaded model and serializes load / unload / generate. All Metal
/// command-buffer waits happen inside this actor, off the main actor; one
/// cooperative-pool thread is occupied for the duration of a generation,
/// which is acceptable for the app's single session. The 8 GB rule lives
/// here: a reload releases the loaded model, runner, and scratch before constructing
/// replacements, so two models are never alive at once.
actor RealInferenceSession {
    private var loadedKey: SessionLoadKey?
    private var ctx: MetalContext?
    private var tokenizer: GFTokenizer?
    private var tokenizerDirectoryCache = TokenizerDirectoryCache()
    private var runner: RealForwardRunner?
    private var scratch: RawCompletionScratch?

    func ensureLoaded(key: SessionLoadKey,
                      onState: @Sendable (AppModelLoadState) -> Void) async throws {
        if loadedKey == key, runner != nil { return }

        runner = nil
        scratch = nil
        loadedKey = nil

        let start = Date()
        do {
            onState(.loading(.validatingDirectory))
            let manifest = key.directory.appendingPathComponent("manifest.json")
            guard FileManager.default.fileExists(atPath: manifest.path) else {
                throw AppInferenceError.modelNotFound(key.directory.path)
            }

            onState(.loading(.tokenizer))
            if tokenizer == nil || tokenizerDirectoryCache.shouldReload(for: key.directory) {
                do {
                    tokenizer = try await Self.loadTokenizer(for: key.directory)
                    tokenizerDirectoryCache.markLoaded(for: key.directory)
                } catch {
                    throw AppInferenceError.tokenizerUnavailable("\(error)")
                }
            }
            try Task.checkCancellation()

            onState(.loading(.verifyingWeights))
            let runtimeConfiguration = try key.options.resolvedRuntimeConfiguration(
                forceLogitsHead: key.forceLogitsHead)
            let context: MetalContext
            if let ctx {
                context = ctx
            } else {
                context = try MetalContext()
                ctx = context
            }
            let loadedModel = try Model.load(
                directoryURL: key.directory,
                device: context.device,
                streamingMode: .pread(slotCount: runtimeConfiguration.expertCacheSlots),
                expertCachePolicy: runtimeConfiguration.modelExpertCachePolicy,
                integrityPolicy: key.options.modelVerification.runtimeValue)
            try Task.checkCancellation()

            onState(.loading(.preparingRunner))
            let loadedRunner = try RealForwardRunner(
                model: loadedModel,
                context: context,
                maxContext: key.maxContext,
                runtimeConfiguration: runtimeConfiguration)
            let loadedScratch = try RawCompletionScratch(context: context,
                                                         vocab: loadedModel.config.vocabSize)
            try Task.checkCancellation()

            runner = loadedRunner
            scratch = loadedScratch
            loadedKey = key
            onState(.ready(modelDirectory: key.directory,
                           loadSeconds: Date().timeIntervalSince(start)))
        } catch is CancellationError {
            throw CancellationError()
        } catch let appError as AppInferenceError {
            onState(.failed(appError))
            throw appError
        } catch {
            let appError = AppInferenceError.modelLoadFailed("\(error)")
            onState(.failed(appError))
            throw appError
        }
    }

    private static func loadTokenizer(for modelDirectory: URL) async throws -> GFTokenizer {
        try await GFTokenizer.load(forModelDirectory: modelDirectory)
    }

    static func forceLogitsHead(for request: AppGenerationRequest) -> Bool {
        !request.isPureGreedy
    }

    static func generationConfig(for request: AppGenerationRequest,
                                 maxNewTokens: Int? = nil) -> GenerationConfig {
        GenerationConfig(maxNewTokens: maxNewTokens ?? request.maxNewTokens,
                         temperature: request.temperature,
                         topK: request.topK,
                         topP: request.topP,
                         repetitionPenalty: request.repetitionPenalty)
    }

    static func effectiveMaxNewTokens(requested: Int,
                                      promptTokenCount: Int,
                                      maxContext: Int) -> Int {
        min(requested, max(0, maxContext - promptTokenCount))
    }

    func unload() {
        runner = nil
        scratch = nil
        tokenizer = nil
        tokenizerDirectoryCache.clear()
        loadedKey = nil
    }

    func run(request: AppGenerationRequest,
             memorySampler: AppMemorySampler,
             continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        let prefillConfig = request.runtimeOptions.prefillConfig
        let progress = ProgressState()
        do {
            try request.validate()
            let executedPrefillMode: PrefillExecutedMode =
                prefillConfig.mode == .chunked ? .chunked : .off
            let prefillDiagnostics = PrefillExecutionDiagnostics(config: prefillConfig,
                                                                 executedMode: executedPrefillMode,
                                                                 kvStorageMode: .fp16)
            let requestKey = SessionLoadKey(
                directory: request.modelDirectory.standardizedFileURL,
                maxContext: request.maxContextTokens,
                options: request.runtimeOptions,
                forceLogitsHead: Self.forceLogitsHead(for: request))
            guard let loadedKey else { throw AppInferenceError.modelNotLoaded }
            guard loadedKey == requestKey else { throw AppInferenceError.reloadRequired }
            guard let runner, let tokenizer, let ctx, let scratch else {
                throw AppInferenceError.modelLoadFailed("session lost its loaded state")
            }

            let renderedPrompt = try tokenizer.applyChatTemplate([
                GFTokenizer.Message(role: .user, content: request.prompt)
            ])
            let promptIds = tokenizer.encode(renderedPrompt, addBOS: false)
            progress.promptTokenCount = promptIds.count
            guard promptIds.count < runner.maxContext else {
                throw AppInferenceError.contextOverflow(prompt: promptIds.count,
                                                        maxNew: request.maxNewTokens,
                                                        maxContext: runner.maxContext)
            }
            memorySampler.resetPeak()
            _ = memorySampler.sample()
            let config = Self.generationConfig(
                for: request,
                maxNewTokens: Self.effectiveMaxNewTokens(
                    requested: request.maxNewTokens,
                    promptTokenCount: promptIds.count,
                    maxContext: runner.maxContext))
            runner.reset()
            progress.prefillStart = Date()

            let result = try await runRawCompletion(
                producer: runner, tokenizer: tokenizer, promptIds: promptIds,
                config: config, context: ctx, scratch: scratch,
                prefillConfig: prefillConfig) { event in
                switch event {
                case .prefill(let done, let total):
                    if done == total {
                        progress.decodeStart = Date()
                        progress.countersAtDecodeStart = RunnerCounterSnapshot(runner)
                    }
                    continuation.yield(.prefillProgress(done: done, total: total))
                case .token(let index, _, let delta):
                    if progress.firstTokenDate == nil { progress.firstTokenDate = Date() }
                    progress.generated = index + 1
                    if index % 8 == 0 { _ = memorySampler.sample() }
                    continuation.yield(.token(AppTokenEvent(
                        index: index,
                        textDelta: delta,
                        elapsedDecodeSeconds: progress.elapsedDecodeSeconds)))
                case .tail(let text):
                    continuation.yield(.token(AppTokenEvent(
                        index: max(progress.generated - 1, 0),
                        textDelta: text,
                        elapsedDecodeSeconds: progress.elapsedDecodeSeconds)))
                }
            }

            let diagnostics = makeDiagnostics(request: request,
                                              memorySampler: memorySampler,
                                              progress: progress,
                                              stopReason: Self.stopReason(result.reason),
                                              prefillSeconds: result.prefillSeconds,
                                              decodeSeconds: result.decodeSeconds,
                                              generated: result.newTokens,
                                              prefill: prefillDiagnostics)
            continuation.yield(.finished(diagnostics))
            continuation.finish()
        } catch is CancellationError {
            let diagnostics = makeDiagnostics(request: request,
                                              memorySampler: memorySampler,
                                              progress: progress,
                                              stopReason: .cancelled,
                                              prefillSeconds: progress.elapsedPrefillSeconds,
                                              decodeSeconds: progress.elapsedDecodeSeconds,
                                              generated: progress.generated,
                                              prefill: PrefillExecutionDiagnostics(
                                                config: prefillConfig,
                                                executedMode: prefillConfig.mode == .chunked ? .chunked : .off,
                                                kvStorageMode: .fp16))
            continuation.yield(.cancelled(diagnostics))
            continuation.finish(throwing: AppInferenceError.cancelled)
        } catch let prefillError as PrefillError {
            let diagnostics = Self.prefillFailureDiagnostics(config: prefillConfig,
                                                             kvStorageMode: .fp16,
                                                             reason: prefillError.description)
            failGeneration(.unknown(prefillError.description),
                           request: request,
                           memorySampler: memorySampler,
                           progress: progress,
                           continuation: continuation,
                           prefill: diagnostics,
                           forcePartialDiagnostics: true)
        } catch let appError as AppInferenceError {
            failGeneration(appError, request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        } catch {
            failGeneration(.unknown("\(error)"), request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        }
    }

    private func failGeneration(_ error: AppInferenceError,
                                request: AppGenerationRequest,
                                memorySampler: AppMemorySampler,
                                progress: ProgressState,
                                continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
                                prefill: PrefillExecutionDiagnostics? = nil,
                                forcePartialDiagnostics: Bool = false) {
        let partial = progress.generated > 0 || forcePartialDiagnostics
            ? makeDiagnostics(request: request, memorySampler: memorySampler,
                              progress: progress, stopReason: .failed,
                              prefillSeconds: progress.elapsedPrefillSeconds,
                              decodeSeconds: progress.elapsedDecodeSeconds,
                              generated: progress.generated,
                              prefill: prefill)
            : nil
        continuation.yield(.failed(error, partial: partial))
        continuation.finish(throwing: error)
    }

    private func makeDiagnostics(request: AppGenerationRequest,
                                 memorySampler: AppMemorySampler,
                                 progress: ProgressState,
                                 stopReason: AppStopReason,
                                 prefillSeconds: Double? = nil,
                                 decodeSeconds: Double,
                                 generated: Int,
                                 prefill: PrefillExecutionDiagnostics? = nil) -> AppDiagnostics {
        _ = memorySampler.sample()
        let ttft: Double?
        if let first = progress.firstTokenDate, let start = progress.decodeStart {
            ttft = first.timeIntervalSince(start)
        } else {
            ttft = nil
        }
        return AppDiagnostics(
            generatedTokens: generated,
            stopReason: stopReason,
            promptTokenCount: progress.promptTokenCount,
            prefillSeconds: prefillSeconds,
            timeToFirstTokenSeconds: ttft,
            decodeSeconds: decodeSeconds,
            tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
            peakMemoryBytes: memorySampler.peakBytes,
            runtimeOptions: request.runtimeOptions,
            prefill: prefill,
            runner: runnerDiagnostics(progress: progress, generated: generated))
    }

    /// Per-token buckets as diffs of the runner's cumulative counters from the
    /// decode start (excludes prefill), divided by the decode forward count.
    /// The forward count is `generated - 1`: each loop iteration that continues
    /// ends with one `produce`; the final sampled token never runs a forward.
    private func runnerDiagnostics(progress: ProgressState, generated: Int) -> AppRunnerDiagnostics? {
        guard let runner, let base = progress.countersAtDecodeStart, generated > 1 else { return nil }
        let now = RunnerCounterSnapshot(runner)
        let forwards = Double(generated - 1)
        func ms(_ end: UInt64, _ start: UInt64) -> Double {
            Double(end &- start) / 1_000_000 / forwards
        }
        return AppRunnerDiagnostics(
            cb1MillisecondsPerToken: ms(now.cb1, base.cb1),
            ioMillisecondsPerToken: ms(now.io, base.io),
            cb2MillisecondsPerToken: ms(now.cb2, base.cb2),
            headMillisecondsPerToken: ms(now.head, base.head),
            rdadviseMillisecondsPerToken: ms(now.rdadvise, base.rdadvise),
            rdadviseCallsPerToken: Double(now.rdadviseCalls &- base.rdadviseCalls) / forwards,
            rdadviseMegabytesPerToken: Double(now.rdadviseBytes &- base.rdadviseBytes) / 1_048_576.0 / forwards,
            rdadviseSkippedPerToken: Double(now.rdadviseSkipped &- base.rdadviseSkipped) / forwards,
            rdadviseFailures: now.rdadviseFailures &- base.rdadviseFailures)
    }

    private static func stopReason(_ reason: StopReason) -> AppStopReason {
        switch reason {
        case .eos: return .eos
        case .endOfTurn: return .endOfTurn
        case .maxTokens: return .maxTokens
        case .stopString: return .stopString
        }
    }

    internal static func prefillFailureDiagnostics(config: PrefillRuntimeConfig,
                                                   kvStorageMode: PrefillKVStorageMode,
                                                   reason: String) -> PrefillExecutionDiagnostics {
        PrefillExecutionDiagnostics.unsupported(config: config,
                                                kvStorageMode: kvStorageMode,
                                                reason: reason)
    }
}

/// Mutable per-generation state shared between the progress callback and the
/// surrounding actor method. Single-threaded: the callback runs synchronously
/// inside `runRawCompletion` on the session actor's task.
private final class ProgressState: @unchecked Sendable {
    var generated = 0
    var promptTokenCount: Int?
    var prefillStart: Date?
    var decodeStart: Date?
    var firstTokenDate: Date?
    var countersAtDecodeStart: RunnerCounterSnapshot?

    var elapsedDecodeSeconds: Double {
        guard let decodeStart else { return 0 }
        return Date().timeIntervalSince(decodeStart)
    }

    var elapsedPrefillSeconds: Double? {
        guard let prefillStart else { return nil }
        let end = decodeStart ?? Date()
        return max(end.timeIntervalSince(prefillStart), 0)
    }
}

private struct RunnerCounterSnapshot {
    let cb1: UInt64
    let io: UInt64
    let cb2: UInt64
    let head: UInt64
    let rdadvise: UInt64
    let rdadviseCalls: UInt64
    let rdadviseBytes: UInt64
    let rdadviseFailures: UInt64
    let rdadviseSkipped: UInt64

    init(_ runner: RealForwardRunner) {
        cb1 = runner.totalCb1Nanos
        io = runner.totalIoNanos
        cb2 = runner.totalCb2Nanos
        head = runner.totalHeadNanos &+ runner.totalHeadFusedNanos
        rdadvise = runner.totalRDAdviseNanos
        rdadviseCalls = runner.totalRDAdviseCalls
        rdadviseBytes = runner.totalRDAdviseBytes
        rdadviseFailures = runner.totalRDAdviseFailures
        rdadviseSkipped = runner.totalRDAdviseSkipped
    }
}
