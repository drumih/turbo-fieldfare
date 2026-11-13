// The loaded model session: decode state, its state, and its initializer.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import TinyTitan

/// Per-generation decode state that `runRawCompletion`'s progress closure
/// mutates. Boxed so the closure captures a reference the compiler can send
/// into the nonisolated call; Swift 6.4 rejects sending the captured mutable
/// struct itself.
///
/// unchecked-invariant: one box per generation, and the coordinator plus the
/// session's slot pool guarantee one generation per occurrence, so exactly one
/// task ever touches a given box.
final class GenerationDecodeState: @unchecked Sendable {
    /// The decoder is per-generation state too, and holding it here is what
    /// lets the progress closure capture only this box: a closure that also
    /// captured the decoder directly is not Sendable, and Swift 6.4 refuses to
    /// send it into the nonisolated completion call.
    ///
    /// `nil` for a raw completion, which is emitted verbatim: a caller that
    /// spelled its own prompt did not ask the engine to interpret think blocks
    /// or tool markers, and the pre-facade CLI streamed them as text.
    let decoder: StructuredAssistantDecoder?
    var output: AssistantOutput
    var decodingError: Error?
    var shouldStop = false
    /// The prompt event is published once, and a fully cached prompt never
    /// produces the prefill progress callback that would otherwise carry it.
    var publishedPromptProgress = false

    init(decoder: StructuredAssistantDecoder?, output: AssistantOutput) {
        self.decoder = decoder
        self.output = output
    }
}

package actor ServerModelSession: ServerInferenceBackend, PromptTokenCounting, PromptCacheDescribing
{
    /// Manifest-derived API model identifier used when --model-id is absent.
    package nonisolated let defaultModelID: String
    /// The session's configured context window; the HTTP layer validates
    /// max_tokens against it (S11).
    package nonisolated var maximumContext: Int { maxContext }
    package nonisolated var samplingDefaults: GenerationDefaults.Sampling {
        profileSampling
    }
    nonisolated let profileSampling: GenerationDefaults.Sampling
    nonisolated let modelFamily: ModelFamily

    let context: MetalContext
    let model: Model
    let tokenizer: GFTokenizer
    /// The tokenizer's vocabulary as byte strings, built the first time a
    /// request asks for structured output and kept for the life of the model.
    /// The bytes of a token id do not change with the reasoning level a request
    /// re-renders at, so one table serves every request this session handles.
    var jsonTokenTable: JSONTokenTable?
    /// Where the tokenizer came from and the reasoning it was rendered at.
    ///
    /// Kept so a request that asks for a different thinking mode or effort can
    /// resolve its own tokenizer through the shared
    /// `(folder, thinking, effort)` cache, instead of being pinned to whatever
    /// the model was loaded with. `nil` reasoning means the session's own.
    let tokenizerFolder: URL
    nonisolated let loadedReasoning: RequestReasoning
    let runner: RealForwardRunner
    /// Non-nil when the session runs more than one slot: decode steps park
    /// here so concurrent slots reach one `produceBatch` together.
    let batchedProducer: (any LogitProducer)?
    let mtpDecoder: StreamingMTPDecoder?
    /// Opt-in wide verify (`TINYTITAN_WIDE_VERIFY=1`), read once at load. It
    /// applies to single-slot runner generations only; `runRawCompletion`
    /// leaves MTP, batched slots and constrained requests on scalar decode.
    nonisolated let wideVerifyOptions = WideVerifyDecodeOptions.fromEnvironment()
    /// One raw-completion scratch per slot: its own logits/probs/token buffers
    /// and its own sampler, so concurrent slots cannot sample from each other's
    /// logits.
    let scratches: [RawCompletionScratch]
    /// Slots not currently held by a generation. Bounded by the coordinator's
    /// width; the waiter queue is a safety net if width ever exceeds slots.
    var freeSlots: [Int]
    var slotWaiters: [SuspensionSlot] = []
    let prefillConfig: PrefillRuntimeConfig
    // Long prompts are prefilled chunk by chunk — small enough to keep expert
    // reads tight.
    package nonisolated let prefillChunkTokens: Int
    /// Routed-expert slots per layer actually in force, so the ready banner can
    /// report the streaming budget rather than leaving the user to infer it.
    package nonisolated let expertCacheSlots: Int
    /// How many sequences this session runs at once.
    package nonisolated let slots: Int
    let maxContext: Int
    package nonisolated let promptCacheMode: ServerPromptCacheMode
    let promptCacheDomain: ServerPromptCacheDomain
    var promptCache: ServerPromptCache
    let promptStateStore: ServerPromptStateStore?
    /// Interior checkpoints (system end, last-turn start). Present only in
    /// multi-prefix mode, where the same storage budget also backs them.
    let promptCheckpointStore: PromptCheckpointStore?
    var activePromptCacheEntryID: UUID?
    /// Concise-mode system prompt injected into every completion, or nil when
    /// concise mode is off. Selected per quantization (see ConcisePrompt).
    nonisolated let concisePrompt: String?
    /// The startup expert warm-up while it may still be running; see
    /// `ServerModelSession+ExpertWarmup.swift`. A request stops it before it
    /// touches the runner, so the two never plan the slot cache at once.
    var expertWarmupTask: Task<ExpertWarmupReport?, Never>?

    deinit {
        // A session dropped mid-warm (a catalog switch, `Engine.unload`) must
        // not keep reading into a cache nobody will use.
        expertWarmupTask?.cancel()
    }

    /// A pure function of its arguments, so a caller can reproduce the
    /// effective cache mode for the startup banner without loading a model.
    package static func effectivePromptCacheMode(
        requested: ServerPromptCacheMode,
        mtpEnabled: Bool,
        slots: Int = 1
    ) -> ServerPromptCacheMode {
        // A target-only snapshot cannot restore the draft stream. Keeping a
        // cache allocated while MTP is active would spend memory on entries
        // that must never be consumed or published.
        guard !mtpEnabled else { return .off }
        // The cache holds one sequence's KV prefix, and its snapshot/restore and
        // `activePromptCacheEntryID` are session-wide. With more than one slot
        // that entry could be restored into the wrong sequence, which produces
        // plausible wrong output rather than an error, so batching runs with the
        // cache off and re-prefills each turn until it is slot-keyed.
        return slots > 1 ? .off : requested
    }

    /// The prompt cache a catalog server's *initial* model will really run,
    /// which is what the routing banner states beside `engine=`.
    ///
    /// The extra input over `effectivePromptCacheMode` is the engine: a CPU
    /// entry has no cache at all, so it reports `.off` rather than a mode the
    /// server asked for that could never exist. The catalog loader never
    /// attaches MTP, so the engine and the width are the whole rule. Kept here
    /// rather than in the executable so both arms are testable without a
    /// catalog on disk.
    package static func initialPromptCacheMode(
        backend: ModelBackend,
        requested: ServerPromptCacheMode,
        maxConcurrentSequences: Int
    ) -> ServerPromptCacheMode {
        guard backend != .cpu else { return .off }
        return effectivePromptCacheMode(
            requested: requested, mtpEnabled: false,
            slots: maxConcurrentSequences)
    }

    /// Tokenize a raw-completion prompt: the text with BOS and no chat
    /// template.
    ///
    /// The one place a facade `.raw` prompt becomes token ids. It runs on the
    /// actor because the tokenizer lives here, so no front end has to own one.
    package func tokenizeRawPrompt(_ text: String) -> [Int32] {
        tokenizer.encode(text, addBOS: true)
    }

    init(
        context: MetalContext,
        model: Model,
        tokenizer: GFTokenizer,
        tokenizerFolder: URL,
        loadedReasoning: RequestReasoning,
        runner: RealForwardRunner,
        mtpDecoder: StreamingMTPDecoder?,
        scratches: [RawCompletionScratch],
        prefillConfig: PrefillRuntimeConfig,
        expertCacheSlots: Int,
        slots: Int,
        maxContext: Int,
        promptCacheMode: ServerPromptCacheMode,
        promptCacheDomain: ServerPromptCacheDomain,
        promptCache: ServerPromptCache,
        promptStateStore: ServerPromptStateStore?,
        promptCheckpointStore: PromptCheckpointStore?,
        concisePrompt: String?
    ) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizerFolder = tokenizerFolder
        self.loadedReasoning = loadedReasoning
        self.modelFamily = model.config.family
        self.profileSampling =
            ModelProfile.resolve(
                modelID: model.modelID, family: model.config.family,
                weightBits: model.routedExpertWeightBits
            ).sampling
        self.defaultModelID = ServerModelIdentity.apiModelID(
            manifestModelID: model.modelID,
            family: model.config.family,
            weightBits: model.routedExpertWeightBits)
        self.runner = runner
        self.batchedProducer =
            slots > 1 ? BatchedLogitProducer(runner: runner, width: slots) : nil
        self.mtpDecoder = mtpDecoder
        self.scratches = scratches
        self.slots = max(1, slots)
        self.freeSlots = Array(0..<max(1, slots))
        self.prefillConfig = prefillConfig
        self.prefillChunkTokens = prefillConfig.chunkTokens
        self.expertCacheSlots = expertCacheSlots
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
        self.promptCache = promptCache
        self.promptStateStore = promptStateStore
        self.promptCheckpointStore = promptCheckpointStore
        self.concisePrompt = concisePrompt
    }
}
