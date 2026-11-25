import Foundation
import TinyTitanCore

/// The facts a caller needs before a model is resident.
///
/// All three are derivable from `manifest.json`, which the installer writes
/// next to the weights — so the startup banner reports real values even when
/// the load has been deferred, instead of placeholders that resolve later.
package struct ModelSessionFacts: Sendable, Equatable {
    package let modelID: String
    package let prefillChunkTokens: Int
    package let promptCacheMode: ServerPromptCacheMode
    /// Routed-expert slots per layer in force, so the banner can state the
    /// streaming budget instead of leaving the user to infer it from a flag they
    /// may not have passed.
    package let expertCacheSlots: Int

    package init(
        modelID: String,
        prefillChunkTokens: Int,
        promptCacheMode: ServerPromptCacheMode,
        expertCacheSlots: Int = 0
    ) {
        self.modelID = modelID
        self.prefillChunkTokens = prefillChunkTokens
        self.promptCacheMode = promptCacheMode
        self.expertCacheSlots = expertCacheSlots
    }
}

/// Everything needed to build a `ServerModelSession`, in one place.
///
/// Both the eager path and the deferred path construct sessions through
/// `makeSession`, so a parameter added to `ServerModelSession.load` cannot be
/// wired into one path and forgotten in the other.
package struct ModelSessionPlan: Sendable {
    package let modelDirectory: URL
    package let maxContext: Int
    package let maxContextExplicit: Bool
    /// How many sequences the session's runner and per-slot scratch support.
    /// One is the historical single-generation server; the coordinator's width
    /// is kept equal to it.
    package let slots: Int
    package let promptCacheMode: ServerPromptCacheMode
    package let promptCacheMaximumEntries: Int
    package let promptCacheMemoryLimitBytes: Int
    package let promptCacheDiskDirectory: URL?
    package let promptCacheDiskLimitBytes: Int
    package let prefillChunkTokens: Int?
    package let kvCachePrecision: KVCachePrecision
    package let hostKVWindowTokens: Int?
    package let ropeScalingMode: RuntimeRoPEScalingMode
    package let thinkingMode: ModelThinkingMode
    /// Reasoning-effort override for effort-aware families; nil keeps the
    /// template default. Family support is validated on load and preview.
    package let reasoningEffort: ModelReasoningEffort?
    package let expertCacheSlots: Int?
    /// Bytes the routed-expert cache may use; slots are derived from it.
    package let expertCacheBudgetBytes: Int?
    package let mtpModelDirectory: URL?
    package let mtpMemoryMiB: Int

    package init(
        modelDirectory: URL,
        maxContext: Int,
        maxContextExplicit: Bool = true,
        slots: Int = 1,
        promptCacheMode: ServerPromptCacheMode,
        promptCacheMaximumEntries: Int,
        promptCacheMemoryLimitBytes: Int,
        promptCacheDiskDirectory: URL?,
        promptCacheDiskLimitBytes: Int,
        prefillChunkTokens: Int?,
        kvCachePrecision: KVCachePrecision = .int8,
        hostKVWindowTokens: Int? = nil,
        ropeScalingMode: RuntimeRoPEScalingMode = .none,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil,
        expertCacheSlots: Int?,
        expertCacheBudgetBytes: Int? = nil,
        mtpModelDirectory: URL?,
        mtpMemoryMiB: Int
    ) {
        self.modelDirectory = modelDirectory
        self.maxContext = maxContext
        self.maxContextExplicit = maxContextExplicit
        self.slots = slots
        self.promptCacheMode = promptCacheMode
        self.promptCacheMaximumEntries = promptCacheMaximumEntries
        self.promptCacheMemoryLimitBytes = promptCacheMemoryLimitBytes
        self.promptCacheDiskDirectory = promptCacheDiskDirectory
        self.promptCacheDiskLimitBytes = promptCacheDiskLimitBytes
        self.prefillChunkTokens = prefillChunkTokens
        self.kvCachePrecision = kvCachePrecision
        self.hostKVWindowTokens = hostKVWindowTokens
        self.ropeScalingMode = ropeScalingMode
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
        self.expertCacheSlots = expertCacheSlots
        self.expertCacheBudgetBytes = expertCacheBudgetBytes
        self.mtpModelDirectory = mtpModelDirectory
        self.mtpMemoryMiB = mtpMemoryMiB
    }

    package func makeSession(
        reusingContext: MetalContext? = nil,
        factory: EngineFactory
    ) async throws -> ServerModelSession {
        try await ServerModelSession.load(
            modelDirectory: modelDirectory,
            factory: factory,
            maxContext: maxContext,
            maxContextExplicit: maxContextExplicit,
            slots: slots,
            promptCacheMode: promptCacheMode,
            promptCacheMaximumEntries: promptCacheMaximumEntries,
            promptCacheMemoryLimitBytes: promptCacheMemoryLimitBytes,
            promptCacheDiskDirectory: promptCacheDiskDirectory,
            promptCacheDiskLimitBytes: promptCacheDiskLimitBytes,
            prefillChunkTokens: prefillChunkTokens,
            kvCachePrecision: kvCachePrecision,
            hostKVWindowTokens: hostKVWindowTokens,
            ropeScalingMode: ropeScalingMode,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort,
            expertCacheSlots: expertCacheSlots,
            expertCacheBudgetBytes: expertCacheBudgetBytes,
            mtpModelDirectory: mtpModelDirectory,
            mtpMemoryMiB: mtpMemoryMiB,
            reusingContext: reusingContext)
    }

    /// Resolve the banner facts by reading `manifest.json` only — no weights
    /// are mapped and no Metal device is created.
    ///
    /// Throwing here also preserves the eager path's behaviour that a bad
    /// `--model` fails at launch rather than on the first request.
    /// The per-family reasoning profile the HTTP layer validates requests
    /// against. Reads `manifest.json` only.
    package func reasoningProfile(factory: EngineFactory) throws -> ServerReasoningProfile {
        let metadata = try factory.metadata(modelDirectory)
        return ServerReasoningProfile(
            metadata: metadata,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
    }

    package func previewFacts(
        modelIDOverride: String? = nil, factory: EngineFactory
    ) throws -> ModelSessionFacts {
        let metadata = try factory.metadata(modelDirectory)
        try factory.validateReasoning(metadata, thinkingMode, reasoningEffort)
        let preview = try factory.preview(modelDirectory, prefillChunkTokens)
        return ModelSessionFacts(
            modelID: modelIDOverride ?? preview.modelID,
            prefillChunkTokens: preview.prefillChunkTokens,
            promptCacheMode: ServerModelSession.effectivePromptCacheMode(
                requested: promptCacheMode,
                mtpEnabled: mtpModelDirectory != nil,
                slots: mtpModelDirectory == nil ? slots : 1))
    }
}
