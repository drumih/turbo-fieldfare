import Foundation
import TinyTitan

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

package enum ServerModelIdentity {
    /// The advertised id, which always ends in the quantization.
    ///
    /// Two installs of the same weights at different widths are different
    /// models to anyone choosing between them, and an id that hides which one
    /// is loaded makes `/v1/models` useless for telling them apart. The width
    /// comes from the manifest's routed-expert slot rather than from parsing
    /// the id, so it is right even when the id says nothing.
    package static func apiModelID(
        manifestModelID: String,
        family: ModelFamily,
        weightBits: Int
    ) -> String {
        base(manifestModelID: manifestModelID, family: family)
            + "_\(weightBits)-Bit"
    }

    /// The id with any quantization the manifest already spelled removed, so
    /// the suffix is added exactly once. The catalog names installs by it.
    package static func base(
        manifestModelID: String,
        family: ModelFamily
    ) -> String {
        for suffix in ["-4bit", "-8bit", "-6bit"]
        where manifestModelID.hasSuffix(suffix) {
            return String(manifestModelID.dropLast(suffix.count))
        }
        if manifestModelID != "unknown/snapshot" {
            return manifestModelID
        }
        switch family {
        case .qwen36: return "qwen3.6-35b-a3b"
        case .qwen36MTP: return "qwen3.6-35b-a3b-mtp"
        case .qwen38flash: return "qwen3.8-flash-next"
        case .qwen38flashMTP: return "qwen3.8-flash-next-mtp"
        case .qwen4flash: return "qwen4-omni-120b-think"
        case .qwen4flashMTP: return "qwen4-omni-120b-think-mtp"
        // Reached only for an install whose manifest carries no model id; the
        // dense converter always writes one.
        case .qwen35Dense: return "qwen3.5-dense"
        case .k2Horizon: return "k2-horizon"
        case .laguna: return "laguna-s-2-1"
        }
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
        reusingContext: MetalContext? = nil
    ) async throws -> ServerModelSession {
        try await ServerModelSession.load(
            modelDirectory: modelDirectory,
            maxContext: maxContext,
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
    package func reasoningProfile() throws -> ServerReasoningProfile {
        let family = try ManifestReader.peekIdentity(directoryURL: modelDirectory).family
        return ServerReasoningProfile(
            family: family,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
    }

    package func previewFacts(modelIDOverride: String? = nil) throws -> ModelSessionFacts {
        let identity = try ManifestReader.peekIdentity(directoryURL: modelDirectory)
        let family = identity.family
        // Fail a lazy-load server at launch, not on the first request, when
        // the installed family's template defines no effort levels.
        try family.validateReasoning(
            thinkingMode: thinkingMode,
            effort: reasoningEffort)
        let defaultModelID = ServerModelIdentity.apiModelID(
            manifestModelID: identity.modelID,
            family: family,
            weightBits: identity.weightBits)
        // Mirrors the precedence in ServerModelSession.load: an explicit
        // --prefill-chunk wins, otherwise qwen36 takes the long-prefill chunk
        // and anything else takes the runtime default. Family is the only
        // input, and family comes from the manifest.
        let resolvedChunk =
            prefillChunkTokens
            ?? ModelProfile.resolve(identity: identity).prefillChunkTokens
            ?? (family == .qwen36
                ? RuntimeConfiguration.qwenLongPrefillChunkTokens
                : RuntimeConfiguration.production.prefillChunkTokens)
        return ModelSessionFacts(
            modelID: modelIDOverride ?? defaultModelID,
            prefillChunkTokens: resolvedChunk,
            promptCacheMode: ServerModelSession.effectivePromptCacheMode(
                requested: promptCacheMode,
                mtpEnabled: mtpModelDirectory != nil,
                slots: mtpModelDirectory == nil ? slots : 1))
    }
}
