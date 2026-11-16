import TinyTitanCore
// Building a `ModelSessionPlan` from the server's launch flags.
//
// Moved out of `ModelSessionPlan.swift` (2026-10-02, phase A1 of
// `docs/plan-embedded-library.md`) when that type moved into `TinyTitanLib`:
// the factory reads `ServerArguments`, which is this server's launch surface
// and not part of the engine an embedder links, so the flags are read here and
// the kit's plan is built through its own initializer.
import Foundation
import TinyTitan
import TinyTitanLib

extension ModelSessionPlan {
    /// The one place a plan is built from the server's arguments.
    ///
    /// Both the single-model path and the catalog loader go through it, so a
    /// parameter added for one cannot be forgotten by the other. That is not
    /// hypothetical: the catalog loader once built its own plan and dropped
    /// `slots`, so `--models-dir` sessions ran one sequence while the
    /// coordinator admitted four, and the four silently serialized.
    package static func from(
        arguments: ServerArguments,
        modelDirectory: URL,
        thinking: ModelThinkingMode,
        reasoningEffort: ModelReasoningEffort?,
        mtpModelDirectory: URL?
    ) -> ModelSessionPlan {
        ModelSessionPlan(
            modelDirectory: modelDirectory,
            maxContext: arguments.maxContext,
            // MTP overrides the requested width inside `sessionSlots`.
            slots: arguments.sessionSlots,
            promptCacheMode: arguments.promptCacheMode,
            promptCacheMaximumEntries: arguments.promptCacheMaximumEntries,
            promptCacheMemoryLimitBytes: arguments.promptCacheMemoryMiB * 1_048_576,
            promptCacheDiskDirectory: arguments.promptCacheDiskDirectory.map {
                URL(fileURLWithPath: $0).standardizedFileURL
            },
            promptCacheDiskLimitBytes: arguments.promptCacheDiskMiB * 1_048_576,
            prefillChunkTokens: arguments.prefillChunkTokens,
            kvCachePrecision: arguments.kvCachePrecision,
            hostKVWindowTokens: arguments.hostKVWindowTokens,
            ropeScalingMode: arguments.ropeScalingMode,
            thinkingMode: thinking,
            reasoningEffort: reasoningEffort,
            expertCacheSlots: arguments.expertCacheSlots,
            expertCacheBudgetBytes: arguments.expertCacheBudgetBytes,
            mtpModelDirectory: mtpModelDirectory,
            mtpMemoryMiB: arguments.mtpMemoryMiB)
    }
}
