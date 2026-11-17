import TinyTitanCore
// Interior-checkpoint capture planning for one generation.
//
// Split out of ServerModelSession+Generation.swift (2026-10-22) under the
// 500-line-per-file rule. The generation orchestrator asks for a plan; this
// file owns the boundary selection, the snapshot capture and the detached
// store writes.
import Foundation

extension ServerModelSession {
    /// What one generation should capture, and the callback that captures it.
    struct InteriorCheckpointCapture: Sendable {
        let positions: [Int]
        let onCheckpoint: (@Sendable (Int) -> Void)?

        static let none = InteriorCheckpointCapture(
            positions: [], onCheckpoint: nil)
    }

    /// Plan the interior checkpoints for this prompt.
    ///
    /// The positions are message boundaries on the ids that were actually
    /// rendered. Capture is skipped when a restore could never succeed (a QSA
    /// or PLE family), when the tier is off, and when another producer (MTP or
    /// a batched head) owns the prefill.
    func interiorCheckpointCapture(
        promptIDs: [Int32],
        messages: [GFTokenizer.Message],
        imStartID: Int32,
        enabled: Bool
    ) -> InteriorCheckpointCapture {
        guard enabled,
            promptCacheMode == .multiPrefix,
            let checkpointStore = promptCheckpointStore,
            runner.supportsInferenceStateRestore
        else { return .none }
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: promptIDs,
            messages: messages,
            imStartID: imStartID,
            gridTokens: Self.checkpointGridTokens,
            includeMessageBoundaries: !Self.checkpointGridOnly)
        guard !boundaries.isEmpty else { return .none }
        let kinds = Dictionary(
            uniqueKeysWithValues: boundaries.map { ($0.position, $0.kind) })
        // The system boundary is captured under its own kind so a later
        // session can tell a shareable prefix from a conversation-local one.
        // The position is the one the planner already uses; a single-user-turn
        // chat labels it `.lastTurnStart`, which is why it is located
        // separately rather than read back from the label.
        let sharedSystemPosition = PromptCheckpointPlanner.systemEndPosition(
            promptIDs: promptIDs, messages: messages, imStartID: imStartID)
        let captureRunner = runner
        let captureDomain = promptCacheDomain
        let capturePromptIDs = promptIDs
        let conversation = PromptCheckpointPlanner.conversationKey(
            domain: captureDomain, messages: messages)
        let onCheckpoint: @Sendable (Int) -> Void = { position in
            do {
                let snapshot = try captureRunner.captureInferenceState(
                    maximumBytes: PromptCheckpointStore.maximumCaptureBytes)
                guard snapshot.descriptor.position == position else {
                    ServerLog.diagnostic(
                        "TinyTitan prompt_checkpoint capture_mismatch "
                            + "expected=\(position) "
                            + "actual=\(snapshot.descriptor.position)")
                    return
                }
                let auxiliary = try captureRunner.captureAuxiliaryState()
                let prefix = Array(capturePromptIDs.prefix(position))
                let key = PromptCheckpointKey.make(
                    domain: captureDomain, prefixTokenIDs: prefix)
                let kind: PromptCheckpointKind =
                    position == sharedSystemPosition
                    ? .sharedSystem : (kinds[position] ?? .lastTurnStart)
                let metadata = PromptCheckpointMetadata(
                    key: key,
                    domain: captureDomain,
                    kind: kind,
                    conversationKey: conversation,
                    prefixTokenIDs: prefix,
                    descriptor: snapshot.descriptor)
                Task.detached(priority: .utility) {
                    guard !(await checkpointStore.contains(key)) else {
                        return
                    }
                    do {
                        try await checkpointStore.save(
                            metadata,
                            snapshot: snapshot,
                            auxiliary: auxiliary,
                            // A system prefix can be small enough to fit
                            // RAM; the disk copy is what lets it survive a
                            // restart, which is the point of sharing it.
                            durable: metadata.kind == .sharedSystem)
                        ServerLog.diagnostic(
                            "TinyTitan prompt_checkpoint stored "
                                + "kind=\(metadata.kind.rawValue) "
                                + "tokens=\(position) "
                                + "state_bytes=\(snapshot.payload.count) "
                                + "aux_bytes=\(auxiliary?.payload.count ?? 0)")
                    } catch {
                        ServerLog.diagnostic(
                            "TinyTitan prompt_checkpoint save_failed "
                                + "tokens=\(position) error=\(error)")
                    }
                }
            } catch {
                // A capture failure must not fail the request: the prefill
                // already succeeded and a full prefill next time is correct,
                // just slower.
                ServerLog.diagnostic(
                    "TinyTitan prompt_checkpoint capture_failed "
                        + "tokens=\(position) error=\(error)")
            }
        }
        return InteriorCheckpointCapture(
            positions: boundaries.map(\.position),
            onCheckpoint: onCheckpoint)
    }

    /// The optional token-grid interval (`TINYTITAN_CHECKPOINT_GRID`).
    ///
    /// Nil in the shipped configuration. T24 uses it to price a 16K/32K grid
    /// against the message-boundary points; T20's design argues a grid should
    /// not be the default, and the benchmark decides whether it earns a switch
    /// rather than an assumption.
    static var checkpointGridOnly: Bool {
        ProcessInfo.processInfo.environment[
            "TINYTITAN_CHECKPOINT_GRID_ONLY"] == "1"
    }

    static var checkpointGridTokens: Int? {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "TINYTITAN_CHECKPOINT_GRID"],
            let value = Int(raw), value >= 1_024
        else { return nil }
        return value
    }

    /// Restore the system prefix a previous conversation captured.
    ///
    /// The generic interior restore runs first, because a conversation
    /// reopened after a model reload may still restore a longer
    /// same-conversation prefix. This is the cross-conversation fallback, and
    /// it only ever restores the system-end boundary: no user or assistant
    /// token can be inside a `.sharedSystem` entry.
    func restoreSharedSystemPrefix(
        promptIDs: [Int32],
        messages: [GFTokenizer.Message]
    ) async -> Int? {
        guard promptCacheMode == .multiPrefix,
            let store = promptCheckpointStore,
            runner.supportsInferenceStateRestore,
            let prefix = PromptCheckpointSharedSystem.prefix(
                promptIDs: promptIDs,
                messages: messages,
                imStartID: tokenizer.imStartID,
                domain: promptCacheDomain)
        else { return nil }
        return await PromptCheckpointSharedSystem.restore(
            store: store,
            domain: promptCacheDomain,
            runner: runner,
            prefix: prefix,
            promptTokenCount: promptIDs.count)
    }
}
