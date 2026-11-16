import TinyTitanCore
import Foundation
import TinyTitan

/// Cross-session reuse of the system-end checkpoint ("warm subagents").
///
/// The turn-end cache and the interior checkpoints describe one conversation.
/// A new session that renders the same system prompt is a different
/// conversation; its store used to be empty, so it prefilled the whole system
/// segment again. This isolates the one boundary that is safe to share -- the
/// end of the system/developer message and its tool schema -- and the
/// capture/restore rules around it.
///
/// The key is the content address the store already uses:
/// `SHA-256(domain + system token IDs)`. Every input that changes the render
/// (model, source snapshot, runtime profile, context, KV layout, template) is
/// part of the domain, so a different system prompt, model or context cannot
/// match. An entry is only ever the prefix *before* `systemEnd`; no user or
/// assistant token is inside it.
enum PromptCheckpointSharedSystem {
    /// One request's system boundary and the store key it addresses.
    struct Prefix: Sendable, Equatable {
        /// The `.sharedSystem` boundary. `position` is the prefix length.
        let boundary: PromptCheckpointBoundary
        let key: PromptCheckpointKey
        let prefixTokenIDs: [Int32]
    }

    /// The system-end boundary of a rendered chat prompt, or nil when the
    /// chat does not open with a system/developer message.
    static func prefix(
        promptIDs: [Int32],
        messages: [GFTokenizer.Message],
        imStartID: Int32,
        domain: ServerPromptCacheDomain
    ) -> Prefix? {
        guard
            let position = PromptCheckpointPlanner.systemEndPosition(
                promptIDs: promptIDs,
                messages: messages,
                imStartID: imStartID),
            position > 0, position < promptIDs.count
        else { return nil }
        let ids = Array(promptIDs.prefix(position))
        return Prefix(
            boundary: PromptCheckpointBoundary(
                kind: .sharedSystem, position: position),
            key: PromptCheckpointKey.make(
                domain: domain, prefixTokenIDs: ids),
            prefixTokenIDs: ids)
    }

    /// Build (but do not save) the metadata for a captured system prefix.
    ///
    /// The runner is only at the boundary for the duration of the prefill
    /// callback, so the snapshot is taken there and this validates that it
    /// really is the prefix the key describes before anything is stored.
    static func claimedMetadata(
        prefix: Prefix,
        domain: ServerPromptCacheDomain,
        conversationKey: String,
        snapshot: InferenceStateSnapshot
    ) throws -> PromptCheckpointMetadata {
        guard snapshot.descriptor.position == prefix.boundary.position else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "snapshot position \(snapshot.descriptor.position) does not "
                    + "match the system boundary \(prefix.boundary.position)")
        }
        return PromptCheckpointMetadata(
            key: prefix.key,
            domain: domain,
            kind: .sharedSystem,
            conversationKey: conversationKey,
            prefixTokenIDs: prefix.prefixTokenIDs,
            descriptor: snapshot.descriptor)
    }

    /// Validate a loaded record against the request it claims to serve.
    ///
    /// Every mismatch throws; the caller treats it as a miss and prefills from
    /// scratch. Note what is *not* trusted: the stored position, prefix, kind
    /// and domain are all re-checked here even though the store validated its
    /// own copy, because a partially restored runner would answer from the
    /// wrong prefix.
    static func validate(
        metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?,
        against prefix: Prefix,
        domain: ServerPromptCacheDomain
    ) throws {
        guard metadata.domain == domain else {
            throw PromptCheckpointStoreError.domainMismatch
        }
        // `.systemEnd` is the pre-K1 spelling of the same record; a single
        // user turn used to label the shared position `.lastTurnStart`. Both
        // are accepted only because position and prefix are checked below:
        // at this boundary the prefix still contains no user token.
        let shareable: Set<PromptCheckpointKind> = [
            .sharedSystem, .systemEnd, .lastTurnStart,
        ]
        guard shareable.contains(metadata.kind) else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "checkpoint kind \(metadata.kind.rawValue) is not a system prefix")
        }
        guard metadata.position == prefix.boundary.position else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "checkpoint position \(metadata.position) does not match the "
                    + "system boundary \(prefix.boundary.position)")
        }
        guard metadata.prefixTokenIDs == prefix.prefixTokenIDs else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "checkpoint prefix does not match the rendered system prefix")
        }
        guard snapshot.descriptor.position == prefix.boundary.position else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "snapshot position \(snapshot.descriptor.position) does not "
                    + "match the system boundary \(prefix.boundary.position)")
        }
        if let auxiliary {
            guard auxiliary.position == prefix.boundary.position else {
                throw PromptCheckpointStoreError.invalidMetadata(
                    "auxiliary position \(auxiliary.position) does not match "
                        + "the system boundary \(prefix.boundary.position)")
            }
        }
    }

    /// Load, validate and restore the shared system prefix.
    ///
    /// Returns the reused prefix length on a hit; nil means "prefill from
    /// scratch" and is the outcome for a miss, a corrupt record, and a runner
    /// that needs auxiliary state the record does not carry -- never a
    /// partially restored sequence. An untrusted record is removed so the next
    /// request does not pay for it again.
    static func restore(
        store: PromptCheckpointStore,
        domain: ServerPromptCacheDomain,
        runner: RealForwardRunner,
        prefix: Prefix,
        promptTokenCount: Int
    ) async -> Int? {
        let loaded:
            (
                metadata: PromptCheckpointMetadata,
                snapshot: InferenceStateSnapshot,
                auxiliary: InferenceAuxiliaryState?,
                tier: String
            )
        do {
            loaded = try await store.load(prefix.key, domain: domain)
            try validate(
                metadata: loaded.metadata,
                snapshot: loaded.snapshot,
                auxiliary: loaded.auxiliary,
                against: prefix,
                domain: domain)
            // A missing QSA/PLE payload is refused by the runner itself, which
            // resets rather than leaving the previous sequence's indexer and
            // n-gram window behind; the throw lands in the catch below.
            try runner.restoreInferenceState(
                loaded.snapshot, auxiliary: loaded.auxiliary)
        } catch PromptCheckpointStoreError.missing(_) {
            return nil
        } catch {
            await store.remove(prefix.key)
            ServerLog.diagnostic(
                "TinyTitan prompt_checkpoint shared_system_restore_failed "
                    + "prefix_tokens=\(prefix.boundary.position) error=\(error)")
            return nil
        }
        ServerLog.diagnostic(
            hitLine(
                tier: loaded.tier,
                prefixTokens: prefix.boundary.position,
                promptTokens: promptTokenCount))
        return prefix.boundary.position
    }

    /// The one line a shared hit emits: where it came from, how much was
    /// reused, and how much prefill the request saved. Built separately from
    /// the logging call so the contract is testable without a process-wide
    /// log sink.
    static func hitLine(
        tier: String,
        prefixTokens: Int,
        promptTokens: Int
    ) -> String {
        "TinyTitan prompt_checkpoint shared_system_hit "
            + "tier=\(tier) "
            + "prefix_tokens=\(prefixTokens) "
            + "reused_tokens=\(prefixTokens) "
            + "suffix_tokens=\(max(0, promptTokens - prefixTokens))"
    }
}
