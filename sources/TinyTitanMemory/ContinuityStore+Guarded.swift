import ContinuityCore
import Foundation

extension ContinuityStore {
    /// A write that reverts a key to a value it already had before is almost
    /// never a real event and almost always a re-derivation -- an inn that
    /// burned coming back as "standing" because a later session's chapters
    /// were written against a stale bootstrap. The value is written, because
    /// it may be right, and the key is marked disputed so the model sees the
    /// conflict instead of inheriting whichever side it read last.
    ///
    /// Returns true when a reversion was flagged.
    @discardableResult
    /// Who a record is attributed to. The person's own statements are the
    /// only thing in a transcript that is not the model's own output, and
    /// the guard is built entirely on being able to tell them apart.
    static func author(of record: MemoryRecord) -> ProvenanceAuthor {
        // Not `isUserAsserted` directly: that is the extraction's claim, and
        // a claim about a value holding several facts cannot be true. See
        // `MemoryRecord.carriesUserAuthority`.
        record.carriesUserAuthority ? .user : .model
    }

    /// The outcome of a guarded write, for the caller to log.
    /// `set`, with the precedence rule the guard adds.
    ///
    /// A user-asserted fact may always be replaced by a newer user-asserted
    /// one -- that is how a state the person changes ("the inn burned in
    /// chapter 34") reaches the store. What is refused is the model
    /// overwriting the person, which is the direction that measured as the
    /// single largest source of wrong facts.
    /// The protocol's guarded write. Reversion flagging is off here: that
    /// heuristic was tuned on consolidation, where a value returning to one
    /// it already had is almost always a re-derivation. A deliberate tool
    /// call is not the same act, and flagging it would put a dispute marker
    /// in front of the next session for a write the model meant to make.
    public func set(
        _ record: MemoryRecord, in scope: MemoryScope,
        guarding: Bool
    ) async throws -> GuardedWrite {
        try await set(record, in: scope, guarding: guarding, flaggingReversions: false)
    }

    /// The protocol's guarded delete. A model retiring a fact the person
    /// established is the same failure as overwriting it, and the same
    /// answer: the fact stays, the disagreement is recorded, and the next
    /// session sees both.
    public func delete(
        _ key: MemoryKey, in scope: MemoryScope,
        guarding: Bool
    ) async throws -> GuardedDelete {
        guard guarding else {
            return try await delete(key, in: scope) ? .deleted : .absent
        }
        let taskID = try await task(for: scope)
        let address = Self.address(for: key)
        let active = await engine.recall(
            taskID: taskID,
            namespace: address.namespace,
            key: address.key)
        if let active, active.provenance?.author == .user,
            active.status.isEligibleForContext
        {
            _ = try? await engine.dispute(
                taskID: taskID,
                namespace: address.namespace,
                key: address.key)
            return .heldByGuard
        }
        return try await delete(key, in: scope) ? .deleted : .absent
    }

    public func set(
        _ record: MemoryRecord, in scope: MemoryScope,
        guarding: Bool,
        flaggingReversions: Bool
    ) async throws -> GuardedWrite {
        guard guarding, !record.isUserAsserted else {
            let reverted = try await set(
                record, in: scope,
                flaggingReversions: flaggingReversions)
            return reverted ? .reverted : .stored
        }
        let taskID = try await task(for: scope)
        let address = Self.address(for: Self.normalize(record).key)
        let active = await engine.recall(
            taskID: taskID,
            namespace: address.namespace,
            key: address.key)
        if let active, active.provenance?.author == .user, active.status != .archived,
            Self.fold(active.value) != Self.fold(record.value)
        {
            _ = try? await engine.dispute(
                taskID: taskID,
                namespace: address.namespace,
                key: address.key)
            return .heldByGuard(existing: active.value)
        }
        let reverted = try await set(
            record, in: scope,
            flaggingReversions: flaggingReversions)
        return reverted ? .reverted : .stored
    }

    public func set(
        _ record: MemoryRecord, in scope: MemoryScope,
        flaggingReversions: Bool
    ) async throws -> Bool {
        guard flaggingReversions else {
            try await set(record, in: scope)
            return false
        }
        let taskID = try await task(for: scope)
        let address = Self.address(for: Self.normalize(record).key)
        let history = await engine.history(
            taskID: taskID, namespace: address.namespace,
            key: address.key)
        let incoming = Self.fold(record.value)
        let current = history.last.map { Self.fold($0.value) }
        let reverts =
            current != nil && current != incoming
            && history.dropLast().contains { Self.fold($0.value) == incoming }
        try await set(record, in: scope)
        if reverts {
            _ = try? await engine.dispute(
                taskID: taskID, namespace: address.namespace,
                key: address.key)
        }
        return reverts
    }

    private static func fold(_ value: String) -> String {
        value.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ".\"'"))
    }

}
