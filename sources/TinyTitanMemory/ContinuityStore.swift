import ContinuityCore
import Foundation

/// The durable memory store, backed by the in-process continuity engine.
///
/// This replaced a Valkey client. Nothing above it changed: the server still
/// talks to `MemoryStore`, and the model still sees the same tools. What went
/// away is a second process, a wire protocol, a connection to lose and a
/// cache to size. The store now lives in the same binary as the model that
/// reads it.
///
/// A scope maps to one continuity task, and a `MemoryKey` maps to one
/// address. Keys are normalized on the way in, and the normalized form is
/// what comes back out, so a key the model reads is always a key it can use.
///
/// The record is stored in the engine's own fields, not as a JSON blob in the
/// value. That is what lets a text search be answered by the engine instead of
/// by decoding every record in the workspace, keeps the value readable in the
/// journal file, and stops importance and tags being written twice.
public actor ContinuityStore: MemoryStore {
    let engine: ContinuityEngine
    let limits: MemoryLimits
    var taskIDs: [MemoryScope: UUID] = [:]
    var sessionIDs: [String: UUID] = [:]
    /// Continuity session to the caller's own session name, so a record can
    /// report who wrote it without a lookup per read.
    var sessionLabels: [UUID: String] = [:]
    var labelsLoadedFor: Set<UUID> = []

    public init(engine: ContinuityEngine, limits: MemoryLimits = .init()) {
        self.engine = engine
        self.limits = limits
    }

    // MARK: - MemoryStore

    public func get(_ key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord? {
        let taskID = try await task(for: scope)
        let address = Self.address(for: key)
        guard
            let item = await engine.recall(
                taskID: taskID,
                namespace: address.namespace,
                key: address.key),
            item.status.isEligibleForContext
        else { return nil }
        await loadLabels(taskID: taskID)
        return try record(from: item)
    }

    public func set(_ record: MemoryRecord, in scope: MemoryScope) async throws {
        try limits.validate(value: record.value)
        let taskID = try await task(for: scope)
        let normalized = Self.normalize(record)
        let address = Self.address(for: normalized.key)
        // The writing session, when the caller named one and a continuity
        // session has been opened for it. That is what makes a record
        // explainable later, so it is carried as provenance rather than
        // duplicated into the value. Resolved live rather than read straight
        // from the cache, because the cache outlives the sessions it names.
        let sessionID = await liveSession(for: normalized.sourceSession, taskID: taskID)
        do {
            if let sessionID {
                try await engine.remember(
                    sessionID: sessionID,
                    namespace: address.namespace,
                    key: address.key,
                    value: normalized.value,
                    author: Self.author(of: normalized),
                    importance: normalized.importance,
                    confidence: normalized.confidence,
                    tags: normalized.tags)
            } else {
                try await engine.remember(
                    taskID: taskID,
                    namespace: address.namespace,
                    key: address.key,
                    value: normalized.value,
                    author: Self.author(of: normalized),
                    importance: normalized.importance,
                    confidence: normalized.confidence,
                    tags: normalized.tags)
            }
        } catch let error as ContinuityError {
            throw Self.translate(error)
        }
    }

    @discardableResult
    public func delete(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        let taskID = try await task(for: scope)
        let address = Self.address(for: key)
        guard
            let existing = await engine.recall(
                taskID: taskID,
                namespace: address.namespace,
                key: address.key),
            existing.status.isEligibleForContext
        else { return false }
        // Archived, not destroyed. A model that deletes a fact in one session
        // and contradicts itself in the next leaves a chain that explains it.
        do {
            try await engine.archive(
                taskID: taskID,
                namespace: address.namespace,
                key: address.key)
        } catch let error as ContinuityError {
            // Every failure, not only `notPersisted`: answering "deleted" after
            // an archive that threw leaves the fact on disk with the model
            // believing it gone, and a journal write error is that same lie with
            // a different cause. `translate` maps `notPersisted` through
            // unchanged, so the message the RAM-only case earns is unchanged.
            throw Self.translate(error)
        }
        return true
    }

    public func exists(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        try await get(key, in: scope) != nil
    }

    public func list(prefix: String, limit: Int, in scope: MemoryScope) async throws -> [MemoryKey]
    {
        let taskID = try await task(for: scope)
        let bound = max(0, min(limit, limits.maximumListResults))
        // The prefix is pushed into the engine as a namespace filter, so a
        // list of one category does not walk the whole workspace. Only the
        // last segment cannot be pushed down, because it is a key prefix
        // rather than a namespace, and that residue is filtered here.
        let plan = Self.pushDown(prefix: prefix)
        var query = ContinuityCore.MemoryQuery(
            namespacePrefix: plan.namespace,
            order: .recency)
        if plan.isExact { query.limit = bound }
        let items = await engine.recall(taskID: taskID, query)
        return
            items
            .filter { plan.matches(Self.keyText(for: $0)) }
            .prefix(bound)
            .compactMap { try? MemoryKey(validating: Self.keyText(for: $0)) }
    }

    public func search(_ query: MemoryQuery, in scope: MemoryScope) async throws -> [MemoryRecord] {
        let taskID = try await task(for: scope)
        // What the ranker cannot redo is pushed into the engine, and ranking
        // stays here because it is the memory layer's own policy, shared with the
        // reference store so the two cannot drift.
        //
        // `text` and `tags` used to be pushed down too, and that is how the two
        // did drift: the engine's predicate is one case-insensitive substring of
        // address and value, so a query for "race sync" had to appear as that
        // phrase, and its tag compare is exact where the ranker lowercases both
        // sides. The prefix and the importance floor stay, because the ranker
        // applies the same rule to those and pushing them down only narrows the
        // scan.
        let plan = Self.pushDown(prefix: query.prefix ?? "")
        let engineQuery = ContinuityCore.MemoryQuery(
            namespacePrefix: plan.namespace,
            minimumImportance: query.minimumImportance,
            limit: maximumScan,
            order: .recency)
        let items = await engine.recall(taskID: taskID, engineQuery)
            .filter { plan.matches(Self.keyText(for: $0)) }
        await loadLabels(taskID: taskID)
        let records = items.compactMap { try? record(from: $0) }
        var bounded = query
        bounded.limit = min(query.limit, limits.maximumSearchResults)
        return MemoryRanking.rank(records, for: bounded)
    }

    /// Ceiling on how many records one query may consider.
    ///
    /// The Valkey backend read a bounded slice of a sorted-set index for the
    /// same reason: one scope's cost must not grow with how much it has
    /// stored. Ranking a few thousand short facts is cheap; ranking every
    /// fact of a two-year project on every search is not.
    let maximumScan = 2_000

    @discardableResult
    public func append(_ text: String, to key: MemoryKey, in scope: MemoryScope) async throws
        -> MemoryRecord
    {
        let existing = try await get(key, in: scope)
        let combined = existing.map { $0.value.isEmpty ? text : $0.value + "\n" + text } ?? text
        try limits.validate(value: combined)
        var record = existing ?? MemoryRecord(key: key, value: "")
        record.value = combined
        record.updatedAt = Date()
        try await set(record, in: scope)
        return record
    }

    public func sessionInit(_ session: MemorySession, in scope: MemoryScope) async throws
        -> MemoryBootstrap
    {
        let taskID = try await task(for: scope)
        // Session identity is derived from the conversation, so the same id
        // can arrive again after a restart or when a client replays a
        // conversation. Reuse the session rather than opening a second one
        // that splits the same conversation's journal in half.
        if let existing = await engine.session(externalID: session.id, taskID: taskID) {
            remember(existing, as: session.id)
        } else {
            let continuity = try await engine.beginSession(
                taskID: taskID,
                model: session.modelID,
                externalID: session.id,
                tag: session.tag)
            remember(continuity, as: session.id)
        }
        // Ranked by what is being asked, not by a static importance. A flat
        // list ranked by importance dropped a character's eye colour out of
        // the window by the fourth session of a novel because running state
        // had been rated higher; the fact was in the store every time. The
        // engine's assembler ranks by priority namespace, then relevance to
        // the request, then importance, and drags dependencies in with
        // what it picks.
        let candidates = max(limits.bootstrapRecords * 8, 200)
        let items = await engine.recall(
            taskID: taskID,
            ContinuityCore.MemoryQuery(
                statuses: [.active, .disputed],
                limit: candidates,
                order: .relevance))
        await loadLabels(taskID: taskID)
        guard let task = await engine.task(taskID) else { return .empty }
        var index: [String: ContinuityCore.MemoryItem] = [:]
        for item in items { index[item.address] = item }
        let request = ContextRequest(
            task: task,
            items: items,
            index: index,
            budget: ContextBudget(
                maxTokens: max(64, limits.bootstrapBytes / 4),
                priorityNamespaces: Self.bootstrapPriority,
                recentTurnCount: 0),
            focus: session.focus)
        let ordered: [ContinuityCore.MemoryItem]
        if let snapshot = try? DefaultContextAssembler().assemble(request) {
            let byID = Dictionary(uniqueKeysWithValues: items.map { ($0.id, $0) })
            ordered = snapshot.memoryItemIDs.compactMap { byID[$0] }
        } else {
            ordered = items
        }
        let records = ordered.compactMap { try? record(from: $0) }
        return MemoryBootstrap.build(
            ordered: records, limits: limits,
            recent: recentlyChanged(
                among: items,
                excluding: sessionIDs[session.id]))
    }

    /// The namespaces a session cannot do without, first. Rules and
    /// constraints bound everything; decisions explain the code; a novel's
    /// characters and setting are its rules. Running state comes last: it is
    /// what most recently changed, and it is exactly what a session is most
    /// likely to be about to change again.
    static let bootstrapPriority: [String] = [
        "k.rules", "k.rule", "k.constraints", "k.constraint",
        "k.decisions", "k.decision", "k.gotchas", "k.gotcha",
        "k.project", "k.setting", "k.characters", "k.character",
        "k.architecture", "k.conventions",
    ]

    /// What the most recent other session wrote, newest first, bounded.
    private func recentlyChanged(
        among items: [ContinuityCore.MemoryItem],
        excluding current: UUID?
    ) -> [MemoryRecord] {
        let newest =
            items
            .filter { $0.provenance?.sessionID != nil && $0.provenance?.sessionID != current }
            .sorted { $0.updatedAt > $1.updatedAt }
        guard let last = newest.first?.provenance?.sessionID else { return [] }
        return newest.filter { $0.provenance?.sessionID == last }
            .prefix(12)
            .compactMap { try? record(from: $0) }
    }

    // MARK: - Beyond the protocol

    /// The continuity session for a memory session, once one has begun. The
    /// journal uses it so both stores describe the same session rather than
    /// two parallel ones.
    public func continuitySession(for id: String) -> UUID? { sessionIDs[id] }

    public func taskID(for scope: MemoryScope) async throws -> UUID {
        try await task(for: scope)
    }

    /// The first journal write the engine could not make, or nil while every
    /// write has reached the file. Read by the service, which is what tells
    /// the model whether its memory outlives the process.
    public var journalFailure: String? {
        get async { await engine.journalFailure }
    }

    /// The last automatic compaction the engine could not make, or nil while
    /// the journal has been collapsing. Read by the service, and read
    /// separately from `journalFailure` on purpose: a refused checkpoint loses
    /// no record, so it says nothing about whether writes reach the file. It
    /// says the file is growing, which is a different thing to tell someone.
    public var compactionFailure: String? {
        get async { await engine.compactionFailure }
    }

    // MARK: - Internals

    private func record(from item: ContinuityCore.MemoryItem) throws -> MemoryRecord {
        let key = try MemoryKey(validating: Self.keyText(for: item))
        var record = MemoryRecord(
            key: key,
            value: item.value,
            importance: item.importance,
            confidence: item.confidence,
            tags: item.tags,
            sourceSession: item.provenance?.sessionID
                .flatMap { sessionLabels[$0] },
            createdAt: item.createdAt,
            updatedAt: item.updatedAt)
        record.isDisputed = item.status == .disputed
        // Authority has to survive a read. Without this, every
        // read-modify-write -- `append` is one -- rewrites a fact the person
        // asserted as though the model had derived it, and the guard can
        // never fire on that address again.
        record.isUserAsserted = item.provenance?.author == .user
        return record
    }

    /// Fills the session-name map for a task, once.
    ///
    /// After a restart the map is empty but the sessions themselves carry
    /// their external names, so one pass rebuilds it. Without this a record
    /// written last week would come back saying nobody wrote it.
    private func loadLabels(taskID: UUID) async {
        guard !labelsLoadedFor.contains(taskID) else { return }
        labelsLoadedFor.insert(taskID)
        for session in await engine.sessions(taskID: taskID) {
            if let external = session.externalID { sessionLabels[session.id] = external }
        }
    }

    /// How a memory-key prefix is answered.
    ///
    /// The namespace filter is only ever an optimization: the residual check
    /// runs against the reconstructed key and decides the result on its own.
    /// Keeping it that way means a prefix that lands mid-segment, like `dec`
    /// for `decisions/sync`, cannot silently be matched against the wrong
    /// part of the address.
    struct PrefixPlan: Equatable {
        /// Namespace prefix the engine can filter on, when the prefix names
        /// whole segments. Nil when it does not.
        let namespace: String?
        /// The normalized prefix, matched against the whole key.
        let residual: String
        /// True when the namespace filter alone is exactly the answer, so the
        /// engine may apply the limit itself.
        var isExact: Bool { namespace != nil && residual.hasSuffix("/") }

        func matches(_ keyText: String) -> Bool {
            residual.isEmpty || keyText.hasPrefix(residual)
        }
    }

}
