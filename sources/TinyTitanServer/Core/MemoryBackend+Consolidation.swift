import TinyTitanCore
import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanMemory

// Memory consolidation and workspace placement: when a session is consolidated,
// how the work is scheduled, and where a request's memory lands.
//
// Split out of `MemoryBackend.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. The actor's stored
// properties and helpers widened from `private` to internal because the
// methods that stay call them.
extension MemoryBackend {

    // MARK: - Consolidation

    /// Arms the idle timer for a session that just gained a turn, and fires
    /// the consolidation of a session that rolled over during this request.
    ///
    /// The turn is recorded and the reply has been returned by the time this
    /// runs, so the person is reading. That is the pause a consolidation is
    /// allowed to use.
    func scheduleConsolidation(after context: MemorySessionContext) {
        guard configuration.sessionConsolidation else { return }
        let scope = context.scope
        unconsolidated[scope] = context
        idleTimers[scope]?.cancel()
        let delay = configuration.consolidationIdleSeconds
        // The timer task only waits. Once the wait is over the consolidation
        // runs in a task of its own, so a turn arriving later -- which
        // cancels the timer -- can never cancel a generation already under
        // way: the gate would release mid-inference and the engine would be
        // asked to abandon a request for no reason.
        idleTimers[scope] = Task { [weak self] in
            if delay > 0 { try? await Task.sleep(for: .seconds(delay)) }
            guard !Task.isCancelled, let self else { return }
            Task { await self.consolidateIfPending(scope: scope, expecting: context.session.id) }
        }
        if let previous = pendingAfterTurn.removeValue(forKey: scope) {
            Task { [weak self] in await self?.consolidate(previous) }
        }
    }

    func consolidateIfPending(scope: MemoryScope, expecting sessionID: String) async {
        guard let pending = unconsolidated[scope], pending.session.id == sessionID else { return }
        await consolidate(pending)
    }

    /// Distils a finished session into memory, after every earlier
    /// distillation in the same scope has finished writing.
    ///
    /// The chaining is the whole point: `runConsolidation` reads what memory
    /// holds to build its prompt, and a read that overtakes an earlier write
    /// is how two sessions came to name one fact under two addresses
    /// (TT-035). Waiting here is cheap — the earlier generation was going to
    /// occupy the one generation gate anyway — and it makes the order of the
    /// reads the order of the writes.
    func consolidate(_ context: MemorySessionContext) async {
        let scope = context.scope
        let previous = consolidationChain[scope]
        let current = Task { [weak self] in
            await previous?.value
            await self?.runConsolidation(context)
        }
        consolidationChain[scope] = current
        await current.value
        if consolidationChain[scope] == current {
            consolidationChain[scope] = nil
        }
    }

    /// Distils a finished session into memory.
    ///
    /// This is the engine writing, not the model choosing to. Measured on a
    /// hundred-chapter novel, a model given the bible in its prompt made zero
    /// writes in that session, then found memory empty in the next and
    /// stored a bible it had invented; a harness that simply forced a
    /// summary at each boundary carried twice as much. The forcing is what
    /// works. Writing the result as addressed facts rather than a note is
    /// what lets a later change supersede an earlier state instead of the
    /// note copying the old state forward, which is how the summary lost
    /// every plot event one session after it happened.
    func runConsolidation(_ context: MemorySessionContext) async {
        let scope = context.scope
        if unconsolidated[scope]?.session.id == context.session.id {
            unconsolidated[scope] = nil
        }
        guard let journal = await service.journalStore(for: scope) else { return }
        let newestFirst: [JournalTurn]
        do {
            newestFirst = try await journal.turns(
                session: context.session.id,
                limit: configuration.consolidationMaximumTurns,
                in: scope)
        } catch {
            // "No new turns" and "the journal would not answer" are different
            // facts, and only the first is a reason to skip. Nothing is lost by
            // returning here: the consolidated head is not advanced on this
            // path, so the next session end reads the journal again.
            ServerLog.memory(
                "consolidation skipped session=\(context.session.id): "
                    + "the journal could not be read (\(error))")
            return
        }
        let chronological = Array(newestFirst.reversed())
        let through = consolidatedThrough[context.session.id] ?? -1
        let fresh = chronological.filter { $0.index > through }
        guard let last = fresh.last else {
            ServerLog.memory("consolidation skipped session=\(context.session.id): no new turns")
            return
        }
        let characters = fresh.reduce(0) { $0 + $1.prompt.count + $1.reply.count }
        guard characters >= configuration.consolidationMinimumCharacters else {
            ServerLog.memory(
                "consolidation skipped session=\(context.session.id): "
                    + "\(characters) new characters, nothing to distil")
            return
        }
        // One already-distilled turn ahead of the new ones, so a reply that
        // answers the previous prompt is read with that prompt.
        let overlap = chronological.last { $0.index <= through }.map { [$0] } ?? []
        let turns = overlap + fresh
        let existing: [MemoryRecord]
        do {
            existing = try await service.recordedFacts(in: scope, limit: 400)
        } catch {
            // Skipping, not consolidating against an empty list. `existing:` is
            // what the extraction is shown as "already on file"; answering it
            // with nothing for a store that would not read is how the same
            // facts come back written again under new keys.
            ServerLog.memory(
                "consolidation skipped session=\(context.session.id): "
                    + "the facts on file could not be read (\(error))")
            return
        }
        let request = ServerMemory.consolidationRequest(
            turns: turns, existing: existing, workspace: scope.workspace)
        let started = Date()
        let completion: ServerCompletion
        do {
            completion = try await gated(request, onEvent: { _ in })
        } catch {
            ServerLog.memory("consolidation failed session=\(context.session.id): \(error)")
            return
        }
        let parsed = ServerMemory.consolidationRecords(from: completion.content)
        let (records, merged) = ServerMemory.reconcile(parsed, existing: existing)
        for (from, to) in merged {
            ServerLog.memory("consolidation routed \(from) -> \(to) session=\(context.session.id)")
        }
        if records.isEmpty {
            // Nothing usable came back. The head of the raw output is the only
            // way to tell an honest "[]" from a truncated array or a refusal.
            let head = completion.content.prefix(200).replacingOccurrences(of: "\n", with: " ")
            ServerLog.memory(
                "consolidation produced no facts session=\(context.session.id) "
                    + "finish=\(completion.finishReason) output=\"\(head)\"")
        }
        let written = await service.storeConsolidation(records, in: context)
        consolidatedThrough[context.session.id] = last.index
        ServerLog.memory(
            "consolidated session=\(context.session.id) turns=\(fresh.count) "
                + "facts=\(written) keys=\(records.map(\.key.rawValue).joined(separator: ","))"
                + " prompt=\(completion.usage.promptTokens) "
                + "completion=\(completion.usage.completionTokens) "
                + "seconds=\(Int(Date().timeIntervalSince(started)))")
    }

    /// Decides the workspace for a request, in this order:
    ///
    /// 1. The `X-TinyTitan-Workspace` header, when the client sent one.
    /// 2. The working directory the client declared in its system prompt.
    ///    This is the one that keeps a novel and a codebase apart with no
    ///    configuration at all: the coding CLIs already say where they are
    ///    on every request, and where they are is the project.
    /// 3. The launch directory.
    ///
    /// A declared directory that is not a project -- the home directory,
    /// the root -- falls through to the launch workspace and is logged once,
    /// rather than being refused: refusing a request over a client's cwd
    /// would turn a memory nicety into a serving failure.
    func resolvePlacement(for request: ValidatedChatRequest) -> Placement {
        if let header = request.workspace {
            return Placement(workspace: header, override: header, tag: header, source: "header")
        }
        guard configuration.allowsPerRequestWorkspace,
            let declared = ServerMemory.declaredWorkingDirectory(in: request.messages)
        else {
            return Placement(
                workspace: configuration.workspace, override: nil,
                tag: nil, source: "launch")
        }
        if let reason = MemoryConfiguration.junkDrawerReason(
            forPath: declared, environment: ["HOME": homeDirectory])
        {
            if refusedDirectories.insert(declared).inserted {
                ServerLog.memory("declared working directory ignored: \(reason)")
            }
            return Placement(
                workspace: configuration.workspace, override: nil,
                tag: nil, source: "launch")
        }
        let workspace = MemoryConfiguration.workspaceIdentifier(forPath: declared)
        let tag = URL(fileURLWithPath: declared).lastPathComponent
        return Placement(
            workspace: workspace, override: workspace, tag: tag,
            source: "declared-cwd")
    }
}
