import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanMemory
import TinyTitanCore

/// Adds persistent memory to any inference backend.
///
/// A decorator rather than a change to generation: it rewrites the request on
/// the way in (the memory instructions and the memory tools), and on the way
/// out it services the memory tool calls the model makes and asks the inner
/// backend to continue. The engine's request lifecycle, prompt cache and
/// tool parsing are untouched, and with memory disabled this type is not
/// constructed at all.
///
/// Why the engine executes these tools when it executes no others: the
/// server's own tools are the client's, and no client knows about TinyTitan
/// memory. A memory tool the client would have to run is a memory tool
/// nothing runs.
public actor MemoryBackend: ServerInferenceBackend, PromptTokenCounting, ResidencyManaging {
    let inner: any ServerInferenceBackend
    let service: MemoryService
    let configuration: MemoryConfiguration
    /// Session contexts by conversation, so a multi-turn conversation keeps
    /// one session and bootstraps once.
    var contexts: [String: MemorySessionContext] = [:]
    /// The exact instruction text installed for a conversation, kept for the
    /// life of that conversation.
    ///
    /// This is the constraint the whole design turns on. The fragment sits at
    /// the head of the prompt, so if it changed between turns the prefix
    /// would change and every cached KV block after it would be invalid.
    /// TinyTitan prefills at roughly twice its decode rate rather than the
    /// hundredfold of a GPU server, so a needless cache miss costs minutes,
    /// not milliseconds. The bootstrap is therefore computed once per
    /// conversation and frozen, even though memory keeps changing underneath
    /// it: a stale bootstrap is cheap, and the model can always call a tool
    /// or read the journal for what is current.
    var installedInstructions: [String: String] = [:]
    /// Turn counter per conversation, for the journal.
    var turnIndex: [String: Int] = [:]
    /// Read once. The workspace guard needs it per request, and reading the
    /// environment per request is the pattern that once cost 40% of a token.
    let homeDirectory = FileManager.default.homeDirectoryForCurrentUser.path
    /// Declared directories already refused, so each is logged once.
    var refusedDirectories: Set<String> = []
    /// Sessions with turns not yet distilled into memory, by scope. One per
    /// scope: a new session in a scope replaces the pending one, and the
    /// replaced one is consolidated on the rollover it just caused.
    var unconsolidated: [MemoryScope: MemorySessionContext] = [:]
    /// The idle timer per scope. Reset on every turn; fires consolidation.
    var idleTimers: [MemoryScope: Task<Void, Never>] = [:]
    /// A session that rolled over before its idle timer fired. Consolidated
    /// as soon as the current request has returned, never before.
    var pendingAfterTurn: [MemoryScope: MemorySessionContext] = [:]
    /// The last journal turn index each session has been distilled through.
    /// Every idle gap used to re-read the newest forty turns of the whole
    /// session and extract them again; a fifty-turn coding session paid that
    /// on every pause. Only turns after this index are read now, plus the
    /// one before them for context.
    var consolidatedThrough: [String: Int] = [:]
    /// The distillation in flight for each scope, so a later one can wait for
    /// it before reading what memory holds.
    ///
    /// The extraction is shown the addresses memory already uses and told to
    /// reuse them; that prompt is built from a read of the store. Two sessions
    /// in one scope are routinely in flight together -- a rolled-over session
    /// is distilled while the new one's idle timer runs -- and a later prompt
    /// built before an earlier write sees an empty store, invents a parallel
    /// namespace, and leaves two live values for one fact. On the `contract`
    /// benchmark this is exactly what happened: session 3's extraction ran
    /// with an empty key list and wrote `agreement/*` beside session 2's
    /// `msa/*`, although the two were requested four seconds apart (TT-035).
    var consolidationChain: [MemoryScope: Task<Void, Never>] = [:]
    /// One generation at a time through this backend.
    ///
    /// The HTTP layer admits one request at a time, but a consolidation is
    /// not an HTTP request: it enters below that gate, and the first smoke
    /// test with it on returned 500s to the user request it overlapped. So
    /// every call into the inner backend, a person's turn or the engine's
    /// own, takes this gate first. A person never waits behind more than one
    /// consolidation, and a consolidation only starts in a pause.
    var innerBusy = false
    var innerWaiters: [SuspensionSlot] = []

    package init(
        wrapping inner: any ServerInferenceBackend,
        service: MemoryService,
        configuration: MemoryConfiguration
    ) {
        self.inner = inner
        self.service = service
        self.configuration = configuration
    }

    public nonisolated var maximumContext: Int { inner.maximumContext }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling { inner.samplingDefaults }

    /// Counts the request as the client sent it. The memory fragment and
    /// bootstrap are not included: they are added per session at generation
    /// time, and a count endpoint that guessed at them would be wrong more
    /// often than it was useful.
    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        guard let counting = inner as? any PromptTokenCounting else {
            throw ServerRequestError.unsupportedOperation("count_tokens")
        }
        return try await counting.countPromptTokens(request)
    }

    /// Forwards to the model underneath. The unload endpoint asks the
    /// outermost backend whether it manages residency, and with memory on
    /// that is this decorator: before it forwarded the question the endpoint
    /// answered false and the model stayed in memory.
    public func unload() async -> Bool {
        guard let managing = inner as? any ResidencyManaging else { return false }
        return await managing.unload()
    }

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        guard let context = await sessionContext(for: request) else {
            return try await gated(request, onEvent: onEvent)
        }
        let conversation = conversationKey(for: request)
        // Frozen on the first turn and reused verbatim thereafter, so the
        // prompt prefix is stable for the life of the conversation.
        let instructions: String
        if let existing = installedInstructions[conversation] {
            instructions = existing
        } else {
            instructions = await service.instructions(for: context)
            installedInstructions[conversation] = instructions
        }
        let memoryTools = ServerMemory.functionDefinitions(await service.toolDefinitions())

        var current = request.replacingMessages(
            ConcisePrompt.appendingSystemPrompt(instructions, to: request.messages),
            tools: ServerMemory.merging(tools: request.tools, memory: memoryTools))
        let startedAt = Date()

        // Memory tool calls are ours to answer, so the client never sees
        // them; anything else, including the client's own tools, passes
        // through untouched.
        let filteredEvents: @Sendable (ServerInferenceEvent) -> Void = { event in
            if case .toolCall(let call) = event, MemoryTools.isMemoryTool(call.name) { return }
            onEvent(event)
        }

        // Only the visible transcript feeds memory -- the journal, the
        // replayed assistant turn, the consolidation that reads them. The
        // thoughts reach the client and nothing else.
        var transcript = ""
        var thoughts = ""
        var rounds = 0
        while true {
            let completion = try await gated(current, onEvent: filteredEvents)
            let memoryCalls = completion.toolCalls.filter { MemoryTools.isMemoryTool($0.name) }
            let otherCalls = completion.toolCalls.filter { !MemoryTools.isMemoryTool($0.name) }
            transcript += completion.content
            thoughts += completion.reasoning

            // Stop when the model is done with memory. A turn that also calls
            // a client tool ends here as well: the client has to run that one,
            // and continuing would strand its result.
            guard !memoryCalls.isEmpty, otherCalls.isEmpty else {
                let finished = Self.settled(
                    completion, content: transcript,
                    reasoning: thoughts, toolCalls: otherCalls)
                await journal(
                    request: request, completion: finished, context: context,
                    conversation: conversation, startedAt: startedAt)
                return finished
            }

            // Rounds exhausted and the model still wants memory. Returning
            // here would hand back whatever preamble preceded the last call --
            // measured, that was a 31-token "I need to check the existing
            // memories" where ten chapters should have been. Instead the last
            // calls are answered, the model is told the rounds are gone, and
            // it gets one more generation to answer with what it has. The
            // tools stay in the request so the prompt prefix does not move;
            // any tool call it makes anyway is dropped.
            if rounds >= configuration.maximumToolRounds {
                var messages = current.messages
                messages.append(
                    ServerMemory.assistantMessage(
                        content: completion.content,
                        calls: memoryCalls))
                for call in memoryCalls {
                    let result = await service.execute(
                        name: call.name,
                        arguments: ServerMemory.arguments(from: call.arguments),
                        in: context)
                    messages.append(ServerMemory.toolResultMessage(call: call, result: result))
                }
                messages.append(
                    GFTokenizer.Message(
                        role: .user,
                        content: "Your memory tool rounds for this turn are used up. Answer the "
                            + "original request now, in full, without calling any tools."))
                current = current.replacingMessages(messages, tools: current.tools)
                let last = try await gated(current, onEvent: filteredEvents)
                transcript += last.content
                thoughts += last.reasoning
                let finished = Self.settled(
                    last, content: transcript, reasoning: thoughts,
                    toolCalls: last.toolCalls.filter { !MemoryTools.isMemoryTool($0.name) },
                    finishReason: transcript.isEmpty ? "length" : last.finishReason)
                ServerLog.memory(
                    "tool rounds exhausted; answered without tools "
                        + "session=\(context.session.id)")
                await journal(
                    request: request, completion: finished, context: context,
                    conversation: conversation, startedAt: startedAt)
                return finished
            }

            rounds += 1
            var messages = current.messages
            messages.append(
                ServerMemory.assistantMessage(
                    content: completion.content,
                    calls: memoryCalls))
            for call in memoryCalls {
                let result = await service.execute(
                    name: call.name,
                    arguments: ServerMemory.arguments(from: call.arguments),
                    in: context)
                messages.append(ServerMemory.toolResultMessage(call: call, result: result))
                ServerLog.memory(
                    "tool=\(call.name) "
                        + (result.isFailure ? "failed" : "ok")
                        + " round=\(rounds) session=\(context.session.id)")
            }
            current = current.replacingMessages(messages, tools: current.tools)
        }
    }

    /// The completion a memory turn returns: the whole turn's visible text
    /// and thoughts in place of the last round's, with that round's usage.
    ///
    /// Rebuilding the completion must not drop what the watchdogs saw: this
    /// path runs for every memory-enabled request, so forgetting it here
    /// once silenced the whole feature whenever memory was on. The stop
    /// string that ended the last round is kept for the same reason: without
    /// it a Messages client saw `end_turn` for a turn its stop sequence ended.
    static func settled(
        _ completion: ServerCompletion,
        content: String,
        reasoning: String,
        toolCalls: [ParsedToolCall],
        finishReason: String? = nil
    ) -> ServerCompletion {
        ServerCompletion(
            content: content,
            toolCalls: toolCalls,
            finishReason: finishReason ?? completion.finishReason,
            usage: completion.usage,
            watchdogTrips: completion.watchdogTrips,
            stopSequence: completion.stopSequence,
            reasoning: reasoning,
            unrequestedReasoning: completion.unrequestedReasoning)
    }

    /// Writes the turn to the journal after the completion is settled.
    ///
    /// This runs after generation and before the completion is returned, so
    /// it is on the request path -- but only for a `write(2)` into the page
    /// cache, which is microseconds. The durability barrier is deliberately
    /// not here: the journal takes it a couple of seconds after the last
    /// append, once the drive is idle, so nothing this does can hold the
    /// answer or contend with the expert streamer. The journal swallows its
    /// own failures. Only the user's prompt and the assistant's reply text
    /// go in; tool definitions, tool calls and tool results never reach it,
    /// which is what keeps a turn at a few kilobytes.
    func journal(
        request: ValidatedChatRequest,
        completion: ServerCompletion,
        context: MemorySessionContext,
        conversation: String,
        startedAt: Date
    ) async {
        let index = (turnIndex[conversation] ?? 0)
        turnIndex[conversation] = index + 1
        let prompt = request.messages.last { $0.role == .user }?.content ?? ""
        await service.recordTurn(
            session: context,
            index: index,
            prompt: prompt,
            reply: completion.content,
            model: nil,
            promptTokens: completion.usage.promptTokens,
            completionTokens: completion.usage.completionTokens,
            latencyMilliseconds: Int(Date().timeIntervalSince(startedAt) * 1_000),
            stopReason: completion.finishReason)
        scheduleConsolidation(after: context)
    }

    // MARK: - The generation gate

    func acquireInner() async {
        while let slot = queueInner() {
            await withCheckedContinuation { slot.handOver($0) }
        }
    }

    /// Take the gate or queue on a fresh slot, in one step with no suspension, so a
    /// release cannot land between the check and the registration. Nil means taken.
    private func queueInner() -> SuspensionSlot? {
        if innerBusy {
            let slot = SuspensionSlot()
            innerWaiters.append(slot)
            return slot
        }
        innerBusy = true
        return nil
    }

    func releaseInner() {
        innerBusy = false
        let waiting = innerWaiters
        innerWaiters.removeAll()
        for waiter in waiting { waiter.signal(.wake) }
    }

    /// Runs one inner generation under the gate.
    func gated(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    )
        async throws -> ServerCompletion
    {
        await acquireInner()
        defer { releaseInner() }
        return try await inner.generate(request, onEvent: onEvent)
    }

    /// Identifies a conversation for the purpose of freezing its prompt and
    /// counting its turns.
    func conversationKey(for request: ValidatedChatRequest) -> String {
        let placement = resolvePlacement(for: request)
        return ServerMemory.sessionIdentifier(
            messages: request.messages,
            workspace: placement.workspace)
    }

    /// Runs the session-end hook, if consolidation is on. The engine calls
    /// this when a conversation is finished with; nothing calls it
    /// automatically, because the API has no end-of-conversation signal.
    public func endSession(conversation id: String) async {
        guard let context = contexts.removeValue(forKey: id) else { return }
        await service.endSession(context)
    }

    /// Resolves, and caches, the memory session for this conversation.
    func sessionContext(for request: ValidatedChatRequest) async
        -> MemorySessionContext?
    {
        let placement = resolvePlacement(for: request)
        let id = ServerMemory.sessionIdentifier(
            messages: request.messages,
            workspace: placement.workspace)
        if let existing = contexts[id] { return existing }
        let focus = request.messages.first { $0.role == .user }?.content
            .map { String($0.prefix(600)) }
        guard
            let context = await service.beginSession(
                id: id,
                workspaceOverride: placement.override,
                modelID: nil,
                tag: placement.tag,
                focus: focus)
        else { return nil }
        contexts[id] = context
        // A new session in a scope whose last session still has undistilled
        // turns is a rollover: the end-of-conversation signal the API never
        // sends. The previous session is consolidated once this request has
        // returned, so the person waiting on it does not pay for it.
        if configuration.sessionConsolidation,
            let previous = unconsolidated[context.scope],
            previous.session.id != context.session.id
        {
            idleTimers[context.scope]?.cancel()
            pendingAfterTurn[context.scope] = previous
            unconsolidated[context.scope] = nil
        }
        ServerLog.memory(
            "session=\(context.session.id) scope=\(context.scope.workspace) "
                + "tag=\(placement.tag ?? "-") via=\(placement.source) "
                + "bootstrap=\(context.bootstrap.records.count) "
                + "durable=\(context.isDurable)")
        return context
    }

    /// Where a conversation's memory lives, and why.
    struct Placement {
        /// The workspace the session is placed in.
        let workspace: String
        /// The override handed to the service; nil means the launch workspace.
        let override: String?
        /// The label recorded on the session.
        let tag: String?
        /// For the log: "header", "declared-cwd" or "launch".
        let source: String
    }

}

extension MemoryBackend {
    /// Flush memory to disk and release the workspace locks.
    ///
    /// Called on the way out of a graceful shutdown. Session boundaries are
    /// the usual durability point, but a server told to stop mid-conversation
    /// has records that have not reached a barrier yet, and those are the
    /// ones a person would most notice losing.
    public func shutDown() async {
        for timer in idleTimers.values { timer.cancel() }
        idleTimers.removeAll()
        await service.shutDown()
    }
}

/// Builds the memory decorator, or returns the backend unchanged.
///
/// The command target calls this so it never has to know how the service is
/// assembled or how memory logs; with memory off it is a pass-through and no
/// memory type is constructed.
public enum ServerMemoryFactory {
    /// - Parameters:
    ///   - modelsDirectory: where a side-engine install is looked up. Nil
    ///     leaves the engine off unless a directory is named in the
    ///     environment, and an engine that is off changes nothing.
    ///   - isClientGenerating: read before every token the side-engine
    ///     produces, so it takes one thread while a person is waiting and the
    ///     performance cores in the gaps. The background retrieval pass gates
    ///     on the same read — `isIdle` is its inverse — so T7 runs only in a
    ///     window where nobody is waiting. Nil leaves the width alone and the
    ///     pass ungated, which is what a test or a benchmark wants.
    package static func wrap(
        _ backend: any ServerInferenceBackend,
        configuration: MemoryConfiguration = .fromEnvironment(),
        modelsDirectory: String? = nil,
        isClientGenerating: (@Sendable () -> Bool)? = nil
    )
        -> any ServerInferenceBackend
    {
        guard configuration.isEnabled else {
            if let reason = configuration.disabledReason { ServerLog.memory(reason) }
            return backend
        }
        // Built before the service because the service holds the port. The
        // weights are not read until the first judgement.
        let sideEngine = ServerSideEngineFactory.make(
            modelsDirectory: modelsDirectory,
            isClientGenerating: isClientGenerating)
        // Spelled out rather than mapped: nesting the closure inside
        // `MemoryService(...)`, or even inside an optional `map`, made the
        // type checker crash rather than infer.
        let isIdle: (@Sendable () -> Bool)?
        if let isClientGenerating {
            isIdle = { !isClientGenerating() }
        } else {
            isIdle = nil
        }
        let service = MemoryService(
            configuration: configuration,
            sideEngine: sideEngine.map { SideEngineMemoryAdapter(engine: $0) },
            isIdle: isIdle
        ) { event in
            ServerLog.memory(event.message)
        }
        ServerLog.memory(configuration.summary)
        // Replay the workspace journal now, at boot, rather than when the
        // first request arrives and the model is about to need the disk.
        Task(priority: .utility) { await service.warmUp() }
        return MemoryBackend(
            wrapping: backend,
            service: service,
            configuration: configuration)
    }
}
