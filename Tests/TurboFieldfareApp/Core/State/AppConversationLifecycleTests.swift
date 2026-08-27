import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The three gaps a review of this branch named: unload with a conversation
/// open, the Cancel semantics the app actually ships, and a reset the service
/// never heard. Each of these had no coverage, and each hid a defect.
@Suite struct AppConversationLifecycleTests {
    @MainActor
    private func readyModel(_ client: FakeInferenceClient) async throws -> AppModel {
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(modelDirectory: directory, client: client)
        model.modelPathText = directory.path
        try await client.ensureLoaded(
            modelDirectory: directory, maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions, forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        return model
    }

    @MainActor
    private func finish(_ model: AppModel) async {
        while model.isRunning { await Task.yield() }
    }

    /// The shipping client stops cooperatively: the turn ends at a token
    /// boundary and arrives as `.finished`, so it is committed and its partial
    /// reply stays. Every other cancel test models the in-process client, which
    /// does the opposite — so this path had no coverage at all.
    @MainActor
    @Test func acooperativeStopCommitsTheTurnAndKeepsWhatItProduced() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(40),
                                         cancelSemantics: .cooperativeStop)
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        while model.isRunning, model.liveTokenCount == 0 { await Task.yield() }
        model.cancel()
        await finish(model)

        // Not counting it would put the app behind the service, and the next
        // turn would be refused.
        #expect(model.conversation.committedTurns == 1)
        #expect(model.conversation.turns.map(\.role) == [.user, .assistant])
        #expect(model.diagnostics?.stopReason == .cancelled)
        #expect(model.error == nil, "stopping is not an error")

        // The conversation continues from it, with a turn that really ran:
        // a double that never lowered its stop flag made the next run break at
        // its first token, and this assertion passed on an empty reply.
        model.promptText = "two"
        model.run()
        await finish(model)
        #expect(model.conversation.committedTurns == 2)
        let reply = try #require(model.conversation.turns.last)
        #expect(reply.role == .assistant)
        #expect(!reply.text.isEmpty, "the follow-up turn produced nothing")
        #expect(model.diagnostics?.stopReason != .cancelled,
                "the follow-up turn was stopped by a flag left set")
    }

    /// Pins the *double's* flag hygiene, which every other cooperative-stop
    /// assertion depends on: a fake that never lowered its stop flag made the
    /// following run break at its first token, quietly emptying them. The real
    /// client's equivalent is pinned in the heavy suite, because only a loaded
    /// model exercises its flags.
    @MainActor
    @Test func astopDoesNotLeakIntoTheNextRun() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(30),
                                         cancelSemantics: .cooperativeStop)
        let model = try await readyModel(client)

        model.promptText = "one"
        model.run()
        while model.isRunning, model.liveTokenCount == 0 { await Task.yield() }
        model.cancel()
        await finish(model)
        #expect(model.diagnostics?.stopReason == .cancelled)

        model.promptText = "two"
        model.run()
        await finish(model)
        #expect(model.diagnostics?.stopReason != .cancelled,
                "the next run inherited the stop")
        #expect((model.diagnostics?.generatedTokens ?? 0) > 1,
                "the next run stopped at one token")
    }

    /// The in-process client rewinds the turn instead, and the app must not
    /// count it. Both clients exist, and the app has to be right for both.
    @MainActor
    @Test func ahardAbortDoesNotCountTheTurnItRewound() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(40),
                                         cancelSemantics: .hardAbort)
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        while model.isRunning, model.livePrefillDone == 0 { await Task.yield() }
        model.cancel()
        await finish(model)

        #expect(model.conversation.committedTurns == 0)
        #expect(model.conversation.turns.isEmpty)
        #expect(model.error == .cancelled)
    }

    /// `currentHandles()` is nil when the connection is gone, not when the model
    /// is unloaded. Returning normally let the app record an epoch the service
    /// had never heard of — and because it then matched, the reset was never
    /// retried and every later turn was refused with no way back.
    @MainActor
    @Test func aresetTheServiceNeverHeardIsNotRecordedAsOpen() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        await finish(model)
        #expect(client.conversationEpochs.count == 1)

        model.newChat()
        client.failNextReset(with: .unknown("the decode service connection is gone"))
        model.promptText = "after a lost reset"
        model.run()
        await finish(model)

        #expect(model.error != nil, "a lost reset was reported as success")
        #expect(model.conversation.committedTurns == 0,
                "the turn was counted against a lineage the service never opened")

        // And the next attempt opens it again rather than short-circuiting.
        model.promptText = "retry"
        model.run()
        await finish(model)
        #expect(client.conversationEpochs.count == 2,
                "the reset was never retried, so the conversation could not recover")
        #expect(model.conversation.committedTurns == 1)
    }
}
