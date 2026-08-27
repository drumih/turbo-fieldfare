import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The app and the decode service count committed turns independently, and the
/// service refuses a turn whose index does not match its own count. Every test
/// here pins one way the two counts used to be able to drift, each of which
/// ends the conversation with a rejection the user cannot act on.
@Suite struct AppModelConversationTests {
    @MainActor
    private func readyModel(_ client: FakeInferenceClient) async throws -> AppModel {
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(modelDirectory: directory, client: client)
        model.modelPathText = directory.path
        try await client.ensureLoaded(
            modelDirectory: directory,
            maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions,
            forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        return model
    }

    @MainActor
    private func finish(_ model: AppModel) async {
        while model.isRunning { await Task.yield() }
    }

    @MainActor
    @Test func afreshModelStartsOnTurnZeroOfItsOwnConversation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "hello"
        #expect(model.conversation.isEmpty)
        #expect(model.conversation.committedTurns == 0)

        let request = try model.makeRequest()
        // No ticket was taken, so the request is not yet a conversation turn.
        #expect(request.conversationEpoch == nil)
        #expect(request.turnIndex == nil)
        #expect(request.conversationTokens == 0)
    }

    @MainActor
    @Test func eachRunCarriesTheNextTurnIndexOfOneEpoch() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)

        model.promptText = "one"
        model.run()
        let firstEpoch = model.conversation.epoch
        #expect(model.conversation.turns.first?.role == .user,
                "the user's message shows while the model is still thinking")
        await finish(model)
        #expect(model.conversation.committedTurns == 1)

        model.promptText = "two"
        model.run()
        await finish(model)
        #expect(model.conversation.committedTurns == 2)
        #expect(model.conversation.epoch == firstEpoch,
                "one conversation, not one per turn")
        #expect(model.conversation.turns.map(\.role) == [.user, .assistant, .user, .assistant])
    }

    @MainActor
    @Test func theconversationIsOpenedOnceAndReusedAcrossTurns() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)

        model.promptText = "one"
        model.run()
        await finish(model)
        model.promptText = "two"
        model.run()
        await finish(model)

        // Opening it again per turn would reset the KV and undo the entire
        // point of the feature.
        #expect(client.conversationEpochs == [model.conversation.epoch])
    }

    @MainActor
    @Test func newChatOpensAFreshEpochAndEmptiesTheTranscript() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        await finish(model)
        let firstEpoch = model.conversation.epoch

        model.newChat()
        #expect(model.conversation.turns.isEmpty)
        #expect(model.conversation.committedTurns == 0)
        #expect(model.conversation.epoch != firstEpoch)

        model.promptText = "fresh"
        model.run()
        await finish(model)
        #expect(client.conversationEpochs.count == 2)
        #expect(client.conversationEpochs.last == model.conversation.epoch)
        #expect(model.conversation.committedTurns == 1)
    }

    /// A load rebuilds the runner and the KV, so the service's gate goes back
    /// to expecting turn zero. An app that kept its turn list would keep
    /// numbering from where it left off, and the gate would refuse that turn
    /// and every turn after it for the rest of the session.
    @MainActor
    @Test func areloadStartsAnEmptyConversationSoTheCountsCannotDesync() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        await finish(model)
        model.promptText = "two"
        model.run()
        await finish(model)
        #expect(model.conversation.committedTurns == 2)
        let before = model.conversation.epoch

        model.applyLoadState(.ready(
            modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1))

        #expect(model.conversation.committedTurns == 0)
        #expect(model.conversation.turns.isEmpty)
        #expect(model.conversation.epoch != before)
        // The transcript stays — lifecycle actions do not discard it — but the
        // turns move out of the model's context and the break says where.
        #expect(model.hasOutputTranscript)
        #expect(model.archivedPairs.count == 2)
        #expect(model.transcriptContextBreak == 2)

        model.promptText = "after reload"
        model.run()
        await finish(model)
        #expect(model.conversation.committedTurns == 1)
        #expect(client.conversationEpochs.last == model.conversation.epoch)
    }

    /// Pointing the app at a different model releases the runner and the KV,
    /// and the service ends its gate's lineage with it. `applyLoadState`'s
    /// branch for this is unreachable — every production transition to
    /// `.notLoaded` assigns `loadState` directly — so the app kept a turn count
    /// and an epoch for a conversation that no longer existed, and drew it with
    /// no break. Every such site now ends the conversation; this is the one
    /// reachable without a real install fixture.
    @MainActor
    @Test func releasingTheKVEndsTheConversationAndMarksTheTranscript() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        await finish(model)
        model.promptText = "two"
        model.run()
        await finish(model)
        #expect(model.conversation.committedTurns == 2)
        let before = model.conversation.epoch

        model.setModelURL(FileManager.default.temporaryDirectory
            .appendingPathComponent("another-model", isDirectory: true))

        #expect(model.conversation.committedTurns == 0)
        #expect(model.conversation.turns.isEmpty)
        #expect(model.conversation.epoch != before)
        // The transcript survives — lifecycle actions do not discard it — but
        // the turns are marked as out of the model's context.
        #expect(model.archivedPairs.count == 2)
        #expect(model.transcriptContextBreak == 2)
    }

    @MainActor
    @Test func areloadEndsTheLineageSoTheNextTurnReopensIt() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        await finish(model)
        #expect(client.conversationEpochs.count == 1)

        // A load rebuilds the runner and the KV. Resuming onto it would resume
        // onto a cache that no longer holds the conversation.
        model.loadState = .ready(
            modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.applyLoadState(.ready(
            modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1))

        model.promptText = "two"
        model.run()
        await finish(model)
        #expect(client.conversationEpochs.count == 2,
                "the turn after a load has to open the conversation again")
    }

    /// The live fields keep holding the last finished turn so its answer stays
    /// on screen. Drawing that pair as history too would show it twice.
    @MainActor
    @Test func historyExcludesTheTurnTheLiveFieldsAreStillShowing() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)

        model.promptText = "one"
        model.run()
        await finish(model)
        #expect(model.conversation.completedPairs.count == 1)
        #expect(model.transcriptHistory.isEmpty,
                "the only finished turn is the one still drawn live")

        model.promptText = "two"
        model.run()
        await finish(model)
        #expect(model.conversation.completedPairs.count == 2)
        #expect(model.transcriptHistory.count == 1)
        #expect(model.transcriptHistory[0].user.text == "one")
    }

    @MainActor
    @Test func copyConversationSpansEveryTurn() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)
        model.promptText = "first question"
        model.run()
        await finish(model)
        model.promptText = "second question"
        model.run()
        await finish(model)

        let text = model.outputConversationPlainText
        #expect(text.contains("first question"))
        #expect(text.contains("second question"))
        let first = try #require(text.range(of: "first question"))
        let second = try #require(text.range(of: "second question"))
        #expect(first.lowerBound < second.lowerBound, "turns are out of order")
    }

    @MainActor
    @Test func newChatEmptiesTheHistoryTheTranscriptDraws() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        await finish(model)
        model.promptText = "two"
        model.run()
        await finish(model)
        #expect(!model.transcriptHistory.isEmpty)

        model.newChat()
        #expect(model.transcriptHistory.isEmpty)
        #expect(model.outputConversationPlainText.isEmpty)
        #expect(!model.hasOutputTranscript)
    }

    /// Stop, then send again. The cancelled turn was rewound by the runtime and
    /// the service did not count it; counting it here put the app one ahead for
    /// the rest of the conversation, and the gate's refusal of the next turn
    /// reached the user as "decode service runtime profile changed during
    /// generation" — a wrong diagnosis for a real fault.
    @MainActor
    @Test func acancelledTurnIsNotCountedSoTheNextOneStillFits() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(60))
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        await finish(model)
        #expect(model.conversation.committedTurns == 1)

        model.promptText = "two"
        model.run()
        // Wait for the run to actually be under way. Cancelling on the same
        // tick races the double's own task registration, not the product.
        while model.isRunning, model.livePrefillDone == 0 { await Task.yield() }
        model.cancel()
        await finish(model)

        #expect(model.conversation.committedTurns == 1,
                "a rewound turn was counted, so the next index will be refused")
        #expect(model.conversation.turns.count == 2,
                "the stopped turn entered the conversation the model can see")
        // What the user was looking at stays on screen as the current turn; it
        // simply never becomes history, because the model cannot see it.
        #expect(model.hasOutputTranscript)
        #expect(model.outputPromptText == "two")
        #expect(!model.conversation.hasTurnInFlight)

        // And the next turn takes the position the stopped one did not consume.
        // The position the stopped turn did not consume is still free, which is
        // the whole point: the service's gate expects exactly this index.
        #expect(model.conversation.committedTurns == 1)
        model.promptText = "three"
        #expect(model.canRun, "the app refused to send after a stop")
    }

    @MainActor
    @Test func arefusedRunGivesThePromptBackAndKeepsTheTurnIndex() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client)
        model.promptText = "one"
        model.run()
        await finish(model)

        // A request the client refuses never reaches the KV, so the app must
        // not count it — the service does not either.
        model.maxContextTokens = model.maxContextTokens * 2
        model.promptText = "two"
        model.run()
        await finish(model)

        #expect(model.conversation.committedTurns == 1)
        #expect(model.conversation.turns.count == 2, "the refused turn left no trace")
        #expect(model.promptText == "two", "the user gets their message back")
    }
}
