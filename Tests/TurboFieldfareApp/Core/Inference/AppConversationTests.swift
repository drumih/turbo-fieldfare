import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The invariant under test throughout: the transcript equals the model's
/// context. The decode service counts committed turns independently, and a
/// count that drifts by one on either side makes the gate reject every later
/// turn of the conversation.
@Suite struct AppConversationTests {
    private func diagnostics(prompt: Int?, cached: Int?, generated: Int,
                             stopReason: AppStopReason = .endOfTurn,
                             omitConversationTokens: Bool = false) -> AppDiagnostics {
        AppDiagnostics(
            generatedTokens: generated,
            stopReason: stopReason,
            promptTokenCount: prompt,
            cachedPromptTokens: cached,
            conversationTokens: omitConversationTokens
                ? nil : prompt.map { $0 + generated },
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
    }

    @Test func afirstTurnIsIndexZeroAndShowsBeforeItsReply() throws {
        var conversation = AppConversation()
        let started = conversation.beginTurn(text: "hello")
        let ticket = try #require(started)
        #expect(ticket.index == 0)
        #expect(ticket.epoch == conversation.epoch)
        // The user's message is on screen while the model is still thinking.
        #expect(conversation.turns.count == 1)
        #expect(conversation.turns[0].role == .user)
        #expect(conversation.committedTurns == 0, "nothing is in the KV yet")
        #expect(conversation.hasTurnInFlight)
    }

    @Test func acompletedTurnAdvancesTheIndexTheServiceChecks() throws {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "one")
        conversation.completeTurn(
            text: "first reply",
            diagnostics: diagnostics(prompt: 12, cached: 0, generated: 5))
        #expect(conversation.committedTurns == 1)
        #expect(conversation.turns.map(\.role) == [.user, .assistant])
        #expect(conversation.turns[0].cachedTokens == 0)
        #expect(conversation.turns[1].text == "first reply")
        #expect(conversation.kvTokens == 17)

        let began = conversation.beginTurn(text: "two")
        let second = try #require(began)
        #expect(second.index == 1)
    }

    @Test func asecondSendIsRefusedWhileATurnIsDecoding() {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "one")
        // A double-tap on Send must not reserve a second position: the service
        // would reject it, and the transcript would show a message that never
        // reached the model.
        #expect(conversation.beginTurn(text: "two") == nil)
        #expect(conversation.turns.count == 1)
    }

    /// A turn that threw was rewound by the runtime, so its tokens are not in
    /// the KV. Leaving it in the transcript would show the user a message the
    /// model never saw, and counting it would push every later turn one
    /// position out of step with the service's gate.
    @Test func anabandonedTurnLeavesNoTraceAndKeepsTheIndex() throws {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "one")
        conversation.completeTurn(
            text: "reply", diagnostics: diagnostics(prompt: 10, cached: 0, generated: 4))

        let began = conversation.beginTurn(text: "two", images: [])
        let ticket = try #require(began)
        #expect(ticket.index == 1)
        let abandoned = conversation.abandonTurn()
        let returned = try #require(abandoned)
        #expect(returned.text == "two", "the composer gets the text back")
        #expect(conversation.turns.count == 2)
        #expect(conversation.committedTurns == 1)
        // The next attempt reuses the same position.
        #expect(conversation.beginTurn(text: "two again")?.index == 1)
    }

    @Test func astoppedTurnStillCommits() throws {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "one")
        // Stopping ends the turn at a token boundary with the partial reply in
        // the KV, so it counts exactly like a finished one.
        conversation.completeTurn(
            text: "partial",
            diagnostics: diagnostics(prompt: 9, cached: 0, generated: 2,
                                     stopReason: .cancelled))
        #expect(conversation.committedTurns == 1)
        #expect(conversation.turns[1].stopReason == .cancelled)
        #expect(conversation.beginTurn(text: "two")?.index == 1)
    }

    @Test func alostLineageKeepsTheTranscriptAndRefusesEverySend() throws {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "one")
        conversation.completeTurn(
            text: "reply", diagnostics: diagnostics(prompt: 10, cached: 0, generated: 4))
        _ = conversation.beginTurn(text: "two")

        let lost = conversation.markLineageLost()
        let returned = try #require(lost)
        #expect(returned.text == "two")
        #expect(conversation.isLineageLost)
        // What the user already read stays readable.
        #expect(conversation.turns.count == 2)
        #expect(!conversation.canSend)
        #expect(conversation.beginTurn(text: "three") == nil)
    }

    @Test func anewChatClearsEverythingAndTakesAFreshEpoch() {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "one")
        conversation.completeTurn(
            text: "reply", diagnostics: diagnostics(prompt: 10, cached: 0, generated: 4))
        conversation.markLineageLost()
        let previous = conversation.epoch

        conversation.startNew()
        #expect(conversation.epoch != previous)
        #expect(conversation.turns.isEmpty)
        #expect(conversation.committedTurns == 0)
        #expect(conversation.kvTokens == 0)
        #expect(!conversation.isLineageLost)
        #expect(conversation.beginTurn(text: "fresh")?.index == 0)
    }

    @Test func cachedTokensGrowWithTheConversation() {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "one")
        conversation.completeTurn(
            text: "a", diagnostics: diagnostics(prompt: 10, cached: 0, generated: 5))
        _ = conversation.beginTurn(text: "two")
        conversation.completeTurn(
            text: "b", diagnostics: diagnostics(prompt: 20, cached: 15, generated: 6))
        // Turn two reused everything turn one left behind. This is the figure
        // the HUD shows, and the only way a regression to full re-prefill is
        // visible without a stopwatch.
        #expect(conversation.turns[2].cachedTokens == 15)
        #expect(conversation.kvTokens == 26)
    }

    @Test(arguments: [
        AppStopReason.endOfTurn,
        .maxTokens,
        .cancelled,
        .stopString
    ])
    func missingCommittedPositionMarksTheKVCountUnknown(_ stopReason: AppStopReason) {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "one")
        conversation.completeTurn(
            text: "a", diagnostics: diagnostics(prompt: 10, cached: 0, generated: 5))
        #expect(conversation.kvTokens == 15)

        _ = conversation.beginTurn(text: "two")
        conversation.completeTurn(
            text: "b",
            diagnostics: diagnostics(
                prompt: 40,
                cached: 15,
                generated: 9,
                stopReason: stopReason,
                omitConversationTokens: true))

        #expect(conversation.kvTokens == nil)
        #expect(conversation.committedTurns == 2)
    }

    @Test func completingWithoutAPendingTurnChangesNothing() {
        var conversation = AppConversation()
        conversation.completeTurn(
            text: "orphan", diagnostics: diagnostics(prompt: 3, cached: 0, generated: 1))
        #expect(conversation.turns.isEmpty)
        #expect(conversation.committedTurns == 0)
    }
}
