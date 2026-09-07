import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The transcript keeps two things per turn that must not be confused: the text
/// the reader sees, and the token IDs the model saw. Storing only the first is
/// what makes a reopened conversation a different conversation, because the
/// pinned template strips historical thought spans when it re-renders.
@Suite struct AppConversationRecordTests {
    private func diagnostics(
        prompt: [Int32]?, generated: [Int32]?,
        boundary: [Int32]? = nil, boundaryNeedsReplay: Bool? = nil,
        conversationTokens: Int? = nil
    ) -> AppDiagnostics {
        var value = AppDiagnostics(
            generatedTokens: generated?.count ?? 0,
            stopReason: .endOfTurn,
            promptTokenCount: prompt?.count,
            conversationTokens: conversationTokens,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        value.promptTokenIDs = prompt
        value.generatedTokenIDs = generated
        value.boundaryTokenIDs = boundary
        value.boundaryNeedsReplay = boundaryNeedsReplay
        return value
    }

    @Test func eachHalfOfATurnKeepsTheIDsItPutInTheKV() throws {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "hello")
        conversation.completeTurn(
            text: "hi there",
            diagnostics: diagnostics(prompt: [2, 105, 7], generated: [40, 41],
                                     conversationTokens: 5))
        #expect(conversation.turns[0].role == .user)
        #expect(conversation.turns[0].tokenIDs == [2, 105, 7])
        #expect(conversation.turns[1].role == .assistant)
        #expect(conversation.turns[1].tokenIDs == [40, 41])
        // Concatenating the record has to reproduce the KV the runtime reports.
        #expect(conversation.turns.compactMap(\.tokenIDs).flatMap { $0 }.count
            == conversation.kvTokens)
    }

    /// A run that stopped on max tokens or was cancelled left one emitted token
    /// outside the KV, and the next turn replays it. A stored conversation that
    /// dropped it would reopen with a context missing a token the model already
    /// produced.
    @Test func theUncommittedBoundaryIsPartOfTheRecord() {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "hello")
        conversation.completeTurn(
            text: "partial",
            diagnostics: diagnostics(prompt: [1], generated: [9],
                                     boundary: [77], boundaryNeedsReplay: true,
                                     conversationTokens: 2))
        #expect(conversation.boundaryTokenIDs == [77])
        #expect(conversation.boundaryNeedsReplay)

        // And a turn that ended cleanly clears it, rather than leaving the
        // previous turn's boundary to be replayed a second time.
        _ = conversation.beginTurn(text: "again")
        conversation.completeTurn(
            text: "done",
            diagnostics: diagnostics(prompt: [2], generated: [10],
                                     conversationTokens: 4))
        #expect(conversation.boundaryTokenIDs.isEmpty)
        #expect(!conversation.boundaryNeedsReplay)
    }

    /// The single-prompt path retains nothing, so it reports no IDs. The record
    /// has to stay nil rather than become an empty array that would later read
    /// as a turn which contributed nothing to the context.
    @Test func aTurnWithNoReportedRecordStoresNothing() {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "hello")
        conversation.completeTurn(
            text: "hi", diagnostics: diagnostics(prompt: nil, generated: nil))
        #expect(conversation.turns[0].tokenIDs == nil)
        #expect(conversation.turns[1].tokenIDs == nil)
    }

    /// A turn the runtime rewound is in neither the KV nor the transcript, so
    /// its IDs must not survive either.
    @Test func anAbandonedTurnTakesItsRecordWithIt() {
        var conversation = AppConversation()
        _ = conversation.beginTurn(text: "hello")
        conversation.abandonTurn()
        #expect(conversation.turns.isEmpty)
        #expect(conversation.committedTurns == 0)
    }
}
