import Foundation
import Testing
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareDecodeService

/// The service appends turns to a KV it cannot rewind past 280 tokens, so a
/// turn admitted against the wrong lineage is not a recoverable error: the
/// model's context permanently holds a message the user never sent in that
/// conversation. Every case here is one way that used to be possible before the
/// gate existed, when `generate` carried no conversation identity at all.
@Suite struct DecodeConversationGateTests {
    private func turn(_ epoch: UUID?, _ index: Int?) -> DecodeGenerationRequest {
        DecodeGenerationRequest(
            prompt: "hello", maxNewTokens: 16, maxContextTokens: 8_192,
            temperature: 0, conversationEpoch: epoch, turnIndex: index)
    }

    @Test func aOneShotRunsWhenNoConversationIsOpen() {
        let gate = DecodeConversationGate()
        #expect(gate.admit(turn(nil, nil)) == .success(.oneShot))
    }

    @Test func aOneShotIsRefusedWhileAConversationIsOpen() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: epoch)
        // A one-shot resets the KV before prefilling. Running it here would
        // wipe the lineage the app still believes it is talking to.
        #expect(gate.admit(turn(nil, nil))
            == .failure(.oneShotDuringConversation(open: epoch)))
    }

    @Test func turnsAreAdmittedInOrderAndOnlyAfterTheyCommit() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: epoch)
        #expect(gate.admit(turn(epoch, 0)) == .success(.turn(epoch: epoch, index: 0)))
        // Not yet committed, so turn 1 is still out of order.
        #expect(gate.admit(turn(epoch, 1))
            == .failure(.outOfOrderTurn(requested: 1, committed: 0)))
        gate.commit(.turn(epoch: epoch, index: 0))
        #expect(gate.committedTurns == 1)
        #expect(gate.admit(turn(epoch, 1)) == .success(.turn(epoch: epoch, index: 1)))
    }

    @Test func areplayedTurnIndexIsRefused() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: epoch)
        gate.commit(.turn(epoch: epoch, index: 0))
        // A retransmitted turn 0 would append the same user message twice,
        // behind its own reply.
        #expect(gate.admit(turn(epoch, 0))
            == .failure(.outOfOrderTurn(requested: 0, committed: 1)))
    }

    @Test func aturnWithoutAnIndexIsRefused() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: epoch)
        #expect(gate.admit(turn(epoch, nil))
            == .failure(.outOfOrderTurn(requested: nil, committed: 0)))
    }

    @Test func aturnFromAReplacedConversationIsRefused() {
        let first = UUID()
        let second = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: first)
        gate.commit(.turn(epoch: first, index: 0))
        gate.reset(to: second)
        #expect(gate.committedTurns == 0, "a reset restarts the turn order")
        #expect(gate.admit(turn(first, 1))
            == .failure(.staleConversation(requested: first, open: second)))
    }

    @Test func arejectedTurnDoesNotAdvanceTheOrder() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: epoch)
        _ = gate.admit(turn(epoch, 7))
        _ = gate.admit(turn(UUID(), 0))
        #expect(gate.committedTurns == 0)
        #expect(gate.admit(turn(epoch, 0)) == .success(.turn(epoch: epoch, index: 0)))
    }

    @Test func committingAnAdmissionFromASupersededEpochIsIgnored() {
        let first = UUID()
        let second = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: first)
        let admission = DecodeConversationGate.Admission.turn(epoch: first, index: 0)
        // The turn was admitted, then the user started a new chat while it was
        // still decoding. Counting it against the new lineage would push every
        // later turn one position out of step.
        gate.reset(to: second)
        gate.commit(admission)
        #expect(gate.committedTurns == 0)
    }

    @Test func endingTheLineageRestoresTheOneShotPath() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: epoch)
        gate.commit(.turn(epoch: epoch, index: 0))
        // Both unload and load reach here: each releases or rebuilds the KV, so
        // the tokens the epoch named are gone.
        gate.endLineage()
        #expect(gate.openEpoch == nil)
        #expect(gate.committedTurns == 0)
        #expect(gate.admit(turn(epoch, 1))
            == .failure(.staleConversation(requested: epoch, open: nil)))
        #expect(gate.admit(turn(nil, nil)) == .success(.oneShot))
    }

    @Test func rejectionsNameBothSidesSoTheLogDiagnosesItself() {
        let requested = UUID()
        let open = UUID()
        let stale = DecodeConversationGate.Rejection
            .staleConversation(requested: requested, open: open)
        #expect(stale.message.contains(requested.uuidString))
        #expect(stale.message.contains(open.uuidString))
        let order = DecodeConversationGate.Rejection
            .outOfOrderTurn(requested: 4, committed: 2)
        #expect(order.message.contains("4"))
        #expect(order.message.contains("2"))
        let unset = DecodeConversationGate.Rejection
            .outOfOrderTurn(requested: nil, committed: 0)
        #expect(unset.message.contains("unset"))
    }
}
