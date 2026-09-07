import Foundation
import Testing
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareDecodeService

/// Reopening a stored conversation puts turns back into a KV the service can
/// only append to. Every case here is a way the reopened lineage could end up
/// disagreeing with the transcript the user is reading, which no later check
/// can detect.
@Suite struct DecodeConversationRestoreTests {
    private func turn(_ epoch: UUID?, _ index: Int?) -> DecodeGenerationRequest {
        DecodeGenerationRequest(
            prompt: "hello", maxNewTokens: 16, maxContextTokens: 8_192,
            temperature: 0, conversationEpoch: epoch, turnIndex: index)
    }

    /// The bug this pins: opening a restored lineage at zero, the way a reset
    /// does, rejects the reopened conversation's next turn as out of order and
    /// keeps rejecting every turn after it.
    @Test func aRestoredLineageContinuesAtItsStoredTurnCount() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.restore(to: epoch, committedTurns: 4)
        #expect(gate.admit(turn(epoch, 4)) == .success(.turn(epoch: epoch, index: 4)))
        #expect(gate.admit(turn(epoch, 3))
            == .failure(.outOfOrderTurn(requested: 3, committed: 4)))
        #expect(gate.admit(turn(epoch, 5))
            == .failure(.outOfOrderTurn(requested: 5, committed: 4)))
    }

    @Test func aTurnForTheReplacedConversationIsStillRefusedAfterARestore() {
        let stale = UUID()
        let restored = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: stale)
        gate.commit(.turn(epoch: stale, index: 0))
        gate.restore(to: restored, committedTurns: 2)
        #expect(gate.admit(turn(stale, 2))
            == .failure(.staleConversation(requested: stale, open: restored)))
        // And a one-shot cannot run over the restored lineage either.
        #expect(gate.admit(turn(nil, nil))
            == .failure(.oneShotDuringConversation(open: restored)))
    }

    /// The count arrives over a socket. A negative one would make `committedTurns`
    /// unreachable by any turn index and lock the conversation out for good.
    @Test func aNegativeStoredTurnCountIsClampedRatherThanTrusted() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.restore(to: epoch, committedTurns: -3)
        #expect(gate.committedTurns == 0)
        #expect(gate.admit(turn(epoch, 0)) == .success(.turn(epoch: epoch, index: 0)))
    }

    /// A restore that fails has already taken the KV it was replacing.
    ///
    /// The session resets before it prefills, so by the time a digest mismatch
    /// or a bad token refuses the record, the previous lineage's tokens are
    /// gone. Leaving its epoch open admitted its next turn onto an empty cache
    /// as an opening turn, under a transcript showing three exchanges.
    @Test func aFailedRestoreEndsTheLineageItReplaced() {
        let previous = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: previous)
        gate.commit(.turn(epoch: previous, index: 0))
        gate.commit(.turn(epoch: previous, index: 1))
        gate.commit(.turn(epoch: previous, index: 2))

        gate.restoreFailed()

        #expect(gate.openEpoch == nil)
        #expect(gate.committedTurns == 0)
        #expect(gate.admit(turn(previous, 3))
            == .failure(.staleConversation(requested: previous, open: nil)))
        // Nothing is open, so a one-shot may reset the cache and run.
        #expect(gate.admit(turn(nil, nil)) == .success(.oneShot))
    }

    @Test func unloadEndsARestoredLineageToo() {
        let epoch = UUID()
        var gate = DecodeConversationGate()
        gate.restore(to: epoch, committedTurns: 2)
        gate.endLineage()
        #expect(gate.openEpoch == nil)
        #expect(gate.admit(turn(epoch, 2))
            == .failure(.staleConversation(requested: epoch, open: nil)))
    }

    /// A refused turn is a lost lineage, not a failure.
    ///
    /// The service used to send `kind: .failed`, which the app turns into a
    /// retryable `.unknown` — so it put the message back in the composer and
    /// offered a retry that the gate refuses identically every time. Nothing
    /// but a new chat can clear a refusal, and `.lineageLost` is the kind that
    /// says so. `DecodeServiceInferenceClient` maps it to
    /// `conversationLineageLost`; `.failed` it does not.
    @Test func agateRefusalIsSentBackAsALostLineage() throws {
        let stale = UUID()
        let open = UUID()
        var gate = DecodeConversationGate()
        gate.reset(to: open)
        guard case .failure(let rejection) = gate.admit(turn(stale, 0)) else {
            Issue.record("a turn from a replaced conversation was admitted")
            return
        }

        let generationID = UUID()
        let event = rejection.terminalEvent(
            generationID: generationID,
            conversationEpoch: gate.openEpoch)

        #expect(event.kind == .lineageLost, "the app would offer a doomed retry")
        #expect(event.generationID == generationID)
        #expect(event.error == rejection.message)
        #expect(event.conversationEpoch == open)
        // And it survives the wire as that kind, which is what the app reads.
        let frame = try DecodeFrameCodec.encode(event)
        let decoded = try JSONDecoder().decode(
            DecodeServiceEvent.self, from: frame.dropFirst(4))
        #expect(decoded.kind == .lineageLost)
        #expect(decoded.error == rejection.message)
    }

    /// Every rejection the gate can produce takes the same route.
    @Test func everyRefusalKindIsSentBackAsALostLineage() {
        let open = UUID()
        let rejections: [DecodeConversationGate.Rejection] = [
            .staleConversation(requested: UUID(), open: open),
            .outOfOrderTurn(requested: 3, committed: 1),
            .oneShotDuringConversation(open: open),
        ]
        for rejection in rejections {
            let event = rejection.terminalEvent(
                generationID: UUID(),
                conversationEpoch: open)
            #expect(event.kind == .lineageLost)
            #expect(event.error == rejection.message)
        }
    }

    @Test func theRestoreRequestSurvivesTheWireWithItsTokenIDs() throws {
        let request = DecodeRestoreConversationRequest(
            tokenIDs: [2, 105, -1, 262_143],
            images: [DecodeReplayImage(
                tokenLowerBound: 12, tokenCount: 256,
                path: "/tmp/a.png", expectedDigest: "ab12")],
            boundaryTokenIDs: [77],
            boundaryNeedsReplay: true,
            committedTurns: 3,
            maxContextTokens: 8_192)
        let frame = try DecodeFrameCodec.encode(
            DecodeServiceCommand.restoreConversation(request))
        let decoded = try JSONDecoder().decode(
            DecodeServiceCommand.self, from: frame.dropFirst(4))
        guard case .restoreConversation(let round) = decoded else {
            Issue.record("the command decoded as something else")
            return
        }
        #expect(round == request)
    }

    /// 8K `Int32` as JSON is about 50 KB against a 4 MiB frame cap, and 64K
    /// about 400 KB. The IDs ride the terminal event for that reason, so the
    /// headroom is worth pinning rather than assuming.
    @Test func aFullContextOfTokenIDsFitsTheFrameCap() throws {
        let event = DecodeServiceEvent(
            kind: .finished, generationID: UUID(),
            promptTokenIDs: (0..<32_768).map { Int32($0) },
            generatedTokenIDs: (0..<32_768).map { Int32(262_143 - $0) })
        let frame = try DecodeFrameCodec.encode(event)
        #expect(frame.count < DecodeFrameCodec.maximumPayloadBytes)
    }

    @Test func aTerminalEventCarriesTheTurnsRecord() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)
        var record = AppDiagnostics(
            generatedTokens: 2,
            stopReason: .endOfTurn,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        record.promptTokenIDs = [1, 2, 3]
        record.generatedTokenIDs = [4, 5]
        record.boundaryTokenIDs = [6]
        record.boundaryNeedsReplay = true

        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()
        outbox.publish(.finished(record))
        let terminal = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        outbox.finish()
        #expect(writerFinished.wait(timeout: .now() + 2) == .success)

        #expect(terminal.kind == .finished)
        #expect(terminal.promptTokenIDs == [1, 2, 3])
        #expect(terminal.generatedTokenIDs == [4, 5])
        #expect(terminal.boundaryTokenIDs == [6])
        #expect(terminal.boundaryNeedsReplay == true)
    }

    /// The service opens whatever path a replay image names, so the store root
    /// is a trust boundary and not a tidiness rule.
    @Test func onlyPathsInsideThisModelsStoreAreAccepted() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("store-\(UUID().uuidString)", isDirectory: true)
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let store = ConversationStoreLocation.url(forModelDirectory: model)
        #expect(store == root.appendingPathComponent("conversations", isDirectory: true))

        let inside = store.appendingPathComponent("abc/images/0.png")
        #expect(ConversationStoreLocation.contains(inside, forModelDirectory: model))

        // A sibling whose name merely starts the same must not pass as a prefix.
        let lookalike = root
            .appendingPathComponent("conversations-elsewhere/abc/images/0.png")
        #expect(!ConversationStoreLocation.contains(lookalike, forModelDirectory: model))
        #expect(!ConversationStoreLocation.contains(
            URL(fileURLWithPath: "/etc/passwd"), forModelDirectory: model))
        // Traversal has to lose: standardizing resolves it before the compare.
        #expect(!ConversationStoreLocation.contains(
            store.appendingPathComponent("../../secret.png"), forModelDirectory: model))
        // The root itself is not a file inside the root.
        #expect(!ConversationStoreLocation.contains(store, forModelDirectory: model))
    }
}
