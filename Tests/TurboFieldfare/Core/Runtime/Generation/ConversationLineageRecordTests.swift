import Foundation
import Testing
@testable import TurboFieldfare

/// The split that turns one turn into two transcript records, and the identity
/// that decides whether those records may be replayed at all.
@Suite struct ConversationLineageRecordTests {
    // MARK: - A record's token IDs are checked before they reach the table

    private struct BadToken: Equatable {
        let index: Int
        let token: Int32
    }

    private func firstBadToken(_ body: () throws -> Void) -> BadToken? {
        do {
            try body()
            return nil
        } catch MultimodalConversationError.invalidReplayToken(
            let index, let token, _) {
            return BadToken(index: index, token: token)
        } catch {
            Issue.record("unexpected error: \(error)")
            return nil
        }
    }

    /// Restore is the one prefill of IDs the tokenizer did not produce, and the
    /// embedding kernel indexes its table by the raw value. A negative ID
    /// became `UInt32(bitPattern:)` of itself, a too-large one read past the
    /// table; both were GPU reads of memory that is not the model.
    @Test func aTokenOutsideTheVocabularyIsRefusedBeforeItReachesTheTable() {
        let vocab = 262_144
        let negative = MultimodalConversationLineage(tokenIDs: [2, -1, 3])
        #expect(firstBadToken { try negative.validateTokenIDs(vocabSize: vocab) }
            == BadToken(index: 1, token: -1))
        let past = MultimodalConversationLineage(tokenIDs: [2, 3, Int32(vocab)])
        #expect(firstBadToken { try past.validateTokenIDs(vocabSize: vocab) }
            == BadToken(index: 2, token: Int32(vocab)))
        let last = MultimodalConversationLineage(tokenIDs: [Int32(vocab - 1), 0])
        #expect(firstBadToken { try last.validateTokenIDs(vocabSize: vocab) } == nil)
    }

    /// The boundary token enters the same prefill, ahead of the next turn.
    @Test func aBoundaryTokenOutsideTheVocabularyIsRefusedToo() {
        let lineage = MultimodalConversationLineage(
            tokenIDs: [1, 2], uncommittedBoundary: [300_000], boundaryNeedsReplay: true)
        #expect(firstBadToken { try lineage.validateTokenIDs(vocabSize: 262_144) }
            == BadToken(index: 2, token: 300_000))
    }

    /// What the next turn's prompt starts with: the KV, plus the boundary only
    /// when it is going to be replayed.
    @Test func thePromptPrefixCountsAReplayedBoundaryOnly() {
        #expect(MultimodalConversationLineage(
            tokenIDs: [1, 2, 3], uncommittedBoundary: [4],
            boundaryNeedsReplay: true).promptPrefixCount == 4)
        #expect(MultimodalConversationLineage(
            tokenIDs: [1, 2, 3], uncommittedBoundary: [4],
            boundaryNeedsReplay: false).promptPrefixCount == 3)
    }

    @Test func generatedTokensAreWhatTheKVKeptPastThePrompt() {
        let kv: [Int32] = [1, 2, 3, 4, 5, 6, 7]
        #expect(MultimodalTurnRecord.generatedTokenIDs(kvTokenIDs: kv, promptCount: 4)
            == [5, 6, 7])
    }

    /// Without the suffix-of-the-record rule this returned the tokens the decode
    /// loop counted, including the ones a stop-string match had just rewound out
    /// of the KV. Replaying that transcript would prefill assistant text the
    /// cache never held and the reader never saw.
    @Test func aStopStringTrimIsAlreadyAppliedToTheGeneratedRecord() {
        let trimmed: [Int32] = [1, 2, 3, 4, 5]
        #expect(MultimodalTurnRecord.generatedTokenIDs(
            kvTokenIDs: trimmed, promptCount: 4) == [5])
    }

    @Test func aTrimBackToThePromptBoundaryLeavesNothingGenerated() {
        let kv: [Int32] = [1, 2, 3, 4]
        #expect(MultimodalTurnRecord.generatedTokenIDs(
            kvTokenIDs: kv, promptCount: 4).isEmpty)
        // A reset-to-empty after a trim leaves fewer tokens than the prompt had.
        #expect(MultimodalTurnRecord.generatedTokenIDs(
            kvTokenIDs: [], promptCount: 4).isEmpty)
    }

    /// Concatenating the per-turn records has to reproduce the KV exactly; that
    /// is the whole reason the record is token IDs rather than text.
    @Test func concatenatingTurnRecordsReproducesTheKV() {
        var kv: [Int32] = []
        var record: [Int32] = []
        for turn in 0..<3 {
            let prompt: [Int32] = [Int32(turn * 10), Int32(turn * 10 + 1)]
            let promptCount = kv.count + prompt.count
            kv += prompt
            kv += [Int32(turn * 10 + 2), Int32(turn * 10 + 3)]
            record += prompt
            record += MultimodalTurnRecord.generatedTokenIDs(
                kvTokenIDs: kv, promptCount: promptCount)
        }
        #expect(record == kv)
    }

    @Test func identityRoundTripsAndDistinguishesEveryField() throws {
        let identity = ConversationIdentity(
            modelID: "google/gemma-4-26B-A4B-it",
            sourceSnapshotHash: "0d77464e",
            templateIdentity: GFTokenizer.chatTemplateIdentity,
            imageProcessingVersion: VisionImageProcessing.version)
        let decoded = try JSONDecoder().decode(
            ConversationIdentity.self, from: try JSONEncoder().encode(identity))
        #expect(decoded == identity)

        #expect(decoded != ConversationIdentity(
            modelID: "other", sourceSnapshotHash: "0d77464e",
            templateIdentity: GFTokenizer.chatTemplateIdentity,
            imageProcessingVersion: VisionImageProcessing.version))
        #expect(decoded != ConversationIdentity(
            modelID: identity.modelID, sourceSnapshotHash: "deadbeef",
            templateIdentity: GFTokenizer.chatTemplateIdentity,
            imageProcessingVersion: VisionImageProcessing.version))
        #expect(decoded != ConversationIdentity(
            modelID: identity.modelID, sourceSnapshotHash: identity.sourceSnapshotHash,
            templateIdentity: "gemma4-it-text-no-tools-v2",
            imageProcessingVersion: VisionImageProcessing.version))
        #expect(decoded != ConversationIdentity(
            modelID: identity.modelID, sourceSnapshotHash: identity.sourceSnapshotHash,
            templateIdentity: GFTokenizer.chatTemplateIdentity,
            imageProcessingVersion: VisionImageProcessing.version + 1))
    }
}
