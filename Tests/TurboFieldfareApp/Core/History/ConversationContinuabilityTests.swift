import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// The rule a sidebar row shows and a reopen obeys. It has to be the same rule
/// the runtime's admission guard uses, or a row offers to continue a chat whose
/// first turn is then refused.
@Suite struct ConversationContinuabilityTests {
    private let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private func meta(kvTokens: Int?,
                      identity: ConversationIdentity? = nil,
                      imageCount: Int = 0,
                      recordedContext: Int = 8_192) -> ConversationMeta {
        ConversationMeta(
            id: UUID(), title: "t", createdAt: Date(), updatedAt: Date(),
            turnCount: 2, kvTokens: kvTokens, imageCount: imageCount,
            identity: identity ?? self.identity,
            session: ConversationSessionSettings(
                contextTokens: recordedContext, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(
                temperature: 0.2, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 4_096))
    }

    @Test func aNewerFormatCannotReplayEvenWhenItsIdentityAndTokensMatch() {
        var newer = meta(kvTokens: 32)
        newer.version = ConversationMeta.currentVersion + 1
        #expect(ConversationContinuability.evaluate(
            meta: newer, currentContext: 8_192, identity: identity)
            == .cannotReplay(reason: .newerFormat))
    }

    /// Both sides of the exact boundary, because an off-by-one here is a row
    /// that promises what the next turn refuses.
    @Test func theBoundaryIsTheSameOneAdmissionUses() {
        let context = 4_096
        let largest = context - ConversationGenerationReserve.tokens
            - ConversationGenerationReserve.turnEnvelope
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: largest), currentContext: context, identity: identity)
            == .continuable)
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: largest + 1), currentContext: context,
            identity: identity)
            == .needsContext(
                required: ConversationGenerationReserve.contextRequired(
                    forLineage: largest + 1)))
        // The lineage the row admits still fits `generate`'s own check once
        // the smallest turn is on it: the rule that used to admit a row the
        // next message was then refused on.
        #expect(ConversationGenerationReserve.fits(
            tokens: largest + ConversationGenerationReserve.turnEnvelope,
            maxContext: context))
    }

    /// A boundary token the last run left outside the KV is replayed ahead of
    /// the next turn, so it is part of what has to fit.
    @Test func aBoundaryTokenAwaitingReplayCountsAgainstTheContext() {
        let context = 4_096
        let largest = context - ConversationGenerationReserve.tokens
            - ConversationGenerationReserve.turnEnvelope
        var withBoundary = meta(kvTokens: largest)
        withBoundary.boundary = ConversationBoundary(tokens: [7], needsReplay: true)
        #expect(ConversationContinuability.evaluate(
            meta: withBoundary, currentContext: context, identity: identity)
            == .needsContext(
                required: ConversationGenerationReserve.contextRequired(
                    forLineage: largest + 1)))
        // A boundary that is not replayed costs nothing.
        var committed = meta(kvTokens: largest)
        committed.boundary = ConversationBoundary(tokens: [7], needsReplay: false)
        #expect(ConversationContinuability.evaluate(
            meta: committed, currentContext: context, identity: identity)
            == .continuable)
    }

    /// The recorded context is informational: a small chat made at 8K continues
    /// at 4K, and refusing it because of the number in the file would be a
    /// refusal with no cause.
    @Test func theRecordedContextDoesNotDecide() {
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: 3_000, recordedContext: 8_192),
            currentContext: 4_096, identity: identity) == .continuable)
    }

    /// Fail closed. A conversation whose count was never reported would
    /// otherwise be restored, admitted, and only then found not to fit.
    @Test func anUnknownTokenCountIsNotContinuable() {
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: nil), currentContext: 8_192, identity: identity)
            == .cannotReplay(reason: .tokenCountUnknown))
    }

    @Test func everyIdentityFieldRefusesReplayOnItsOwn() {
        func recorded(_ change: (inout ConversationIdentity) -> Void)
            -> ConversationIdentity {
            var value = identity
            change(&value)
            return value
        }
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: 100, identity: recorded { $0 = ConversationIdentity(
                modelID: "other", sourceSnapshotHash: $0.sourceSnapshotHash,
                templateIdentity: $0.templateIdentity,
                imageProcessingVersion: $0.imageProcessingVersion) }),
            currentContext: 8_192, identity: identity)
            == .cannotReplay(reason: .differentModel))
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: 100, identity: recorded { $0 = ConversationIdentity(
                modelID: $0.modelID, sourceSnapshotHash: "deadbeef",
                templateIdentity: $0.templateIdentity,
                imageProcessingVersion: $0.imageProcessingVersion) }),
            currentContext: 8_192, identity: identity)
            == .cannotReplay(reason: .differentCheckpoint))
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: 100, identity: recorded { $0 = ConversationIdentity(
                modelID: $0.modelID, sourceSnapshotHash: $0.sourceSnapshotHash,
                templateIdentity: "gemma4-it-text-no-tools-v2",
                imageProcessingVersion: $0.imageProcessingVersion) }),
            currentContext: 8_192, identity: identity)
            == .cannotReplay(reason: .differentTemplate))
    }

    /// A preprocessing bump only matters to a conversation that has images.
    /// Refusing a text chat for it would be a refusal the user cannot act on.
    @Test func anImageProcessingBumpOnlyRefusesConversationsWithImages() {
        let bumped = ConversationIdentity(
            modelID: identity.modelID,
            sourceSnapshotHash: identity.sourceSnapshotHash,
            templateIdentity: identity.templateIdentity,
            imageProcessingVersion: identity.imageProcessingVersion + 1)
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: 100, identity: bumped, imageCount: 0),
            currentContext: 8_192, identity: identity) == .continuable)
        #expect(ConversationContinuability.evaluate(
            meta: meta(kvTokens: 100, identity: bumped, imageCount: 2),
            currentContext: 8_192, identity: identity)
            == .cannotReplay(reason: .differentImageProcessing))
    }
}

@Suite struct ConversationTitleTests {
    @Test func aShortMessageIsItsOwnTitle() {
        #expect(ConversationTitle.fromFirstMessage("Explain RoPE")
            == "Explain RoPE")
    }

    @Test func whitespaceIsCollapsedSoARowIsOneLine() {
        #expect(ConversationTitle.fromFirstMessage("  Explain\n\n  RoPE  ")
            == "Explain RoPE")
    }

    @Test func anEmptyMessageFallsBackRatherThanShowingABlankRow() {
        #expect(ConversationTitle.fromFirstMessage("   \n ")
            == ConversationTitle.untitled)
    }

    /// Cut on a word boundary: clipping mid-word reads as damage rather than as
    /// a title.
    @Test func aLongMessageIsCutOnAWordBoundary() {
        let text = String(repeating: "alpha beta ", count: 20)
        let title = ConversationTitle.fromFirstMessage(text)
        #expect(title.count <= ConversationTitle.maximumCharacters + 1)
        #expect(title.hasSuffix("\u{2026}"))
        #expect(!title.dropLast().hasSuffix(" "))
        #expect(title.dropLast().split(separator: " ").allSatisfy {
            $0 == "alpha" || $0 == "beta"
        })
    }

    /// One very long word has no boundary to cut at, and must still produce a
    /// title rather than an ellipsis alone.
    @Test func aSingleLongWordIsStillATitle() {
        let title = ConversationTitle.fromFirstMessage(
            String(repeating: "x", count: 200))
        #expect(title.count == ConversationTitle.maximumCharacters + 1)
        #expect(title.hasPrefix("xxxx"))
    }
}
