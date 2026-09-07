import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Every cell of `(screen, event)`, one row each.
///
/// The machine exists because the six fields it replaces were never enumerated:
/// each defect was one more combination nobody had thought about, and the fix
/// for each was one more line in one more branch. A table cannot be fixed that
/// way — a cell either has a row or the completeness test at the bottom fails —
/// and a new screen case or a new event fails to compile until every cell it
/// creates is written down.
@Suite struct ConversationScreenMachineTests {
    @Test(arguments: [false, true])
    func adversarialOldReadCannotReplaceReopenedSameChat(failure: Bool) {
        var machine = ConversationScreenMachine()
        let firstRead = UUID()
        let latestRead = Self.renderA
        for (id, readID) in [(Self.row, firstRead), (Self.other, UUID()), (Self.row, latestRead)] {
            _ = machine.apply(.rowClicked(
                id: id, heldID: Self.held, kvMatchesHeld: true,
                state: .continuable, renderID: readID))
        }
        _ = machine.apply(.documentLoaded(
            id: Self.row, renderID: Self.renderA, document: Self.reloaded, state: .continuable))
        let latest = machine.screen
        let effects = failure
            ? machine.apply(.documentFailed(id: Self.row, renderID: firstRead, Self.unreadableError))
            : machine.apply(.documentLoaded(
                id: Self.row, renderID: firstRead, document: Self.stored, state: .continuable))
        #expect(machine.screen == latest)
        #expect(effects.isEmpty)
    }

    @Test func adversarialLateDocumentDuringReplayCannotResetSampling() {
        var machine = ConversationScreenMachine()
        _ = machine.apply(Self.clicked(Self.row))
        _ = machine.apply(.sendRequested(Self.pending))
        let effects = machine.apply(.documentLoaded(
            id: Self.row, renderID: Self.renderB, document: Self.stored, state: .continuable))
        #expect(machine.screen.isReplaying)
        #expect(machine.screen.document == Self.stored)
        #expect(!effects.contains(.applySampling(Self.sampling)))
        if case .replaying(_, _, let pending, _) = machine.screen {
            #expect(pending == Self.pending)
        }
    }

    // MARK: - Fixtures

    private static let held = UUID()
    private static let row = UUID()
    private static let other = UUID()
    private static let renderA = UUID()
    private static let renderB = UUID()
    private static let epoch = UUID()

    private static let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private static let sampling = ConversationSampling(
        temperature: 0.2, topKEnabled: true, topK: 64,
        topPEnabled: true, topP: 0.95, maxNewTokens: 8_192)

    private static func document(_ id: UUID, text: String) -> ConversationDocument {
        let meta = ConversationMeta(
            id: id, title: "stored", createdAt: Date(), updatedAt: Date(),
            identity: identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "on-demand"),
            sampling: sampling)
        let records: [TranscriptRecord] = [
            .header(ConversationHeaderRecord(id: id, createdAt: Date())),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: text, tokens: [1, 2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer", tokens: [3])),
        ]
        return ConversationDocument.load(
            ConversationOpenResult(meta: meta, records: records,
                                   isReadOnly: false, droppedTornFinalLine: false),
            directory: URL(fileURLWithPath: "/tmp/conversations/\(id.uuidString)",
                           isDirectory: true))
    }

    private static let stored = document(row, text: "as it was clicked")
    private static let reloaded = document(row, text: "as the load found it")
    private static let elsewhere = document(other, text: "another chat")

    private static let pending = PreparedTurn(
        prompt: "the message that started the replay", images: [])
    private static let unreadableError =
        AppInferenceError.conversationUnreadable("its final record will not read")
    private static let restoreError =
        AppInferenceError.conversationRestoreFailed("image support is unavailable")
    private static let recordError = AppInferenceError.conversationRestoreFailed(
        ConversationRecordError.turnWithoutTokens.description)

    private static let live = ConversationScreen.live
    private static let reading = ConversationScreen.reading(
        id: row, document: stored, state: .continuable, renderID: renderA)
    private static let readingHeld = ConversationScreen.reading(
        id: held, document: stored, state: .continuable, renderID: renderA)
    private static let readingRefused = ConversationScreen.reading(
        id: row, document: stored, state: .needsContext(required: 9_000),
        renderID: renderA)
    private static let replaying = ConversationScreen.replaying(
        id: row, document: stored, pending: pending, renderID: renderA)
    private static let unreadable = ConversationScreen.unreadable(
        id: row, error: unreadableError)

    /// A click on `id`, with the KV holding `held` and matching it.
    private static func clicked(
        _ id: UUID, state: ConversationContinuability = .continuable,
        heldID: UUID? = held, kvMatchesHeld: Bool = true
    ) -> ConversationScreenMachine.Event {
        .rowClicked(id: id, heldID: heldID, kvMatchesHeld: kvMatchesHeld,
                    state: state, renderID: renderB)
    }

    struct Row: Sendable, CustomTestStringConvertible {
        let name: String
        let screen: ConversationScreen
        let event: ConversationScreenMachine.Event
        let expected: ConversationScreen
        let effects: [ConversationScreenMachine.Effect]

        var testDescription: String { name }
    }

    // MARK: - The matrix

    static let rows: [Row] = [
        // MARK: live
        Row(name: "live + a click on another row reads it",
            screen: live, event: clicked(other),
            expected: .reading(id: other, document: nil, state: .continuable,
                               renderID: renderB),
            effects: [.select(other), .dropOutOfContextTurns,
                      .loadDocument(other, renderID: renderB)]),
        Row(name: "live + a click on the row the KV holds rebuilds nothing",
            screen: live, event: clicked(held),
            expected: .live,
            effects: [.select(held), .dropOutOfContextTurns]),
        // The record's replayability is beside the point: the chat is in the
        // KV. Reading it as a copy instead locked the user out of the
        // conversation the model was holding.
        Row(name: "live + a click on the held row whose record cannot be replayed still rebuilds nothing",
            screen: live,
            event: clicked(held, state: .cannotReplay(reason: .imageRecordMissing)),
            expected: .live,
            effects: [.select(held), .dropOutOfContextTurns]),
        Row(name: "live + a click on the held row whose KV was rebuilt reads it",
            screen: live, event: clicked(held, kvMatchesHeld: false),
            expected: .reading(id: held, document: nil, state: .continuable,
                               renderID: renderB),
            effects: [.select(held), .dropOutOfContextTurns,
                      .loadDocument(held, renderID: renderB)]),
        Row(name: "live + a click on the held row that no longer fits reads it",
            screen: live,
            event: clicked(held, state: .needsContext(required: 9_000)),
            expected: .reading(id: held, document: nil,
                               state: .needsContext(required: 9_000),
                               renderID: renderB),
            effects: [.select(held), .dropOutOfContextTurns,
                      .loadDocument(held, renderID: renderB)]),
        Row(name: "live + a context change redraws nothing",
            screen: live,
            event: .contextChanged(state: .continuable, heldID: held,
                                   kvMatchesHeld: true, renderID: renderB),
            expected: .live, effects: []),
        Row(name: "live + a document that landed after the screen went live",
            screen: live, event: .documentLoaded(id: row, renderID: Self.renderA, document: stored,
                                                 state: .continuable),
            expected: .live, effects: []),
        Row(name: "live + a read that failed after the screen went live",
            screen: live,
            event: .documentFailed(id: row, renderID: Self.renderA, unreadableError),
            expected: .live, effects: []),
        Row(name: "live + a send is an ordinary turn",
            screen: live, event: .sendRequested(pending),
            expected: .live, effects: []),
        Row(name: "live + a replay that landed after the screen went live",
            screen: live,
            event: .replaySucceeded(id: row, document: stored, epoch: epoch,
                                    kvTokens: 12),
            expected: .live, effects: []),
        Row(name: "live + a replay failure that landed after the screen went live",
            screen: live, event: .replayFailed(id: row, error: restoreError),
            expected: .live, effects: []),
        Row(name: "live + a replay that never started, after the screen went live",
            screen: live, event: .replayNotStarted(id: row, error: restoreError),
            expected: .live, effects: []),
        Row(name: "live + a refused record that landed after the screen went live",
            screen: live, event: .replayRefused(id: row, error: recordError),
            expected: .live, effects: []),
        Row(name: "live + New Chat selects no row",
            screen: live, event: .newChat,
            expected: .live,
            effects: [.releaseImagesOfHeldConversation(includingLiveTurn: true),
                      .select(nil)]),
        Row(name: "live + deleting a chat that is not on screen",
            screen: live, event: .deleted(id: other, heldID: held),
            expected: .live, effects: []),
        Row(name: "live + the KV going ends the chat and its selection",
            screen: live, event: .lineageEnded,
            expected: .live, effects: [.select(nil)]),

        // MARK: reading
        Row(name: "reading + a click on a third row replaces what is drawn",
            screen: reading, event: clicked(other),
            expected: .reading(id: other, document: nil, state: .continuable,
                               renderID: renderB),
            effects: [.select(other), .dropOutOfContextTurns,
                      .loadDocument(other, renderID: renderB)]),
        Row(name: "reading + a click back on the held row goes live",
            screen: reading, event: clicked(held),
            expected: .live,
            effects: [.select(held), .dropOutOfContextTurns]),
        Row(name: "reading + a context change redraws the row against the new answer",
            screen: reading,
            event: .contextChanged(state: .needsContext(required: 9_000),
                                   heldID: held, kvMatchesHeld: true,
                                   renderID: renderB),
            expected: .reading(id: row, document: nil,
                               state: .needsContext(required: 9_000),
                               renderID: renderB),
            effects: [.select(row), .dropOutOfContextTurns,
                      .loadDocument(row, renderID: renderB)]),
        Row(name: "reading the held row + a context change that lets it fit again",
            screen: readingHeld,
            event: .contextChanged(state: .continuable, heldID: held,
                                   kvMatchesHeld: true, renderID: renderB),
            expected: .live,
            effects: [.select(held), .dropOutOfContextTurns]),
        Row(name: "reading + its document lands and moves the sliders with it",
            screen: reading, event: .documentLoaded(id: row, renderID: Self.renderA, document: reloaded,
                                                    state: .continuable),
            expected: .reading(id: row, document: reloaded, state: .continuable,
                               renderID: renderA),
            effects: [.applySampling(sampling)]),
        Row(name: "reading + a document for a row that is no longer on screen",
            screen: reading, event: .documentLoaded(id: other, renderID: Self.renderA, document: elsewhere,
                                                    state: .continuable),
            expected: reading, effects: []),
        Row(name: "reading a row that cannot be continued + its document lands",
            screen: readingRefused, event: .documentLoaded(
                id: row, renderID: Self.renderA, document: reloaded, state: .continuable),
            expected: .reading(id: row, document: reloaded,
                               state: .continuable,
                               renderID: renderA),
            effects: [.applySampling(sampling)]),
        Row(name: "reading + the read failed, so the row says why",
            screen: reading,
            event: .documentFailed(id: row, renderID: Self.renderA, unreadableError),
            expected: .unreadable(id: row, error: unreadableError),
            effects: [.reportError(unreadableError)]),
        Row(name: "reading + a failure for a row that is no longer on screen",
            screen: reading,
            event: .documentFailed(id: other, renderID: Self.renderA, unreadableError),
            expected: reading, effects: []),
        Row(name: "reading + a send replays the row first",
            screen: reading, event: .sendRequested(pending),
            expected: .replaying(id: row, document: stored, pending: pending,
                                 renderID: renderA),
            effects: [.replay(id: row, pending: pending)]),
        // Refused at the machine as well as at the composer. Replaying a row
        // the notice has refused put tokens recorded under another checkpoint
        // into the KV, or dropped the held chat for a restore the context
        // could not hold, and a failure then relabelled the row continuable.
        Row(name: "reading a row that cannot be continued + a send is refused",
            screen: readingRefused, event: .sendRequested(pending),
            expected: readingRefused, effects: []),
        Row(name: "reading + a replay that belongs to an earlier send",
            screen: reading,
            event: .replaySucceeded(id: row, document: stored, epoch: epoch,
                                    kvTokens: 12),
            expected: reading, effects: []),
        Row(name: "reading + a replay failure that belongs to an earlier send",
            screen: reading, event: .replayFailed(id: row, error: restoreError),
            expected: reading, effects: []),
        Row(name: "reading + a replay that never started, from an earlier send",
            screen: reading, event: .replayNotStarted(id: row, error: restoreError),
            expected: reading, effects: []),
        Row(name: "reading + a refused record that belongs to an earlier send",
            screen: reading, event: .replayRefused(id: row, error: recordError),
            expected: reading, effects: []),
        Row(name: "reading + New Chat leaves the row",
            screen: reading, event: .newChat,
            expected: .live,
            effects: [.releaseImagesOfHeldConversation(includingLiveTurn: true),
                      .select(nil)]),
        Row(name: "reading + deleting the row selects the chat that comes back",
            screen: reading, event: .deleted(id: row, heldID: held),
            expected: .live, effects: [.select(held)]),
        Row(name: "reading + deleting another chat leaves the screen alone",
            screen: reading, event: .deleted(id: other, heldID: held),
            expected: reading, effects: []),
        Row(name: "reading + the KV going takes the read copy with it",
            screen: reading, event: .lineageEnded,
            expected: .live, effects: [.select(nil)]),

        // MARK: replaying
        Row(name: "replaying + a click is refused",
            screen: replaying, event: clicked(other),
            expected: replaying, effects: []),
        Row(name: "replaying + a context change is refused",
            screen: replaying,
            event: .contextChanged(state: .continuable, heldID: held,
                                   kvMatchesHeld: true, renderID: renderB),
            expected: replaying, effects: []),
        Row(name: "replaying + the click's read lands under it",
            screen: replaying, event: .documentLoaded(id: row, renderID: Self.renderA, document: reloaded,
                                                      state: .continuable),
            expected: .replaying(id: row, document: reloaded, pending: pending,
                                 renderID: renderA),
            effects: []),
        Row(name: "replaying + a document for another row",
            screen: replaying, event: .documentLoaded(id: other, renderID: Self.renderA, document: elsewhere,
                                                      state: .continuable),
            expected: replaying, effects: []),
        Row(name: "replaying + the click's read failed, which the replay reports itself",
            screen: replaying,
            event: .documentFailed(id: row, renderID: Self.renderA, unreadableError),
            expected: replaying, effects: []),
        Row(name: "replaying + a second send is refused",
            screen: replaying,
            event: .sendRequested(PreparedTurn(
                prompt: "again", images: [])),
            expected: replaying, effects: []),
        Row(name: "replaying + it landed, so the held turn starts",
            screen: replaying,
            event: .replaySucceeded(id: row, document: reloaded, epoch: epoch,
                                    kvTokens: 12),
            expected: .live,
            effects: [.releaseImagesOfHeldConversation(includingLiveTurn: false),
                      .adoptRestored(reloaded, epoch: epoch, kvTokens: 12),
                      .select(row),
                      .startTurn(pending)]),
        Row(name: "replaying + a replay of a chat that is no longer on screen",
            screen: replaying,
            event: .replaySucceeded(id: other, document: elsewhere, epoch: epoch,
                                    kvTokens: 12),
            expected: replaying, effects: []),
        Row(name: "replaying + it failed on the runtime, so the row stays continuable",
            screen: replaying, event: .replayFailed(id: row, error: restoreError),
            expected: .reading(id: row, document: stored, state: .continuable,
                               renderID: renderA),
            effects: [.endFailedReplayLineage,
                      .restoreComposer(pending, restoreError)]),
        // No lineage end: the service was never asked, so the held chat and
        // its pictures are exactly where they were. Ending it here deleted
        // the held chat's staged images for a restore that had not happened.
        Row(name: "replaying + it was refused before the service, so the held chat stays",
            screen: replaying,
            event: .replayNotStarted(id: row, error: restoreError),
            expected: .reading(id: row, document: stored, state: .continuable,
                               renderID: renderA),
            effects: [.restoreComposer(pending, restoreError)]),
        Row(name: "replaying + a refusal for a chat that is no longer on screen",
            screen: replaying,
            event: .replayNotStarted(id: other, error: restoreError),
            expected: replaying, effects: []),
        Row(name: "replaying + the record cannot be replayed, so the row says so",
            screen: replaying, event: .replayRefused(id: row, error: recordError),
            expected: .reading(id: row, document: stored,
                               state: .cannotReplay(reason: .tokenCountUnknown),
                               renderID: renderA),
            effects: [.restoreComposer(pending, recordError)]),
        Row(name: "replaying + New Chat, whose replay can no longer land",
            screen: replaying, event: .newChat,
            expected: .live,
            effects: [.releaseImagesOfHeldConversation(includingLiveTurn: true),
                      .select(nil)]),
        Row(name: "replaying + deleting the chat being replayed hands the message back",
            screen: replaying, event: .deleted(id: row, heldID: held),
            expected: .live,
            effects: [.restoreComposer(pending, nil), .select(held)]),
        Row(name: "replaying + a refused record whose fresh read said why",
            screen: replaying,
            event: .replayRefused(id: row, error: recordError,
                                  state: .needsContext(required: 9_000)),
            expected: .reading(id: row, document: stored,
                               state: .needsContext(required: 9_000),
                               renderID: renderA),
            effects: [.restoreComposer(pending, recordError)]),
        // The replay can no longer land, and the send waiting on it stops
        // when it finds the screen gone. The message goes back here or it
        // goes nowhere: dropped with the screen, the prompt and its staged
        // pictures were never handed back to anything.
        Row(name: "replaying + the KV going under it hands the message back",
            screen: replaying, event: .lineageEnded,
            expected: .live,
            effects: [.restoreComposer(pending, nil), .select(nil)]),

        // MARK: unreadable
        Row(name: "unreadable + a click on another row reads that one",
            screen: unreadable, event: clicked(other),
            expected: .reading(id: other, document: nil, state: .continuable,
                               renderID: renderB),
            effects: [.select(other), .dropOutOfContextTurns,
                      .loadDocument(other, renderID: renderB)]),
        Row(name: "unreadable + a context change cannot make it readable",
            screen: unreadable,
            event: .contextChanged(state: .continuable, heldID: held,
                                   kvMatchesHeld: true, renderID: renderB),
            expected: unreadable, effects: []),
        Row(name: "unreadable + a document that landed after the failure",
            screen: unreadable, event: .documentLoaded(id: row, renderID: Self.renderA, document: stored,
                                                       state: .continuable),
            expected: unreadable, effects: []),
        Row(name: "unreadable + a second failure for the same row",
            screen: unreadable,
            event: .documentFailed(id: row, renderID: Self.renderA, unreadableError),
            expected: unreadable, effects: []),
        // Nothing to replay and nothing a message could go to: the composer
        // is closed for this screen, and a send that reached the pipeline
        // anyway is handed back rather than run on the held chat under the
        // row that could not be read.
        Row(name: "unreadable + a send is refused",
            screen: unreadable, event: .sendRequested(pending),
            expected: unreadable, effects: []),
        Row(name: "unreadable + a replay that landed after the failure",
            screen: unreadable,
            event: .replaySucceeded(id: row, document: stored, epoch: epoch,
                                    kvTokens: 12),
            expected: unreadable, effects: []),
        Row(name: "unreadable + a replay failure that landed after it",
            screen: unreadable, event: .replayFailed(id: row, error: restoreError),
            expected: unreadable, effects: []),
        Row(name: "unreadable + a replay that never started, landing after it",
            screen: unreadable,
            event: .replayNotStarted(id: row, error: restoreError),
            expected: unreadable, effects: []),
        Row(name: "unreadable + a refused record that landed after it",
            screen: unreadable, event: .replayRefused(id: row, error: recordError),
            expected: unreadable, effects: []),
        Row(name: "unreadable + New Chat clears the notice",
            screen: unreadable, event: .newChat,
            expected: .live,
            effects: [.releaseImagesOfHeldConversation(includingLiveTurn: true),
                      .select(nil)]),
        Row(name: "unreadable + deleting the chat that could not be read",
            screen: unreadable, event: .deleted(id: row, heldID: held),
            expected: .live, effects: [.select(held)]),
        Row(name: "unreadable + the KV going",
            screen: unreadable, event: .lineageEnded,
            expected: .live, effects: [.select(nil)]),
    ]

    // MARK: - The table is the test

    @Test(arguments: rows)
    func themachineMakesExactlyTheTransitionTheTableStates(_ row: Row) {
        var machine = ConversationScreenMachine(screen: row.screen)

        let effects = machine.apply(row.event)

        #expect(machine.screen == row.expected)
        #expect(effects == row.effects)
    }

    /// A cell with no row is a transition nobody decided, which is what the six
    /// fields were made of. Adding a screen case or an event fails here until
    /// its cells are written down.
    @Test func everyCellOfTheMatrixHasARow() {
        var covered: Set<String> = []
        for row in Self.rows {
            covered.insert("\(row.screen.shape.rawValue) + \(row.event.shape.rawValue)")
        }
        var missing: [String] = []
        for screen in ConversationScreen.allShapes {
            for event in ConversationScreenMachine.Event.allShapes
            where !covered.contains("\(screen.rawValue) + \(event.rawValue)") {
                missing.append("\(screen.rawValue) + \(event.rawValue)")
            }
        }
        #expect(missing.isEmpty, "no row for \(missing.joined(separator: ", "))")
        #expect(covered.count
            == ConversationScreen.allShapes.count
                * ConversationScreenMachine.Event.allShapes.count)
    }

    /// A fresh machine shows the live conversation, which is what a window with
    /// no history at all shows.
    @Test func amachineStartsOnTheLiveConversation() {
        #expect(ConversationScreenMachine().screen == .live)
    }

    /// Two clicks on the same row are two different things to draw.
    ///
    /// Keyed on the conversation's own id instead, re-opening it after the
    /// context changed was not a change the renderer could see, so a chat that
    /// had just become continuable kept the boundary rule saying its earlier
    /// turns were out of context.
    @Test func reopeningTheSameRowIsANewThingToDraw() {
        var machine = ConversationScreenMachine()
        let first = UUID()
        let second = UUID()
        _ = machine.apply(.rowClicked(id: Self.row, heldID: Self.held,
                                      kvMatchesHeld: true, state: .continuable,
                                      renderID: first))
        guard case .reading(_, _, _, let one) = machine.screen else {
            Issue.record("the click did not read the row")
            return
        }
        _ = machine.apply(.rowClicked(id: Self.row, heldID: Self.held,
                                      kvMatchesHeld: true, state: .continuable,
                                      renderID: second))
        guard case .reading(_, _, _, let two) = machine.screen else {
            Issue.record("the second click did not read the row")
            return
        }
        #expect(one == first)
        #expect(two == second)
        #expect(one != two)
    }


    /// The composer's answer is the notice's answer. `conversation.canSend`
    /// alone let a row the notice had refused be sent anyway.
    @Test func onlyTheLiveChatAndAContinuableRowAllowASend() {
        #expect(Self.live.allowsSend)
        #expect(Self.reading.allowsSend)
        #expect(!Self.readingRefused.allowsSend)
        #expect(!ConversationScreen.reading(
            id: Self.row, document: Self.stored,
            state: .cannotReplay(reason: .differentCheckpoint),
            renderID: Self.renderA).allowsSend)
        #expect(!Self.replaying.allowsSend)
        #expect(!Self.unreadable.allowsSend)
    }

    /// A send holds its render identity, so the transcript is not rebuilt at the
    /// moment the replay starts.
    @Test func asendKeepsTheTranscriptThatIsAlreadyOnScreen() {
        var machine = ConversationScreenMachine(screen: Self.reading)
        _ = machine.apply(.sendRequested(Self.pending))
        guard case .replaying(_, _, _, let renderID) = machine.screen else {
            Issue.record("the send did not start a replay")
            return
        }
        #expect(renderID == Self.renderA)
    }
}
