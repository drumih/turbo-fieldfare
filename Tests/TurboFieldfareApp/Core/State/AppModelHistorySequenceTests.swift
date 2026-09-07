import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareValidationSupport
@testable import TurboFieldfareAppCore

/// Sequences, not steps.
///
/// Every defect this feature shipped needed three or four operations in a row —
/// delete the chat the KV is holding and send again, read one chat then another
/// then the first, send two pictures and reopen — and every test written
/// afterwards is one of those sequences, written once the defect was known.
/// This suite writes them down as fixed cases and then generates the ones
/// nobody has thought of yet, checking the same seven invariants after every
/// operation.
///
/// Serialized: each case drives an `AppModel` on the main actor and waits for
/// its tasks to finish, and cases competing for that actor turn a settle
/// deadline into a flake rather than a finding.
@Suite(.serialized) struct AppModelHistorySequenceTests {
    // MARK: - The defects that reached the user

    /// C23. Deleting the chat the KV is holding used to keep the conversation
    /// and let the next turn open a fresh file, whose meta then described
    /// everything behind it: a row reading 1,065 tokens over 22 tokens of
    /// record.
    @MainActor
    @Test func deletingTheHeldChatDoesNotCarryItsCountsIntoTheNextOne() async throws {
        let harness = try await HistoryHarness(label: "c23")
        defer { harness.tearDown() }

        try await harness.apply(.send(label: 1))
        try await harness.apply(.delete(row: 0))
        try await harness.apply(.send(label: 2))
    }

    /// The duplicated transcript. Opening a chat left part of the previous one
    /// on screen, so two conversations were drawn at once — the stored pairs
    /// were the one list not cleared, and the branch for a chat that could not
    /// be continued cleared nothing at all.
    @MainActor
    @Test func openingAChatDrawsThatChatAndNothingElse() async throws {
        let harness = try await HistoryHarness(label: "duplicate")
        defer { harness.tearDown() }

        // One long exchange, so this chat stops fitting when the context drops.
        try await harness.apply(.sendLong(label: 1))
        try await harness.apply(.newChat)
        try await harness.apply(.send(label: 2))
        try await harness.apply(.newChat)
        try await harness.apply(.send(label: 3))
        try await harness.apply(.setContext(tokens: AppContextLengthOption.fourK.tokens))
        // The long one, now refused for context: it draws through the read-only
        // path, which fills a different list from the one a continuable row
        // fills.
        try await harness.apply(.open(row: try harness.row(ofCreated: 0)))
        // And then one that is neither held nor refused, which has to replace
        // what the refused one left on screen rather than draw under it.
        try await harness.apply(.open(row: try harness.row(ofCreated: 1)))
    }

    /// A conversation with a picture in each of two turns could never be
    /// continued again: the second image-bearing turn looked for its
    /// placeholders at a position that only existed if every earlier image had
    /// been in the same turn.
    @MainActor
    @Test func aChatWithPicturesInTwoTurnsReplaysAndContinues() async throws {
        guard HistoryImageFixture.isAvailable else { return }
        let harness = try await HistoryHarness(label: "two-images")
        defer { harness.tearDown() }

        try await harness.apply(.sendWithImage(label: 1))
        try await harness.apply(.sendWithImage(label: 2))
        try await harness.apply(.newChat)
        try await harness.apply(.open(row: 0))
        try await harness.apply(.send(label: 3))
    }

    /// The release paths deleted a stored conversation's pictures off disk.
    /// Reopening one puts its images into the live conversation, and New Chat
    /// releases every image the live conversation holds.
    @MainActor
    @Test func reopeningAChatAndLeavingItKeepsItsPictures() async throws {
        guard HistoryImageFixture.isAvailable else { return }
        let harness = try await HistoryHarness(label: "image-lifetime")
        defer { harness.tearDown() }

        try await harness.apply(.sendWithImage(label: 1))
        try await harness.apply(.newChat)
        try await harness.apply(.open(row: 0))
        // The replay is what puts the stored copies into the live conversation.
        try await harness.apply(.send(label: 2))
        try await harness.apply(.newChat)
    }

    // MARK: - The defects the walk found

    /// Raising the context back to where the chat fits, and coming back to the
    /// chat the KV was holding, draws it once.
    ///
    /// Without the fix: the branch that returns to the held conversation
    /// cleared the stored pairs but not the archived ones, and a row refused
    /// for context draws through the archived list — so the transcript showed
    /// the chat from disk above the same chat live. Found by the seeded walk at
    /// seed 6, step 28.
    @MainActor
    @Test func comingBackToTheHeldChatAfterRaisingTheContextDrawsItOnce() async throws {
        let harness = try await HistoryHarness(label: "held-redraw")
        defer { harness.tearDown() }

        try await harness.apply(.sendLong(label: 1))
        // Too big for 4K, so opening it draws the read-only copy.
        try await harness.apply(.setContext(tokens: AppContextLengthOption.fourK.tokens))
        try await harness.apply(.open(row: 0))
        // And now it fits again, so the same click goes back to the live one.
        try await harness.apply(.setContext(tokens: AppContextLengthOption.eightK.tokens))
    }

    /// A replay that fails leaves the chat the KV is still holding with its
    /// pictures.
    ///
    /// Without the fix: the staged copies of the conversation being left were
    /// released at the top of the replay, before anything had been given up. A
    /// restore that then failed left that conversation held and on screen with
    /// every image in it deleted off disk. Found by the seeded walk at seed 2,
    /// step 17.
    @MainActor
    @Test func aFailedReplayLeavesTheHeldChatsPicturesAlone() async throws {
        guard HistoryImageFixture.isAvailable else { return }
        let harness = try await HistoryHarness(label: "failed-replay-images")
        defer { harness.tearDown() }

        try await harness.apply(.send(label: 1))
        try await harness.apply(.newChat)
        try await harness.apply(.sendWithImage(label: 2))
        // The first chat, which is not the one being held.
        try await harness.apply(.open(row: try harness.row(ofCreated: 0)))
        try await harness.apply(.failNextRestore)
        try await harness.apply(.send(label: 3))
    }

    /// Deleting the chat being read selects the chat that comes back on screen.
    ///
    /// Without the fix: the selection was cleared outright, so the window
    /// showed the conversation the KV was holding with no row highlighted for
    /// it — the one state every other path in this file is careful not to
    /// produce. Found by the seeded walk at seed 3, step 29.
    @MainActor
    @Test func deletingTheChatBeingReadSelectsTheOneThatComesBack() async throws {
        let harness = try await HistoryHarness(label: "delete-read")
        defer { harness.tearDown() }

        try await harness.apply(.send(label: 1))
        try await harness.apply(.newChat)
        try await harness.apply(.send(label: 2))
        try await harness.apply(.open(row: try harness.row(ofCreated: 0)))
        try await harness.apply(.delete(row: try harness.row(ofCreated: 0)))
    }

    /// Reloading while reading a chat that cannot be continued leaves that
    /// chat's transcript behind.
    ///
    /// Without the fix: a load rebuilds the KV, so the conversation it was
    /// holding moved out of context into the one array that also held the
    /// read-only copy of a chat refused for context. The copy being read was
    /// still in it, so the window drew a conversation the user was only
    /// looking at as the live one's own out-of-context turns, under a context
    /// break, and Re-read into a new chat would have built its prompt from it.
    /// Found by the seeded walk at seed 17, step 23, and again at seed 22 with
    /// an unload in place of the reload. The two are separate values now — the
    /// screen and `AppConversation.outOfContextPairs` — so this case is what
    /// keeps them separate.
    @MainActor
    @Test func reloadingWhileReadingAChatPreservesTheViewedOneOnScreen() async throws {
        let harness = try await HistoryHarness(label: "reload-reading")
        defer { harness.tearDown() }

        // One long chat, so it stops fitting when the context drops, and one
        // short chat for the KV to be holding when the reload lands.
        try await harness.apply(.sendLong(label: 1))
        try await harness.apply(.newChat)
        try await harness.apply(.send(label: 2))
        try await harness.apply(.setContext(tokens: AppContextLengthOption.fourK.tokens))
        // The refused row remains selected; it must not become the released
        // live conversation's out-of-context turns or be replaced by that row.
        try await harness.apply(.open(row: try harness.row(ofCreated: 0)))
        try await harness.apply(.reload)
    }

    /// An unload takes the same path.
    @MainActor
    @Test func unloadingWhileReadingAChatPreservesTheViewedOneOnScreen() async throws {
        let harness = try await HistoryHarness(label: "unload-reading")
        defer { harness.tearDown() }

        try await harness.apply(.sendLong(label: 1))
        try await harness.apply(.newChat)
        try await harness.apply(.send(label: 2))
        try await harness.apply(.setContext(tokens: AppContextLengthOption.fourK.tokens))
        try await harness.apply(.open(row: try harness.row(ofCreated: 0)))
        try await harness.apply(.unload)
    }

    /// A reload ends the in-memory lineage on both sides, but the durable row
    /// remains selected. The next send replays it and appends to that same row.
    @MainActor
    @Test func sendingAfterAReloadReplaysAndContinuesTheStoredConversation() async throws {
        let harness = try await HistoryHarness(label: "reload-send")
        defer { harness.tearDown() }

        try await harness.apply(.send(label: 1))
        try await harness.apply(.reload)
        try await harness.apply(.send(label: 2))
    }

    // MARK: - The ones nobody has thought of yet

    /// The reference model archived a durable row after reload as if its KV
    /// still existed. Deleting the row exposed those invented transcript turns.
    @MainActor
    @Test func failedReplayAfterReloadDoesNotInventOutOfContextTurns() async throws {
        let harness = try await HistoryHarness(label: "reload-failed-replay-delete")
        defer { harness.tearDown() }
        try await harness.apply(.send(label: 1))
        try await harness.apply(.failNextRestore)
        try await harness.apply(.reload)
        try await harness.apply(.send(label: 4))
        try await harness.apply(.delete(row: 0))
    }

    @MainActor
    @Test(arguments: 0..<64)
    func aSeededWalkHoldsEveryInvariant(seed: Int) async throws {
        var generator = SplitMix64(seed: UInt64(seed))
        let harness = try await HistoryHarness(label: "seed-\(seed)")
        defer { harness.tearDown() }

        for step in 1...40 {
            try await harness.apply(
                HistoryOperation.next(step: step, using: &generator))
        }
    }

    /// Replays one seed on its own and prints the trace.
    ///
    /// `TURBO_FIELDFARE_HISTORY_SEED=17 TURBO_FIELDFARE_HISTORY_STEPS=12
    /// swift test --filter aNamedSeedReplaysWithItsTrace`. The seed and the step
    /// index are the whole reproduction: there is no shrinking, so a walk that
    /// finds something is narrowed by re-running it shorter and then written
    /// down by hand as one of the cases above.
    @MainActor
    @Test func aNamedSeedReplaysWithItsTrace() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let raw = environment["TURBO_FIELDFARE_HISTORY_SEED"],
              let seed = UInt64(raw) else { return }
        let steps = environment["TURBO_FIELDFARE_HISTORY_STEPS"].flatMap(Int.init) ?? 40
        var generator = SplitMix64(seed: seed)
        let harness = try await HistoryHarness(label: "replay-\(seed)")
        defer {
            print("seed \(seed), \(harness.trace.count) step(s):")
            for line in harness.trace { print("  \(line)") }
            harness.tearDown()
        }

        for step in 1...steps {
            try await harness.apply(
                HistoryOperation.next(step: step, using: &generator))
        }
    }
}
