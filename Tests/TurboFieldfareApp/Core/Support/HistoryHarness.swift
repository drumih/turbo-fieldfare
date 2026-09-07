import Foundation
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// The one-pixel PNG, preprocessed once, so the harness knows what a stored
/// image record will claim before it writes one.
///
/// The soft-token count matters twice: the record stores it, and the double has
/// to emit that many placeholder IDs in the turn that carries the image or the
/// conversation can never be replayed — replay locates an image by finding its
/// placeholder run inside the turn that cites it. Measuring it here rather than
/// hard-coding it keeps the two ends of that from drifting apart.
enum HistoryImageFixture {
    static let softTokens: Int? = {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-image-probe-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            let store = AppImageAttachmentStore(
                directoryURL: directory.appendingPathComponent("staged", isDirectory: true))
            let staged = try store.stage(AppModelHistoryTests.pngFixture())
            let record = try ConversationImageWriter().write(
                attachment: staged,
                into: directory.appendingPathComponent("images", isDirectory: true))
            return record.softTokens
        } catch {
            // No Metal device, or a preprocessor that will not take the
            // fixture. The image operations stand down rather than the walk
            // failing for a reason that is not about history.
            return nil
        }
    }()

    static var isAvailable: Bool { softTokens != nil }
}

/// Drives `AppModel` and `ConversationStore` through a sequence of operations
/// and checks every invariant after each one.
///
/// It owns the app, the double, the store root and the reference model, and it
/// is the only thing in the app tests that knows how to wait for this feature
/// to be at rest: every write here finishes in a task the caller did not start,
/// so "the operation returned" and "the operation happened" are different
/// moments, and a check taken between them reads a conversation that is half
/// written.
@MainActor
final class HistoryHarness {
    enum Failure: Error, CustomStringConvertible {
        case notSettled(step: String, reason: String, trace: [String])
        case invariant(step: String, failures: [String], trace: [String])
        case storeUnavailable(String)

        var description: String {
            switch self {
            case .notSettled(let step, let reason, let trace):
                return "\(step) did not settle within 5 s: \(reason)\n"
                    + Self.render(trace)
            case .invariant(let step, let failures, let trace):
                return "\(step) broke \(failures.count) invariant(s):\n"
                    + failures.map { "  - \($0)" }.joined(separator: "\n") + "\n"
                    + Self.render(trace)
            case .storeUnavailable(let reason):
                return reason
            }
        }

        private static func render(_ trace: [String]) -> String {
            "  trace:\n" + trace.map { "    \($0)" }.joined(separator: "\n")
        }
    }

    private static let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    let root: URL
    let modelDirectory: URL
    let storeRoot: URL
    let client: FakeInferenceClient
    private(set) var model: AppModel!
    private(set) var reference: HistoryReferenceModel
    private(set) var trace: [String] = []
    private var attachmentStore = AppImageAttachmentStore()
    private var step = 0

    init(label: String) async throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-walk-\(label)-\(UUID().uuidString)",
                                    isDirectory: true)
        modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        storeRoot = ConversationStoreLocation.url(forModelDirectory: modelDirectory)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        client = FakeInferenceClient(eventDelay: .milliseconds(1))
        client.setImageSoftTokens(HistoryImageFixture.softTokens ?? 0)
        // The context the app comes up at with no settings file, taken from
        // the same defaults it takes it from. Restating the number here made
        // the relaunch check a check on the harness rather than on the app.
        reference = HistoryReferenceModel(
            maxContextTokens: MacAppSettings().contextTokens)
        try await start()
    }

    /// Everything this harness put on disk. Called from the test's `defer`,
    /// because a walk that throws still has to leave the machine clean.
    func tearDown() {
        model?.releaseAllAttachments()
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - Lifecycle

    private func start() async throws {
        attachmentStore = AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent(
                "staged-\(UUID().uuidString)", isDirectory: true))
        let model = AppModel(modelDirectory: modelDirectory, client: client,
                             attachmentStore: attachmentStore,
                             settingsPersistenceEnabled: true)
        model.modelPathText = modelDirectory.path
        // Read from the installed model's manifest in the app; a temporary
        // directory has none, and identity is not what this covers.
        model.conversationIdentity = Self.identity
        model.installationStatus = .complete
        model.loadState = .ready(modelDirectory: modelDirectory, loadSeconds: 1)
        // A long message is thousands of words, and the double answers with the
        // prompt echoed back one token at a time. Eight is enough reply to
        // pair every turn and keeps a long send as cheap as a short one.
        model.maxNewTokensOverride = 8
        self.model = model
        try await client.ensureLoaded(
            modelDirectory: modelDirectory,
            maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions,
            forceLogitsHead: true) { _ in }
        try await waitForWritableStore()
        try await settle("start")
        if let restored = reference.reading {
            try await settle("restore selection") {
                model.screen.conversationID == restored
                    && model.screen.document != nil
            }
        }
    }

    private func waitForWritableStore() async throws {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if !model.history.isReadOnlyStore { return }
            // The previous instance's `flock` is released when its store
            // deallocates, which is not synchronous with dropping the model.
            model.reacquireStoreIfPossible()
            try await Task.sleep(for: .milliseconds(1))
        }
        throw Failure.storeUnavailable(
            "the relaunched instance never took the writer lock")
    }

    // MARK: - Applying an operation

    func apply(_ operation: HistoryOperation) async throws {
        step += 1
        trace.append("\(step). \(operation)")
        switch operation {
        case .newChat: try await applyNewChat()
        case .send(let label): try await applySend(label: label)
        case .sendLong(let label): try await applySend(label: label, long: true)
        case .sendWithImage(let label): try await applySend(label: label, withImage: true)
        case .open(let row): try await applyOpen(row: row)
        case .delete(let row): try await applyDelete(row: row)
        case .rename(let row, let label): try await applyRename(row: row, label: label)
        case .setContext(let tokens): try await applySetContext(tokens)
        case .failNextRestore: try await applyFailNextRestore()
        case .relaunch: try await applyRelaunch()
        case .reload: try await applyReload()
        case .unload: try await applyUnload()
        case .stopMidTurn(let label): try await applySend(label: label, stopping: true)
        }
        try checkInvariants()
    }

    private func note(_ text: String) {
        trace[trace.count - 1] += " [\(text)]"
    }

    private func applyNewChat() async throws {
        guard model.canStartNewChat else {
            note("refused")
            return
        }
        model.newChat()
        reference.newChat()
        try await settle("newChat")
    }

    private func applySend(label: Int, long: Bool = false,
                           withImage: Bool = false,
                           stopping: Bool = false) async throws {
        // The composer is closed for a row that cannot be continued, so a send
        // there is not something a user can do. Driving `run()` anyway would
        // exercise a replay the window never offers.
        if model.isShowingStoredCopy, model.openedConversationState != .continuable {
            note("skipped: the row on screen cannot be continued")
            return
        }
        let word = "m\(label)"
        let prompt = long
            ? Array(repeating: word, count: HistoryOperation.longMessageWords)
                .joined(separator: " ")
            : word
        model.promptText = prompt
        guard model.canRun else {
            model.promptText = ""
            note("skipped: canRun is false")
            return
        }
        if withImage, HistoryImageFixture.isAvailable, model.maximumImageAttachments >= 1 {
            let staged = try attachmentStore.stage(AppModelHistoryTests.pngFixture())
            model.setComposerAttachmentsForTesting([staged])
        } else if withImage {
            note("no image: \(HistoryImageFixture.isAvailable ? "the context has no room" : "no preprocessor")")
        }
        // Whatever the composer is actually carrying, which is not only what
        // this step attached: a replay that failed hands the message and its
        // pictures back, and the next send is the one that delivers them.
        let images = model.imageAttachments.count

        // Stopping a turn that is carrying pictures would leave the write that
        // was started for it with nothing to finish into, and this operation is
        // about the turn, not the images. The composer can be holding some
        // without this step having attached any: a replay that failed hands the
        // message and its pictures back.
        let stops = stopping && images == 0

        let readingBefore = readingID
        let failsToReplay = readingBefore != nil && reference.restoreFailureArmed
        let heldBefore = reference.held
        let target = readingBefore ?? heldBefore
        let expectedRecords = ((target.map { reference.exchanges(in: $0) }) ?? 0) * 2 + 2

        model.send()
        guard model.isTurnInFlight else {
            // `send()` takes the composer synchronously, so the only way it
            // returns with nothing in flight is the `canRun` guard above.
            model.promptText = ""
            note("refused before it started")
            try await settle("send")
            return
        }
        if stopping, !stops { note("not stopped: the turn is carrying pictures") }
        if stops {
            // Stop reaches a decoding turn, not a replay: the button is not
            // offered while a reopened conversation is being prefilled. The
            // send runs the stages in its own task, so the button becomes live
            // a hop or two after `send()` returns — and whether the stop then
            // reaches the run before it finishes is the race
            // `applyStopOutcome` reads off the transcript.
            try await waitForCancellableTurn()
            if model.canCancel { model.cancel() }
        }

        if failsToReplay {
            reference.failRestoreDestructively()
            try await settle("send") { [self] in
                // The replay hands the message back to the composer, which is
                // the observable end of that path.
                !model.isTurnInFlight && model.promptText == prompt
            }
            model.promptText = ""
            note("replay refused")
            return
        }

        if readingBefore != nil {
            // The replay put the conversation back into the KV, so nothing on
            // screen is out of context any more.
            reference.adoptRestored()
        }
        if stops {
            try await applyStopOutcome(prompt: prompt, target: target,
                                       expectedRecords: expectedRecords)
            return
        }
        try await settle("send")
        if model.promptText == prompt {
            // The message came back, so a stage refused it: the request did not
            // validate, because what the conversation already holds leaves no
            // room for the images attached to this turn. Read off the composer
            // rather than off a synchronous return, because the stages that can
            // refuse now run in the send's own task.
            model.promptText = ""
            note("refused")
            try await settle("send")
            return
        }
        try await settle("send") { [self] in
            guard let id = model.storedConversationID else { return false }
            return model.history.entry(id)?.turnCount == expectedRecords
        }
        guard let landed = model.storedConversationID else {
            throw Failure.invariant(
                step: "send", failures: ["the turn was written to no conversation"],
                trace: trace)
        }
        if let target {
            guard landed == target else {
                throw Failure.invariant(
                    step: "send",
                    failures: ["the turn landed in \(landed) rather than \(target)"],
                    trace: trace)
            }
        } else {
            guard reference.conversations[landed] == nil else {
                throw Failure.invariant(
                    step: "send",
                    failures: ["a new chat reused the existing conversation \(landed)"],
                    trace: trace)
            }
            reference.create(landed)
        }
        reference.appendExchange(to: landed, userText: prompt, images: images)
        note("into \(short(landed))")
    }

    /// Records what a stopped turn actually did.
    ///
    /// Whether the stop reaches the run before it produces its whole answer is
    /// a race the harness must not pretend to win: a hard abort rewinds the
    /// turn and writes nothing, and a stop that lands after the terminal event
    /// leaves an ordinary completed exchange. Which of the two happened is read
    /// off the transcript, not off the app — the app's own answer would pass
    /// for a model that had silently done neither.
    private func applyStopOutcome(prompt: String, target: UUID?,
                                  expectedRecords: Int) async throws {
        try await settle("stopMidTurn")
        guard let landed = model.storedConversationID else {
            // The turn never reached the point where the directory is created.
            note("stopped before the chat was written")
            return
        }
        if let target {
            guard landed == target else {
                throw Failure.invariant(
                    step: "stopMidTurn",
                    failures: ["the turn landed in \(landed) rather than \(target)"],
                    trace: trace)
            }
        } else if reference.conversations[landed] == nil {
            reference.create(landed)
        }
        let userTexts = try diskConversations()[landed]?.turns
            .filter { $0.role == .user }.map(\.text) ?? []
        if userTexts.count == reference.exchanges(in: landed) + 1 {
            // The stop lost the race; the turn is a committed exchange like any
            // other, and the record has to say so before the invariants run.
            try await settle("stopMidTurn") { [self] in
                model.history.entry(landed)?.turnCount == expectedRecords
            }
            reference.appendExchange(to: landed, userText: prompt, images: 0)
            note("finished before the stop landed, into \(short(landed))")
            return
        }
        reference.stopTurn(in: landed, userText: prompt)
        note("rewound in \(short(landed))")
    }

    /// Waits until the send has a generation Stop can reach, or until it has
    /// ended without one.
    private func waitForCancellableTurn() async throws {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if model.canCancel || !model.isTurnInFlight { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        throw Failure.notSettled(
            step: "stopMidTurn", reason: "the send never reached a turn Stop "
                + "could cancel", trace: trace)
    }

    private func applyReload() async throws {
        guard !model.isTurnInFlight, model.loadState.isReady || model.canLoadModel else {
            note("refused")
            return
        }
        // What the window's Reload does: the runner and its KV go, and a new
        // pair is built. Driven as the two actions the app exposes because the
        // Reload button itself is offered only for a session whose settings
        // have gone stale, and this harness deliberately never lets one.
        let released = reference.reading ?? reference.held
        if model.loadState.isReady {
            model.unloadModel()
            try await settle("reload") { [self] in model.loadState == .notLoaded }
        }
        model.loadModel()
        try await settle("reload") { [self] in model.loadState.isReady }
        reference.releaseKV()
        try await settle("reload") { [self] in
            guard let released else { return true }
            return model.screen.conversationID == released
                && model.screen.document != nil
        }
        note(released == nil ? "no durable row" : "durable row awaits replay")
    }

    private func applyUnload() async throws {
        guard !model.isTurnInFlight, model.canUnloadModel else {
            note("refused")
            return
        }
        let released = reference.reading ?? reference.held
        model.unloadModel()
        try await settle("unload") { [self] in model.loadState == .notLoaded }
        reference.releaseKV()
        try await settle("unload") { [self] in
            guard let released else { return true }
            return model.screen.conversationID == released
                && model.screen.document != nil
        }
        note(released == nil ? "no durable row" : "durable row awaits replay")
    }

    private func applyOpen(row: Int) async throws {
        guard let id = rowID(row) else {
            note("no rows")
            return
        }
        guard !model.isTurnInFlight else {
            note("refused")
            return
        }
        let state = continuability(of: id)
        // The one the KV is already holding stays live: browsing back to it
        // rebuilds nothing, whatever its record says. Only a context it no
        // longer fits reads it as a copy, because then the copy's notice is
        // the truth.
        var staysLive = id == model.storedConversationID
            && model.serviceEpoch == model.conversation.epoch
        if case .needsContext = state { staysLive = false }
        let exchanges = reference.exchanges(in: id)

        model.openConversation(id: id)

        if staysLive {
            reference.stopReading()
            try await settle("open") { [self] in
                !model.isShowingStoredCopy && model.history.selection == id
            }
            note("live \(short(id))")
            return
        }
        reference.read(id)
        try await settle("open") { [self] in
            guard exchanges > 0 else { return true }
            // One list now, whichever state the row is in: what the window
            // draws is the document the click asked for.
            return !model.transcriptHistory.isEmpty
        }
        note("reading \(short(id)) \(state)")
    }

    private func applyDelete(row: Int) async throws {
        guard let id = rowID(row) else {
            note("no rows")
            return
        }
        model.deleteConversation(id: id)
        reference.delete(id)
        try await settle("delete") { [self] in model.history.entry(id) == nil }
        note("removed \(short(id))")
    }

    private func applyRename(row: Int, label: Int) async throws {
        guard let id = rowID(row) else {
            note("no rows")
            return
        }
        let title = "t\(label)"
        let before = model.history.entry(id)?.updatedAt
        model.renameConversation(id: id, to: title)
        reference.rename(id, to: title)
        try await settle("rename") { [self] in model.history.entry(id)?.title == title }
        if let before, model.history.entry(id)?.updatedAt != before {
            throw Failure.invariant(
                step: "rename",
                failures: ["renaming \(short(id)) moved it up the list"], trace: trace)
        }
        note("renamed \(short(id))")
    }

    private func applySetContext(_ tokens: Int) async throws {
        guard tokens != model.maxContextTokens else {
            note("already \(tokens)")
            return
        }
        let reading = readingID
        model.setMaxContextTokens(tokens)
        reference.maxContextTokens = tokens
        reference.recordSettingsPersisted()
        // The double's loaded session has to move with the setting or every
        // later turn comes back as a runtime mismatch. Deliberately not a
        // load: a real one ends the lineage, and taking the KV away from the
        // conversation the window is still holding is what the `reload`
        // operation is for.
        client.setLoadedContext(tokens)

        guard let reading else {
            try await settle("setContext")
            note("live")
            return
        }
        // Re-opening the row rebuilds it from the answer that now applies, and
        // that answer can put it back into the live conversation.
        let state = continuability(of: reading)
        let staysLive = reading == model.storedConversationID
            && model.serviceEpoch == model.conversation.epoch
            && state == .continuable
        let exchanges = reference.exchanges(in: reading)
        if staysLive { reference.stopReading() }
        try await settle("setContext") { [self] in
            if staysLive { return !model.isShowingStoredCopy }
            guard exchanges > 0 else { return true }
            return !model.transcriptHistory.isEmpty
        }
        note("redrew \(short(reading)) as \(state)")
    }

    private func applyFailNextRestore() async throws {
        client.failNextRestore(with: .conversationRestoreFailed("harness"))
        reference.armRestoreFailure()
        try await settle("failNextRestore")
    }

    private func applyRelaunch() async throws {
        let before = try diskConversations()
        // What quitting does: the staged copies of this session go, the service
        // ends its lineage, and the window is gone before the next one starts.
        model.releaseAllAttachments()
        await client.unload()
        model = nil
        reference.relaunch()
        try await start()
        guard model.maxContextTokens == reference.maxContextTokens else {
            throw Failure.invariant(
                step: "relaunch",
                failures: ["the context came back as \(model.maxContextTokens), "
                    + "not the \(reference.maxContextTokens) that was saved"],
                trace: trace)
        }
        let after = try diskConversations()
        var failures: [String] = []
        if Set(before.keys) != Set(after.keys) {
            failures.append("the store lost or gained conversations across a relaunch")
        }
        let store = try requireStore()
        for id in after.keys.sorted(by: { $0.uuidString < $1.uuidString }) {
            guard let opened = try? await store.open(id: id) else {
                failures.append("\(short(id)) could not be opened after a relaunch")
                continue
            }
            if opened.droppedTornFinalLine {
                failures.append("\(short(id)) had a torn final line after a clean quit")
            }
        }
        if !failures.isEmpty {
            throw Failure.invariant(step: "relaunch", failures: failures, trace: trace)
        }
        note("\(after.count) row(s) intact")
    }

    // MARK: - Settling

    /// Waits until nothing is in flight, with a deadline that throws.
    ///
    /// The existing `waitUntil` returns silently when its deadline expires,
    /// which turns a hang into a passing assertion about whatever state the app
    /// happened to be in. Nothing here may do that.
    func settle(_ what: String,
                _ extra: @MainActor () -> Bool = { true }) async throws {
        await model.persistenceTail?.value
        let deadline = Date().addingTimeInterval(5)
        var reason = "unknown"
        while Date() < deadline {
            if let pending = inFlightReason() {
                reason = pending
            } else if !extra() {
                reason = "the operation's own effect has not landed"
            } else {
                // A send can enqueue its final write after the initial wait.
                // Read the committed transcript only after that write settles.
                await model.persistenceTail?.value
                if inFlightReason() != nil || !extra() { continue }
                return
            }
            try await Task.sleep(for: .milliseconds(1))
        }
        throw Failure.notSettled(step: what, reason: reason, trace: trace)
    }

    private func inFlightReason() -> String? {
        if model.isRunning { return "a generation is running" }
        if model.screen.isReplaying { return "a replay is running" }
        // The send is one pipeline in its own task now, and its stages — the
        // replay, the ticket, the request, the retain, the hand-back — are not
        // all covered by the two flags above.
        if model.isTurnInFlight { return "a send is on its way" }
        if client.isGenerating { return "the client still holds a generation" }
        if model.pendingTurnImageWrite != nil { return "a turn's images are being written" }
        // Names only. Decoding every transcript on a one-millisecond poll made
        // settling cost more than the operations it was waiting for.
        guard let onDisk = try? diskConversationIDs() else {
            return "the store could not be read"
        }
        if Set(model.history.entries.map(\.id)) != onDisk {
            return "the list and the disk disagree"
        }
        return nil
    }

    // MARK: - Invariants

    func checkInvariants() throws {
        let disk: [UUID: StoredConversation]
        do { disk = try diskConversations() } catch {
            throw Failure.invariant(
                step: "invariants", failures: ["the store could not be read: \(error)"],
                trace: trace)
        }
        var failures: [String] = []
        failures += diskAgreesWithItself(disk)
        failures += oneThingIsOnScreen(disk)
        failures += theAppAndTheGateAgree()
        failures += everyReferencedImageExists(disk)
        failures += noOrphanImages(disk)
        failures += theListIsTheDisk(disk)
        guard failures.isEmpty else {
            throw Failure.invariant(
                step: trace.last ?? "step", failures: failures, trace: trace)
        }
    }

    /// 1. Every conversation on disk is what its own records say it is, and
    /// what the harness asked for.
    private func diskAgreesWithItself(_ disk: [UUID: StoredConversation]) -> [String] {
        var failures: [String] = []
        for id in disk.keys.sorted(by: { $0.uuidString < $1.uuidString }) {
            let stored = disk[id]!
            let name = short(id)
            let turns = stored.turns
            let images = turns.reduce(0) { $0 + $1.images.count }
            // One check for the whole file. `conversation.json` is a cache of
            // the projection over the records, so every field it carries has
            // exactly one right answer and anything else is drift — the class
            // that produced a row reading 1,065 tokens over 22 tokens of
            // record. `legacy: nil` on purpose: a store this harness wrote
            // carries its creation facts in its header, and letting the cache
            // answer for itself would pass for any field that had drifted.
            do {
                let projected = try ConversationMetaProjection.project(
                    directoryID: id, records: stored.records, legacy: nil)
                let differing = Self.fieldsDiffering(projected, stored.meta)
                if !differing.isEmpty {
                    failures.append("\(name): the file disagrees with its records "
                        + "about \(differing.joined(separator: ", "))")
                }
            } catch {
                failures.append("\(name): its records do not project to a "
                    + "conversation: \(error)")
            }
            guard let expected = reference.conversations[id] else {
                failures.append("\(name): on disk but not expected")
                continue
            }
            let userTexts = turns.filter { $0.role == .user }.map(\.text)
            if userTexts != expected.userTexts {
                failures.append("\(name): holds \(userTexts.count) message(s), "
                    + "\(expected.userTexts.count) expected")
            }
            if images != expected.imageCount {
                failures.append("\(name): holds \(images) image(s), "
                    + "\(expected.imageCount) expected")
            }
            if let title = reference.expectedTitle(of: id), stored.meta.title != title {
                failures.append("\(name): titled \(stored.meta.title), \(title) expected")
            }
        }
        for id in reference.conversations.keys where disk[id] == nil {
            failures.append("\(short(id)): expected on disk and not there")
        }
        return failures
    }

    /// 2. Exactly one conversation is on screen, and it is the one the window
    /// says it is.
    private func oneThingIsOnScreen(_ disk: [UUID: StoredConversation]) -> [String] {
        var failures: [String] = []
        if model.screen.isReplaying {
            failures.append("a replay was still running after the step settled")
        }
        let onScreen = model.transcriptHistory
        let ids = onScreen.map(\.user.id)
        if Set(ids).count != ids.count {
            failures.append("a turn is drawn twice in the transcript")
        }
        if case .unreadable(let id, let cause) = model.screen {
            failures.append("the window could not read \(short(id)): \(cause)")
            return failures
        }
        // A stored copy on screen, whether it is merely being read or is being
        // put back into the KV. There is no second list to check any more: the
        // screen either names a conversation or it does not.
        if let reading = model.screen.conversationID {
            if reference.reading != reading {
                failures.append("the window is reading \(short(reading)), "
                    + "\(reference.reading.map(short) ?? "nothing") expected")
            }
            guard let stored = disk[reading] else {
                failures.append("the window is reading \(short(reading)), which is "
                    + "not on disk")
                return failures
            }
            if model.history.selection != reading {
                failures.append("the row selected is "
                    + "\(model.history.selection.map(short) ?? "none") while "
                    + "\(short(reading)) is on screen")
            }
            let expected = stored.turns.filter { $0.role == .user }.map(\.text)
            if onScreen.map(\.user.text) != expected {
                failures.append("the transcript shows \(onScreen.count) exchange(s) of "
                    + "\(short(reading)), which holds \(expected.count)")
            }
            return failures
        }
        // Live: the window is showing the conversation the KV is holding.
        if reference.reading != nil {
            failures.append("the window went live while \(short(reference.reading!)) "
                + "was expected on screen")
        }
        // Empty unless a load or an unload released the KV under a conversation
        // that was on screen: those turns stay drawn, above a context break,
        // and belong to no live conversation any more. A stored copy cannot be
        // drawn under them any more — there is nowhere left to hold one while
        // the screen is live — so the check for that is the compiler's now.
        let archived = model.conversation.outOfContextPairs.map(\.user.text)
        if archived != reference.outOfContext {
            failures.append("the transcript holds \(archived.count) exchange(s) out "
                + "of context, \(reference.outOfContext.count) expected")
        }
        if model.transcriptContextBreak != (archived.isEmpty ? nil : archived.count) {
            failures.append("the context break is drawn at "
                + "\(model.transcriptContextBreak.map(String.init) ?? "nowhere")")
        }
        if model.storedConversationID != reference.held {
            failures.append("the held conversation is "
                + "\(model.storedConversationID.map(short) ?? "none"), "
                + "\(reference.held.map(short) ?? "none") expected")
        }
        if model.history.selection != model.storedConversationID {
            failures.append("the row selected is "
                + "\(model.history.selection.map(short) ?? "none") while the window "
                + "shows \(model.storedConversationID.map(short) ?? "an empty chat")")
        }
        let expected = reference.held.flatMap { reference.conversations[$0]?.userTexts } ?? []
        let live = model.conversation.completedPairs.map(\.user.text)
        if live != expected {
            failures.append("the live conversation holds \(live.count) exchange(s), "
                + "\(expected.count) expected")
        }
        if onScreen.map(\.user.text) != reference.outOfContext + Array(live.dropLast()) {
            failures.append("the transcript above the live turn is not the rest of "
                + "the conversation")
        }
        return failures
    }

    /// 3. The app's turn numbering and the service's are the same numbering.
    private func theAppAndTheGateAgree() -> [String] {
        var failures: [String] = []
        for rejection in client.gateRejections {
            failures.append("the service refused a turn: \(rejection)")
        }
        if model.conversation.isLineageLost {
            failures.append("the live conversation's lineage was lost")
        }
        if model.conversation.committedTurns != reference.committedTurns {
            failures.append("the app has committed \(model.conversation.committedTurns) "
                + "turn(s), \(reference.committedTurns) expected")
        }
        guard let epoch = model.serviceEpoch, epoch == model.conversation.epoch else {
            return failures
        }
        if client.gateOpenEpoch != epoch {
            failures.append("the service is holding "
                + "\(client.gateOpenEpoch.map(short) ?? "no conversation") while the "
                + "app is sending against \(short(epoch))")
        }
        if client.gateCommittedTurns != model.conversation.committedTurns {
            failures.append("the service has committed \(client.gateCommittedTurns) "
                + "turn(s) and the app \(model.conversation.committedTurns)")
        }
        return failures
    }

    /// 4. Nothing on screen or on disk points at a picture that is gone.
    private func everyReferencedImageExists(_ disk: [UUID: StoredConversation]) -> [String] {
        var failures: [String] = []
        for (id, stored) in disk {
            for turn in stored.turns {
                for image in turn.images {
                    for file in [image.pixelsFile, image.thumbnailFile] {
                        let url = stored.directory.appendingPathComponent(file)
                        if !FileManager.default.fileExists(atPath: url.path) {
                            failures.append("\(short(id)) cites \(file), which is gone")
                        }
                    }
                }
            }
        }
        var shown: [ChatImage] = model.imageAttachments.map(ChatImage.staged)
        shown += model.outputImageAttachments
        shown += model.conversation.turns.flatMap(\.images)
        shown += model.conversation.outOfContextPairs
            .flatMap { $0.user.images + $0.assistant.images }
        shown += (model.screen.document?.pairs ?? [])
            .flatMap { $0.user.images + $0.assistant.images }
        for attachment in shown
        where !FileManager.default.fileExists(atPath: attachment.fileURL.path) {
            failures.append("the window is holding \(attachment.id), whose file is gone")
        }
        return failures
    }

    /// 5. A conversation's image directory holds nothing no record names.
    private func noOrphanImages(_ disk: [UUID: StoredConversation]) -> [String] {
        var failures: [String] = []
        for (id, stored) in disk {
            let images = stored.directory.appendingPathComponent("images", isDirectory: true)
            guard let files = try? FileManager.default.contentsOfDirectory(
                at: images, includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]) else { continue }
            var referenced: Set<String> = []
            for turn in stored.turns {
                for image in turn.images {
                    referenced.insert(image.pixelsFile)
                    referenced.insert(image.thumbnailFile)
                }
            }
            for file in files where !referenced.contains("images/" + file.lastPathComponent) {
                failures.append("\(short(id)) keeps \(file.lastPathComponent), which no "
                    + "record names")
            }
        }
        return failures
    }

    /// 6. The sidebar is the store, in the order the store defines.
    private func theListIsTheDisk(_ disk: [UUID: StoredConversation]) -> [String] {
        var failures: [String] = []
        let entries = model.history.entries
        if Set(entries.map(\.id)) != Set(disk.keys) {
            failures.append("the list holds \(entries.count) row(s) over "
                + "\(disk.count) conversation(s) on disk")
            return failures
        }
        for entry in entries where entry != disk[entry.id]?.meta {
            failures.append("the row for \(short(entry.id)) is not what the file says")
        }
        // Ordered by when a conversation was last used. Compared as a run
        // rather than against a sorted copy, because conversations written in
        // the same second tie and the sort is not specified to be stable.
        for (older, newer) in zip(entries.dropFirst(), entries)
        where newer.updatedAt < older.updatedAt {
            failures.append("the list is not ordered by when a chat was last used")
            break
        }
        if model.history.unreadableCount != 0 {
            failures.append("\(model.history.unreadableCount) conversation(s) could "
                + "not be read")
        }
        return failures
    }

    // MARK: - Reading the store directly

    struct StoredConversation {
        let directory: URL
        let meta: ConversationMeta
        let records: [TranscriptRecord]

        var turns: [ConversationTurnRecord] {
            records.compactMap { if case .turn(let turn) = $0 { return turn } else { return nil } }
        }

    }

    /// Which fields of the cache and the projection disagree.
    ///
    /// Named rather than compared whole so a failure says what drifted; the
    /// values themselves are not printed, because a title is content.
    private static func fieldsDiffering(_ projected: ConversationMeta,
                                        _ stored: ConversationMeta) -> [String] {
        var names: [String] = []
        if projected.version != stored.version { names.append("version") }
        if projected.id != stored.id { names.append("id") }
        if projected.title != stored.title { names.append("title") }
        if projected.titleSource != stored.titleSource { names.append("titleSource") }
        if projected.createdAt != stored.createdAt { names.append("createdAt") }
        if projected.updatedAt != stored.updatedAt { names.append("updatedAt") }
        if projected.turnCount != stored.turnCount {
            names.append("turnCount (\(stored.turnCount) stored, "
                + "\(projected.turnCount) recorded)")
        }
        if projected.kvTokens != stored.kvTokens {
            names.append("kvTokens (\(stored.kvTokens.map(String.init) ?? "none") "
                + "stored, \(projected.kvTokens.map(String.init) ?? "none") recorded)")
        }
        if projected.imageCount != stored.imageCount {
            names.append("imageCount (\(stored.imageCount) stored, "
                + "\(projected.imageCount) recorded)")
        }
        if projected.identity != stored.identity { names.append("identity") }
        if projected.session != stored.session { names.append("session") }
        if projected.sampling != stored.sampling { names.append("sampling") }
        if projected.boundary != stored.boundary { names.append("boundary") }
        if projected.imageWriteFailed != stored.imageWriteFailed {
            names.append("imageWriteFailed")
        }
        if projected.pinned != stored.pinned { names.append("pinned") }
        return names
    }

    /// The conversations on disk by name alone, without opening any of them.
    func diskConversationIDs() throws -> Set<UUID> {
        guard FileManager.default.fileExists(atPath: storeRoot.path) else { return [] }
        return Set(try FileManager.default.contentsOfDirectory(
            at: storeRoot, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            .compactMap { UUID(uuidString: $0.lastPathComponent) })
    }

    /// Reads the files, not the store.
    ///
    /// Deliberately its own decoder: an invariant that asked `ConversationStore`
    /// what is on disk would pass for any bug the two share.
    func diskConversations() throws -> [UUID: StoredConversation] {
        guard FileManager.default.fileExists(atPath: storeRoot.path) else { return [:] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var result: [UUID: StoredConversation] = [:]
        for entry in try FileManager.default.contentsOfDirectory(
            at: storeRoot, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) {
            guard let id = UUID(uuidString: entry.lastPathComponent) else { continue }
            let metaData = try Data(
                contentsOf: entry.appendingPathComponent(ConversationMeta.fileName))
            let meta = try decoder.decode(ConversationMeta.self, from: metaData)
            let transcript = try Data(
                contentsOf: entry.appendingPathComponent("transcript.jsonl"))
            var records: [TranscriptRecord] = []
            for line in transcript.split(separator: 0x0A, omittingEmptySubsequences: true) {
                records.append(try decoder.decode(TranscriptRecord.self, from: Data(line)))
            }
            result[id] = StoredConversation(
                directory: entry, meta: meta, records: records)
        }
        return result
    }

    // MARK: - Small helpers

    private func requireStore() throws -> ConversationStore {
        guard let store = model.conversationStore else {
            throw Failure.storeUnavailable("the app has no conversation store")
        }
        return store
    }

    /// The row a conversation is drawn at right now.
    ///
    /// A fixed sequence names a conversation by when it was created; the
    /// sidebar orders by when a chat was last used, and `updatedAt` is written
    /// to whole seconds — so two chats used in the same second tie, and which
    /// of them is row 0 then depends on the order the directory enumerated. A
    /// literal row index is not a stable way to name one of them.
    func row(ofCreated index: Int) throws -> Int {
        guard index < reference.created.count else {
            throw Failure.storeUnavailable(
                "the harness has only \(reference.created.count) conversation(s)")
        }
        let id = reference.created[index]
        guard let row = model.history.entries.firstIndex(where: { $0.id == id }) else {
            throw Failure.storeUnavailable("\(short(id)) is not in the list")
        }
        return row
    }

    private func rowID(_ row: Int) -> UUID? {
        let entries = model.history.entries
        guard !entries.isEmpty else { return nil }
        return entries[row % entries.count].id
    }

    /// The stored conversation the window is drawing from disk, if any.
    ///
    /// Deliberately not `screen.conversationID`, which also names the row of a
    /// chat that could not be read: what the callers ask is whether a send has
    /// a conversation to replay first.
    private var readingID: UUID? {
        if case .reading(let id, _, _, _) = model.screen { return id }
        return nil
    }

    private func continuability(of id: UUID) -> ConversationContinuability {
        guard let meta = model.history.entry(id) else {
            return .cannotReplay(reason: .tokenCountUnknown)
        }
        return ConversationContinuability.evaluate(
            meta: meta, currentContext: reference.maxContextTokens,
            identity: Self.identity)
    }

    private func short(_ id: UUID) -> String { String(id.uuidString.prefix(8)) }
}
