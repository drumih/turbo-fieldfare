import Foundation
@testable import TurboFieldfareAppCore

/// What the store and the window should hold, kept independently of both.
///
/// Deliberately not derived from `AppModel`: a check that reads the app's own
/// answer back to itself passes for a model that has silently done nothing. The
/// only thing learned from the app is the identifier of a conversation the app
/// creates, because the app picks it — and the invariants then require that
/// identifier to be new when a new chat was expected and unchanged when it was
/// not.
struct HistoryReferenceModel {
    struct ExpectedConversation: Equatable {
        /// The user half of every stored exchange, in order. The assistant half
        /// is the double's fixed reply and carries no information.
        var userTexts: [String] = []
        var imageCount = 0
        /// The last title a rename recorded. Nil while the title is still the
        /// one derived from the first message.
        var renamedTo: String?
    }

    private(set) var conversations: [UUID: ExpectedConversation] = [:]
    /// The order they were created in, which is not the order they are drawn
    /// in: the sidebar sorts by when a chat was last used.
    private(set) var created: [UUID] = []
    /// The conversation the KV is holding and the next turn is written to.
    private(set) var held: UUID?
    private var heldIsInKV = false
    /// A stored conversation on screen in place of the live one.
    private(set) var reading: UUID?
    /// The user half of every exchange the window still draws but the model can
    /// no longer see: a load or an unload rebuilt or released the KV while the
    /// conversation was on screen. They stay above the live conversation with a
    /// context break between, and nothing appends to their file again.
    private(set) var outOfContext: [String] = []
    /// The message of a turn that was stopped and is still drawn as the current
    /// one. A rewound turn is in neither the KV nor the transcript, so it is not
    /// an exchange — but the window keeps showing it, and a release that finds
    /// nothing else in the conversation carries exactly that one turn out of
    /// context with it.
    private(set) var stoppedTurnOnScreen: String?
    /// The next replay will fail, because a test armed it.
    private(set) var restoreFailureArmed = false
    /// The context in force, and separately the one a send has written to the
    /// settings file — a relaunch comes back with the second, not the first.
    var maxContextTokens: Int
    private(set) var persistedContextTokens: Int

    init(maxContextTokens: Int) {
        self.maxContextTokens = maxContextTokens
        self.persistedContextTokens = maxContextTokens
    }

    /// Turns the service's gate should be holding for the live conversation.
    var committedTurns: Int {
        guard heldIsInKV, let held else { return 0 }
        return conversations[held]?.userTexts.count ?? 0
    }

    func exchanges(in id: UUID) -> Int { conversations[id]?.userTexts.count ?? 0 }

    mutating func newChat() {
        held = nil
        heldIsInKV = false
        reading = nil
        outOfContext = []
        stoppedTurnOnScreen = nil
    }

    mutating func create(_ id: UUID) {
        conversations[id] = ExpectedConversation()
        created.append(id)
    }

    /// A send lands on `id`: it becomes the held conversation, nothing is being
    /// read any more, and the exchange is on disk.
    mutating func appendExchange(to id: UUID, userText: String, images: Int) {
        conversations[id, default: ExpectedConversation()].userTexts.append(userText)
        conversations[id, default: ExpectedConversation()].imageCount += images
        held = id
        heldIsInKV = true
        reading = nil
        stoppedTurnOnScreen = nil
        recordSettingsPersisted()
    }

    /// A send that started and produced no exchange: the turn was stopped and
    /// the runtime rewound it. The conversation exists — its directory is
    /// created when the send starts, not when the turn lands — and the window
    /// is holding it.
    mutating func stopTurn(in id: UUID, userText: String) {
        held = id
        heldIsInKV = true
        reading = nil
        stoppedTurnOnScreen = userText
        recordSettingsPersisted()
    }

    /// Every send persists the settings, so this is the context a relaunch
    /// comes back with. A stopped turn persists them too: it happens when the
    /// run starts, not when it finishes.
    mutating func recordSettingsPersisted() {
        persistedContextTokens = maxContextTokens
    }

    /// Opening a row replaces everything the window was drawing, including the
    /// out-of-context turns of the live conversation.
    mutating func read(_ id: UUID) {
        reading = id
        outOfContext = []
    }

    mutating func stopReading() {
        reading = nil
        outOfContext = []
    }

    /// A replay put a stored conversation back into the KV, so nothing on
    /// screen is out of context any more.
    mutating func adoptRestored() {
        outOfContext = []
        heldIsInKV = true
    }

    /// A load or an unload rebuilt or released the KV.
    ///
    /// Keep the viewed conversation, or the held one when viewing it live.
    /// Its next send replays the exact record before appending to the same file.
    mutating func releaseKV() {
        reading = reading ?? held
        heldIsInKV = false
        outOfContext = []
        stoppedTurnOnScreen = nil
    }

    mutating func rename(_ id: UUID, to title: String) {
        conversations[id]?.renamedTo = title
    }

    /// Deleting held KV must not clear a different conversation being read.
    mutating func delete(_ id: UUID) {
        conversations[id] = nil
        created.removeAll { $0 == id }
        if held == id {
            let remainingRead = reading == id ? nil : reading
            newChat()
            reading = remainingRead
        } else if reading == id {
            reading = heldIsInKV ? nil : held
        }
    }

    mutating func armRestoreFailure() { restoreFailureArmed = true }

    mutating func consumeRestoreFailure() { restoreFailureArmed = false }

    mutating func failRestoreDestructively() {
        restoreFailureArmed = false
        // After reload, held identifies a durable row but its KV is gone.
        // A failed replay cannot archive turns from that empty cache.
        if heldIsInKV {
            outOfContext += held.flatMap { conversations[$0]?.userTexts } ?? []
        }
        held = nil
        heldIsInKV = false
        stoppedTurnOnScreen = nil
    }

    /// Quitting and starting again. The store and selected row survive; no KV
    /// does, so the selection returns as a stored copy awaiting replay.
    mutating func relaunch() {
        let selected = reading ?? held
        held = nil
        heldIsInKV = false
        reading = selected.flatMap { conversations[$0] == nil ? nil : $0 }
        outOfContext = []
        stoppedTurnOnScreen = nil
        maxContextTokens = persistedContextTokens
    }

    /// The title the meta must carry: the rename if there was one, otherwise
    /// the one derived from the first message.
    func expectedTitle(of id: UUID) -> String? {
        guard let conversation = conversations[id] else { return nil }
        if let renamedTo = conversation.renamedTo { return renamedTo }
        guard let first = conversation.userTexts.first else { return nil }
        return ConversationTitle.fromFirstMessage(first)
    }
}
