import Foundation
import Observation

/// What the sidebar shows: the stored conversations, which one is open, and
/// whether this instance may change them.
///
/// Separate from `AppConversation`, which is the live chat and whose invariant
/// is that its turns are exactly the model's context. This is a list of things
/// on disk, most of which the model has never seen.
@MainActor
@Observable
public final class ConversationHistoryState {
    public private(set) var entries: [ConversationMeta] = []
    /// The stored conversation the transcript is showing, or nil for a new chat
    /// that has not been written yet.
    public var selection: UUID?
    /// Another instance of the app holds the writer lock. The list still works;
    /// nothing in it can be changed.
    public private(set) var isReadOnlyStore = false

    public init() {}

    public func replaceEntries(_ entries: [ConversationMeta]) {
        self.entries = entries
    }

    public func replaceEntryAndResort(_ entry: ConversationMeta) {
        entries.removeAll { $0.id == entry.id }
        entries.append(entry)
        entries.sort {
            if $0.updatedAt == $1.updatedAt { return $0.id.uuidString < $1.id.uuidString }
            return $0.updatedAt > $1.updatedAt
        }
    }

    /// How many stored conversations the last listing could not read. The
    /// sidebar says so: they are absent from the list, and absent with no
    /// explanation is indistinguishable from deleted.
    public private(set) var unreadableCount = 0
    public private(set) var discardedLegacyTrashCount = 0

    public func setUnreadableCount(_ count: Int) {
        unreadableCount = count
    }

    public func setDiscardedLegacyTrashCount(_ count: Int) {
        discardedLegacyTrashCount = count
    }

    public func setReadOnlyStore(_ isReadOnly: Bool) {
        isReadOnlyStore = isReadOnly
    }

    public func entry(_ id: UUID) -> ConversationMeta? {
        entries.first { $0.id == id }
    }
}
