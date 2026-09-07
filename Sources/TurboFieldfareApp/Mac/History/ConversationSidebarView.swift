import AppKit
import SwiftUI
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation

/// The list of stored conversations, built as a panel rather than a
/// `NavigationSplitView` column.
///
/// The split view supplied a sidebar toggle of its own beside ours, moved our
/// toolbar group out from the traffic lights whenever the list was open, and
/// left its sidebar column inert to the mouse. A panel is the same shape the
/// Inspector already had on the other side, and every part of it — the search
/// field, the rows, the selection — is ours, so a click does what the code says
/// it does.
///
/// Rows are buttons, not `List` selection. Selection in a `List` is a binding
/// SwiftUI drives, and the row still has to report which conversation to open;
/// a button says that directly, and it cannot render as selectable while
/// refusing to be selected.
struct ConversationSidebarView: View {
    let model: AppModel
    @State private var search = ""
    @State private var renameTarget: ConversationMeta?
    @State private var renameText = ""
    @State private var deleteTarget: ConversationMeta?

    var body: some View {
        VStack(spacing: 0) {
            searchField

            if model.history.isReadOnlyStore {
                readOnlyNotice
            }

            if model.history.unreadableCount > 0 {
                unreadableNotice
            }

            if model.history.discardedLegacyTrashCount > 0 {
                Text("Removed \(model.history.discardedLegacyTrashCount) chat(s) left in legacy trash.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
            }

            if visibleSections.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .alert("Rename Chat", isPresented: isRenaming, presenting: renameTarget) { target in
            TextField("Name", text: $renameText)
                .accessibilityIdentifier(.historyRenameField)
            Button("Rename") {
                model.renameConversation(id: target.id, to: renameText)
                renameTarget = nil
            }
            .keyboardShortcut(.defaultAction)
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .accessibilityIdentifier(.historyRenameConfirm)
            Button("Cancel", role: .cancel) { renameTarget = nil }
        } message: { target in
            Text("Rename \u{201C}\(ConversationRowPresentation.title(target))\u{201D}.")
        }
        // Asked, because it cannot be taken back. Delete used to move the
        // conversation to a `.trash` directory an Undo item could restore from,
        // which meant every deleted transcript and image stayed on disk for the
        // life of the store with nothing to collect them. Deleting for real is
        // the right behaviour; the cost is that this is the last chance to stop.
        .alert("Delete this chat?", isPresented: isDeleting, presenting: deleteTarget) { target in
            Button("Delete", role: .destructive) {
                model.deleteConversation(id: target.id)
                deleteTarget = nil
            }
            .disabled(!model.canMutateConversation(id: target.id))
            .accessibilityIdentifier(.historyDeleteConfirm)
            Button("Cancel", role: .cancel) { deleteTarget = nil }
        } message: { target in
            Text("\u{201C}\(ConversationRowPresentation.title(target))\u{201D} and "
                 + "everything in it will be deleted. This cannot be undone.")
        }
    }

    private var isDeleting: Binding<Bool> {
        Binding(
            get: { deleteTarget != nil },
            set: { presented in if !presented { deleteTarget = nil } })
    }

    /// Renaming is a dialog, not an editable row.
    ///
    /// In place it was unusable: putting a focused `TextField` inside the
    /// `LazyVStack` made every row draw its selected background, and clicking
    /// away from the field left it focused with no way to commit or cancel
    /// except the keyboard. A dialog has one field, one Return and one Escape,
    /// and the list keeps rendering as a list.
    private var isRenaming: Binding<Bool> {
        Binding(
            get: { renameTarget != nil },
            set: { presented in if !presented { renameTarget = nil } })
    }

    // MARK: - Search

    /// Built to the pill's measurements, not to its own.
    ///
    /// The two boxes sit on the same row of the window either side of the
    /// divider, so any difference between them reads as a mistake. At 24 points
    /// against the pill's 50 this one looked like it had floated to the top of
    /// the sidebar. Same top inset, same height, same capsule.
    private var searchField: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
                .font(.callout)
            TextField("Search Chats", text: $search)
                .textFieldStyle(.plain)
                .font(.callout)
                .accessibilityIdentifier(.historySearch)
            if !search.isEmpty {
                Button {
                    search = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Clear search")
                .accessibilityIdentifier(.historySearchClear)
            }
        }
        .frame(height: Self.chromeHeight)
        .padding(.horizontal, 12)
        .padding(.vertical, Self.chromeVerticalPadding)
        .background(Capsule().fill(Color(nsColor: .quaternarySystemFill)))
        .padding(.horizontal, 12)
        .padding(.top, Self.chromeTopInset)
        .padding(.bottom, 6)
    }

    /// `StatusHUDView`'s pill: 30 points of content, 10 above and below, 10 from
    /// the top of the window.
    private static let chromeHeight: CGFloat = 30
    private static let chromeVerticalPadding: CGFloat = 10
    private static let chromeTopInset: CGFloat = 10

    private var readOnlyNotice: some View {
        Label(
            "Another window has these chats open. This one can read them but "
                + "not change them.",
            systemImage: "lock")
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 12)
            .padding(.bottom, 6)
    }

    /// Says that chats are missing from the list, because they are.
    ///
    /// A directory whose meta will not decode is skipped so one bad
    /// conversation cannot hide the rest — right, but on its own it means a
    /// chat can vanish with nothing anywhere to say it existed. The count is
    /// all this can honestly offer: the file that would have carried the title
    /// is the file that would not read.
    private var unreadableNotice: some View {
        Label(
            model.history.unreadableCount == 1
                ? "1 chat on disk could not be read and is not listed."
                : "\(model.history.unreadableCount) chats on disk could not be "
                    + "read and are not listed.",
            systemImage: "exclamationmark.triangle")
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 12)
            .padding(.bottom, 6)
    }

    @ViewBuilder
    private var emptyState: some View {
        Spacer(minLength: 0)
        VStack(spacing: 6) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.largeTitle)
                .foregroundStyle(.tertiary)
            Text(search.isEmpty ? "No Chats Yet" : "No Matches")
                .font(.headline)
            Text(search.isEmpty
                 ? "Chats are kept here once you send a message."
                 : "No chat title contains \u{201C}\(search)\u{201D}.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, 20)
        Spacer(minLength: 0)
    }

    // MARK: - List

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2, pinnedViews: [.sectionHeaders]) {
                ForEach(visibleSections, id: \.group) { section in
                    Section {
                        ForEach(section.entries, id: \.id) { entry in
                            row(entry)
                        }
                    } header: {
                        Text(section.group.rawValue)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 16)
                            .padding(.top, 8)
                            .padding(.bottom, 2)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(nsColor: .windowBackgroundColor))
                    }
                }
            }
            .padding(.bottom, 8)
        }
        .scrollContentBackground(.hidden)
    }

    private var visibleSections: [(group: ConversationDateGroup, entries: [ConversationMeta])] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        let entries = query.isEmpty
            ? model.history.entries
            : model.history.entries.filter {
                $0.title.localizedCaseInsensitiveContains(query)
            }
        return ConversationDateGroup.sections(entries, now: Date())
    }

    private func row(_ entry: ConversationMeta) -> some View {
        let state = model.continuability(of: entry)
        let isSelected = model.history.selection == entry.id
        return HStack(spacing: 4) {
            Button {
                model.openConversation(id: entry.id)
            } label: {
                VStack(alignment: .leading, spacing: 2) {
                    Text(ConversationRowPresentation.title(entry))
                        .lineLimit(1)
                        .foregroundStyle(.primary)
                    Text(ConversationRowPresentation.subtitle(entry, state: state))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                // The whole row is the target, not just the text it happens
                // to contain.
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier(AccessibilityID.row(entry.id))

            // Rename and Delete on an explicit menu rather than only on a
            // right-click. A plain-styled button takes the control-click for
            // itself, so a context menu alone left both actions with no
            // reliable way in.
            Menu {
                Button("Rename") {
                    renameText = entry.title
                    renameTarget = entry
                }
                .disabled(!model.canMutateConversation(id: entry.id))
                // Not gated on the store being writable: looking at a
                // conversation's files is a read, and a second window holding
                // the lock is no reason to refuse it.
                Button("Open Folder in Finder") { openFolder(entry) }
                Divider()
                Button("Delete\u{2026}", role: .destructive) {
                    deleteTarget = entry
                }
                .disabled(!model.canMutateConversation(id: entry.id))
            } label: {
                Image(systemName: "ellipsis")
                    .foregroundStyle(.secondary)
                    .frame(width: 22, height: 22)
                    .contentShape(Rectangle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            // The window's accent tint reaches in here and painted the glyph
            // green, which reads as a primary action rather than the secondary
            // affordance it is.
            .tint(.secondary)
            .help("Rename or delete this chat")
            .accessibilityIdentifier(AccessibilityID.rowMenu(entry.id))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(rowBackground(isSelected: isSelected))
        .padding(.horizontal, 8)
    }

    /// Opens the folder itself rather than selecting it in its parent. The
    /// files are the point — `transcript.jsonl`, `conversation.json` and
    /// `images/` — and revealing the directory in the enclosing store would
    /// show a list of UUIDs instead.
    private func openFolder(_ entry: ConversationMeta) {
        Task {
            guard let url = await model.conversationFolderURL(for: entry.id) else {
                return
            }
            NSWorkspace.shared.open(url)
        }
    }

    private func rowBackground(isSelected: Bool) -> some View {
        RoundedRectangle(cornerRadius: 7, style: .continuous)
            .fill(isSelected
                  ? Color(nsColor: .selectedContentBackgroundColor).opacity(0.35)
                  : Color.clear)
    }
}
