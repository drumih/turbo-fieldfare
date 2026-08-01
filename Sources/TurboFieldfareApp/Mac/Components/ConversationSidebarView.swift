import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ConversationSidebarView: View {
    let model: AppModel
    @State private var renamingID: UUID?
    @State private var renameText = ""
    @State private var pendingDeletionID: UUID?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            list
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text("Chats")
                .font(.headline)
            Spacer()
            Button(action: model.newConversation) {
                Label("New Chat", systemImage: "square.and.pencil")
                    .labelStyle(.iconOnly)
                    .frame(width: 26, height: 26)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .disabled(model.isRunning)
            .help("New Chat")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                ForEach(model.conversationsByRecency) { conversation in
                    row(conversation)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
        }
    }

    @ViewBuilder
    private func row(_ conversation: Conversation) -> some View {
        let isActive = conversation.id == model.activeConversationID
        Group {
            if renamingID == conversation.id {
                TextField("Title", text: $renameText)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout)
                    .onSubmit { commitRename(conversation.id) }
                    .onExitCommand { renamingID = nil }
            } else {
                Button {
                    model.selectConversation(conversation.id)
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(conversation.title)
                            .font(.callout)
                            .lineLimit(1)
                            .truncationMode(.tail)
                            .foregroundStyle(isActive ? Color.primary : .secondary)
                        Text(subtitle(conversation))
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background {
                        RoundedRectangle(cornerRadius: 7)
                            .fill(isActive
                                  ? TurboFieldfareMacTheme.accentColor.opacity(0.14)
                                  : Color.clear)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(model.isRunning && !isActive)
                .contextMenu {
                    Button("Rename…") {
                        renameText = conversation.title
                        renamingID = conversation.id
                    }
                    Button("Delete", role: .destructive) {
                        pendingDeletionID = conversation.id
                    }
                    .disabled(model.isRunning)
                }
            }
        }
        .confirmationDialog(
            "Delete this conversation?",
            isPresented: Binding(
                get: { pendingDeletionID == conversation.id },
                set: { if !$0 { pendingDeletionID = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                model.deleteConversation(conversation.id)
                pendingDeletionID = nil
            }
            Button("Cancel", role: .cancel) { pendingDeletionID = nil }
        } message: {
            Text("\(conversation.title) will be removed permanently.")
        }
    }

    private func subtitle(_ conversation: Conversation) -> String {
        guard !conversation.isEmpty else { return "Empty" }
        let exchanges = conversation.turns.filter { $0.role == .user }.count
        return exchanges == 1 ? "1 message" : "\(exchanges) messages"
    }

    private func commitRename(_ id: UUID) {
        model.renameConversation(id, to: renameText)
        renamingID = nil
    }
}
