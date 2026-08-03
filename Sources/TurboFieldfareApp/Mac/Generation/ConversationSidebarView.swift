import SwiftUI
import TurboFieldfareAppCore

/// Sidebar showing the conversation history.
struct ConversationSidebarView: View {
    @Bindable var model: AppModel
    @State private var searchText: String = ""
    @State private var showingClearConfirmation = false

    private var filteredConversations: [Conversation] {
        let items = model.conversationStore.conversations
        guard !searchText.isEmpty else { return items }
        let query = searchText.lowercased()
        return items.filter {
            $0.title.lowercased().contains(query)
                || $0.prompt.lowercased().contains(query)
                || $0.response.lowercased().contains(query)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar: new conversation + search
            VStack(spacing: 8) {
                HStack {
                    Button {
                        model.newConversation()
                    } label: {
                        Label("New", systemImage: "square.and.pencil")
                            .font(.callout.weight(.medium))
                    }
                    .buttonStyle(.borderless)
                    .disabled(model.isRunning)

                    Spacer()

                    if !model.conversationStore.conversations.isEmpty {
                        Button(role: .destructive) {
                            showingClearConfirmation = true
                        } label: {
                            Label("Clear all", systemImage: "trash")
                                .labelStyle(.iconOnly)
                        }
                        .buttonStyle(.borderless)
                        .help("Clear all history")
                    }
                }

                TextField("Search…", text: $searchText)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)

            Divider()

            // Conversation list
            if filteredConversations.isEmpty {
                ContentUnavailableView(
                    searchText.isEmpty ? "No conversations" : "No results",
                    systemImage: "bubble.left.and.bubble.right",
                    description: Text(
                        searchText.isEmpty
                            ? "Run a generation to create your first conversation."
                            : "No conversations match “\(searchText)”."
                    )
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(selection: .constant(model.activeConversationID)) {
                    ForEach(filteredConversations) { conversation in
                        ConversationRowView(conversation: conversation,
                                            isActive: model.activeConversationID == conversation.id,
                                            onSelect: { model.loadConversation(conversation) },
                                            onDelete: { model.deleteConversation(conversation) })
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
            }
        }
        .alert("Clear all history?", isPresented: $showingClearConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Clear", role: .destructive) {
                model.conversationStore.clearAll()
                model.newConversation()
            }
        } message: {
            Text("This action cannot be undone.")
        }
    }
}

/// A row in the conversation list.
struct ConversationRowView: View {
    let conversation: Conversation
    let isActive: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void

    var body: some View {
        Button(action: onSelect) {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(conversation.title)
                        .font(.callout.weight(isActive ? .semibold : .regular))
                        .lineLimit(1)
                    Spacer()
                    Text(conversation.updatedAt, style: .relative)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text(conversation.responsePreview)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                if let tokens = conversation.generatedTokenCount {
                    Text("\(tokens) tokens")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.vertical, 2)
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button("Open", action: onSelect)
            Divider()
            Button("Delete", role: .destructive, action: onDelete)
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
            Button(role: .destructive, action: onDelete) {
                Label("Delete", systemImage: "trash")
            }
        }
    }
}

#Preview {
    ConversationSidebarView(model: AppModel())
        .frame(width: 260, height: 400)
}