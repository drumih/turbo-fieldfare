import SwiftUI
import UniformTypeIdentifiers
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation

/// Sidebar showing the conversation history.
struct ConversationSidebarView: View {
    @Bindable var model: AppModel
    @State private var searchText: String = ""
    @State private var showingClearConfirmation = false
    @State private var exportDocument: ConversationsExportDocument?
    @State private var showingExporter = false
    @State private var showingImporter = false
    @State private var importError: String?
    @State private var importedCount: Int?
    @FocusState private var searchFocused: Bool
    @State private var showingCorruptionWarning = false
    @State private var markdownDocument: MarkdownExportDocument?
    @State private var showingMarkdownExporter = false
    @State private var markdownFilename = "conversation"
    @State private var showingTemplatesOnly = false
    @State private var editingTagsConversation: Conversation?
    @State private var tagsDraft = ""

    private var filteredConversations: [Conversation] {
        var items = model.conversationStore.conversations
        if showingTemplatesOnly {
            items = items.filter(\.isTemplate)
        }
        guard !searchText.isEmpty else { return items }
        let query = searchText.lowercased()
        if query.hasPrefix("#") {
            let tag = String(query.dropFirst())
            return items.filter { conversation in
                conversation.tags.contains { $0.lowercased().contains(tag) }
            }
        }
        return items.filter {
            $0.title.lowercased().contains(query)
                || $0.turns.contains { turn in
                    turn.text.lowercased().contains(query)
                }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar: new conversation + search + export/import
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
                    .keyboardShortcut("n", modifiers: .command)

                    Spacer()

                    if model.conversationStore.conversations.contains(where: \.isTemplate) {
                        Button {
                            showingTemplatesOnly.toggle()
                        } label: {
                            Label("Templates", systemImage: "square.stack.3d.up")
                                .labelStyle(.iconOnly)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(showingTemplatesOnly
                                         ? TurboFieldfareMacTheme.accentColor
                                         : Color.secondary)
                        .help(showingTemplatesOnly
                              ? "Show all conversations"
                              : "Show templates only")
                    }

                    if !model.conversationStore.conversations.isEmpty {
                        Menu {
                            Button("Export…") { prepareExport() }
                            Button("Import…") { showingImporter = true }
                            Divider()
                            Button("Clear all", role: .destructive) {
                                showingClearConfirmation = true
                            }
                        } label: {
                            Label("More", systemImage: "ellipsis.circle")
                                .labelStyle(.iconOnly)
                        }
                        .menuStyle(.borderlessButton)
                        .menuIndicator(.hidden)
                        .fixedSize()
                        .help("Export, import, or clear history")
                    }
                }

                TextField("Search…", text: $searchText)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout)
                    .focused($searchFocused)
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
                        ConversationRowView(
                            conversation: conversation,
                            isActive: model.activeConversationID == conversation.id,
                            onSelect: { model.loadConversation(conversation) },
                            onDelete: { model.deleteConversation(conversation) },
                            onRename: { model.renameConversation(conversation, title: $0) },
                            onTogglePin: { model.togglePin(conversation) },
                            onFork: { model.forkConversation(conversation) },
                            onToggleTemplate: {
                                model.setTemplate(conversation,
                                                  isTemplate: !conversation.isTemplate)
                            },
                            onEditTags: {
                                tagsDraft = conversation.tags.joined(separator: ", ")
                                editingTagsConversation = conversation
                            },
                            onExportMarkdown: {
                                markdownFilename = Self.safeFilename(conversation.title)
                                markdownDocument = MarkdownExportDocument(
                                    text: model.exportConversationMarkdown(conversation))
                                showingMarkdownExporter = true
                            }
                        )
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
            }
        }
        .background {
            // Hidden command: focus the search field (⌘F).
            Button("") { searchFocused = true }
                .keyboardShortcut("f", modifiers: .command)
                .opacity(0)
                .frame(width: 0, height: 0)
                .accessibilityHidden(true)
        }
        .onAppear {
            if model.conversationStore.didLoadFromCorrupted {
                showingCorruptionWarning = true
            }
        }
        .alert("History file unreadable", isPresented: $showingCorruptionWarning) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("The previous history file was corrupted and was moved aside; a fresh empty history was started. The unreadable file was renamed to conversations.json.corrupted.")
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
        .alert("Edit tags", isPresented: .init(
            get: { editingTagsConversation != nil },
            set: { if !$0 { editingTagsConversation = nil } }
        )) {
            TextField("Tags (comma separated)", text: $tagsDraft)
            Button("Save") {
                if let conversation = editingTagsConversation {
                    let tags = tagsDraft.split(separator: ",")
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    model.setTags(conversation, tags: tags)
                }
                editingTagsConversation = nil
            }
            Button("Cancel", role: .cancel) {
                editingTagsConversation = nil
            }
        } message: {
            Text("Tags are used to filter the history (search #tag).")
        }
        .alert("Import failed", isPresented: .init(
            get: { importError != nil },
            set: { if !$0 { importError = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(importError ?? "")
        }
        .alert("Import complete", isPresented: .init(
            get: { importedCount != nil },
            set: { if !$0 { importedCount = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Imported \(importedCount ?? 0) conversation(s).")
        }
        .fileExporter(
            isPresented: $showingExporter,
            document: exportDocument,
            contentType: .json,
            defaultFilename: "turbofieldfare-conversations"
        ) { _ in
            exportDocument = nil
        }
        .fileExporter(
            isPresented: $showingMarkdownExporter,
            document: markdownDocument,
            contentType: .plainText,
            defaultFilename: markdownFilename
        ) { _ in
            markdownDocument = nil
        }
        .fileImporter(
            isPresented: $showingImporter,
            allowedContentTypes: [.json]
        ) { result in
            switch result {
            case .success(let url):
                do {
                    importedCount = try model.importConversations(from: url)
                } catch {
                    importError = error.localizedDescription
                }
            case .failure(let error):
                importError = error.localizedDescription
            }
        }
    }

    private func prepareExport() {
        guard let data = try? model.conversationStore.exportJSON() else { return }
        exportDocument = ConversationsExportDocument(data: data)
        showingExporter = true
    }

    /// Builds a filesystem-safe filename from a conversation title.
    static func safeFilename(_ title: String) -> String {
        let invalid = CharacterSet(charactersIn: "/\\:?%*|\"<>")
        let cleaned = title.components(separatedBy: invalid).joined()
        let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "conversation" : trimmed
    }
}

/// Markdown document backing the per-conversation export sheet.
struct MarkdownExportDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText] }

    var text: String

    init(text: String) {
        self.text = text
    }

    init(configuration: ReadConfiguration) throws {
        text = String(data: configuration.file.regularFileContents ?? Data(),
                      encoding: .utf8) ?? ""
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

/// JSON document backing the export sheet.
struct ConversationsExportDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }

    var data: Data

    init(data: Data) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

/// A row in the conversation list.
struct ConversationRowView: View {
    let conversation: Conversation
    let isActive: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void
    let onRename: (String) -> Void
    let onTogglePin: () -> Void
    let onFork: () -> Void
    let onToggleTemplate: () -> Void
    let onEditTags: () -> Void
    let onExportMarkdown: () -> Void
    @State private var isRenaming = false
    @State private var draftTitle = ""

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onSelect) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        if conversation.isPinned {
                            Image(systemName: "pin.fill")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        if isRenaming {
                            TextField("Title", text: $draftTitle)
                                .textFieldStyle(.roundedBorder)
                                .font(.callout.weight(.semibold))
                                .onSubmit { commitRename() }
                                .onExitCommand { isRenaming = false }
                        } else {
                            Text(conversation.title)
                                .font(.callout.weight(isActive ? .semibold : .regular))
                                .lineLimit(1)
                                .onTapGesture(count: 2) { beginRename() }
                        }
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
        }
        .contextMenu {
            Button("Open", action: onSelect)
            Button(conversation.isPinned ? "Unpin" : "Pin", action: onTogglePin)
            Button(conversation.isTemplate ? "Remove template" : "Save as template",
                   action: onToggleTemplate)
            Button("Edit Tags…", action: onEditTags)
            Button("Rename", action: beginRename)
            Button("Fork", action: onFork)
            Divider()
            Button("Export Markdown…", action: onExportMarkdown)
            Divider()
            Button("Delete", role: .destructive, action: onDelete)
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
            Button(role: .destructive, action: onDelete) {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    private func beginRename() {
        draftTitle = conversation.title
        isRenaming = true
    }

    private func commitRename() {
        isRenaming = false
        onRename(draftTitle)
    }
}

#Preview {
    ConversationSidebarView(model: AppModel())
        .frame(width: 260, height: 400)
}
