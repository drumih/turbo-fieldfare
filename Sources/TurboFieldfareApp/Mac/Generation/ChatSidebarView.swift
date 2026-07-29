import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ChatSidebarView: View {
    @Bindable var model: AppModel
    @AppStorage(AppAppearance.storageKey)
    private var appearanceRawValue = AppAppearance.system.rawValue

    @State private var hoveredChatID: AppChat.ID?
    @State private var chatBeingRenamed: AppChat?
    @State private var renameText = ""
    @State private var chatPendingDeletion: AppChat?

    var body: some View {
        VStack(spacing: 0) {
            header
            newChatButton
                .padding(.horizontal, 12)
                .padding(.bottom, 10)
            Divider()
            chatList
            Divider()
            footer
        }
        .alert(
            "Rename chat",
            isPresented: renameAlertPresented,
            presenting: chatBeingRenamed
        ) { chat in
            TextField("Chat name", text: $renameText)
            Button("Cancel", role: .cancel) {}
            Button("Rename") {
                model.renameChat(id: chat.id, title: renameText)
            }
            .disabled(renameText.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty)
        } message: { _ in
            Text("Choose a name that identifies this chat's context.")
        }
        .alert(
            "Delete chat?",
            isPresented: deletionAlertPresented,
            presenting: chatPendingDeletion
        ) { chat in
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                model.deleteChat(id: chat.id)
            }
        } message: { chat in
            Text("“\(chat.title)” and its saved document context will be removed.")
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "bolt.horizontal.circle.fill")
                .font(.title3)
                .foregroundStyle(TurboFieldfareMacTheme.accentColor)
            Text("TurboFieldfare")
                .font(.headline)
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.top, 38)
        .padding(.bottom, 12)
        .gesture(WindowDragGesture())
    }

    private var newChatButton: some View {
        Button {
            model.createChat()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "square.and.pencil")
                Text("New chat")
                    .fontWeight(.medium)
                Spacer()
                Text("⌘N")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .frame(maxWidth: .infinity, minHeight: 42)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background(
            Color(nsColor: .controlBackgroundColor),
            in: .rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .stroke(.separator.opacity(0.45), lineWidth: 0.5)
        }
        .disabled(model.isRunning)
        .help("Create a chat with separate context")
    }

    private var chatList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 3) {
                Text("Chats")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.top, 10)
                    .padding(.bottom, 3)

                ForEach(sortedChats) { chat in
                    chatRow(chat)
                }
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func chatRow(_ chat: AppChat) -> some View {
        let isSelected = chat.id == model.selectedChatID
        let showsActions = hoveredChatID == chat.id || isSelected

        return HStack(spacing: 4) {
            Button {
                model.selectChat(id: chat.id)
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isSelected
                          ? "bubble.left.fill"
                          : "bubble.left")
                        .font(.caption)
                        .foregroundStyle(isSelected
                                         ? Color.accentColor
                                         : Color.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 5) {
                            Text(chat.title)
                                .font(.callout.weight(
                                    isSelected ? .semibold : .regular))
                                .foregroundStyle(.primary)
                                .lineLimit(1)
                            if chat.contextSummary?.isEmpty == false {
                                Image(systemName: "brain")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .help(
                                        "Older turns are kept in compressed memory")
                            }
                        }
                        if !chat.preview.isEmpty {
                            Text(chat.preview)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    Spacer(minLength: 0)
                }
                .padding(.leading, 9)
                .padding(.vertical, 8)
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .disabled(model.isRunning && !isSelected)

            Menu {
                Button("Rename", systemImage: "pencil") {
                    renameText = chat.title
                    chatBeingRenamed = chat
                }
                Divider()
                Button("Delete", systemImage: "trash", role: .destructive) {
                    chatPendingDeletion = chat
                }
            } label: {
                Label("Chat actions", systemImage: "ellipsis")
                    .labelStyle(.iconOnly)
                    .frame(width: 26, height: 26)
                    .contentShape(Circle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .opacity(showsActions ? 1 : 0)
            .disabled(model.isRunning)
            .accessibilityHidden(!showsActions)
            .padding(.trailing, 5)
        }
        .background(
            isSelected
                ? Color.accentColor.opacity(0.12)
                : Color.clear,
            in: .rect(cornerRadius: 9))
        .contentShape(.rect(cornerRadius: 9))
        .onHover { isHovering in
            hoveredChatID = isHovering ? chat.id : nil
        }
        .contextMenu {
            Button("Rename") {
                renameText = chat.title
                chatBeingRenamed = chat
            }
            .disabled(model.isRunning)
            Button("Delete", role: .destructive) {
                chatPendingDeletion = chat
            }
            .disabled(model.isRunning)
        }
    }

    private var footer: some View {
        HStack(spacing: 7) {
            Image(systemName: "internaldrive")
            Text("\(model.chats.count) local \(model.chats.count == 1 ? "chat" : "chats")")
            Spacer()
            appearanceMenu
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .frame(height: 42)
    }

    private var appearanceMenu: some View {
        let appearance = AppAppearance.resolve(appearanceRawValue)
        return Menu {
            Picker("Appearance", selection: $appearanceRawValue) {
                ForEach(AppAppearance.allCases) { option in
                    Label(option.label, systemImage: option.systemImage)
                        .tag(option.rawValue)
                }
            }
        } label: {
            Label("Appearance", systemImage: appearance.systemImage)
                .labelStyle(.iconOnly)
                .frame(width: 26, height: 26)
                .contentShape(Circle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Appearance: \(appearance.label)")
        .accessibilityLabel("Appearance")
        .accessibilityValue(appearance.label)
    }

    private var sortedChats: [AppChat] {
        model.chats.sorted { lhs, rhs in
            if lhs.updatedAt == rhs.updatedAt {
                return lhs.createdAt > rhs.createdAt
            }
            return lhs.updatedAt > rhs.updatedAt
        }
    }

    private var renameAlertPresented: Binding<Bool> {
        Binding(
            get: { chatBeingRenamed != nil },
            set: { if !$0 { chatBeingRenamed = nil } })
    }

    private var deletionAlertPresented: Binding<Bool> {
        Binding(
            get: { chatPendingDeletion != nil },
            set: { if !$0 { chatPendingDeletion = nil } })
    }
}
