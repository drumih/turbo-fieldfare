import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ChatSidebarView: View {
    let model: AppModel
    @Binding var isCollapsed: Bool
    @State private var chatToRename: AppChatThread?
    @State private var renameText = ""

    var body: some View {
        Group {
            if isCollapsed {
                collapsedContent
            } else {
                expandedContent
            }
        }
        .frame(maxHeight: .infinity, alignment: .top)
        .background(Color(nsColor: .windowBackgroundColor))
        .alert("Rename chat", isPresented: renameAlertIsPresented) {
            TextField("Chat title", text: $renameText)
            Button("Cancel", role: .cancel) {
                chatToRename = nil
            }
            Button("Save") {
                if let chatToRename {
                    model.renameChat(chatToRename.id, to: renameText)
                }
                chatToRename = nil
            }
        }
    }

    private var collapsedContent: some View {
        VStack(spacing: 12) {
            sidebarToggle

            Button(action: model.newChat) {
                Label("New chat", systemImage: "square.and.pencil")
                    .labelStyle(.iconOnly)
                    .frame(width: 32, height: 32)
                    .contentShape(RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.borderedProminent)
            .tint(TurboFieldfareMacTheme.accentColor)
            .disabled(!model.canManageChats)
            .help("New chat")

            Spacer()
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
    }

    private var expandedContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                sidebarToggle

                Text("Chats")
                    .font(.headline)
                Spacer()
            }

            Button(action: model.newChat) {
                Label("New chat", systemImage: "square.and.pencil")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
            }
            .buttonStyle(.borderedProminent)
            .tint(TurboFieldfareMacTheme.accentColor)
            .disabled(!model.canManageChats)

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 3) {
                    ForEach(model.chats) { chat in
                        chatRow(chat)
                    }
                }
            }
        }
        .padding(12)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    private var sidebarToggle: some View {
        Button {
            isCollapsed.toggle()
        } label: {
            Label(
                isCollapsed ? "Expand chat sidebar" : "Collapse chat sidebar",
                systemImage: "sidebar.left")
                .labelStyle(.iconOnly)
                .frame(width: 32, height: 32)
                .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.borderless)
        .help(isCollapsed ? "Expand sidebar" : "Collapse sidebar")
    }

    private func chatRow(_ chat: AppChatThread) -> some View {
        Button {
            model.selectChat(chat.id)
        } label: {
            VStack(alignment: .leading, spacing: 3) {
                Text(chat.title)
                    .font(.callout.weight(
                        chat.id == model.selectedChatID ? .semibold : .regular))
                    .lineLimit(1)
                if !chat.preview.isEmpty {
                    Text(chat.preview)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 9)
            .padding(.vertical, 8)
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .background(
                chat.id == model.selectedChatID
                    ? TurboFieldfareMacTheme.accentColor.opacity(0.14)
                    : .clear,
                in: RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .disabled(!model.canManageChats)
        .contextMenu {
            Button("Rename") {
                renameText = chat.title
                chatToRename = chat
            }
            .disabled(!model.canManageChats)

            Divider()

            Button("Delete", role: .destructive) {
                model.deleteChat(chat.id)
            }
            .disabled(!model.canManageChats)
        }
    }

    private var renameAlertIsPresented: Binding<Bool> {
        Binding(
            get: { chatToRename != nil },
            set: { isPresented in
                if !isPresented { chatToRename = nil }
            })
    }
}
