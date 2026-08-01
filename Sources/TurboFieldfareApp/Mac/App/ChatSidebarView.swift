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
        .background(TurboFieldfareMacTheme.sidebarBackground.opacity(0.82))
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
        VStack(spacing: 14) {
            sidebarMark
            sidebarToggle

            Button(action: model.newChat) {
                Label("New chat", systemImage: "plus")
                    .labelStyle(.iconOnly)
                    .font(.headline.weight(.medium))
                    .frame(width: 34, height: 34)
                    .contentShape(RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .background(TurboFieldfareMacTheme.accentColor, in: RoundedRectangle(cornerRadius: 10))
            .disabled(!model.canManageChats)
            .help("New chat")

            Spacer()
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
    }

    private var expandedContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                sidebarMark

                VStack(alignment: .leading, spacing: 1) {
                    Text("TurboFieldfare")
                        .font(.headline.weight(.semibold))
                    Text("LOCAL AI WORKSPACE")
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .tracking(1.1)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 8)
                sidebarToggle
            }
            .padding(.bottom, 20)

            Button(action: model.newChat) {
                Label("New chat", systemImage: "plus")
                    .font(.callout.weight(.semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 13)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .background(TurboFieldfareMacTheme.accentColor, in: RoundedRectangle(cornerRadius: 10))
            .overlay {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(.white.opacity(0.15), lineWidth: 0.5)
            }
            .disabled(!model.canManageChats)
            .keyboardShortcut("n", modifiers: [.command, .shift])

            HStack {
                Text("Recent chats")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(model.chats.count)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
            .padding(.top, 22)
            .padding(.bottom, 8)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 3) {
                    ForEach(model.chats) { chat in
                        chatRow(chat)
                    }
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 14)
        .padding(.bottom, 12)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    private var sidebarMark: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            TurboFieldfareMacTheme.accentColor,
                            TurboFieldfareMacTheme.accentColor.opacity(0.64),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing))
            Image(systemName: "sparkles")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
        }
        .frame(width: 30, height: 30)
        .shadow(color: TurboFieldfareMacTheme.accentColor.opacity(0.2), radius: 6, y: 2)
        .accessibilityHidden(true)
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
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: chat.id == model.selectedChatID
                      ? "bubble.left.and.bubble.right.fill"
                      : "bubble.left.and.bubble.right")
                    .font(.caption)
                    .foregroundStyle(chat.id == model.selectedChatID
                                     ? TurboFieldfareMacTheme.accentColor
                                     : .secondary)
                    .frame(width: 17)

                VStack(alignment: .leading, spacing: 4) {
                    Text(chat.title)
                        .font(.callout.weight(
                            chat.id == model.selectedChatID ? .semibold : .regular))
                        .lineLimit(1)
                    HStack(spacing: 5) {
                        if !chat.preview.isEmpty {
                            Text(chat.preview)
                                .lineLimit(1)
                        }
                        if !chat.preview.isEmpty {
                            Text("·")
                        }
                        Text(chat.updatedAt, style: .relative)
                            .lineLimit(1)
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.vertical, 9)
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .background(
                chat.id == model.selectedChatID
                    ? TurboFieldfareMacTheme.accentSurface
                    : .clear,
                in: RoundedRectangle(cornerRadius: 8))
            .overlay(alignment: .leading) {
                if chat.id == model.selectedChatID {
                    Capsule()
                        .fill(TurboFieldfareMacTheme.accentColor)
                        .frame(width: 3, height: 24)
                        .offset(x: 1)
                }
            }
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
