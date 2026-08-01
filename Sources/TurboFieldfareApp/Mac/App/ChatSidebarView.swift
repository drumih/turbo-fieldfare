import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ChatSidebarView: View {
    let model: AppModel
    @Binding var isCollapsed: Bool
    @State private var chatToRename: AppChatThread?
    @State private var renameText = ""
    @State private var hoveredChatID: UUID?

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
            .background(TurboFieldfareMacTheme.accentGradient, in: RoundedRectangle(cornerRadius: 10))
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
            .background(TurboFieldfareMacTheme.accentGradient, in: RoundedRectangle(cornerRadius: 10))
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
        let isSelected = chat.id == model.selectedChatID
        let isHovered = chat.id == hoveredChatID

        return HStack(spacing: 2) {
            Button {
                model.selectChat(chat.id)
            } label: {
                chatRowContent(chat, isSelected: isSelected)
            }
            .buttonStyle(.plain)
            .disabled(!model.canManageChats)

            Menu {
                chatActions(for: chat)
            } label: {
                Label("Chat actions", systemImage: "ellipsis")
                    .labelStyle(.iconOnly)
                    .font(.caption.weight(.semibold))
                    .frame(width: 26, height: 28)
                    .contentShape(Circle())
            }
            .menuStyle(.borderlessButton)
            .help("Chat actions")
            .opacity(isSelected || isHovered ? 1 : 0)
            .allowsHitTesting(isSelected || isHovered)
            .disabled(!model.canManageChats)
        }
        .padding(.trailing, 3)
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .background(
            isSelected
                ? TurboFieldfareMacTheme.accentSurface
                : isHovered ? TurboFieldfareMacTheme.hoverSurface : .clear,
            in: RoundedRectangle(cornerRadius: 10))
        .overlay(alignment: .leading) {
            if isSelected {
                Capsule()
                    .fill(TurboFieldfareMacTheme.accentColor)
                    .frame(width: 3, height: 24)
                    .offset(x: 1)
            }
        }
        .onHover { isHovering in
            hoveredChatID = isHovering ? chat.id : nil
        }
        .animation(.easeOut(duration: 0.14), value: isHovered)
        .contextMenu {
            chatActions(for: chat)
        }
    }

    private func chatRowContent(_ chat: AppChatThread, isSelected: Bool) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: isSelected
                  ? "bubble.left.and.bubble.right.fill"
                  : "bubble.left.and.bubble.right")
                .font(.caption)
                .foregroundStyle(isSelected
                                 ? TurboFieldfareMacTheme.accentColor
                                 : .secondary)
                .frame(width: 17)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(chat.title)
                        .font(.callout.weight(isSelected ? .semibold : .regular))
                        .lineLimit(1)
                    Spacer(minLength: 4)
                    ChatRelativeTime(date: chat.updatedAt)
                }
                Text(chat.preview.isEmpty ? "Empty chat" : chat.preview)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .contentShape(RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func chatActions(for chat: AppChatThread) -> some View {
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

    private var renameAlertIsPresented: Binding<Bool> {
        Binding(
            get: { chatToRename != nil },
            set: { isPresented in
                if !isPresented { chatToRename = nil }
            })
    }
}

private struct ChatRelativeTime: View {
    let date: Date

    var body: some View {
        TimelineView(.periodic(from: .now, by: 60)) { context in
            Text(label(for: date, relativeTo: context.date))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .monospacedDigit()
        }
        .accessibilityLabel("Last updated")
        .accessibilityValue(label(for: date, relativeTo: Date()))
    }

    private func label(for date: Date, relativeTo now: Date) -> String {
        let elapsed = max(0, now.timeIntervalSince(date))
        switch elapsed {
        case ..<60:
            return "now"
        case ..<3_600:
            return "\(Int(elapsed / 60))m"
        case ..<86_400:
            return "\(Int(elapsed / 3_600))h"
        default:
            return "\(Int(elapsed / 86_400))d"
        }
    }
}
