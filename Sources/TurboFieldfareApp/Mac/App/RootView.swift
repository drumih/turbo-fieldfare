import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0
    @AppStorage("chatSidebarCollapsed") private var isChatSidebarCollapsed = false

    private let readableContentWidth: CGFloat = 820

    var body: some View {
        HStack(spacing: 0) {
            ChatSidebarView(
                model: model,
                isCollapsed: $isChatSidebarCollapsed)
                .frame(width: isChatSidebarCollapsed ? 58 : 258)

            paneDivider

            primaryContent
                .frame(minWidth: 600, maxWidth: .infinity, maxHeight: .infinity)

            paneDivider

            InspectorView(model: model)
                .frame(width: 316)
                .frame(maxHeight: .infinity)
                .background(TurboFieldfareMacTheme.sidebarBackground.opacity(0.5))
        }
        .containerBackground(for: .window) {
            TurboFieldfareMacTheme.appBackground
        }
        .tint(TurboFieldfareMacTheme.accentColor)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
        .animation(.smooth(duration: 0.2), value: model.presentation.conversationAction)
        .animation(.smooth(duration: 0.22), value: isChatSidebarCollapsed)
        .transaction { transaction in
            if model.isRunning {
                transaction.animation = nil
            }
        }
    }

    private var paneDivider: some View {
        Rectangle()
            .fill(TurboFieldfareMacTheme.border)
            .frame(width: 1)
            .ignoresSafeArea()
    }

    private var primaryContent: some View {
        Group {
            if model.requiresModelInstallation {
                ModelInstallView(model: model)
            } else {
                conversationView
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            StatusHUDView(model: model)
        }
    }

    private var conversationView: some View {
        ZStack(alignment: .bottom) {
            OutputPaneView(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.bottom, conversationChromeHeight)

            conversationChrome
                .background {
                    GeometryReader { chromeGeometry in
                        Color.clear.preference(
                            key: ConversationChromeHeightKey.self,
                            value: chromeGeometry.size.height)
                    }
                }
        }
        .onPreferenceChange(ConversationChromeHeightKey.self) { height in
            guard height > 0 else { return }
            var transaction = Transaction()
            transaction.disablesAnimations = true
            withTransaction(transaction) {
                conversationChromeHeight = height
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            ConversationHeaderView(model: model)
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ModelReadinessPromptView(model: model)
            ErrorBanner(model: model)
            if model.promptText.isEmpty
                && model.showPromptExamples
                && !model.hasOutputTranscript {
                PromptExamplesView { preset in
                    model.promptText = preset.prompt
                }
            }
            PromptComposerView(model: model)
        }
        .frame(maxWidth: readableContentWidth)
        .padding(.horizontal, 20)
        .padding(.bottom, 18)
        .frame(maxWidth: .infinity)
        .animation(
            .smooth(duration: 0.2),
            value: model.promptText.isEmpty
                && model.showPromptExamples
                && !model.hasOutputTranscript)
    }

}

private struct ConversationHeaderView: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.selectedChatTitle)
                    .font(.headline.weight(.semibold))
                    .lineLimit(1)
                HStack(spacing: 7) {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 9, weight: .semibold))
                    Text("Local conversation")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer(minLength: 16)

            if model.hasSystemPrompt {
                Label("Instructions", systemImage: "slider.horizontal.3")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(TurboFieldfareMacTheme.accentSurface, in: Capsule())
            }

            if model.completedTurnCount > 0 {
                Text("\(model.completedTurnCount) "
                     + (model.completedTurnCount == 1 ? "turn" : "turns"))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Menu {
                Button("Copy conversation") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(
                        model.outputConversationPlainText,
                        forType: .string)
                }
                .disabled(model.outputConversationPlainText.isEmpty)

                Button("Regenerate response", action: model.regenerateLastResponse)
                    .disabled(!model.canRegenerate)

                Divider()

                Button("New chat", action: model.newChat)
                    .disabled(!model.canManageChats)
            } label: {
                Image(systemName: "ellipsis")
                    .font(.headline.weight(.semibold))
                    .frame(width: 30, height: 30)
                    .contentShape(Circle())
            }
            .menuStyle(.borderlessButton)
            .foregroundStyle(.secondary)
            .help("Conversation actions")
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 11)
        .frame(maxWidth: 820)
        .frame(maxWidth: .infinity)
        .background(TurboFieldfareMacTheme.appBackground.opacity(0.96))
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(TurboFieldfareMacTheme.border)
                .frame(height: 1)
        }
    }
}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
