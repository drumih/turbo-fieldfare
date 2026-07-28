import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0
    @AppStorage("TurboFieldfare.chatSidebarVisible")
    private var isChatSidebarVisible = true
    @AppStorage("TurboFieldfare.inspectorVisible")
    private var isInspectorVisible = true

    var body: some View {
        HStack(spacing: 0) {
            if isChatSidebarVisible {
                ChatSidebarView(model: model)
                    .frame(width: CGFloat(AppChromeLayout.chatSidebarWidth))
                    .frame(maxHeight: .infinity)
                    .background(Color(nsColor: .underPageBackgroundColor))
                    .transition(.move(edge: .leading).combined(with: .opacity))

                Divider()
            }

            primaryContent
                .frame(
                    minWidth: CGFloat(AppChromeLayout.primaryMinimumWidth),
                    maxWidth: .infinity,
                    maxHeight: .infinity)

            if isInspectorVisible {
                Divider()

                InspectorView(model: model)
                    .frame(width: CGFloat(AppChromeLayout.inspectorWidth))
                    .frame(maxHeight: .infinity)
                    .background(Color(nsColor: .windowBackgroundColor))
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .frame(
            minWidth: CGFloat(AppChromeLayout.minimumWindowWidth(
                isChatSidebarVisible: isChatSidebarVisible,
                isInspectorVisible: isInspectorVisible)),
            minHeight: CGFloat(AppChromeLayout.minimumHeight))
        .containerBackground(for: .window) {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).mix(
                        with: TurboFieldfareMacTheme.accentColor,
                        by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(TurboFieldfareMacTheme.accentColor)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
        .animation(.smooth(duration: 0.22), value: isChatSidebarVisible)
        .animation(.smooth(duration: 0.22), value: isInspectorVisible)
        .transaction { transaction in
            if model.isRunning {
                transaction.animation = nil
            }
        }
        .onReceive(
            NotificationCenter.default.publisher(
                for: NSApplication.willTerminateNotification)
        ) { _ in
            model.flushChatPersistence()
        }
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
            StatusHUDView(
                model: model,
                isChatSidebarVisible: isChatSidebarVisible,
                isInspectorVisible: isInspectorVisible,
                toggleChatSidebar: { isChatSidebarVisible.toggle() },
                toggleInspector: { isInspectorVisible.toggle() })
        }
    }

    private var conversationView: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                if model.hasOutputTranscript {
                    OutputPaneView(model: model)
                        .padding(.bottom, conversationChromeHeight)
                } else if conversationChromeHeight > 0 {
                    OutputPaneView(model: model)
                        .frame(
                            height: max(
                                0,
                                geometry.size.height - conversationChromeHeight))
                        .frame(maxHeight: .infinity, alignment: .top)
                }

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
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ErrorBanner(model: model)
            if model.showsPromptExamples {
                PromptExamplesView { preset in
                    model.promptText = preset.prompt
                }
            }
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
        .animation(.smooth(duration: 0.2), value: model.showsPromptExamples)
    }
}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
