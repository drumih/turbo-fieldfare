import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0
    @AppStorage("chatSidebarCollapsed") private var isChatSidebarCollapsed = false

    var body: some View {
        HStack(spacing: 0) {
            ChatSidebarView(
                model: model,
                isCollapsed: $isChatSidebarCollapsed)
                .frame(width: isChatSidebarCollapsed ? 56 : 240)

            Divider()

            primaryContent
                .frame(minWidth: 600, maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            InspectorView(model: model)
                .frame(width: 320)
                .frame(maxHeight: .infinity)
                .background(Color(nsColor: .windowBackgroundColor))
        }
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
        .animation(.smooth(duration: 0.2), value: model.presentation.conversationAction)
        .animation(.smooth(duration: 0.22), value: isChatSidebarCollapsed)
        .transaction { transaction in
            if model.isRunning {
                transaction.animation = nil
            }
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
            StatusHUDView(model: model)
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
            ModelReadinessPromptView(model: model)
            ErrorBanner(model: model)
            if model.promptText.isEmpty
                && model.showPromptExamples
                && !model.hasOutputTranscript {
                PromptExamplesView { preset in
                    model.promptText = preset.prompt
                }
            }
            ModelActionBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
        .animation(
            .smooth(duration: 0.2),
            value: model.promptText.isEmpty
                && model.showPromptExamples
                && !model.hasOutputTranscript)
    }

}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
