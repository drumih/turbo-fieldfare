import TurboFieldfareAppCore
import SwiftUI

struct RootView: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 0) {
            primaryContent
                .frame(minWidth: 720, maxWidth: .infinity, maxHeight: .infinity)

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
                    Color(nsColor: .windowBackgroundColor).mix(with: .indigo, by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(.indigo)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
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
        OutputPaneView(model: model)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 10) {
                    ErrorBanner(model: model)
                    if model.promptText.isEmpty {
                        PromptExamplesView { preset in
                            model.promptText = preset.prefix
                        }
                    }
                    PromptComposerView(model: model)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 16)
            }
            .animation(.smooth(duration: 0.2), value: model.promptText.isEmpty)
    }
}
