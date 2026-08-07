import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ModelActionBanner: View {
    @Bindable var model: AppModel

    var body: some View {
        if model.hasOutputTranscript,
           let action = model.presentation.conversationAction {
            HStack(spacing: 10) {
                Image(systemName: iconName(for: action))
                    .foregroundStyle(iconColor(for: action))
                    .accessibilityHidden(true)
                Text(message(for: action))
                    .font(.callout.weight(.medium))
                    .lineLimit(2)
                Spacer(minLength: 8)
                Button(buttonTitle(for: action)) {
                    model.perform(action)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .accessibilityLabel(buttonTitle(for: action))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background {
                RoundedRectangle(cornerRadius: 14)
                    .fill(TurboFieldfareMacTheme.elevatedSurface)
                    .overlay {
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(iconColor(for: action).opacity(0.75), lineWidth: 1.5)
                    }
                    .shadow(color: iconColor(for: action).opacity(0.12), radius: 6, y: 1)
            }
            .accessibilityElement(children: .contain)
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    private func message(for action: AppModelAction) -> String {
        switch action {
        case .load:
            return "The model is unloaded. Load it to generate."
        case .retryLoad:
            return model.presentation.detail ?? "The model could not be loaded."
        case .reload:
            return "Settings changed. Reload the model to continue."
        case .install, .cancelInstall, .cancelLoad, .unload:
            return model.presentation.label
        }
    }

    private func buttonTitle(for action: AppModelAction) -> String {
        switch action {
        case .load: return "Load Model"
        case .retryLoad: return "Retry Load"
        case .reload: return "Reload Model"
        case .install: return "Install"
        case .cancelInstall: return "Cancel"
        case .cancelLoad: return "Cancel Load"
        case .unload: return "Unload Model"
        }
    }

    private func iconName(for action: AppModelAction) -> String {
        switch action {
        case .retryLoad: return "exclamationmark.triangle.fill"
        case .reload: return "arrow.clockwise.circle.fill"
        case .load, .install, .cancelInstall, .cancelLoad, .unload:
            return "externaldrive.fill"
        }
    }

    private func iconColor(for action: AppModelAction) -> Color {
        action == .retryLoad ? .red : .orange
    }
}
