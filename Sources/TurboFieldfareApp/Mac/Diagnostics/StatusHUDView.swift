import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct StatusHUDView: View {
    let model: AppModel
    let isChatSidebarVisible: Bool
    let isInspectorVisible: Bool
    let toggleChatSidebar: () -> Void
    let toggleInspector: () -> Void

    var body: some View {
        strip
            .padding(.top, 10)
            .padding(.leading, CGFloat(AppChromeLayout.headerLeadingPadding(
                isChatSidebarVisible: isChatSidebarVisible)))
            .padding(.trailing, 20)
    }

    private var strip: some View {
        HStack(spacing: 12) {
            chatSidebarToggle
            Divider().frame(height: 16)
            ModelStatusBadge(model: model)
            selectedChatTitle
            Divider().frame(height: 16)
            PhaseLabel(model: model)
            if let action = model.presentation.primaryAction {
                HeaderModelActionButton(model: model, action: action)
            }
            Spacer(minLength: 12)
            if showsMetrics {
                HUDMetricView(value: rateText, label: "tok/s", animated: !model.isRunning)
                HUDMetricView(value: tokensText, label: "tokens", animated: !model.isRunning)
                HUDMetricView(value: memoryText, label: "memory", animated: !model.isRunning)
            }
            inspectorToggle
        }
        .frame(height: 30)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background {
            Capsule()
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    Capsule().stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .gesture(WindowDragGesture())
    }

    private var chatSidebarToggle: some View {
        let presentation = AppSidebarControlPresentation(
            sidebar: .chats,
            isVisible: isChatSidebarVisible)
        return Button(action: toggleChatSidebar) {
            Label(
                presentation.title,
                systemImage: presentation.systemImage)
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(isChatSidebarVisible ? .primary : .secondary)
        .keyboardShortcut("s", modifiers: [.command, .control])
        .help(presentation.help)
        .accessibilityValue(presentation.accessibilityValue)
    }

    private var selectedChatTitle: some View {
        HStack(spacing: 6) {
            Image(systemName: "bubble.left")
                .foregroundStyle(.secondary)
            Text(model.selectedChat.title)
                .font(.callout.weight(.medium))
                .lineLimit(1)
            if model.selectedChat.contextSummary?.isEmpty == false {
                Image(systemName: "brain")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .help("Older turns are kept in compressed memory")
            }
        }
        .frame(maxWidth: 180)
        .help(model.selectedChat.title)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Current chat")
        .accessibilityValue(model.selectedChat.title)
    }

    private var inspectorToggle: some View {
        let presentation = AppSidebarControlPresentation(
            sidebar: .inspector,
            isVisible: isInspectorVisible)
        return Button(action: toggleInspector) {
            Label(
                presentation.title,
                systemImage: presentation.systemImage)
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(isInspectorVisible ? .primary : .secondary)
        .keyboardShortcut("i", modifiers: [.command, .shift])
        .help(presentation.help)
        .accessibilityValue(presentation.accessibilityValue)
    }

    private var rateText: String {
        if model.phase == .decode { return MetricFormat.rate(model.liveTokensPerSecond) }
        if let d = model.diagnostics { return MetricFormat.rate(d.tokensPerSecond) }
        return "\u{2014}"
    }

    private var tokensText: String {
        if model.isRunning { return "\(model.liveTokenCount)" }
        if let d = model.diagnostics { return "\(d.generatedTokens)" }
        return "\u{2014}"
    }

    private var memoryText: String {
        MetricFormat.memory(model.currentProcessMemoryBytes)
    }

    private var showsMetrics: Bool {
        model.loadState.isReady || model.isRunning || model.diagnostics != nil
    }
}

private struct HeaderModelActionButton: View {
    let model: AppModel
    let action: AppModelAction

    var body: some View {
        let presentation = AppModelActionPresentation(action: action)
        Button {
            model.perform(action)
        } label: {
            Label(presentation.title, systemImage: presentation.systemImage)
                .lineLimit(1)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.small)
        .tint(presentation.isCancellation
              ? .orange
              : TurboFieldfareMacTheme.accentColor)
        .help(presentation.help)
        .accessibilityHint(presentation.help)
    }
}

private struct PhaseLabel: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            switch content {
            case .loading(let label):
                ProgressView().controlSize(.mini)
                Text(label)
            case .pulse(let label):
                PulsingDot()
                Text(label)
                    .monospacedDigit()
                    .contentTransition(.numericText())
            case .steady(let label):
                Circle().fill(TurboFieldfareMacTheme.accentColor).frame(width: 7, height: 7)
                Text(label).contentTransition(.opacity)
            case .quiet(let label):
                Text(label)
                    .foregroundStyle(.secondary)
                    .contentTransition(.opacity)
            }
        }
        .font(.caption.weight(.medium))
        .lineLimit(1)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Model status")
        .accessibilityValue(model.presentation.label)
    }

    private enum Content {
        case loading(String)
        case pulse(String)
        case steady(String)
        case quiet(String)
    }

    private var content: Content {
        let presentation = model.presentation
        if presentation.showsActivity { return .loading(presentation.label) }
        if model.isRunning
            && (model.phase == .prefill || model.phase == .compressing) {
            return .pulse(presentation.label)
        }
        if model.isRunning && model.phase == .decode { return .steady(presentation.label) }
        return .quiet(presentation.label)
    }
}

private struct PulsingDot: View {
    var body: some View {
        Circle()
            .fill(TurboFieldfareMacTheme.accentColor)
            .frame(width: 7, height: 7)
            .phaseAnimator([0.4, 1.0]) { dot, opacity in
                dot.opacity(opacity)
            } animation: { _ in
                .easeInOut(duration: 0.7)
            }
    }
}
