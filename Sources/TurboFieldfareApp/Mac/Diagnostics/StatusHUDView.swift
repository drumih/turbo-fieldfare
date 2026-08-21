import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct StatusHUDView: View {
    let model: AppModel

    var body: some View {
        strip
            .padding(.top, 10)
            .padding(.leading, 84)
            .padding(.trailing, 20)
    }

    private var strip: some View {
        HStack(spacing: 12) {
            ModelStatusBadge(model: model)
            Divider().frame(height: 16)
            PhaseLabel(model: model)
            Spacer(minLength: 12)
            if showsMetrics {
                HUDMetricView(value: rateText, label: "tok/s", animated: !model.isRunning)
                HUDMetricView(value: tokensText, label: "tokens", animated: !model.isRunning)
                HStack(spacing: 2) {
                    HUDMetricView(value: memoryText, label: "memory", animated: !model.isRunning)
                        .help(memoryHelp)
                    InfoPopoverButton(subject: "Memory", text: memoryHelp, arrowEdge: .bottom)
                }
            }
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

    /// `phys_footprint`: what this process is charged. Measured, not assumed —
    /// the model and image tower weights are memory-mapped and read by the GPU,
    /// and no per-process counter attributes them: 1,144 MB of tower retained
    /// moves the footprint by 5 MB. Those bytes are page cache, owned by the
    /// kernel and reclaimable, so the popover explains them in words and the
    /// diagnostics section carries their measured row.
    private var memoryText: String {
        MetricFormat.memory(model.currentProcessMemoryBytes)
    }

    private var memoryHelp: String {
        MemoryFootprintExplanation.text(chargedBytes: model.currentProcessMemoryBytes)
    }

    private var showsMetrics: Bool {
        model.loadState.isReady || model.isRunning || model.diagnostics != nil
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
        if model.isRunning && model.phase == .prefill { return .pulse(presentation.label) }
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
