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
                if showsContext {
                    // No info button beside this one. The memory figure has one
                    // because `phys_footprint` is genuinely misread; "20/8.2K"
                    // is not, and the same words are on hover.
                    HUDMetricView(value: contextText, label: "context",
                                  animated: !model.isRunning)
                        .help(contextHelp)
                }
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

    /// Shown once a conversation is holding anything. A gauge that reads 0 for
    /// the whole of a single-prompt session is noise.
    private var showsContext: Bool {
        liveContextTokens.map { $0 > 0 } ?? !model.conversation.isEmpty
    }

    /// The KV position now, not at the end of the last turn.
    private var liveContextTokens: Int? {
        guard model.isRunning else { return model.conversation.kvTokens }
        guard model.livePrefillDone > 0 || model.liveTokenCount > 0 else {
            return model.conversation.kvTokens
        }
        return ConversationContextPresentation.liveTokens(
            prefillDone: model.livePrefillDone,
            prefillTotal: model.livePrefillTotal,
            generated: model.liveTokenCount,
            committed: model.conversation.kvTokens ?? 0)
    }

    private var contextText: String {
        guard let liveContextTokens else { return "\u{2014}" }
        return ConversationContextPresentation.gauge(
            kvTokens: liveContextTokens,
            maxContext: model.effectiveMaxContextTokens)
    }

    private var contextHelp: String {
        guard let liveContextTokens else {
            return "The decode service did not report the committed context position. "
                + "Start a new chat before adding images."
        }
        return ConversationContextPresentation.explanation(
            kvTokens: liveContextTokens,
            maxContext: model.effectiveMaxContextTokens,
            cachedTokens: model.diagnostics?.cachedPromptTokens)
    }

    private var rateText: String {
        if model.phase == .decode { return MetricFormat.rate(model.liveTokensPerSecond) }
        if let d = model.diagnostics { return MetricFormat.rate(d.tokensPerSecond) }
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
