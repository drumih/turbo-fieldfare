import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct GenerateControl: View {
    let model: AppModel
    private let controlHeight: CGFloat = 34

    var body: some View {
        if model.isRunning {
            runningPill
        } else {
            generateButton
        }
    }

    private var generateButton: some View {
        Button {
            model.run()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "arrow.up")
                    .font(.callout.weight(.semibold))
                    .accessibilityHidden(true)
                Text("Generate")
                    .font(.callout.weight(.semibold))
                Text(shortcutHint)
                    .font(.caption.weight(.medium))
                    .opacity(0.78)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 18)
            .frame(minWidth: 124, minHeight: controlHeight)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(model.canRun ? Color.white : Color.primary.opacity(0.55))
        .background {
            Capsule()
                .fill(model.canRun
                      ? TurboFieldfareMacTheme.accentColor
                      : Color.primary.opacity(0.08))
                .overlay {
                    Capsule().stroke(
                        model.canRun
                            ? Color.white.opacity(0.16)
                            : TurboFieldfareMacTheme.fieldBorder,
                        lineWidth: 1)
                }
        }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!model.canRun)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Generate")
        .accessibilityValue(shortcutHint)
        .accessibilityHint(generateAccessibilityHint)
        .accessibilityAddTraits(.isButton)
        .help(generateHelp)
    }

    private var runningPill: some View {
        Button {
            model.cancel()
        } label: {
            HStack(spacing: 10) {
                if model.isCancellationPending {
                    Text("Stopping")
                        .font(.callout.weight(.medium))
                } else if model.phase == .prefill {
                    Text(model.presentation.label)
                        .font(.callout.weight(.medium))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                } else {
                    Text("\(MetricFormat.rate(model.liveTokensPerSecond)) tok/s")
                        .font(.callout.weight(.semibold))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                }
                Label("Stop generation", systemImage: "stop.fill")
                    .labelStyle(.iconOnly)
                    .font(.callout)
                    .frame(width: 28, height: 28)
            }
            .padding(.leading, 18)
            .padding(.trailing, 4)
            .frame(minWidth: 140, minHeight: controlHeight)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(TurboFieldfareMacTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.cancelAction)
        .disabled(!model.canCancel)
        .accessibilityLabel(model.isCancellationPending ? "Stopping generation" : "Stop generation")
        .help("Stop generation")
        .animation(.smooth(duration: 0.2), value: model.presentation.label)
    }

    private var shortcutHint: String {
        switch model.newlineShortcut {
        case .return: return "⌘↩"
        case .shiftReturn: return "↩"
        }
    }

    private var generateHelp: String {
        if let blocked = generateBlockedReason {
            return blocked
        }
        switch model.newlineShortcut {
        case .return:
            return "Generate a response (⌘↩)"
        case .shiftReturn:
            return "Generate a response (Return). Use Shift-Return for a new line."
        }
    }

    private var generateAccessibilityHint: String {
        generateHelp
    }

    private var generateBlockedReason: String? {
        guard !model.canRun else { return nil }
        // Prefer load/runtime blockers over empty-prompt so the tip matches the HUD.
        if model.loadState.isLoading {
            return "Wait for the model to finish loading"
        }
        if !model.isModelAvailable {
            return "Load the model before generating"
        }
        if model.hasStaleLoadedRuntime {
            return "Reload the model to apply changed settings"
        }
        if model.promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Enter a prompt to generate"
        }
        return "Generate is unavailable right now"
    }
}
