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
            Image(systemName: "arrow.up")
                .font(.callout.weight(.bold))
                .frame(width: controlHeight, height: controlHeight)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(TurboFieldfareMacTheme.accentColor, in: Circle())
        .overlay {
            Circle().stroke(.white.opacity(0.18), lineWidth: 0.5)
        }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!model.canRun)
        .opacity(model.canRun ? 1 : 0.62)
        .accessibilityLabel("Generate")
        .help("Generate response (⌘↵)")
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
            .padding(.leading, 15)
            .padding(.trailing, 4)
            .frame(minWidth: 132, minHeight: controlHeight)
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
        .help("Stop generation")
        .animation(.smooth(duration: 0.2), value: model.presentation.label)
    }
}
