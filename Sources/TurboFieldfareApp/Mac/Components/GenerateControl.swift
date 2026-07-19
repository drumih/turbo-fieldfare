import TurboFieldfareAppCore
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
            Label("Generate", systemImage: "arrow.up")
                .font(.callout.weight(.semibold))
                .padding(.horizontal, 24)
                .frame(minWidth: 124, minHeight: controlHeight)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(.indigo, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!model.canRun)
        .opacity(model.canRun ? 1 : 0.62)
    }

    private var runningPill: some View {
        HStack(spacing: 10) {
            if model.isCancellationPending {
                Text("Stopping")
                    .font(.callout.weight(.medium))
            } else if model.phase == .prefill {
                Text(model.presentation.label)
                    .font(.callout.weight(.medium))
                    .monospacedDigit()
            } else {
                Text("\(MetricFormat.rate(model.liveTokensPerSecond)) tok/s")
                    .font(.callout.weight(.semibold))
                    .monospacedDigit()
            }
            Button {
                model.cancel()
            } label: {
                Label("Stop generation", systemImage: "stop.fill")
                    .labelStyle(.iconOnly)
                    .font(.callout)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)
            .disabled(!model.canCancel)
            .help("Stop generation")
        }
        .padding(.leading, 18)
        .padding(.trailing, 4)
        .frame(minWidth: 140, minHeight: controlHeight)
        .contentShape(Capsule())
        .foregroundStyle(.white)
        .background(.indigo, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
    }
}
