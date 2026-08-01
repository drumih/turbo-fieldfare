import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

/// Shows how much of the context window the conversation already occupies, so
/// filling it is never a surprise. Hidden while no model is resident: "0 used"
/// would be a claim we cannot make before the runtime reports anything.
struct ContextMeterView: View {
    let model: AppModel

    var body: some View {
        if model.loadState.isReady || model.isRunning {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Text("Context")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(usageText)
                        .monospacedDigit()
                        .foregroundStyle(color)
                }
                .font(.caption)

                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(.quaternary)
                        Capsule()
                            .fill(color)
                            .frame(width: max(0, geometry.size.width * fraction))
                    }
                }
                .frame(height: 4)
            }
            .help(model.isContextUsageEstimated
                  ? "Estimated until the first reply reports an exact count"
                  : "Exact prompt size reported by the last reply")
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Context usage")
            .accessibilityValue(usageText)
        }
    }

    private var fraction: Double {
        guard model.contextCapacityTokens > 0 else { return 0 }
        return min(1, Double(model.contextUsedTokens)
                   / Double(model.contextCapacityTokens))
    }

    private var usageText: String {
        let prefix = model.isContextUsageEstimated && model.contextUsedTokens > 0
            ? "~" : ""
        return "\(prefix)\(compact(model.contextUsedTokens))"
            + " / \(compact(model.contextCapacityTokens))"
    }

    private var color: Color {
        switch fraction {
        case ..<0.7: return .secondary
        case ..<0.9: return .orange
        default: return .red
        }
    }

    private func compact(_ tokens: Int) -> String {
        guard tokens >= 1_000 else { return "\(tokens)" }
        let thousands = Double(tokens) / 1_024
        return thousands < 10
            ? String(format: "%.1fK", thousands)
            : String(format: "%.0fK", thousands)
    }
}
