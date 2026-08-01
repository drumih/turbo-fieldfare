import TurboFieldfareAppCore
import SwiftUI

/// Deliberately not routed through `model.error`: a full conversation is an
/// expected limit with its own remedies, not a failed run, and conflating the
/// two would let dismissing one clear the other.
struct ContextOverflowBanner: View {
    let model: AppModel

    var body: some View {
        if model.isContextOverflowNoticeVisible {
            HStack(spacing: 8) {
                Image(systemName: "gauge.with.dots.needle.100percent")
                    .foregroundStyle(.orange)
                Text(message)
                    .font(.callout)
                    .lineLimit(2)
                Spacer(minLength: 8)
                Button("New Chat") { model.newConversation() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                if let next = nextContextOption {
                    Button("Use \(next.shortLabel)") {
                        model.maxContextTokens = next.tokens
                        model.dismissContextOverflowNotice()
                    }
                    .controlSize(.small)
                    .help("Raising the context length requires reloading the model")
                }
                Button {
                    model.dismissContextOverflowNotice()
                } label: {
                    Label("Dismiss", systemImage: "xmark")
                        .labelStyle(.iconOnly)
                        .font(.caption.weight(.semibold))
                        .frame(width: 28, height: 28)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background {
                Capsule()
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .overlay {
                        Capsule().stroke(.orange.opacity(0.55), lineWidth: 1)
                    }
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    private var message: String {
        "This chat has filled the \(shortLabel(model.contextCapacityTokens)) context — not enough room left for a reply."
    }

    private var contextOptions: [AppContextOption] {
        AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: AppMemoryBudget.residentWeightEstimateBytes,
            installedRAMBytes: AppMemoryBudget.installedRAMBytes)
    }

    /// Only offers a length the machine's budget can actually hold — suggesting
    /// one that would be greyed out in the inspector would send the user to a
    /// control they cannot use.
    private var nextContextOption: AppContextOption? {
        contextOptions.first { $0.tokens > model.maxContextTokens && $0.isEnabled }
    }

    private func shortLabel(_ tokens: Int) -> String {
        contextOptions.first { $0.tokens == tokens }?.shortLabel ?? "\(tokens)"
    }
}
