import TurboFieldfareAppCore
import SwiftUI

struct ModelStatusBadge: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 8) {
            statusDot
            Text("Gemma 4 26B")
                .font(.callout.weight(.semibold))
                .lineLimit(1)
                .help(model.installDescriptor.repoID)
                .accessibilityLabel("Model")
                .accessibilityValue(model.installDescriptor.repoID)
        }
    }

    @ViewBuilder
    private var statusDot: some View {
        switch model.presentation.severity {
        case .neutral: dot(.gray)
        case .active, .warning: dot(.orange)
        case .success: dot(.green)
        case .error: dot(.red)
        }
    }

    private func dot(_ color: Color) -> some View {
        ZStack {
            Circle()
                .fill(color.opacity(0.22))
                .frame(width: 14, height: 14)
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
        }
        .accessibilityHidden(true)
    }
}
