import TurboFieldfareAppCore
import SwiftUI

struct ModelStatusBadge: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
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
        Circle().fill(color).frame(width: 8, height: 8).accessibilityHidden(true)
    }
}
