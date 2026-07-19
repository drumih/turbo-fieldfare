import TurboFieldfareAppCore
import SwiftUI

struct ErrorBanner: View {
    @Bindable var model: AppModel

    var body: some View {
        if let error = model.error, error != .cancelled {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(error.userMessage)
                    .font(.callout)
                    .lineLimit(2)
                Spacer(minLength: 8)
                Button {
                    model.error = nil
                } label: {
                    Label("Dismiss error", systemImage: "xmark")
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
                        Capsule().stroke(.red.opacity(0.55), lineWidth: 1)
                    }
            }
            .help(error.technicalDetail)
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }
}
