import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ErrorBanner: View {
    @Bindable var model: AppModel

    var body: some View {
        if let error = model.error,
           GenericErrorBannerPolicy.shouldShow(error: error, loadState: model.loadState) {
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
                    Image(systemName: "xmark")
                        .font(.caption.weight(.semibold))
                        .frame(width: 28, height: 28)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Dismiss error")
                .help("Dismiss error")
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
