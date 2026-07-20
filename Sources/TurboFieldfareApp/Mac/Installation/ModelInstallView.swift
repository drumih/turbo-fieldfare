import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ModelInstallView: View {
    let model: AppModel

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                identity
                storageCard
                progressArea
                actions
            }
            .frame(maxWidth: 560)
            .padding(.horizontal, 28)
            .padding(.vertical, 48)
            .frame(maxWidth: .infinity)
        }
    }

    private var identity: some View {
        VStack(spacing: 10) {
            Image(systemName: "externaldrive.badge.plus")
                .font(.system(.largeTitle, design: .rounded))
                .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                .accessibilityHidden(true)
            Text("Model required")
                .font(.title.bold())
                .accessibilityHeading(.h1)
            Text("TurboFieldfare needs \(model.installDescriptor.displayName) before it can generate text.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }

    private var storageCard: some View {
        VStack(spacing: 12) {
            if let requirement = model.installRequirement {
                StorageRow(label: "Space required",
                           value: MetricFormat.storage(requirement.requiredBytes))
                StorageRow(label: "Available on this Mac",
                           value: MetricFormat.storage(requirement.availableBytes))
                capacityStatus(requirement)
            } else if case .failed(let message) = model.installReadiness {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(message).foregroundStyle(.secondary)
                    Spacer()
                }
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(readinessLabel)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
            Text(model.modelPathText)
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(2)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(18)
        .background {
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    @ViewBuilder
    private func capacityStatus(_ requirement: AppModelInstallRequirement) -> some View {
        if requirement.canInstall {
            Label("Enough space to install", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Label("\(MetricFormat.storage(requirement.shortfallBytes)) more is required",
                  systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private var progressArea: some View {
        if model.isInstallingModel {
            VStack(spacing: 10) {
                if let fraction = model.installProgressFraction,
                   let downloaded = model.installDownloadedBytes,
                   let total = model.installTotalBytes {
                    ProgressView(value: fraction)
                        .accessibilityLabel("Model download")
                        .accessibilityValue(Text(MetricFormat.percent(fraction * 100)))
                    HStack {
                        Text("Downloaded \(MetricFormat.storage(downloaded)) of \(MetricFormat.storage(total))")
                        Spacer()
                        Text(MetricFormat.percent(fraction * 100))
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                } else {
                    Text(model.presentation.label)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
        } else if case .cancelled = model.installState {
            Label("Installation cancelled", systemImage: "xmark.circle")
                .foregroundStyle(.secondary)
        } else if case .failed(let message) = model.installState {
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
        }
    }

    private var actions: some View {
        HStack(spacing: 12) {
            if model.isInstallingModel {
                Button("Cancel", action: model.cancelInstall)
                    .buttonStyle(.bordered)
                    .keyboardShortcut(.cancelAction)
                    .disabled(!model.canCancelInstall)
            } else {
                Button("Check Again", action: model.recheckModelAtCurrentLocation)
                .buttonStyle(.bordered)
                .disabled(model.isInstallingModel)

                Button("Install", action: model.installModel)
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canInstallModel)
            }
        }
        .controlSize(.large)
    }

    private var readinessLabel: String {
        switch model.installReadiness {
        case .checking:
            return "Checking available space"
        case .failed(let message):
            return message
        case .ready, .insufficientSpace:
            return "Checking available space"
        }
    }
}

private struct StorageRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(value)
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
        }
    }
}
