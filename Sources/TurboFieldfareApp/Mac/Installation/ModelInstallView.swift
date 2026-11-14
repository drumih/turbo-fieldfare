import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ModelInstallView: View {
    let model: AppModel
    @StoredState private var showingDiscardConfirmation = false

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                identity
                ModelLocationView(model: model)
                actions
                VStack(alignment: .leading, spacing: 8) {
                    ModelLocationNotices(model: model, showsVolumeWarning: false)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                progressArea
                storageCard
            }
            .frame(maxWidth: 560)
            .padding(.horizontal, 28)
            .padding(.vertical, 48)
            .frame(maxWidth: .infinity)
        }
        .defaultScrollAnchor(.top, for: .sizeChanges)
        .confirmationDialog(
            "Discard the saved model download?",
            isPresented: $showingDiscardConfirmation,
            titleVisibility: .visible
        ) {
            Button("Discard Download", role: .destructive) {
                model.discardModelDownload()
            }
            Button("Keep Download", role: .cancel) {}
        } message: {
            Text("Downloaded ranges will be removed. The installed model, if any, is preserved.")
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
                StorageRow(label: "Available on selected volume",
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
            HStack {
                Button("Check Again", action: model.recheckModelAtCurrentLocation)
                    .buttonStyle(.borderless)
                    .disabled(model.isInstallingModel)
                    .accessibilityIdentifier(.installCheckAgain)
                Spacer()
                if model.hasPartialModelDownload && !model.isInstallingModel {
                    Button("Discard Download", role: .destructive) {
                        showingDiscardConfirmation = true
                    }
                    .buttonStyle(.borderless)
                    .disabled(!model.canDiscardModelDownload)
                    .accessibilityIdentifier(.installDiscard)
                }
            }
            .controlSize(.small)
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
            VStack(alignment: .leading, spacing: 8) {
                if let fraction = model.installProgressFraction,
                   let downloaded = model.installDownloadedBytes,
                   let total = model.installTotalBytes {
                    ProgressView(value: fraction)
                        .accessibilityLabel("Model download")
                        .accessibilityValue(Text(accessibleProgressValue(fraction: fraction)))
                    HStack {
                        Text("Downloaded \(MetricFormat.storage(downloaded)) of \(MetricFormat.storage(total))")
                        Spacer()
                        Text(MetricFormat.percent(fraction * 100))
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    HStack(alignment: .firstTextBaseline, spacing: 16) {
                        if let reused = model.installReusedBytes, reused > 0 {
                            Text("Reused \(MetricFormat.storage(reused)) from the saved download")
                                .font(.caption)
                        }
                        Spacer(minLength: 16)
                        if let eta = model.installETAText {
                            Text(eta)
                                .font(.caption.monospacedDigit())
                        }
                    }
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                } else {
                    Text(model.presentation.label)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
        } else if case .cancelled = model.installState {
            Label("Download paused", systemImage: "pause.circle")
                .foregroundStyle(.secondary)
        } else if case .failed(let message) = model.installState {
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
        } else if case .recoverable(let message) = model.installState {
            Label(message, systemImage: "exclamationmark.triangle")
                .font(.callout)
                .foregroundStyle(.orange)
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
                    .accessibilityIdentifier(.installCancel)
            } else {
                Button(model.hasPartialModelDownload ? "Resume Download" : "Download Model",
                       action: model.installModel)
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canInstallModel)
                    .accessibilityIdentifier(.installDownload)
            }
            Button("Use Existing Model…") {
                ModelLocationPanels.choose(model: model, intent: .existing)
            }
            .buttonStyle(.bordered)
            .tint(.primary)
            .disabled(!model.canChangeModelLocation)
            .accessibilityIdentifier(.modelUseExisting)
        }
        .controlSize(.large)
    }

    private var readinessLabel: String {
        switch model.installReadiness {
        case .installed:
            return "Installed"
        case .checking:
            return "Checking available space"
        case .failed(let message):
            return message
        case .ready, .insufficientSpace:
            return "Checking available space"
        }
    }

    private func accessibleProgressValue(fraction: Double) -> String {
        let percent = MetricFormat.percent(fraction * 100)
        guard let eta = model.installETAText else { return percent }
        return "\(percent), \(eta)"
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
