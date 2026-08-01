import SwiftUI
import TurboFieldfareAppCore

struct ModelPickerView: View {
    @Bindable var model: AppModel
    @State private var isAddingCustomModel = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Models").font(.headline)
                Spacer()
                Button("Add Custom Model…") { isAddingCustomModel = true }
                    .controlSize(.small)
            }
            ForEach(model.catalog.entries) { entry in
                ModelPickerRow(
                    entry: entry,
                    installState: model.installStates.state(for: entry.repoID),
                    isSelected: model.selectedRepoID == entry.repoID,
                    isLoaded: model.selectedRepoID == entry.repoID && model.loadState.isReady,
                    onDownload: { model.startInstall(for: entry) },
                    onCancel: { model.cancelInstall() },
                    onLoad: { model.switchModel(to: entry) },
                    onDelete: { model.deleteInstall(for: entry) })
                if entry.id != model.catalog.entries.last?.id {
                    Divider()
                }
            }
        }
        .padding()
        .sheet(isPresented: $isAddingCustomModel) {
            AddCustomModelSheet(model: model, isPresented: $isAddingCustomModel)
        }
    }
}

private struct ModelPickerRow: View {
    let entry: ModelCatalogEntry
    let installState: AppModelInstallState
    let isSelected: Bool
    let isLoaded: Bool
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onLoad: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(entry.displayName).font(.body.weight(.medium))
                    TrustBadge(tier: entry.trustTier)
                    if isLoaded {
                        Text("Loaded")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                }
                Text(entry.repoID)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                statusLine
            }
            Spacer(minLength: 8)
            actions
        }
        .padding(.vertical, 6)
        .background(isSelected ? Color.accentColor.opacity(0.06) : .clear)
    }

    @ViewBuilder private var statusLine: some View {
        switch installState {
        case .idle, .cancelled:
            Text("Not installed \u{00B7} \(formattedBytes(entry.approximateDownloadBytes)) download")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .installed:
            Text("Installed \u{00B7} \(formattedBytes(entry.installedBytes)) on disk")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(.red).lineLimit(2)
        case .recoverable(let message):
            Text("Interrupted \u{00B7} \(message)")
                .font(.caption)
                .foregroundStyle(.orange)
                .lineLimit(2)
        case .copyingPayload(let reused, let downloaded, let total):
            ProgressView(value: Double(reused + downloaded), total: Double(max(total, 1)))
                .controlSize(.small)
                .frame(maxWidth: 220)
        default:
            ProgressView().controlSize(.small)
        }
    }

    @ViewBuilder private var actions: some View {
        switch installState {
        case .installed:
            Button("Load", action: onLoad).disabled(isLoaded)
            Button("Delete", action: onDelete)
                .disabled(isLoaded)
                .help(isLoaded ? "Unload this model before deleting it." : "")
        case .idle, .cancelled, .failed:
            Button("Download", action: onDownload)
        case .recoverable:
            Button("Resume", action: onDownload)
        default:
            Button("Cancel", action: onCancel)
        }
    }

    private func formattedBytes(_ bytes: UInt64) -> String {
        let gigabytes = Double(bytes) / (1_024 * 1_024 * 1_024)
        return String(format: "%.1f GB", gigabytes)
    }
}

private struct TrustBadge: View {
    let tier: ModelTrustTier

    var body: some View {
        Text(tier == .curated ? "Verified" : "Unverified")
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tier == .curated
                ? Color.green.opacity(0.15)
                : Color.orange.opacity(0.15))
            .clipShape(Capsule())
            .help(tier == .curated
                ? "Pinned and verified by the project."
                : "Added by you. Contents are not verified by the project.")
    }
}
