import SwiftUI
import TurboFieldfareAppCore

struct ModelPickerView: View {
    @Bindable var model: AppModel
    @State private var isAddingCustomModel = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ViewThatFits(in: .horizontal) {
                HStack {
                    Text("Models").font(.headline)
                    Spacer(minLength: 8)
                    Button("Add Custom Model…") { isAddingCustomModel = true }
                        .controlSize(.small)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("Models").font(.headline)
                    Button("Add Custom Model…") { isAddingCustomModel = true }
                        .controlSize(.small)
                }
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

    /// Laid out vertically on purpose. This row also lives in the inspector
    /// sidebar, which is far too narrow for a name, a badge, and two buttons on
    /// one line — a horizontal layout there compresses the title column until
    /// it wraps one character per line.
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(entry.displayName)
                    .font(.body.weight(.medium))
                    .fixedSize(horizontal: false, vertical: true)
                TrustBadge(tier: entry.trustTier)
                if isLoaded {
                    Text("Loaded")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .fixedSize()
                }
                Spacer(minLength: 0)
            }
            Text(entry.repoID)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(entry.repoID)
            statusLine
            HStack(spacing: 8) {
                Spacer(minLength: 0)
                actions
            }
            .padding(.top, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 6)
        .background(isSelected ? Color.accentColor.opacity(0.06) : .clear)
    }

    @ViewBuilder private var statusLine: some View {
        switch installState {
        case .idle, .cancelled:
            statusText("Not installed \u{00B7} \(formattedBytes(entry.approximateDownloadBytes)) download")
        case .installed:
            statusText("Installed \u{00B7} \(formattedBytes(entry.installedBytes)) on disk")
        case .failed(let message):
            statusText(message, tint: .red)
        case .recoverable(let message):
            statusText("Interrupted \u{00B7} \(message)", tint: .orange)
        case .copyingPayload(let reused, let downloaded, let total):
            ProgressView(value: Double(reused + downloaded), total: Double(max(total, 1)))
                .controlSize(.small)
                .frame(maxWidth: .infinity)
        default:
            ProgressView().controlSize(.small)
        }
    }

    private func statusText(_ text: String, tint: Color = .secondary) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(tint)
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
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
            // Without this the capsule is squeezed to a single glyph column in
            // a narrow sidebar and the label reads vertically.
            .fixedSize()
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
