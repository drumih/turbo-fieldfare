import SwiftUI
import TurboFieldfareAppCore

/// Adds a user-supplied Hugging Face repository.
///
/// The architecture pre-flight runs before anything is downloaded, so pasting a
/// model this build cannot execute costs seconds rather than a multi-gigabyte
/// transfer. That is the common case: most open MoE checkpoints are not Gemma.
struct AddCustomModelSheet: View {
    @Bindable var model: AppModel
    @Binding var isPresented: Bool

    @State private var repoID = ""
    @State private var revision = "main"
    @State private var token = ""
    @State private var stage: Stage = .entry
    @State private var errorText: String?

    private enum Stage: Equatable {
        case entry
        case checking
        case consent(facts: ModelSourceProbe.SourceFacts)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Add Custom Model").font(.headline)

            switch stage {
            case .entry, .checking:
                entryForm
            case .consent(let facts):
                consentPanel(facts: facts)
            }

            if let errorText {
                Text(errorText)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Button("Cancel") { isPresented = false }
                Spacer()
                primaryButton
            }
        }
        .padding(20)
        .frame(width: 480)
    }

    private var entryForm: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("owner/model", text: $repoID)
                .textFieldStyle(.roundedBorder)
            TextField("Revision (branch, tag, or commit)", text: $revision)
                .textFieldStyle(.roundedBorder)
            TextField("Hugging Face token (only for gated repos)", text: $token)
                .textFieldStyle(.roundedBorder)
            Text("Only Gemma 4 26B-A4B and its finetunes can run in this build.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func consentPanel(facts: ModelSourceProbe.SourceFacts) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Architecture supported", systemImage: "checkmark.seal")
                .foregroundStyle(.green)
            LabeledContent("Repository", value: repoID)
            LabeledContent("Revision", value: revision)
            LabeledContent("Index SHA-256", value: String(facts.indexSHA256.prefix(16)) + "…")
            Text("This model is not verified by the project. Its contents will be "
                + "recorded now, and you will be warned if the repository changes later.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var primaryButton: some View {
        switch stage {
        case .entry:
            Button("Check Model") { Task { await check() } }
                .buttonStyle(.borderedProminent)
                .disabled(repoID.isEmpty || revision.isEmpty)
        case .checking:
            ProgressView().controlSize(.small)
        case .consent(let facts):
            Button("Add Model") { add(facts: facts) }
                .buttonStyle(.borderedProminent)
        }
    }

    private func check() async {
        errorText = nil
        let trimmed = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        // Validated before any request so a hostile ID never reaches a URL or a
        // directory name.
        do {
            _ = try ModelSlug.make(repoID: trimmed)
        } catch let failure as ModelSlug.InvalidRepositoryID {
            errorText = failure.reason
            return
        } catch {
            errorText = "\(error)"
            return
        }
        repoID = trimmed
        stage = .checking
        do {
            let facts = try await ModelSourceProbe.probe(
                repoID: trimmed,
                revision: revision,
                token: token.isEmpty ? nil : token)
            switch facts.architecture {
            case .supported:
                stage = .consent(facts: facts)
            case .unsupported(let modelType):
                stage = .entry
                errorText = "This model's architecture is \"\(modelType)\", which this "
                    + "build does not support. Only Gemma 4 26B-A4B and its finetunes can run. "
                    + "Nothing was downloaded."
            }
        } catch let failure as ModelSourceProbe.ProbeFailure {
            stage = .entry
            errorText = failure.reason
        } catch {
            stage = .entry
            errorText = "Could not reach Hugging Face: \(error.localizedDescription)"
        }
    }

    private func add(facts: ModelSourceProbe.SourceFacts) {
        let curated = ModelCatalog.curated.first
        let entry = ModelCatalogEntry(
            displayName: repoID.split(separator: "/").last.map(String.init) ?? repoID,
            repoID: repoID,
            revision: revision,
            trustTier: .custom,
            recordedIndexSHA256: facts.indexSHA256,
            // Size estimates borrow the curated model's figures: a finetune of
            // the same architecture has the same tensor shapes, and the real
            // numbers arrive from the repacker once the install starts.
            approximateDownloadBytes: curated?.approximateDownloadBytes ?? 0,
            installedBytes: curated?.installedBytes ?? 0,
            reserveBytes: curated?.reserveBytes ?? 1_073_741_824)
        guard model.addCustomModel(entry) else {
            errorText = "\(repoID) is already in the catalog."
            return
        }
        isPresented = false
    }
}
