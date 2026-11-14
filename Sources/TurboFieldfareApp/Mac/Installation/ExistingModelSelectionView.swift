import SwiftUI
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation

struct ExistingModelSelectionView: View {
    let model: AppModel
    let review: ExistingModelReview
    @StoredState private var selectedID: String?

    init(model: AppModel, review: ExistingModelReview) {
        self.model = model
        self.review = review
        _selectedID = StoredState(wrappedValue: review.initialSelection)
    }

    private var selected: ExistingModelCandidate? {
        review.candidates.first { $0.id == selectedID }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Choose a model").font(.title2.bold())
            Text(review.folder.path)
                .font(.caption).foregroundStyle(.secondary)
                .lineLimit(2).truncationMode(.middle).help(review.folder.path)
            ScrollView {
                Picker("Text model", selection: $selectedID) {
                    ForEach(Array(review.candidates.enumerated()), id: \.element.id) { index, candidate in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(candidate.textURL.lastPathComponent).font(.headline)
                            Text(candidate.textURL.deletingLastPathComponent().path)
                                .font(.caption).foregroundStyle(.secondary)
                                .lineLimit(1).truncationMode(.middle).help(candidate.textURL.path)
                            Text(candidate.canSelect ? candidate.visionSummary : candidate.textSummary)
                                .font(.caption)
                                .foregroundStyle(candidate.canSelect ? Color.secondary : Color.red)
                        }
                        .tag(Optional(candidate.id))
                        .disabled(!candidate.canSelect)
                        .accessibilityIdentifier(AccessibilityID.Prefix.modelCandidate.rawValue + String(index))
                    }
                }
                .pickerStyle(.radioGroup)
                .accessibilityIdentifier(.modelReviewPicker)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: min(CGFloat(review.candidates.count) * 94, 200))
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if let selected {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Text model").font(.headline)
                            Text(selected.textURL.path).textSelection(.enabled)
                            Text("Vision pack").font(.headline)
                            Text(selected.visionURL.path).textSelection(.enabled)
                            Text(selected.visionSummary)
                                .foregroundStyle(selected.visionStatus == .complete ? Color.secondary : Color.orange)
                        }
                        .font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                    } else {
                        Text("Select a text model above. No model will be chosen automatically.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    if let message = model.modelLocationDiagnostic {
                        Text(message).font(.caption).foregroundStyle(.red)
                    }
                    if !model.canChangeModelLocation {
                        Text("Wait for the current model operation to finish before confirming.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 180)
            HStack {
                Spacer()
                Button("Cancel") { model.existingModelReview = nil }
                    .keyboardShortcut(.cancelAction)
                    .disabled(model.isChangingModelLocation)
                    .accessibilityIdentifier(.modelReviewCancel)
                Button("Use Model") {
                    guard let selected else { return }
                    Task { await model.confirmExistingModelSelection(selected) }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(selected?.canSelect != true || !model.canChangeModelLocation)
                .accessibilityIdentifier(.modelReviewConfirm)
            }
        }
        .padding(24)
        .frame(width: 520)
        .tint(TurboFieldfareMacTheme.accentColor)
        .interactiveDismissDisabled(model.isChangingModelLocation)
    }
}
