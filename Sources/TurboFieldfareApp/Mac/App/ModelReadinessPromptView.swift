import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ModelReadinessPromptView: View {
    let model: AppModel

    var body: some View {
        if shouldShow {
            HStack(spacing: 12) {
                Image(systemName: model.loadState.isFailed
                      ? "exclamationmark.triangle"
                      : "cube.transparent")
                    .font(.title3)
                    .foregroundStyle(model.loadState.isFailed ? .orange : .secondary)

                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.callout.weight(.semibold))
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer(minLength: 12)

                action
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .stroke(.separator.opacity(0.55), lineWidth: 0.5)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel(title)
            .accessibilityValue(detail)
        }
    }

    private var shouldShow: Bool {
        !model.loadState.isReady || model.hasStaleLoadedRuntime
    }

    private var title: String {
        if model.hasStaleLoadedRuntime { return "Reload the model to continue" }
        if model.loadState.isFailed { return "The model could not be loaded" }
        if model.loadState.isLoading { return "Loading model" }
        return "Load the model to start chatting"
    }

    private var detail: String {
        if let presentationDetail = model.presentation.detail,
           model.loadState.isFailed {
            return presentationDetail
        }
        if model.hasStaleLoadedRuntime {
            return "This chat is ready, but the current runtime settings need a reload."
        }
        if model.loadState.isLoading {
            return model.presentation.label
        }
        return "Your conversation is safe. Load Gemma 4 to generate the next response."
    }

    @ViewBuilder
    private var action: some View {
        if model.loadState.isLoading {
            ProgressView()
                .controlSize(.small)
        } else if model.hasStaleLoadedRuntime, model.canReloadModel {
            Button("Reload Model", action: model.reloadModel)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        } else if model.canLoadModel {
            Button(model.loadState.isFailed ? "Retry Load" : "Load Model",
                   action: model.loadModel)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        }
    }
}
