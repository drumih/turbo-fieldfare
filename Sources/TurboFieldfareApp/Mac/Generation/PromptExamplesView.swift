import TurboFieldfareAppCore
import SwiftUI

struct PromptExamplesView: View {
    let select: (AppPromptPreset) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Try an example")
                        .font(.headline)
                    Text("Choose a prompt, edit it, or write your own.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                moreExamples
            }

            LazyVGrid(columns: columns, alignment: .leading, spacing: 10) {
                primaryCards
            }
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 18)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .transition(.opacity.combined(with: .move(edge: .bottom)))
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private var primaryCards: some View {
        ForEach(AppPromptPreset.primary) { preset in
            Button {
                select(preset)
            } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(preset.title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                    Text(preset.prompt)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                        .lineLimit(3)
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity, minHeight: 96, alignment: .topLeading)
                .padding(10)
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .background(.quaternary.opacity(0.25), in: .rect(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(.separator.opacity(0.35), lineWidth: 0.5)
            }
            .accessibilityLabel(preset.title)
            .accessibilityValue(preset.prompt)
            .accessibilityHint("Copies this prompt into the prompt editor")
        }
    }

    private var columns: [GridItem] {
        Array(repeating: GridItem(.flexible(minimum: 0), spacing: 10, alignment: .top), count: 3)
    }

    private var moreExamples: some View {
        Menu("More examples") {
            ForEach(AppPromptPreset.secondary) { preset in
                Button {
                    select(preset)
                } label: {
                    VStack(alignment: .leading) {
                        Text(preset.title)
                        Text(preset.prompt)
                    }
                }
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .accessibilityHint("Shows additional prompts")
    }
}
