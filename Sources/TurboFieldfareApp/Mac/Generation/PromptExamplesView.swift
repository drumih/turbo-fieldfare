import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct PromptExamplesView: View {
    let select: (AppPromptPreset) -> Void
    @State private var hoveredPresetID: String?

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
                .fill(TurboFieldfareMacTheme.elevatedSurface.opacity(0.7))
                .overlay {
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(TurboFieldfareMacTheme.cardBorder, lineWidth: 0.5)
                }
        }
        .transition(.opacity.combined(with: .move(edge: .bottom)))
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private var primaryCards: some View {
        ForEach(AppPromptPreset.primary) { preset in
            let isHovered = hoveredPresetID == preset.id
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
            .background(
                isHovered
                    ? TurboFieldfareMacTheme.hoverSurface
                    : TurboFieldfareMacTheme.surface.opacity(0.45),
                in: .rect(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(
                        isHovered
                            ? TurboFieldfareMacTheme.accentColor.opacity(0.34)
                            : TurboFieldfareMacTheme.border,
                        lineWidth: isHovered ? 1 : 0.5)
            }
            .scaleEffect(isHovered ? 1.012 : 1)
            .shadow(
                color: isHovered ? TurboFieldfareMacTheme.accentColor.opacity(0.08) : .clear,
                radius: 8,
                y: 3)
            .onHover { isHovering in
                hoveredPresetID = isHovering ? preset.id : nil
            }
            .animation(.easeOut(duration: 0.14), value: isHovered)
            .accessibilityLabel(preset.title)
            .accessibilityValue(preset.prompt)
            .accessibilityHint("Copies this prompt into the prompt editor")
        }
    }

    private var columns: [GridItem] {
        [GridItem(.adaptive(minimum: 170), spacing: 10, alignment: .top)]
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
