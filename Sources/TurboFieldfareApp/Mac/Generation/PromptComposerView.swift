import TurboFieldfareAppCore
import SwiftUI

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            editor
            footer
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    private var editor: some View {
        TextEditor(text: $model.promptText)
            .accessibilityLabel("Prompt")
            .font(.body)
            .scrollContentBackground(.hidden)
            .focused($promptFocused)
            .frame(height: editorHeight)
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    // Matches the NSTextView text origin: 5pt line fragment
                    // padding, no vertical inset.
                    Text("Prompt")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
    }

    private var editorHeight: CGFloat {
        model.promptText.isEmpty ? 46 : 84
    }

    private var footer: some View {
        HStack(spacing: 10) {
            promptTips
            Spacer()
            clearAction
            GenerateControl(model: model)
        }
    }

    private var promptTips: some View {
        Button {
            showingPromptTips.toggle()
        } label: {
            Label("Prompt tips", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Prompt tips")
        .popover(isPresented: $showingPromptTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            promptGuide
        }
    }

    private var promptGuide: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting this model")
                .font(.headline)

            tipSection("It continues text",
                       "This is a base completion model, not a chat assistant. Write the beginning of the output you want, such as “A concise explanation:” or a function signature.")
            tipSection("If it repeats or runs on",
                       "Keep Temperature at 1.0, retry, shorten Max response, or enable Repetition penalty.")
            tipSection("If it drifts or invents facts",
                       "Put required facts, constraints, and the desired structure directly in the prompt. Keep the requested continuation focused and verify factual output.")
            tipSection("For code or exact formats",
                       "Include the signature, schema, or a short example. Lower temperature can reduce variation, but it may increase repetition and does not guarantee correct code.")
            tipSection("If it does not answer",
                       "Turn the question into an answer prefix. For example, use “The reason this happens is” instead of “Why does this happen?”")
        }
        .font(.callout)
        .frame(width: 390, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .fontWeight(.semibold)
            Text(detail)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label("Clear output", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear output")
        } else if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused = true
            } label: {
                Label("Clear prompt", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear prompt")
        }
    }
}
