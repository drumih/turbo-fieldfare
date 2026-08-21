import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text("Prompt")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                if promptFocused {
                    Text(shortcutCaption)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .accessibilityHidden(true)
                }
            }
            editor
            footer
        }
        .padding(14)
        .turboElevatedCard(cornerRadius: 22)
        .onChange(of: model.loadState.isReady) { _, isReady in
            if isReady && model.promptText.isEmpty {
                promptFocused = true
            }
        }
    }

    private var editor: some View {
        TextEditor(text: $model.promptText)
            .accessibilityLabel("Prompt")
            .font(.body)
            .scrollContentBackground(.hidden)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .focused($promptFocused)
            .onKeyPress(.return, phases: [.down, .repeat]) { keyPress in
                switch PromptSubmissionPolicy.decision(
                    newlineShortcut: model.newlineShortcut,
                    modifiers: keyPress.modifiers,
                    canRun: model.canRun,
                    hasMarkedText: promptHasMarkedText,
                    isRepeat: keyPress.phase.contains(.repeat)) {
                case .submit:
                    model.run()
                    return .handled
                case .consume:
                    return .handled
                case .deferToEditor:
                    return .ignored
                }
            }
            .frame(minHeight: editorHeight, maxHeight: 160)
            .frame(height: editorHeight)
            .background {
                RoundedRectangle(cornerRadius: 12)
                    .fill(TurboFieldfareMacTheme.fieldSurface)
                    .overlay {
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(
                                promptFocused
                                    ? TurboFieldfareMacTheme.fieldBorderFocused
                                    : TurboFieldfareMacTheme.fieldBorder,
                                lineWidth: promptFocused ? 1.5 : 1)
                    }
            }
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    Text("Ask anything…")
                        .font(.body)
                        .foregroundStyle(TurboFieldfareMacTheme.fieldPlaceholder)
                        .padding(.leading, 13)
                        .padding(.top, 8)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
    }

    private var promptHasMarkedText: Bool {
        (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() == true
    }

    private var editorHeight: CGFloat {
        // Larger empty hit target so the field is easy to find and click.
        model.promptText.isEmpty ? 64 : 96
    }

    private var shortcutCaption: String {
        switch model.newlineShortcut {
        case .return: return "⌘↩ to generate"
        case .shiftReturn: return "↩ to generate · ⇧↩ for newline"
        }
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
            Label("Tips", systemImage: "questionmark.circle")
                .turboQuietControlLabel()
        }
        .turboQuietControlChrome()
        .accessibilityLabel("Prompt tips")
        .accessibilityHint("Shows guidance for writing effective prompts")
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

            tipSection("Ask for a clear task",
                       "Say what you want the model to create, explain, plan, or transform. Put the essential context in the same prompt.")
            tipSection("Shape the answer",
                       "Specify a useful length, sections, tone, or output format. Concrete constraints work better than a long list of vague preferences.")
            tipSection("Anchor important facts",
                       "Include facts the answer must preserve and say what should be checked. Generated factual claims can still be wrong or outdated.")
            tipSection("For code and calculations",
                       "Provide types, dimensions, interfaces, edge cases, or a small scaffold. Compile or run the result before relying on it.")
            tipSection("Try a focused revision",
                       "If the answer drifts, shorten the task and make the missing requirement explicit. The default temperature is 0.20 for steadier responses.")
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
                Label("Clear", systemImage: "trash")
                    .turboQuietControlLabel()
            }
            .turboQuietControlChrome()
            .accessibilityLabel("Clear output")
            .accessibilityHint("Removes the conversation transcript and last-run metrics")
            .help("Clear output")
        } else if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused = true
            } label: {
                Label("Clear", systemImage: "xmark.circle.fill")
                    .symbolRenderingMode(.hierarchical)
                    .turboQuietControlLabel()
            }
            .turboQuietControlChrome()
            .accessibilityLabel("Clear prompt")
            .accessibilityHint("Clears the prompt editor")
            .help("Clear prompt")
        }
    }
}
