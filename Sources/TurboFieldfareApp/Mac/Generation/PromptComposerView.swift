import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var showingChatInstructions = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let image = model.pendingImage {
                pendingImagePreview(image)
            }
            if let attachmentError = model.attachmentError {
                attachmentErrorView(attachmentError)
            }
            editor
            footer
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(TurboFieldfareMacTheme.elevatedSurface.opacity(0.92))
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(
                            promptFocused
                                ? TurboFieldfareMacTheme.accentColor.opacity(0.6)
                                : TurboFieldfareMacTheme.border,
                            lineWidth: promptFocused ? 1 : 0.5)
                }
        }
        .shadow(
            color: promptFocused
                ? TurboFieldfareMacTheme.accentColor.opacity(0.18)
                : .black.opacity(0.08),
            radius: promptFocused ? 14 : 12,
            y: 5)
        .animation(.easeOut(duration: 0.16), value: promptFocused)
    }

    private var editor: some View {
        TextEditor(text: $model.promptText)
            .accessibilityLabel("Prompt")
            .font(.body)
            .scrollContentBackground(.hidden)
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
            .frame(height: editorHeight)
            .onKeyPress { press in
                guard press.key == .return,
                      !press.modifiers.contains(.shift) else {
                    return .ignored
                }
                guard model.canRun else { return .handled }
                model.run()
                promptFocused = true
                return .handled
            }
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    // Matches the NSTextView text origin: 5pt line fragment
                    // padding, no vertical inset.
                    Text("Message Gemma…")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
    }

    private var promptHasMarkedText: Bool {
        (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() == true
    }

    private var editorHeight: CGFloat {
        let explicitLineCount = max(
            1,
            model.promptText.split(separator: "\n", omittingEmptySubsequences: false).count)
        // This keeps the composer comfortable for a longer thought while
        // deliberately capping it so the conversation remains visible.
        let estimatedWrappedLineCount = max(
            explicitLineCount,
            max(1, (model.promptText.count + 67) / 68))
        return min(168, max(46, CGFloat(estimatedWrappedLineCount) * 22 + 22))
    }

    private var footer: some View {
        HStack(spacing: 10) {
            imagePicker
            promptTips
            chatInstructions
            if model.completedTurnCount > 0 {
                Text("\(model.completedTurnCount) "
                     + (model.completedTurnCount == 1 ? "turn" : "turns"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("↵ Send  ·  ⇧↵ New line")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            clearAction
            GenerateControl(model: model)
        }
    }

    private var imagePicker: some View {
        Button(action: chooseImage) {
            Group {
                if model.isImportingImage {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Adding image")
                } else {
                    Label("Image", systemImage: "photo.badge.plus")
                        .labelStyle(.titleAndIcon)
                        .font(.caption.weight(.medium))
                }
            }
            .frame(minWidth: 68, minHeight: 28)
            .padding(.horizontal, 5)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(TurboFieldfareMacTheme.accentColor)
        .background(
            TurboFieldfareMacTheme.accentColor.opacity(0.12),
            in: Capsule())
        .overlay {
            Capsule()
                .stroke(TurboFieldfareMacTheme.accentColor.opacity(0.28), lineWidth: 0.75)
        }
        .layoutPriority(1)
        .disabled(!model.canAttachImage)
        .help(imagePickerHelp)
    }

    private var imagePickerHelp: String {
        if model.isImportingImage {
            return "Adding image…"
        }
        if model.pendingImage != nil {
            return "Remove the current image before adding another"
        }
        let count = model.conversation.reduce(0) { $0 + $1.images.count }
        if count >= AppGenerationRequest.maximumImageAttachments {
            return "This chat has reached its \(AppGenerationRequest.maximumImageAttachments)-image limit"
        }
        return "Add an image"
    }

    private func pendingImagePreview(_ image: AppImageAttachment) -> some View {
        HStack(spacing: 10) {
            LocalImageThumbnailView(
                fileURL: model.imageURL(for: image),
                maximumPixelSize: 240)
                .frame(width: 78, height: 58)
                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))

            VStack(alignment: .leading, spacing: 3) {
                Text(image.originalFilename)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
                Text("\(image.pixelWidth) × \(image.pixelHeight)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            Button(action: model.removePendingImage) {
                Label("Remove image", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Remove image")
        }
        .padding(8)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .contain)
    }

    private func attachmentErrorView(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 8)
            Button(action: model.dismissAttachmentError) {
                Label("Dismiss image error", systemImage: "xmark")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
        }
        .padding(.horizontal, 8)
    }

    private func chooseImage() {
        guard model.canAttachImage else { return }
        let panel = NSOpenPanel()
        panel.title = "Choose an image"
        panel.prompt = "Add Image"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.resolvesAliases = true
        panel.allowedContentTypes = [
            .png,
            .jpeg,
            .heic,
            .webP,
        ]
        guard panel.runModal() == .OK,
              let url = panel.url else { return }
        Task {
            _ = await model.attachImage(at: url)
            promptFocused = true
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

    private var chatInstructions: some View {
        Button {
            showingChatInstructions.toggle()
        } label: {
            Label("Chat instructions", systemImage: "slider.horizontal.3")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(model.hasSystemPrompt
                         ? TurboFieldfareMacTheme.accentColor
                         : .secondary)
        .help(model.hasSystemPrompt ? "Chat instructions are active" : "Chat instructions")
        .popover(isPresented: $showingChatInstructions,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Chat instructions")
                    .font(.headline)
                Text("Applied to every answer in this chat. Start a new chat to remove them.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                TextEditor(text: $model.systemPromptText)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(width: 360, height: 120)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityLabel("Chat instructions")
            }
            .padding(16)
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
                model.newChat()
            } label: {
                Label("New chat", systemImage: "plus.bubble")
                    .labelStyle(.iconOnly)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("New chat")
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
