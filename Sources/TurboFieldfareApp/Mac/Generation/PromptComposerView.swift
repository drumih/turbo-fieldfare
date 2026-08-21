import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var isImportingDocuments = false
    @State private var isExtractingDocuments = false
    @State private var documentImportError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.promptAttachments.isEmpty {
                attachments
            }
            if let documentImportError {
                Text(documentImportError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
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
        .fileImporter(
            isPresented: $isImportingDocuments,
            allowedContentTypes: DocumentTextExtractor.supportedContentTypes,
            allowsMultipleSelection: true,
            onCompletion: handleDocumentSelection)
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

    private var promptHasMarkedText: Bool {
        (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() == true
    }

    private var editorHeight: CGFloat {
        model.promptText.isEmpty ? 46 : 84
    }

    private var footer: some View {
        HStack(spacing: 10) {
            attachDocumentAction
            promptTips
            Spacer()
            clearAction
            GenerateControl(model: model)
        }
    }

    private var attachments: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(model.promptAttachments) { attachment in
                    HStack(spacing: 7) {
                        Image(systemName: "doc.text")
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(attachment.fileName)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                            Text(attachmentDetail(attachment))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            model.removePromptAttachment(id: attachment.id)
                        } label: {
                            Label("Remove \(attachment.fileName)", systemImage: "xmark.circle.fill")
                                .labelStyle(.iconOnly)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .disabled(model.isRunning)
                    }
                    .padding(.leading, 10)
                    .padding(.trailing, 7)
                    .padding(.vertical, 7)
                    .background(.quaternary.opacity(0.35), in: .capsule)
                    .overlay {
                        Capsule().stroke(.separator.opacity(0.4), lineWidth: 0.5)
                    }
                    .help(attachment.wasTruncatedDuringExtraction
                          ? "Text was truncated during local extraction."
                          : "Text extracted locally for this prompt.")
                }
            }
        }
        .scrollIndicators(.hidden)
    }

    private var attachDocumentAction: some View {
        Button {
            documentImportError = nil
            isImportingDocuments = true
        } label: {
            Group {
                if isExtractingDocuments {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Label("Attach documents", systemImage: "paperclip")
                        .labelStyle(.iconOnly)
                }
            }
            .frame(width: 28, height: 28)
            .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .disabled(model.isRunning || isExtractingDocuments)
        .help("Attach PDF, Word, PowerPoint, or Excel files")
        .accessibilityLabel(isExtractingDocuments
                            ? "Extracting document text"
                            : "Attach documents")
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

    private func attachmentDetail(_ attachment: AppPromptAttachment) -> String {
        let count = attachment.characterCount.formatted(.number.notation(.compactName))
        let suffix = attachment.wasTruncatedDuringExtraction ? " • truncated" : ""
        return "\(attachment.formatLabel) • \(count) chars\(suffix)"
    }

    private func handleDocumentSelection(_ result: Result<[URL], any Error>) {
        switch result {
        case .success(let urls):
            importDocuments(urls)
        case .failure(let error):
            documentImportError = error.localizedDescription
        }
    }

    private func importDocuments(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        isExtractingDocuments = true
        documentImportError = nil

        Task {
            let outcomes = await Task.detached(priority: .userInitiated) {
                urls.map { url -> DocumentImportOutcome in
                    do {
                        return .success(try DocumentTextExtractor.extract(from: url))
                    } catch {
                        return .failure(fileName: url.lastPathComponent,
                                        message: error.localizedDescription)
                    }
                }
            }.value

            var failures: [String] = []
            for outcome in outcomes {
                switch outcome {
                case .success(let document):
                    model.addPromptAttachment(AppPromptAttachment(
                        fileName: document.fileName,
                        formatLabel: document.formatLabel,
                        extractedText: document.text,
                        wasTruncatedDuringExtraction: document.wasTruncated))
                case .failure(let fileName, let message):
                    failures.append("\(fileName): \(message)")
                }
            }
            documentImportError = failures.isEmpty
                ? nil
                : failures.joined(separator: "\n")
            isExtractingDocuments = false
        }
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning && !model.promptText.isEmpty {
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
        } else if !model.isRunning && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label("Clear chat history", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear chat history")
        }
    }
}

private enum DocumentImportOutcome: Sendable {
    case success(ExtractedPromptDocument)
    case failure(fileName: String, message: String)
}
