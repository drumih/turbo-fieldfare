import AppKit
import TurboFieldfareAppCore
import SwiftUI

struct OutputPaneView: View {
    let model: AppModel

    var body: some View {
        Group {
            if model.isRunning && !hasVisibleCompletion {
                pendingGeneration
            } else if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .contextMenu {
            Button("Copy All") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(model.outputPlainText, forType: .string)
            }
            .disabled(!model.hasOutputTranscript)
            Button("Clear") { model.clearOutput() }
                .disabled(model.isRunning || !model.hasOutputTranscript)
        }
    }

    @ViewBuilder
    private var placeholder: some View {
        ScrollView {
            emptyPlaceholder
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .defaultScrollAnchor(.bottom, for: .sizeChanges)
    }

    private var transcript: some View {
        IncrementalTranscriptView(
            prompt: model.outputPromptText,
            output: model.outputText,
            mailbox: model.generationTranscriptMailbox)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
    }

    private var pendingGeneration: some View {
        VStack(spacing: 10) {
            Text("Processing prompt")
                .font(.callout)
                .foregroundStyle(.secondary)
            if model.livePrefillTotal > 0 {
                Text("\(model.livePrefillDone) of \(model.livePrefillTotal) tokens")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    private var hasVisibleCompletion: Bool {
        model.outputText.contains { !$0.isWhitespace }
    }

    private var emptyPlaceholder: some View {
        VStack(spacing: 8) {
            Image(systemName: placeholderSymbol)
                .font(.title2)
                .foregroundStyle(.quaternary)
            if !needsModelLoad {
                Text("Give TurboFieldfare a beginning, and it will continue the text from there.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Text(placeholderHint)
                .font(.callout)
                .foregroundStyle(.tertiary)
            if let detail = model.presentation.detail {
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(model.presentation.severity == .error ? .red : .secondary)
                    .multilineTextAlignment(.center)
            }
            if model.canLoadModel {
                Button(model.loadState.isFailed ? "Retry Load" : "Load Model",
                       action: model.loadModel)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
            } else if model.canReloadModel {
                Button("Reload Model", action: model.reloadModel)
                    .buttonStyle(.borderedProminent)
            } else if model.canCancelLoad {
                Button("Cancel Load", action: model.cancelLoad)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 240)
    }

    private var needsModelLoad: Bool {
        !model.loadState.isReady
    }

    private var placeholderSymbol: String {
        needsModelLoad ? "cube.transparent" : "text.cursor"
    }

    private var placeholderHint: String {
        if model.loadState.isFailed { return "The model could not be loaded" }
        if model.hasStaleLoadedRuntime { return "Reload the model to use changed settings" }
        return needsModelLoad
            ? "Load the model to begin"
            : "Enter a prompt and press \u{2318}\u{21A9} to generate"
    }
}

private struct IncrementalTranscriptView: NSViewRepresentable {
    var prompt: String
    var output: String
    var mailbox: GenerationTranscriptMailbox?

    @MainActor
    final class Coordinator: NSObject {
        var prompt = ""
        var output = ""
        weak var scrollView: NSScrollView?
        weak var textView: NSTextView?
        var mailbox: GenerationTranscriptMailbox?
        var timer: Timer?

        func attach(scrollView: NSScrollView, textView: NSTextView) {
            self.scrollView = scrollView
            self.textView = textView
            guard timer == nil else { return }
            let timer = Timer(timeInterval: 0.1, target: self,
                              selector: #selector(drainMailbox),
                              userInfo: nil, repeats: true)
            timer.tolerance = 0.02
            RunLoop.main.add(timer, forMode: .common)
            self.timer = timer
        }

        func synchronize(prompt: String, output: String,
                         mailbox: GenerationTranscriptMailbox?) {
            self.mailbox = mailbox
            let completion = mailbox?.drain().completeText ?? output
            apply(prompt: prompt, output: completion)
        }

        @objc private func drainMailbox() {
            guard let mailbox else { return }
            let snapshot = mailbox.drain()
            guard !snapshot.pendingText.isEmpty || snapshot.completeText != output else { return }
            apply(prompt: prompt, output: snapshot.completeText)
        }

        func invalidate() {
            timer?.invalidate()
            timer = nil
            mailbox = nil
        }

        private func apply(prompt: String, output: String) {
            guard let scrollView, let textView, let storage = textView.textStorage else {
                return
            }
            let wasAtBottom = isAtBottom(scrollView)
            let selection = textView.selectedRanges

            storage.beginEditing()
            if prompt != self.prompt || !output.hasPrefix(self.output) {
                let replacement = NSMutableAttributedString(
                    string: prompt,
                    attributes: promptAttributes())
                replacement.append(NSAttributedString(
                    string: output,
                    attributes: outputAttributes()))
                storage.setAttributedString(replacement)
            } else if output.count > self.output.count {
                let delta = String(output.dropFirst(self.output.count))
                storage.append(NSAttributedString(
                    string: delta,
                    attributes: outputAttributes()))
            }
            storage.endEditing()

            self.prompt = prompt
            self.output = output
            let restoredSelection = selection.map { value in
                let range = value.rangeValue
                let location = min(range.location, storage.length)
                let length = min(range.length, storage.length - location)
                return NSValue(range: NSRange(location: location, length: length))
            }
            if restoredSelection.isEmpty {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.selectedRanges = restoredSelection
            }
            if wasAtBottom { textView.scrollToEndOfDocument(nil) }
        }

        private func isAtBottom(_ scrollView: NSScrollView) -> Bool {
            guard let document = scrollView.documentView else { return true }
            let visible = scrollView.contentView.bounds
            return visible.maxY >= document.bounds.maxY - 24
        }

        private func outputAttributes() -> [NSAttributedString.Key: Any] {
            [.font: NSFont.monospacedSystemFont(
                ofSize: NSFont.systemFontSize, weight: .regular),
             .foregroundColor: NSColor.labelColor,
             .paragraphStyle: paragraphStyle()]
        }

        private func promptAttributes() -> [NSAttributedString.Key: Any] {
            [.font: NSFont.monospacedSystemFont(
                ofSize: NSFont.systemFontSize, weight: .bold),
             .foregroundColor: NSColor.labelColor,
             .paragraphStyle: paragraphStyle()]
        }

        private func paragraphStyle() -> NSParagraphStyle {
            let style = NSMutableParagraphStyle()
            style.lineSpacing = 3
            return style
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = false

        let textView = NSTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.textContainerInset = NSSize(width: 0, height: 0)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = 0
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }
        context.coordinator.attach(scrollView: scrollView, textView: textView)
        context.coordinator.synchronize(
            prompt: prompt, output: output, mailbox: mailbox)
    }

    static func dismantleNSView(_ nsView: NSScrollView,
                                coordinator: Coordinator) {
        coordinator.invalidate()
    }
}
