import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct OutputPaneView: View {
    let model: AppModel
    @State private var responseCopyFeedbackID: UUID?

    var body: some View {
        Group {
            if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .task(id: responseCopyFeedbackID) {
            guard let feedbackID = responseCopyFeedbackID else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled, responseCopyFeedbackID == feedbackID else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                responseCopyFeedbackID = nil
            }
        }
        .contextMenu {
            Button("Copy response") {
                copyResponse()
            }
            .disabled(model.outputResponsePlainText.isEmpty)

            Button("Copy prompt") {
                copy(model.outputPromptText)
            }
            .disabled(model.outputPromptText.isEmpty)

            Button("Copy conversation") {
                copy(model.outputConversationPlainText)
            }
            .disabled(model.outputConversationPlainText.isEmpty)

            Divider()

            Button("Clear") { model.clearOutput() }
                .disabled(model.isRunning || !model.hasOutputTranscript)
        }
    }

    private var placeholder: some View {
        EmptyConversationLayout(spacing: 8) {
            EmptyPlaceholderIcon(systemName: placeholderSymbol)
                .frame(width: 32, height: 32)

            emptyPlaceholderContent
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var transcript: some View {
        IncrementalTranscriptView(
            prompt: model.outputPromptText,
            output: model.outputText,
            mailbox: model.generationTranscriptMailbox,
            isTerminal: !model.isRunning,
            showsPrefillPlaceholder: model.isRunning
                && model.outputResponsePlainText.isEmpty)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .topTrailing) {
                if !model.isRunning && !model.outputResponsePlainText.isEmpty {
                    copyResponseButton
                        .padding(8)
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
    }

    private var copyResponseButton: some View {
        Button {
            copyResponse()
        } label: {
            Image(systemName: responseCopyFeedbackID == nil
                  ? "doc.on.doc"
                  : "checkmark.circle.fill")
                .font(.callout.weight(.medium))
                .contentTransition(.symbolEffect(.replace))
                .foregroundStyle(responseCopyFeedbackID == nil
                                 ? Color.secondary
                                 : TurboFieldfareMacTheme.accentColor)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
                .background(.regularMaterial, in: Circle())
                .overlay {
                    Circle().stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(responseCopyFeedbackID == nil
                            ? "Copy response"
                            : "Response copied")
        .accessibilityHint("Copies only the generated answer")
        .help(responseCopyFeedbackID == nil
              ? "Copy response"
              : "Response copied")
    }

    private var emptyPlaceholderContent: some View {
        VStack(spacing: 8) {
            if !needsModelLoad {
                Text("Choose a predefined example or write your own prompt.")
                    .font(.headline)
                Text("Describe the goal, relevant context, and any constraints.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            if isLoadingModel {
                LoadingModelText()
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            } else if let placeholderHint {
                Text(placeholderHint)
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            }
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
            } else if isLoadingModel {
                Button("Load Model", action: {})
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .hidden()
                    .accessibilityHidden(true)
            } else if model.canReloadModel {
                Button("Reload Model", action: model.reloadModel)
                    .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var needsModelLoad: Bool {
        !model.loadState.isReady
    }

    private var isLoadingModel: Bool {
        if case .loading = model.loadState { return true }
        return false
    }

    private var placeholderSymbol: String {
        "cube.transparent"
    }

    private var placeholderHint: String? {
        if model.loadState.isFailed { return "The model could not be loaded" }
        if model.hasStaleLoadedRuntime { return "Reload the model to use changed settings" }
        return needsModelLoad ? "Load the model to begin" : nil
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func copyResponse() {
        copy(model.outputResponsePlainText)
        withAnimation(.easeIn(duration: 0.15)) {
            responseCopyFeedbackID = UUID()
        }
    }
}

private struct EmptyPlaceholderIcon: View {
    let systemName: String

    var body: some View {
        Image(systemName: systemName)
            .font(.title2)
            .foregroundStyle(.quaternary)
            .accessibilityHidden(true)
    }
}

private struct EmptyConversationLayout: Layout {
    let spacing: CGFloat

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        proposal.replacingUnspecifiedDimensions()
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        guard subviews.count == 2 else { return }

        let iconSize = subviews[0].sizeThatFits(.unspecified)
        let iconCenter = CGPoint(x: bounds.midX, y: bounds.midY)
        subviews[0].place(
            at: iconCenter,
            anchor: .center,
            proposal: ProposedViewSize(
                width: iconSize.width,
                height: iconSize.height))

        subviews[1].place(
            at: CGPoint(
                x: bounds.midX,
                y: iconCenter.y + iconSize.height / 2 + spacing),
            anchor: .top,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
    }
}

private struct LoadingModelText: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var animationStart = Date()

    var body: some View {
        if reduceMotion {
            label(dotCount: 3)
        } else {
            TimelineView(.periodic(from: .now, by: 0.25)) { context in
                let elapsed = max(0, context.date.timeIntervalSince(animationStart))
                label(dotCount: Int(elapsed / 0.25) % 4)
            }
        }
    }

    private func label(dotCount: Int) -> some View {
        ZStack(alignment: .leading) {
            Text("Loading Model...").hidden()
            Text("Loading Model" + String(repeating: ".", count: dotCount))
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Loading Model")
    }
}

private struct IncrementalTranscriptView: NSViewRepresentable {
    var prompt: String
    var output: String
    var mailbox: GenerationTranscriptMailbox?
    var isTerminal: Bool
    var showsPrefillPlaceholder: Bool

    @MainActor
    final class Coordinator: NSObject {
        weak var scrollView: NSScrollView?
        weak var textView: NSTextView?
        var mailbox: GenerationTranscriptMailbox?
        var prompt = ""
        var isTerminal = false
        var showsPrefillPlaceholder = false
        var timer: Timer?
        var prefillAnimationTimer: Timer?
        let documentController = InstructionTranscriptDocumentController()
        /// Whether new output should keep the viewport pinned to the bottom.
        /// Driven by the reader's own scrolling: true while they are at the
        /// bottom, false the moment they scroll up to read earlier output.
        private var autoFollow = true

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

            let clipView = scrollView.contentView
            clipView.postsBoundsChangedNotifications = true
            NotificationCenter.default.addObserver(
                self, selector: #selector(clipViewDidScroll),
                name: NSView.boundsDidChangeNotification, object: clipView)
        }

        @objc private func clipViewDidScroll() {
            guard let scrollView else { return }
            autoFollow = isAtBottom(scrollView)
        }

        func synchronize(
            prompt: String,
            output: String,
            mailbox: GenerationTranscriptMailbox?,
            isTerminal: Bool,
            showsPrefillPlaceholder: Bool
        ) {
            self.mailbox = mailbox
            self.prompt = prompt
            self.isTerminal = isTerminal
            self.showsPrefillPlaceholder = showsPrefillPlaceholder
            let response = mailbox?.drain().completeText ?? output
            apply(
                prompt: prompt,
                response: response,
                isTerminal: isTerminal,
                showsPrefillPlaceholder: showsPrefillPlaceholder)
        }

        @objc private func drainMailbox() {
            guard let mailbox else { return }
            let snapshot = mailbox.drain()
            guard !snapshot.pendingText.isEmpty
                    || snapshot.completeText != documentController.response else {
                return
            }
            apply(prompt: prompt,
                  response: snapshot.completeText,
                  isTerminal: isTerminal,
                  showsPrefillPlaceholder: showsPrefillPlaceholder)
        }

        @objc private func animatePrefillPlaceholderIfNeeded() {
            guard documentController.showsPrefillPlaceholder,
                  let scrollView,
                  let textView,
                  let storage = textView.textStorage else { return }
            let wasAtBottom = isAtBottom(scrollView)
            let selection = textView.selectedRanges.map(\.rangeValue)

            storage.beginEditing()
            let changed = documentController.advancePrefillAnimation(storage: storage)
            storage.endEditing()
            guard changed else { return }

            let restored = InstructionTranscriptDocumentController.clampedRanges(
                selection,
                toLength: storage.length)
            if restored.isEmpty {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.selectedRanges = restored.map(NSValue.init(range:))
            }
            if wasAtBottom { textView.scrollToEndOfDocument(nil) }
        }

        func invalidate() {
            NotificationCenter.default.removeObserver(self)
            timer?.invalidate()
            timer = nil
            stopPrefillAnimationTimer()
            mailbox = nil
        }

        private func updatePrefillAnimationTimer() {
            if documentController.showsPrefillPlaceholder {
                guard prefillAnimationTimer == nil else { return }
                let timer = Timer(
                    timeInterval: 0.25,
                    target: self,
                    selector: #selector(animatePrefillPlaceholderIfNeeded),
                    userInfo: nil,
                    repeats: true)
                timer.tolerance = 0.025
                RunLoop.main.add(timer, forMode: .common)
                prefillAnimationTimer = timer
            } else {
                stopPrefillAnimationTimer()
            }
        }

        private func stopPrefillAnimationTimer() {
            prefillAnimationTimer?.invalidate()
            prefillAnimationTimer = nil
        }

        private func apply(
            prompt: String,
            response: String,
            isTerminal: Bool,
            showsPrefillPlaceholder: Bool
        ) {
            guard let textView, let storage = textView.textStorage else { return }
            let selection = textView.selectedRanges.map(\.rangeValue)

            storage.beginEditing()
            let update = documentController.synchronize(
                storage: storage,
                prompt: prompt,
                response: response,
                isTerminal: isTerminal,
                showsPrefillPlaceholder: showsPrefillPlaceholder)
            storage.endEditing()
            updatePrefillAnimationTimer()

            guard update.mutation != .none else { return }
            // A fresh prompt/response starts pinned to the bottom again.
            if update.mutation == .rebuilt { autoFollow = true }

            let restored = InstructionTranscriptDocumentController.clampedRanges(
                selection,
                toLength: storage.length)
            if restored.isEmpty {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.selectedRanges = restored.map(NSValue.init(range:))
            }

            // Only the changed suffix was mutated, so content the reader scrolled
            // up to is undisturbed. Follow the bottom solely when they are already
            // there; never move their viewport otherwise.
            guard autoFollow else { return }
            if let textContainer = textView.textContainer {
                textView.layoutManager?.ensureLayout(for: textContainer)
            }
            textView.scrollToEndOfDocument(nil)
        }

        private func isAtBottom(_ scrollView: NSScrollView) -> Bool {
            guard let document = scrollView.documentView else { return true }
            let visible = scrollView.contentView.bounds
            return visible.maxY >= document.bounds.maxY - 24
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = false

        // Build an explicit TextKit 1 stack so the transcript can draw rounded
        // containers behind fenced code blocks via TranscriptLayoutManager.
        let textStorage = NSTextStorage()
        let layoutManager = TranscriptLayoutManager()
        textStorage.addLayoutManager(layoutManager)
        let textContainer = NSTextContainer(
            size: NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude))
        textContainer.widthTracksTextView = true
        textContainer.lineFragmentPadding = 0
        layoutManager.addTextContainer(textContainer)

        let textView = NSTextView(frame: .zero, textContainer: textContainer)
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = true
        textView.drawsBackground = false
        textView.textContainerInset = NSSize(width: 0, height: 4)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        // A manually built text view (unlike `NSTextView()`) does not get a
        // growable maxSize, so without this it stays capped at the viewport
        // height and the transcript cannot scroll once content overflows.
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude)
        textContainer.heightTracksTextView = false
        textView.isAutomaticLinkDetectionEnabled = false
        textView.isAutomaticDataDetectionEnabled = false
        textView.setAccessibilityLabel("Conversation transcript")
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }
        context.coordinator.attach(scrollView: scrollView, textView: textView)
        context.coordinator.synchronize(
            prompt: prompt,
            output: output,
            mailbox: mailbox,
            isTerminal: isTerminal,
            showsPrefillPlaceholder: showsPrefillPlaceholder)
    }

    static func dismantleNSView(_ nsView: NSScrollView, coordinator: Coordinator) {
        coordinator.invalidate()
    }
}

#if DEBUG
private struct TranscriptPreview: View {
    let response: String
    let isTerminal: Bool
    var showsPrefillPlaceholder = false

    var body: some View {
        IncrementalTranscriptView(
            prompt: "Explain this clearly.",
            output: response,
            mailbox: nil,
            isTerminal: isTerminal,
            showsPrefillPlaceholder: showsPrefillPlaceholder)
            .padding(24)
            .frame(width: 720, height: 420)
    }
}

#Preview("Empty") {
    VStack(spacing: 8) {
        Image(systemName: "cube.transparent")
            .font(.title2)
            .foregroundStyle(.quaternary)
        Text("Choose a predefined example or write your own prompt.")
            .font(.headline)
        Text("Describe the goal, relevant context, and any constraints.")
            .foregroundStyle(.secondary)
    }
    .frame(width: 720, height: 420)
}

#Preview("Streaming") {
    TranscriptPreview(
        response: "A response arriving one readable piece at a time...",
        isTerminal: false)
}

#Preview("Prefilling") {
    TranscriptPreview(
        response: "",
        isTerminal: false,
        showsPrefillPlaceholder: true)
}

#Preview("Completed prose") {
    TranscriptPreview(
        response: "# A clear answer\n\nHere is a concise explanation with **useful emphasis**.\n\n- First point\n- Second point",
        isTerminal: true)
}

#Preview("Completed code") {
    TranscriptPreview(
        response: "Use `fibonacci(7)`:\n\n```python\ndef fibonacci(n: int) -> list[int]:\n    return []\n```",
        isTerminal: true)
}

#Preview("Incomplete Markdown fallback") {
    TranscriptPreview(
        response: "The partial answer remains readable.\n\n```python\nprint('unfinished')",
        isTerminal: true)
}
#endif
