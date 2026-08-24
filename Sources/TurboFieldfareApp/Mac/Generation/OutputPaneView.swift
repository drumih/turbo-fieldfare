import AppKit
import TurboFieldfare
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
            images: model.outputImageAttachments,
            output: model.outputText,
            mailbox: model.generationTranscriptMailbox,
            isTerminal: !model.isRunning,
            showsPrefillPlaceholder: model.isRunning
                && model.outputResponsePlainText.isEmpty,
            runIdentity: model.runIdentity)
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

struct SubmittedImageThumbnail: View {
    let attachment: AppImageAttachment
    let maximumSize: CGSize
    @State private var image: NSImage?

    init(
        attachment: AppImageAttachment,
        maximumSize: CGSize = CGSize(width: 48, height: 48)
    ) {
        self.attachment = attachment
        self.maximumSize = maximumSize
    }

    var body: some View {
        Group {
            if let image {
                // Filled and cropped to a tile, not fitted inside one. Fitting
                // gave every attachment a different height — a screenshot came
                // out a third the height of a portrait photo — so a row of them
                // was ragged, and the remove badge, pinned to the tile, floated
                // clear of the short ones.
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: maximumSize.width, height: maximumSize.height)
                    .clipped()
            } else {
                Image(systemName: "photo")
                    .foregroundStyle(.tertiary)
                    .frame(width: maximumSize.width, height: maximumSize.height)
            }
        }
        .frame(width: maximumSize.width, height: maximumSize.height)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(.separator.opacity(0.5), lineWidth: 0.5)
        }
        .accessibilityLabel("Attached image \(attachment.displayName)")
        .task(id: "\(attachment.id)-\(maximumSize.width)x\(maximumSize.height)") {
            let url = attachment.fileURL
            let key = attachment.sha256
            let pixels = Int(ceil(max(maximumSize.width, maximumSize.height) * 2))
            // This closure runs on the main actor, and a source near the decode
            // budget takes long enough that decoding here stalled the window.
            // The decode is done off the main thread and left in the cache; the
            // read below is then a lookup.
            await Task.detached(priority: .userInitiated) {
                _ = Self.loadThumbnail(
                    at: url, maximumPixelSize: pixels, cacheKey: key)
            }.value
            image = Self.loadThumbnail(
                at: url, maximumPixelSize: pixels, cacheKey: key)
        }
    }

    /// The transcript lays images out inline, where cropping would hide part of
    /// what was sent, so that path fits rather than fills.
    static func fittedSize(_ source: CGSize, within maximumSize: CGSize) -> CGSize {
        TranscriptImageTile.fittedSize(source, within: maximumSize)
    }

    /// Every attached-image decode in the app goes through here, so the decode
    /// budget cannot be applied to one caller and forgotten on the next. Nil
    /// means refused or unreadable; both callers draw their placeholder for it.
    nonisolated static func loadThumbnail(
        at url: URL,
        maximumPixelSize: Int,
        cacheKey: String? = nil
    ) -> NSImage? {
        TranscriptImageLoader.thumbnail(
            at: url,
            maximumPixelSize: maximumPixelSize,
            budget: decodeBudget,
            cacheKey: cacheKey)
    }

    /// The single point where the app binds the runtime's limits, so the two
    /// cannot drift apart: see `VisionImageLimits` in
    /// Runtime/Vision/Preprocessing/ImageMetadataReader.swift.
    nonisolated private static let decodeBudget: TranscriptImageLoader.Budget = {
        let limits = VisionImageLimits()
        return TranscriptImageLoader.Budget(
            maximumSourcePixels: limits.maximumSourcePixels,
            maximumSourceDimension: limits.maximumSourceDimension,
            maximumDecodedBytes: limits.maximumDecodedBytes,
            allowedTypeIdentifiers: limits.allowedTypeIdentifiers)
    }()
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
    var images: [AppImageAttachment] = []
    var output: String
    var mailbox: GenerationTranscriptMailbox?
    var isTerminal: Bool
    var showsPrefillPlaceholder: Bool
    var runIdentity: Int

    @MainActor
    final class Coordinator: NSObject {
        weak var scrollView: NSScrollView?
        weak var textView: NSTextView?
        var mailbox: GenerationTranscriptMailbox?
        var prompt = ""
        var promptPrefix = NSAttributedString()
        var promptPrefixIdentifier = ""

        /// The identifier the document controller is told about — empty until
        /// the prefix it names actually exists.
        ///
        /// `synchronize` records the new identifier and clears the prefix before
        /// starting the async image build, so telling the controller the final
        /// identifier up front made every term of its rebuild test false when
        /// the built prefix arrived: same prompt, same identifier, same response.
        /// The strip was dropped for the rest of the run, and a coordinator
        /// recreated against a finished transcript never drew images at all.
        ///
        /// `apply` reads this itself rather than taking it as an argument. It
        /// used to be passed in, and one of the three call sites passed the raw
        /// identifier instead — the same defect again, in the code written to
        /// prevent it. A caller that cannot name the identifier cannot get it
        /// wrong.
        var appliedPromptPrefixIdentifier: String {
            promptPrefix.length == 0 ? "" : promptPrefixIdentifier
        }
        var isTerminal = false
        var showsPrefillPlaceholder = false
        var runIdentity = 0
        /// Holds the view at the bottom from the moment a run starts until its
        /// answer begins. One scroll is not enough: image thumbnails finish
        /// loading after it and push the content back down, which is exactly
        /// the case this exists for. The decision itself lives in
        /// `TranscriptScrollFollow`, where it can be tested.
        var follow = TranscriptScrollFollow()
        var timer: Timer?
        var prefillAnimationTimer: Timer?
        let documentController = InstructionTranscriptDocumentController()

        func attach(scrollView: NSScrollView, textView: NSTextView) {
            self.scrollView = scrollView
            self.textView = textView
            guard timer == nil else { return }
            // Auto-follow yields the instant the reader scrolls. Without this,
            // holding the view at the bottom through prefill fought anyone
            // trying to look back at what they had sent.
            scrollView.contentView.postsBoundsChangedNotifications = true
            NotificationCenter.default.addObserver(
                self, selector: #selector(readerTookOver),
                name: NSScrollView.willStartLiveScrollNotification,
                object: scrollView)
            NotificationCenter.default.addObserver(
                self, selector: #selector(readerTookOver),
                name: NSScrollView.didLiveScrollNotification,
                object: scrollView)
            let timer = Timer(timeInterval: 0.1, target: self,
                              selector: #selector(drainMailbox),
                              userInfo: nil, repeats: true)
            timer.tolerance = 0.02
            RunLoop.main.add(timer, forMode: .common)
            self.timer = timer
        }

        func synchronize(
            prompt: String,
            images: [AppImageAttachment],
            output: String,
            mailbox: GenerationTranscriptMailbox?,
            isTerminal: Bool,
            showsPrefillPlaceholder: Bool,
            runIdentity: Int
        ) {
            // A new run always goes to the bottom, whatever the reader was
            // looking at: it is the thing they just asked for.
            let startedNewRun = runIdentity != self.runIdentity
            self.runIdentity = runIdentity
            self.mailbox = mailbox
            self.prompt = prompt
            let prefixIdentifier = images.map {
                "\($0.id.uuidString):\($0.sha256)"
            }.joined(separator: ",")
            if prefixIdentifier != promptPrefixIdentifier {
                promptPrefixIdentifier = prefixIdentifier
                promptPrefix = NSAttributedString()
                buildPromptPrefix(images, identifier: prefixIdentifier)
            }
            self.isTerminal = isTerminal
            self.showsPrefillPlaceholder = showsPrefillPlaceholder
            let response = mailbox?.drain().completeText ?? output
            apply(
                prompt: prompt,
                response: response,
                isTerminal: isTerminal,
                showsPrefillPlaceholder: showsPrefillPlaceholder,
                promptPrefix: promptPrefix)
            if startedNewRun { follow.beginRun() }
            // Once the answer has text, or the run is over, the reader is in
            // charge again; the usual follow-the-bottom rule takes over from a
            // view that is already at the bottom.
            if !response.isEmpty || isTerminal { follow.end() }
            if startedNewRun || shouldFollowNow() { scrollToBottom() }
        }

        func scrollToBottom() {
            guard let textView else { return }
            if let textContainer = textView.textContainer {
                textView.layoutManager?.ensureLayout(for: textContainer)
            }
            textView.scrollToEndOfDocument(nil)
            recordScrollPosition()
        }

        private func recordScrollPosition() {
            guard let scrollView else { return }
            follow.recordScroll(
                origin: scrollView.contentView.bounds.origin.y,
                documentHeight: scrollView.documentView?.bounds.height ?? 0)
        }

        private func shouldFollowNow() -> Bool {
            guard let scrollView else { return false }
            return follow.shouldScrollToBottom(
                origin: scrollView.contentView.bounds.origin.y,
                documentHeight: scrollView.documentView?.bounds.height ?? 0)
        }

        @objc private func drainMailbox() {
            // Keep the newest turn in view while its images lay out, even
            // between synchronize calls — but never against the reader.
            //
            // "Not at the bottom any more" is NOT the test for that. Images lay
            // out after the scroll and grow the document, which leaves the view
            // above the bottom through no act of the reader's; treating that as
            // a reader scroll ended the follow on exactly the turns that needed
            // it, and a prompt with several images stayed scrolled off the top.
            // A reader moving the view changes the scroll origin while the
            // document height stays put, so that is what ends it.
            if shouldFollowNow() { scrollToBottom() }
            guard let mailbox else { return }
            let snapshot = mailbox.drain()
            guard !snapshot.pendingText.isEmpty
                    || snapshot.completeText != documentController.response else {
                return
            }
            apply(prompt: prompt,
                  response: snapshot.completeText,
                  isTerminal: isTerminal,
                  showsPrefillPlaceholder: showsPrefillPlaceholder,
                  promptPrefix: promptPrefix)
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
            if wasAtBottom {
                textView.scrollToEndOfDocument(nil)
                recordScrollPosition()
            }
        }

        @objc private func readerTookOver() {
            follow.end()
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
            showsPrefillPlaceholder: Bool,
            promptPrefix: NSAttributedString
        ) {
            guard let scrollView, let textView, let storage = textView.textStorage else { return }
            let wasAtBottom = isAtBottom(scrollView)
            let selection = textView.selectedRanges.map(\.rangeValue)

            storage.beginEditing()
            let update = documentController.synchronize(
                storage: storage,
                prompt: prompt,
                response: response,
                isTerminal: isTerminal,
                showsPrefillPlaceholder: showsPrefillPlaceholder,
                promptPrefix: promptPrefix,
                promptPrefixIdentifier: appliedPromptPrefixIdentifier)
            storage.endEditing()
            updatePrefillAnimationTimer()

            guard update.mutation != .none else { return }
            let restored = InstructionTranscriptDocumentController.clampedRanges(
                selection,
                toLength: storage.length)
            if restored.isEmpty {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.selectedRanges = restored.map(NSValue.init(range:))
            }
            if InstructionTranscriptDocumentController.shouldScrollToBottom(
                wasAtBottom: wasAtBottom,
                mutation: update.mutation
            ) {
                if let textContainer = textView.textContainer {
                    textView.layoutManager?.ensureLayout(for: textContainer)
                }
                textView.scrollToEndOfDocument(nil)
                // Every programmatic scroll updates the baseline, or the next
                // comparison reads our own move as the reader's.
                recordScrollPosition()
            }
        }

        private func isAtBottom(_ scrollView: NSScrollView) -> Bool {
            guard let document = scrollView.documentView else { return true }
            let visible = scrollView.contentView.bounds
            return visible.maxY >= document.bounds.maxY - 24
        }

        /// Decoding the submitted images ran inside `updateNSView`'s render
        /// pass, so several photos near the decode budget stalled the window at
        /// the moment Run was pressed. Only the decode moves off the main
        /// thread — it lands in the loader's cache, and the attributed string is
        /// then assembled from cached copies, which is cheap and stays here
        /// where AppKit's drawing belongs. A prefix the transcript has since
        /// stopped wanting is dropped rather than applied. Images arriving after
        /// the first paint is the case `follow` already exists for.
        private func buildPromptPrefix(
            _ images: [AppImageAttachment], identifier: String
        ) {
            guard !images.isEmpty else { return }
            Task { [weak self] in
                await Task.detached(priority: .userInitiated) {
                    for attachment in images {
                        _ = SubmittedImageThumbnail.loadThumbnail(
                            at: attachment.fileURL,
                            maximumPixelSize: 720,
                            cacheKey: attachment.sha256)
                    }
                }.value
                guard let self, self.promptPrefixIdentifier == identifier else { return }
                let prefix = Self.makePromptPrefix(images)
                self.promptPrefix = prefix
                self.apply(
                    prompt: self.prompt,
                    response: self.documentController.response,
                    isTerminal: self.isTerminal,
                    showsPrefillPlaceholder: self.showsPrefillPlaceholder,
                    promptPrefix: prefix)
                if self.shouldFollowNow() { self.scrollToBottom() }
            }
        }

        private static func makePromptPrefix(
            _ images: [AppImageAttachment]
        ) -> NSAttributedString {
            let result = NSMutableAttributedString()
            for attachment in images {
                // A refused or unreadable image is dropped from the transcript,
                // which is the same degradation the composer's placeholder tile
                // gives. The separator therefore keys off what has actually
                // been written, not off the attachment's index: keyed off the
                // index, a first image the decode budget refused left the line
                // starting with a bare gap.
                guard let image = SubmittedImageThumbnail.loadThumbnail(
                    at: attachment.fileURL,
                    maximumPixelSize: 720,
                    cacheKey: attachment.sha256) else { continue }
                image.size = SubmittedImageThumbnail.fittedSize(
                    image.size,
                    within: CGSize(width: 360, height: 240))
                let textAttachment = NSTextAttachment()
                textAttachment.attachmentCell = NSTextAttachmentCell(
                    imageCell: Self.rounded(image))
                if result.length > 0 {
                    result.append(NSAttributedString(string: "  "))
                }
                result.append(NSAttributedString(attachment: textAttachment))
            }
            return result
        }

        /// The transcript draws its images as text attachments, which cannot be
        /// clipped by the view the way the composer's thumbnails are, so the
        /// corners have to be drawn into the image itself.
        private static func rounded(_ image: NSImage) -> NSImage {
            let size = image.size
            guard size.width > 1, size.height > 1 else { return image }
            // Proportional rather than fixed, so a small thumbnail and a large
            // one look like the same shape; capped so wide images do not turn
            // into lozenges.
            let radius = min(12, min(size.width, size.height) * 0.08)
            let rounded = NSImage(size: size)
            rounded.lockFocus()
            defer { rounded.unlockFocus() }
            NSGraphicsContext.current?.imageInterpolation = .high
            let bounds = NSRect(origin: .zero, size: size)
            let path = NSBezierPath(roundedRect: bounds,
                                    xRadius: radius, yRadius: radius)
            path.addClip()
            image.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1)
            return rounded
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
        textView.isRichText = true
        textView.drawsBackground = false
        textView.textContainerInset = NSSize(width: 0, height: 4)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = 0
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
            images: images,
            output: output,
            mailbox: mailbox,
            isTerminal: isTerminal,
            showsPrefillPlaceholder: showsPrefillPlaceholder,
            runIdentity: runIdentity)
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
            showsPrefillPlaceholder: showsPrefillPlaceholder,
            runIdentity: 0)
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
