import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

/// A chat-style transcript: user turns are compact trailing bubbles while
/// assistant turns use the full Markdown-rendered reading column.
struct ChatTranscriptView: View {
    let prompt: String
    let messages: [AppChatMessage]
    let attachmentRootURL: URL
    let output: String
    let isRunning: Bool
    let canRegenerate: Bool
    let canEditLastPrompt: Bool
    let canSubmitEditedLastPrompt: Bool
    let onRegenerate: () -> Void
    let onEditLastPrompt: (String) -> Bool

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isFollowingLatest = true
    @State private var hasUnseenContent = false
    @State private var hoveredUserMessageIndex: Int?
    @State private var hoveredAssistantID: String?
    @State private var copiedAssistantID: String?
    @State private var editingUserMessageIndex: Int?
    @State private var editedPrompt = ""
    @State private var responseBuffer = StreamingResponseBuffer()
    @State private var responseRevealTask: Task<Void, Never>?
    @State private var responseRevealGeneration: UInt = 0
    @FocusState private var editedPromptFocused: Bool

    var body: some View {
        transcriptContent(response: responseBuffer.displayedText)
            .textSelection(.enabled)
            .onAppear(perform: synchronizeInitialResponse)
            .onChange(of: output) { _, source in
                synchronizeResponse(source)
            }
            .onChange(of: isRunning) { _, isRunning in
                if isRunning {
                    cancelPromptEditing()
                    beginStreamingResponse()
                } else {
                    finishStreamingResponse()
                }
            }
            .onChange(of: messages) { _, _ in
                cancelPromptEditing()
                guard !isRunning else { return }
                showFinishedResponse()
            }
            .onDisappear {
                cancelPromptEditing()
                cancelResponseReveal()
            }
    }

    private func transcriptContent(response: String) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 26) {
                    messageList

                    if messages.isEmpty, !prompt.isEmpty {
                        userMessage(
                            AppChatMessage(role: .user, content: prompt),
                            index: nil,
                            isEditable: false)
                    }

                    if isRunning || !response.isEmpty {
                        assistantMessage(
                            response,
                            id: "active-response",
                            isActive: true)
                            .id("active-response")
                    }

                    Color.clear
                        .frame(height: 1)
                        .id("transcript-bottom")
                }
                .frame(maxWidth: 820)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.horizontal, 28)
                .padding(.vertical, 28)
            }
            .scrollIndicators(.automatic)
            .onScrollPhaseChange { _, phase, context in
                switch phase {
                case .tracking, .interacting, .decelerating:
                    // Only a human scroll should opt out of following the
                    // answer. Content growth alone must not stop streaming.
                    isFollowingLatest = false
                case .idle:
                    let isAtBottom = context.geometry.visibleRect.maxY
                        >= context.geometry.contentSize.height - 36
                    isFollowingLatest = isAtBottom
                    if isAtBottom { hasUnseenContent = false }
                case .animating:
                    break
                }
            }
            .onChange(of: messages) { _, _ in
                followNewContent(using: proxy)
            }
            .onChange(of: response) { _, _ in
                followNewContent(using: proxy)
            }
            .onAppear {
                scrollToBottom(proxy, animated: false)
            }
            .overlay(alignment: .bottom) {
                if hasUnseenContent {
                    Button {
                        isFollowingLatest = true
                        hasUnseenContent = false
                        scrollToBottom(proxy)
                    } label: {
                        Label("Jump to latest", systemImage: "arrow.down")
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.primary)
                    .background(.regularMaterial, in: Capsule())
                    .overlay {
                        Capsule().stroke(TurboFieldfareMacTheme.border, lineWidth: 0.5)
                    }
                    .shadow(color: .black.opacity(0.12), radius: 8, y: 3)
                    .padding(.bottom, 16)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                    .accessibilityHint("Scrolls to the newest response")
                }
            }
        }
    }

    @ViewBuilder
    private var messageList: some View {
        ForEach(Array(messages.enumerated()), id: \.offset) { index, message in
            switch message.role {
            case .system:
                systemMessage(message.content)
            case .user:
                userMessage(
                    message,
                    index: index,
                    isEditable: index == latestUserMessageIndex)
            case .assistant:
                assistantMessage(message.content, id: "assistant-\(index)", isActive: false)
            }
        }
    }

    private var latestUserMessageIndex: Int? {
        LatestPromptEditPolicy.editableUserMessageIndex(
            in: messages,
            canEditLastPrompt: canEditLastPrompt)
    }

    private func userMessage(
        _ message: AppChatMessage,
        index: Int?,
        isEditable: Bool
    ) -> some View {
        HStack {
            Spacer(minLength: 48)
            if let index, editingUserMessageIndex == index {
                promptEditor(index: index)
            } else {
                VStack(alignment: .trailing, spacing: 5) {
                    VStack(alignment: .leading, spacing: 10) {
                        if let image = message.images.first {
                            LocalImageThumbnailView(
                                fileURL: imageURL(for: image),
                                maximumPixelSize: 720)
                                .frame(maxWidth: 420, maxHeight: 300)
                                .clipShape(RoundedRectangle(
                                    cornerRadius: 13,
                                    style: .continuous))
                                .accessibilityLabel(
                                    "Attached image \(image.originalFilename)")
                        }

                        Text(message.content)
                            .font(.body)
                            .foregroundStyle(.primary)
                            .lineSpacing(3)
                            .textSelection(.enabled)
                    }
                    .frame(maxWidth: 620, alignment: .leading)
                    .padding(.horizontal, 17)
                    .padding(.vertical, 12)
                    .background(
                        TurboFieldfareMacTheme.accentSurface,
                        in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .stroke(
                                TurboFieldfareMacTheme.accentColor.opacity(0.14),
                                lineWidth: 0.5)
                    }

                    if let index, isEditable {
                        Button {
                            beginPromptEditing(message.content, index: index)
                        } label: {
                            Label("Edit latest prompt", systemImage: "pencil")
                                .labelStyle(.iconOnly)
                                .frame(width: 26, height: 24)
                                .contentShape(Circle())
                        }
                        .buttonStyle(.plain)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .opacity(hoveredUserMessageIndex == index ? 1 : 0.68)
                        .help("Edit latest prompt")
                    }
                }
                .onHover { isHovering in
                    guard let index else { return }
                    if isHovering {
                        hoveredUserMessageIndex = index
                    } else if hoveredUserMessageIndex == index {
                        hoveredUserMessageIndex = nil
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
        .accessibilityElement(children: .contain)
    }

    private func imageURL(for attachment: AppImageAttachment) -> URL? {
        let root = attachmentRootURL.standardizedFileURL
        let candidate = root
            .appendingPathComponent(attachment.relativePath, isDirectory: false)
            .standardizedFileURL
        let rootPrefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return candidate.path.hasPrefix(rootPrefix) ? candidate : nil
    }

    private func promptEditor(index: Int) -> some View {
        VStack(alignment: .trailing, spacing: 10) {
            TextEditor(text: $editedPrompt)
                .font(.body)
                .lineSpacing(3)
                .scrollContentBackground(.hidden)
                .focused($editedPromptFocused)
                .frame(minWidth: 360, maxWidth: 620, minHeight: editedPromptEditorHeight)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(
                    TurboFieldfareMacTheme.elevatedSurface,
                    in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(
                            TurboFieldfareMacTheme.accentColor.opacity(0.55),
                            lineWidth: 1)
                }
                .onKeyPress(.return, phases: [.down, .repeat]) { keyPress in
                    guard !keyPress.modifiers.contains(.shift) else { return .ignored }
                    guard !keyPress.phase.contains(.repeat) else { return .handled }
                    guard (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() != true
                    else { return .ignored }
                    submitPromptEdit()
                    return .handled
                }
                .accessibilityLabel("Edit latest prompt")

            HStack(spacing: 8) {
                Text(canSubmitEditedLastPrompt
                     ? "Shift-Return for a new line"
                     : "Load or reload the model to update")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)

                Spacer(minLength: 16)

                Button("Cancel", action: cancelPromptEditing)
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                Button("Update", action: submitPromptEdit)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(!canSubmitPromptEdit)
                    .help(canSubmitEditedLastPrompt
                          ? "Update prompt and regenerate"
                          : "Load or reload the model first")
            }
        }
        .frame(minWidth: 360, maxWidth: 620)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Editing latest prompt")
        .id("prompt-editor-\(index)")
    }

    private func assistantMessage(
        _ content: String,
        id: String,
        isActive: Bool
    ) -> some View {
        let streams = isActive && isRunning
        let showsActions = !content.isEmpty
            && (hoveredAssistantID == id || copiedAssistantID == id)

        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                ZStack {
                    Circle()
                        .fill(TurboFieldfareMacTheme.accentSurface)
                    Image(systemName: "sparkles")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                }
                .frame(width: 24, height: 24)

                Text("Gemma 4 26B")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                if streams {
                    StreamingStatusLabel()
                }
            }

            if content.isEmpty && streams {
                ProgressView()
                    .controlSize(.small)
                    .tint(TurboFieldfareMacTheme.accentColor)
                    .padding(.leading, 32)
                    .padding(.top, 2)
            } else if !content.isEmpty {
                MarkdownMessageText(content: content, isStreaming: streams)
                    .equatable()
                    .font(.body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.leading, 32)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .topTrailing) {
            if showsActions {
                assistantActions(for: content, id: id, isActive: isActive)
                    .transition(.opacity.combined(with: .scale(scale: 0.92)))
            }
        }
        .onHover { isHovering in
            if isHovering {
                hoveredAssistantID = id
            } else if hoveredAssistantID == id {
                hoveredAssistantID = nil
            }
        }
        .animation(.easeOut(duration: 0.14), value: showsActions)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Assistant response")
    }

    private func assistantActions(
        for content: String,
        id: String,
        isActive: Bool
    ) -> some View {
        HStack(spacing: 2) {
            Button {
                copyAssistantMessage(content, id: id)
            } label: {
                Label(
                    copiedAssistantID == id ? "Copied" : "Copy response",
                    systemImage: copiedAssistantID == id ? "checkmark" : "doc.on.doc")
                    .labelStyle(.iconOnly)
                    .frame(width: 26, height: 26)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(copiedAssistantID == id ? "Copied" : "Copy response")

            if isActive && !isRunning && canRegenerate {
                Button(action: onRegenerate) {
                    Label("Regenerate response", systemImage: "arrow.clockwise")
                        .labelStyle(.iconOnly)
                        .frame(width: 26, height: 26)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Regenerate response")
            }
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(.secondary)
        .padding(3)
        .background(.regularMaterial, in: Capsule())
        .overlay {
            Capsule().stroke(TurboFieldfareMacTheme.border, lineWidth: 0.5)
        }
    }

    private func systemMessage(_ content: String) -> some View {
        Text(content)
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
    }

    private func followNewContent(using proxy: ScrollViewProxy) {
        guard isFollowingLatest else {
            hasUnseenContent = true
            return
        }
        // A new token arrives frequently while decoding. Animating every
        // scroll update makes the transcript shimmer and fight the user's
        // scroll position, so content-following stays immediate.
        scrollToBottom(proxy, animated: false)
    }

    private func scrollToBottom(
        _ proxy: ScrollViewProxy,
        animated: Bool = true
    ) {
        if animated {
            withAnimation(.easeOut(duration: 0.18)) {
                proxy.scrollTo("transcript-bottom", anchor: .bottom)
            }
        } else {
            proxy.scrollTo("transcript-bottom", anchor: .bottom)
        }
    }

    private var editedPromptEditorHeight: CGFloat {
        let explicitLines = max(
            1,
            editedPrompt.split(separator: "\n", omittingEmptySubsequences: false).count)
        let wrappedLines = max(explicitLines, max(1, (editedPrompt.count + 67) / 68))
        return min(176, max(72, CGFloat(wrappedLines) * 22 + 28))
    }

    private var canSubmitPromptEdit: Bool {
        canSubmitEditedLastPrompt
            && !editedPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func beginPromptEditing(_ content: String, index: Int) {
        guard canEditLastPrompt, index == latestUserMessageIndex else { return }
        editedPrompt = content
        editingUserMessageIndex = index
        Task { @MainActor in
            editedPromptFocused = true
        }
    }

    private func cancelPromptEditing() {
        editedPromptFocused = false
        editingUserMessageIndex = nil
        editedPrompt = ""
    }

    private func submitPromptEdit() {
        guard canSubmitPromptEdit else { return }
        let replacement = editedPrompt
        guard onEditLastPrompt(replacement) else { return }
        cancelPromptEditing()
    }

    private func copyAssistantMessage(_ content: String, id: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(content, forType: .string)
        copiedAssistantID = id
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.5))
            if copiedAssistantID == id { copiedAssistantID = nil }
        }
    }

    private func synchronizeInitialResponse() {
        if isRunning {
            beginStreamingResponse()
        } else {
            showFinishedResponse()
        }
    }

    private func synchronizeResponse(_ source: String) {
        if isRunning {
            responseBuffer.receive(source)
            scheduleResponseRevealIfNeeded()
        } else {
            showFinishedResponse()
        }
    }

    private func beginStreamingResponse() {
        cancelResponseReveal()
        responseBuffer.begin(with: output)
        scheduleResponseRevealIfNeeded()
    }

    private func finishStreamingResponse() {
        cancelResponseReveal()
        responseBuffer.receive(output)
        responseBuffer.finish()
    }

    private func showFinishedResponse() {
        cancelResponseReveal()
        responseBuffer.begin(with: output)
        responseBuffer.finish()
    }

    private func scheduleResponseRevealIfNeeded() {
        guard responseBuffer.hasPendingText else { return }
        revealResponseBatch()
        guard responseBuffer.hasPendingText, responseRevealTask == nil else { return }

        let generation = responseRevealGeneration
        responseRevealTask = Task { @MainActor in
            defer {
                if generation == responseRevealGeneration {
                    responseRevealTask = nil
                    // A source snapshot can arrive in the narrow interval
                    // after this task observes an empty buffer and before its
                    // deferred cleanup runs. Start the successor here so that
                    // batch never waits for another model token.
                    if responseBuffer.hasPendingText {
                        scheduleResponseRevealIfNeeded()
                    }
                }
            }

            while !Task.isCancelled, generation == responseRevealGeneration {
                guard responseBuffer.hasPendingText else { return }
                do {
                    try await Task.sleep(for: .milliseconds(45))
                } catch {
                    return
                }
                guard !Task.isCancelled, generation == responseRevealGeneration else {
                    return
                }
                revealResponseBatch()
            }
        }
    }

    private func revealResponseBatch() {
        let pending = responseBuffer.pendingCharacterCount
        guard pending > 0 else { return }
        let count = reduceMotion
            ? pending
            : StreamingResponseBuffer.recommendedBatchSize(for: pending)
        _ = responseBuffer.revealNext(maximumCharacterCount: count)
    }

    private func cancelResponseReveal() {
        responseRevealGeneration &+= 1
        responseRevealTask?.cancel()
        responseRevealTask = nil
    }
}

private struct MarkdownMessageText: View, Equatable {
    let content: String
    let isStreaming: Bool

    private let renderer = ResponseMarkdownRenderer()

    var body: some View {
        if isStreaming {
            Text(renderer.streamingText(content))
                .lineSpacing(3)
        } else {
            Text(AttributedString(renderer.render(content).attributedString))
        }
    }

    nonisolated static func == (lhs: MarkdownMessageText, rhs: MarkdownMessageText) -> Bool {
        lhs.content == rhs.content && lhs.isStreaming == rhs.isStreaming
    }
}

private struct StreamingStatusLabel: View {
    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(TurboFieldfareMacTheme.accentColor)
                .frame(width: 5, height: 5)
            Text("Generating")
        }
        .font(.system(size: 9, weight: .semibold, design: .rounded))
        .tracking(0.45)
        .foregroundStyle(TurboFieldfareMacTheme.accentColor)
        .accessibilityLabel("Generating response")
    }
}
