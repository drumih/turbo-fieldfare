import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

/// A chat-style transcript: user turns are compact trailing bubbles while
/// assistant turns use the full Markdown-rendered reading column.
struct ChatTranscriptView: View {
    let prompt: String
    let messages: [AppChatMessage]
    let output: String
    let isRunning: Bool
    let canRegenerate: Bool
    let onRegenerate: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isFollowingLatest = true
    @State private var hasUnseenContent = false
    @State private var hoveredAssistantID: String?
    @State private var copiedAssistantID: String?
    @State private var responseBuffer = StreamingResponseBuffer()
    @State private var responseRevealTask: Task<Void, Never>?
    @State private var responseRevealGeneration: UInt = 0

    var body: some View {
        transcriptContent(response: responseBuffer.displayedText)
            .textSelection(.enabled)
            .onAppear(perform: synchronizeInitialResponse)
            .onChange(of: output) { _, source in
                synchronizeResponse(source)
            }
            .onChange(of: isRunning) { _, isRunning in
                if isRunning {
                    beginStreamingResponse()
                } else {
                    finishStreamingResponse()
                }
            }
            .onChange(of: messages) { _, _ in
                guard !isRunning else { return }
                showFinishedResponse()
            }
            .onDisappear(perform: cancelResponseReveal)
    }

    private func transcriptContent(response: String) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 26) {
                    messageList

                    if messages.isEmpty, !prompt.isEmpty {
                        userMessage(prompt)
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
                userMessage(message.content)
            case .assistant:
                assistantMessage(message.content, id: "assistant-\(index)", isActive: false)
            }
        }
    }

    private func userMessage(_ content: String) -> some View {
        HStack {
            Spacer(minLength: 48)
            Text(content)
                .font(.body)
                .foregroundStyle(.primary)
                .lineSpacing(3)
                .textSelection(.enabled)
                .frame(maxWidth: 620, alignment: .leading)
                .padding(.horizontal, 17)
                .padding(.vertical, 12)
                .background(
                    TurboFieldfareMacTheme.accentSurface,
                    in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(TurboFieldfareMacTheme.accentColor.opacity(0.14), lineWidth: 0.5)
                }
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Your message")
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
