import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

/// A chat-style transcript: user turns are compact trailing bubbles while
/// assistant turns use the full Markdown-rendered reading column.
struct ChatTranscriptView: View {
    let prompt: String
    let messages: [AppChatMessage]
    let output: String
    let mailbox: GenerationTranscriptMailbox?
    let isRunning: Bool

    private let renderer = ResponseMarkdownRenderer()

    var body: some View {
        Group {
            if isRunning {
                TimelineView(.periodic(from: .now, by: 0.1)) { _ in
                    transcriptContent(response: effectiveOutput)
                }
            } else {
                transcriptContent(response: effectiveOutput)
            }
        }
        .textSelection(.enabled)
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
                        assistantMessage(response)
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
            .onChange(of: messages) { _, _ in
                scrollToBottom(proxy)
            }
            .onChange(of: response) { _, _ in
                scrollToBottom(proxy)
            }
            .onAppear {
                scrollToBottom(proxy, animated: false)
            }
        }
    }

    @ViewBuilder
    private var messageList: some View {
        ForEach(Array(messages.enumerated()), id: \.offset) { _, message in
            switch message.role {
            case .system:
                systemMessage(message.content)
            case .user:
                userMessage(message.content)
            case .assistant:
                assistantMessage(message.content)
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

    private func assistantMessage(_ content: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
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

                if isRunning && content == effectiveOutput {
                    Text("GENERATING")
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .tracking(0.8)
                        .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                }
            }

            if content.isEmpty && isRunning {
                ProgressView()
                    .controlSize(.small)
                    .tint(TurboFieldfareMacTheme.accentColor)
                    .padding(.leading, 32)
                    .padding(.top, 2)
            } else if !content.isEmpty {
                Text(AttributedString(renderer.render(content).attributedString))
                    .font(.body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.leading, 32)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Assistant response")
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

    private var effectiveOutput: String {
        let mailboxText = mailbox?.completeText ?? ""
        return mailboxText.isEmpty ? output : mailboxText
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
}
