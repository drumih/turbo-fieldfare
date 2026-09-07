import SwiftUI
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation

/// The banner above the composer when the chat on screen cannot be continued.
///
/// Above the composer rather than in an alert: alerts are not for information,
/// and this is information about something the user is looking at. It names the
/// consequence of the fix as well as the fix — raising the context reloads the
/// model and ends the open chat, which is not a thing to find out afterwards.
struct ConversationStateNoticeView: View {
    let model: AppModel

    var body: some View {
        if let content = ConversationStateNotice.content(
            for: model.openedConversationState,
            currentContext: model.maxContextTokens) {
            VStack(alignment: .leading, spacing: 8) {
                Text(content.title).font(.headline)
                Text(content.message)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 12) {
                    if let required = content.requiredContext,
                       let option = AppContextLengthOption.allCases.first(where: {
                           $0.tokens >= required
                       }) {
                        Button("Raise Context to \(option.shortLabel)") {
                            model.setMaxContextTokens(option.tokens)
                        }
                        .accessibilityIdentifier(.noticeRaiseContext)
                    }
                    Button("Start New Chat") { model.newChat() }
                        .disabled(!model.canStartNewChat)
                        .accessibilityIdentifier(.noticeNewChat)
                    if content.offersReread {
                        Button("Re-read Into a New Chat") { model.rereadIntoNewChat() }
                            .disabled(!model.canRereadIntoNewChat)
                            .accessibilityIdentifier(.noticeReread)
                            .help("Starts a different conversation from the same "
                                + "text. The model reads it fresh, so its replies "
                                + "will not be the ones above.")
                    }
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color(nsColor: .controlBackgroundColor)))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Color.secondary.opacity(0.25)))
        }
    }
}
