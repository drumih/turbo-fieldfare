import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0

    var body: some View {
        // Three columns in one HStack, not a NavigationSplitView. The split
        // view brought a second sidebar toggle of its own, moved our controls
        // whenever the sidebar opened, and left the sidebar column inert to the
        // mouse. Both side panels are now the same shape — a fixed-width column
        // beside a flexible middle — which is also what keeps a narrow window
        // from slicing one in half: the sides hold their width and the
        // transcript gives up the space.
        HStack(spacing: 0) {
            // Collapsed by animating an outer frame to zero and clipping, not
            // by leaving the hierarchy. A conditional panel with a `.move`
            // transition is torn down and rebuilt on every toggle, so its
            // contents relaid out mid-slide and the sibling `Divider()` — which
            // had no transition of its own — popped in and out a frame ahead of
            // the panel. Here the panel keeps its own width throughout, its
            // rule is part of it, and only the window the two are seen through
            // changes size.
            ConversationSidebarView(model: model)
                .frame(width: Self.sidebarWidth)
                .frame(maxHeight: .infinity)
                .background(Color(nsColor: .windowBackgroundColor))
                .overlay(alignment: .trailing) { verticalRule }
                // Trailing, so the panel slides out past the left edge rather
                // than being wiped away in place.
                .frame(width: model.isSidebarVisible ? Self.sidebarWidth : 0,
                       alignment: .trailing)
                .clipped()

            detail
        }
        .containerBackground(for: .window) {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).mix(
                        with: TurboFieldfareMacTheme.accentColor,
                        by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(TurboFieldfareMacTheme.accentColor)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
        .animation(.smooth(duration: 0.2), value: model.presentation.conversationAction)
        .transaction { transaction in
            if model.isRunning {
                transaction.animation = nil
            }
        }
    }

    static let sidebarWidth: CGFloat = 260
    static let inspectorWidth: CGFloat = 320
    /// How a side panel opens and closes.
    ///
    /// Carried by `withAnimation` around the state change itself, not by an
    /// `.animation(value:)` on this view. Attached further down — to the panel
    /// frames — the surrounding `HStack` took the final widths immediately and
    /// only the panel slid. Attached here at the root it animated the whole
    /// subtree, and that swept up anything else changing in the same run loop
    /// turn: clicking a toggle put its own button into the pressed state, whose
    /// release then faded back in over the length of the slide, so the control
    /// just clicked read as having disappeared. An explicit transaction covers
    /// the state change and nothing else.
    static let panelSlide: Animation = .smooth(duration: 0.22)
    /// What the transcript keeps when both panels are open at the window's
    /// minimum width. Lower than the 720 the old two-column layout could
    /// afford, because a third panel has to come from somewhere — and the
    /// alternative, letting a panel be clipped, is the thing this layout exists
    /// to prevent.
    ///
    /// Measured against the status row, which is the widest thing this column
    /// has to hold: the two chat controls, the pill, and the Inspector toggle.
    /// At 440 the pill ran out of room and truncated the model's own name to
    /// "Gem…", which is the one string on that row that has to stay readable.
    static let transcriptMinimumWidth: CGFloat = 530

    private var detail: some View {
        HStack(spacing: 0) {
            primaryContent
                .frame(minWidth: Self.transcriptMinimumWidth,
                       maxWidth: .infinity, maxHeight: .infinity)
                // Whatever this column holds stays inside it, including for
                // the frame or two of a slide when its width has not caught up.
                .clipped()

            InspectorView(model: model)
                .frame(width: Self.inspectorWidth)
                .frame(maxHeight: .infinity)
                .background(Color(nsColor: .windowBackgroundColor))
                .overlay(alignment: .leading) { verticalRule }
                .frame(width: model.isInspectorVisible ? Self.inspectorWidth : 0,
                       alignment: .leading)
                .clipped()
        }
    }

    /// The rule between two columns, drawn as part of the panel it belongs to.
    ///
    /// `Divider()` has no orientation of its own in an overlay — it reads the
    /// enclosing layout, and an overlay has none to read — so this is an
    /// explicit one-point rule in the separator colour.
    private var verticalRule: some View {
        Rectangle()
            .fill(Color(nsColor: .separatorColor))
            .frame(width: 1)
    }

    /// The status row is stacked above the content, not attached as a safe-area
    /// inset.
    ///
    /// `safeAreaInset` resolves its content's width a layout pass behind the
    /// container's, which is invisible at rest and wrong for the length of a
    /// panel slide: the row stayed at the width it had before the column
    /// started moving, so its leading control fell outside the column and was
    /// clipped away mid-animation. A stack is laid out in the same pass as the
    /// column that holds it.
    private var primaryContent: some View {
        VStack(spacing: 0) {
            StatusHUDView(model: model)

            Group {
                if model.requiresModelInstallation {
                    ModelInstallView(model: model)
                } else {
                    conversationView
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var conversationView: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                if model.hasOutputTranscript {
                    OutputPaneView(model: model)
                        .padding(.bottom, conversationChromeHeight)
                } else if conversationChromeHeight > 0 {
                    OutputPaneView(model: model)
                        .frame(
                            height: max(
                                0,
                                geometry.size.height - conversationChromeHeight))
                        .frame(maxHeight: .infinity, alignment: .top)
                }

                conversationChrome
                    .background {
                        GeometryReader { chromeGeometry in
                            Color.clear.preference(
                                key: ConversationChromeHeightKey.self,
                                value: chromeGeometry.size.height)
                        }
                    }
            }
            .onPreferenceChange(ConversationChromeHeightKey.self) { height in
                guard height > 0 else { return }
                var transaction = Transaction()
                transaction.disablesAnimations = true
                withTransaction(transaction) {
                    conversationChromeHeight = height
                }
            }
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ErrorBanner(model: model)
            ConversationStateNoticeView(model: model)
            if model.shouldShowPromptExamples {
                PromptExamplesView { preset in
                    model.promptText = preset.prompt
                }
            }
            ModelActionBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
        .animation(.smooth(duration: 0.2), value: model.promptText.isEmpty)
        .animation(.smooth(duration: 0.2), value: model.showPromptExamples)
    }

}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
