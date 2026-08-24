import SwiftUI

/// An `info.circle` that both hovers and clicks: the tooltip alone was
/// invisible, so the explanation it carried never reached the people who read
/// the figure next to it as a bug.
struct InfoPopoverButton: View {
    let subject: String
    let text: String
    var arrowEdge: Edge = .trailing
    @State private var isShowingPopover = false

    var body: some View {
        Button {
            isShowingPopover.toggle()
        } label: {
            Image(systemName: "info.circle")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .frame(width: 16, height: 16)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(text)
        .popover(isPresented: $isShowingPopover, arrowEdge: arrowEdge) {
            Text(text)
                .font(.callout)
                .frame(width: 280, alignment: .leading)
                .padding()
        }
        .accessibilityLabel("\(subject) information")
        .accessibilityHint(text)
    }
}
