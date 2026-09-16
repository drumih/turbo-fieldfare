import SwiftUI

public struct EmptyConversationLayout: Layout {
    let spacing: CGFloat

    public init(spacing: CGFloat) { self.spacing = spacing }

    public func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        proposal.replacingUnspecifiedDimensions()
    }

    public func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        guard subviews.count == 2 else { return }

        let iconSize = subviews[0].sizeThatFits(.unspecified)
        let contentSize = subviews[1].sizeThatFits(
            ProposedViewSize(width: bounds.width, height: nil))
        let iconCenter = CGPoint(
            x: bounds.midX,
            y: Self.iconCenterY(in: bounds, iconHeight: iconSize.height,
                                contentHeight: contentSize.height, spacing: spacing))
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
    static func iconCenterY(
        in bounds: CGRect, iconHeight: CGFloat, contentHeight: CGFloat, spacing: CGFloat
    ) -> CGFloat {
        // Keep the familiar centered icon when space permits, but reserve room
        // for the model action above the composer in compact windows.
        max(bounds.minY + iconHeight / 2,
            min(bounds.midY, bounds.maxY - contentHeight - spacing - iconHeight / 2))
    }
}
