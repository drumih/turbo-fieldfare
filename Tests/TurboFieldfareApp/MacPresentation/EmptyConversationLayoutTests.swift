import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct EmptyConversationLayoutTests {
    @Test func compactPlaceholderReservesRoomForItsModelAction() {
        // Centering only the icon put the load button behind the examples at 200%.
        let bounds = CGRect(x: 0, y: 20, width: 600, height: 140)
        let center = EmptyConversationLayout.iconCenterY(
            in: bounds, iconHeight: 32, contentHeight: 64, spacing: 8)
        #expect(center == 72)
        #expect(center + 16 + 8 + 64 == bounds.maxY)
    }

    @Test func roomyPlaceholderKeepsItsCenteredIcon() {
        #expect(EmptyConversationLayout.iconCenterY(
            in: CGRect(x: 0, y: 20, width: 600, height: 300),
            iconHeight: 32, contentHeight: 64, spacing: 8) == 170)
    }

    @Test func exactFitAndUndersizedAreasKeepTheIconWithinTheTopEdge() {
        for height: CGFloat in [104, 60, 0] {
            #expect(EmptyConversationLayout.iconCenterY(
                in: CGRect(x: 0, y: 20, width: 600, height: height),
                iconHeight: 32, contentHeight: 64, spacing: 8) == 36)
        }
    }
}
