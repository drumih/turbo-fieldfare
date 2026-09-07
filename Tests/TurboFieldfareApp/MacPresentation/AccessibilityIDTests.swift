import Foundation
import Testing
import TurboFieldfareMacPresentation

struct AccessibilityIDTests {
    /// The driver matches identifiers exactly and the coverage script greps
    /// them, so a name outside the `surface.control` shape would be reachable
    /// by neither.
    @Test func everyIdentifierIsNamespacedBySurface() {
        for id in AccessibilityID.allCases {
            #expect(id.rawValue.range(of: #"^[a-z]+(\.[A-Za-z]+)+$"#, options: .regularExpression) != nil,
                    "\(id.rawValue) is not surface.control")
        }
        for prefix in AccessibilityID.Prefix.allCases {
            #expect(prefix.rawValue.hasSuffix("."), "\(prefix.rawValue) must end in a dot")
            #expect(!AccessibilityID.allCases.contains { $0.rawValue == String(prefix.rawValue.dropLast()) },
                    "\(prefix.rawValue) collides with a static identifier")
        }
    }

    /// A row and its menu share the row prefix; the driver tells them apart by
    /// exact match, and a case tells them apart by the family name.
    @Test func mintedIdentifiersCarryTheirFamilyPrefix() {
        let id = UUID()
        #expect(AccessibilityID.row(id) == "history.row." + id.uuidString)
        #expect(AccessibilityID.rowMenu(id) == "history.row.menu." + id.uuidString)
        #expect(AccessibilityID.remove("staged-1") == "composer.remove.staged-1")
        #expect(AccessibilityID.example("code") == "examples.code")
    }
}
