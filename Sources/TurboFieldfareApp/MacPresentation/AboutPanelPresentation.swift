import AppKit
import Foundation

public struct AboutPanelContent: Equatable, Sendable {
    public let applicationName: String
    public let shortVersion: String

    public init(applicationName: String, shortVersion: String) {
        self.applicationName = applicationName
        self.shortVersion = shortVersion
    }
}

public enum AboutPanelPresentation {
    public static let applicationName = "TurboFieldfare"

    // Most users build from a clone, where there is no Info.plist to read a
    // version from, so this constant is what they see. Scripts/check_app_version.rb
    // fails CI when it falls behind the newest published release.
    public static let fallbackShortVersion = "0.7.0"

    private static let licenseURL = URL(
        string: "https://github.com/drumih/turbo-fieldfare/blob/main/LICENSE")!

    public static let repositoryURL = URL(string: "https://github.com/drumih/turbo-fieldfare")!

    public static let repositoryLinkText = "github.com/drumih/turbo-fieldfare"

    private static let licenseName = "Apache License 2.0"

    public static func content(infoDictionary: [String: Any]?) -> AboutPanelContent {
        AboutPanelContent(
            applicationName: applicationName,
            shortVersion: shortVersion(infoDictionary: infoDictionary))
    }

    public static func shortVersion(infoDictionary: [String: Any]?) -> String {
        let bundleVersion = (infoDictionary?["CFBundleShortVersionString"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let bundleVersion, !bundleVersion.isEmpty else { return fallbackShortVersion }
        return bundleVersion
    }

    @MainActor
    public static func options(infoDictionary: [String: Any]?,
                               icon: NSImage?) -> [NSApplication.AboutPanelOptionKey: Any] {
        let content = content(infoDictionary: infoDictionary)
        var options: [NSApplication.AboutPanelOptionKey: Any] = [
            .applicationName: content.applicationName,
            .applicationVersion: content.shortVersion,
            .credits: credits(),
        ]
        // The documented fallback is NSImage(named: "NSApplicationIcon"), which a
        // bare executable cannot resolve, so pass the Dock icon explicitly.
        if let icon { options[.applicationIcon] = icon }
        return options
    }

    @MainActor
    public static func credits() -> NSAttributedString {
        let paragraph = NSMutableParagraphStyle()
        paragraph.alignment = .center
        // "Apache License 2.0" is a link, and letting it wrap mid-phrase reads as
        // a broken line, so the break is explicit.
        let licenseLead = "TurboFieldfare is licensed under the"
        let text = """
            \(repositoryLinkText)
            \(licenseLead)
            \(licenseName).
            """
        let credits = NSMutableAttributedString(
            string: text,
            attributes: [
                .font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize),
                .foregroundColor: NSColor.secondaryLabelColor,
                .paragraphStyle: paragraph,
            ])
        link(licenseName, to: licenseURL, in: credits, text: text)
        link(repositoryLinkText, to: repositoryURL, in: credits, text: text)
        if let range = text.range(of: licenseLead) {
            let spaced = NSMutableParagraphStyle()
            spaced.alignment = .center
            spaced.paragraphSpacingBefore = 8
            credits.addAttribute(.paragraphStyle,
                                 value: spaced,
                                 range: NSRange(range, in: text))
        }
        return credits
    }

    private static func link(_ substring: String,
                             to url: URL,
                             in credits: NSMutableAttributedString,
                             text: String) {
        guard let range = text.range(of: substring) else { return }
        credits.addAttribute(.link, value: url, range: NSRange(range, in: text))
    }
}
