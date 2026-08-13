import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct AboutPanelPresentationTests {
    @Test func packagedBundleVersionWins() {
        let content = AboutPanelPresentation.content(
            infoDictionary: ["CFBundleShortVersionString": "0.4.2"])
        #expect(content.shortVersion == "0.4.2")
        #expect(content.applicationName == "TurboFieldfare")
    }

    @Test func cloneBuildFallsBackToTheCompiledVersion() {
        #expect(AboutPanelPresentation.shortVersion(infoDictionary: nil)
            == AboutPanelPresentation.fallbackShortVersion)
        #expect(AboutPanelPresentation.shortVersion(infoDictionary: [:])
            == AboutPanelPresentation.fallbackShortVersion)
    }

    @Test(arguments: ["", "   ", "\n"])
    func blankBundleVersionFallsBack(value: String) {
        #expect(AboutPanelPresentation.shortVersion(
            infoDictionary: ["CFBundleShortVersionString": value])
            == AboutPanelPresentation.fallbackShortVersion)
    }

    @Test func nonStringBundleVersionFallsBack() {
        #expect(AboutPanelPresentation.shortVersion(
            infoDictionary: ["CFBundleShortVersionString": 42])
            == AboutPanelPresentation.fallbackShortVersion)
    }

    @Test func bundleVersionIsTrimmed() {
        #expect(AboutPanelPresentation.shortVersion(
            infoDictionary: ["CFBundleShortVersionString": " 1.2.3\n"]) == "1.2.3")
    }

    @MainActor
    @Test func optionsCarryNameVersionAndIconWhenAvailable() {
        let options = AboutPanelPresentation.options(
            infoDictionary: ["CFBundleShortVersionString": "0.4.2"],
            icon: nil)
        #expect(options[.applicationName] as? String == "TurboFieldfare")
        #expect(options[.applicationVersion] as? String == "0.4.2")
        #expect(options[.applicationIcon] == nil)
        #expect(options[.credits] != nil)
    }

    @MainActor
    @Test func creditsLinkToTheLicenseAndTheRepository() {
        let credits = AboutPanelPresentation.credits()
        let text = credits.string
        #expect(text.contains("Apache License 2.0"))
        #expect(text.contains(AboutPanelPresentation.repositoryLinkText))

        var links: [URL] = []
        credits.enumerateAttribute(.link,
                                   in: NSRange(location: 0, length: credits.length)) { value, _, _ in
            if let url = value as? URL { links.append(url) }
        }
        #expect(links.count == 2)
        #expect(links.contains(AboutPanelPresentation.repositoryURL))
    }
}
