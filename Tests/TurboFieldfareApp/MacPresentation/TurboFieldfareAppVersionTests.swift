import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct TurboFieldfareAppVersionTests {
    @Test func fallbackIsUsedWhenBundleHasNoVersion() {
        let bundle = Bundle(for: BundleToken.self)
        #expect(TurboFieldfareAppVersion.resolve(bundle: bundle) == "0.4.1")
    }

    @Test func bundleVersionTakesPrecedence() {
        let bundle = Bundle(for: BundleToken.self)
        let overrides: [String: Any] = ["CFBundleShortVersionString": "1.2.3"]
        let overridden = OverrideBundle(inner: bundle, overrides: overrides)
        #expect(TurboFieldfareAppVersion.resolve(bundle: overridden) == "1.2.3")
    }

    @Test func customFallbackIsHonored() {
        let bundle = Bundle(for: BundleToken.self)
        #expect(TurboFieldfareAppVersion.resolve(
            bundle: bundle,
            fallback: "0.0.0-beta") == "0.0.0-beta")
    }
}

private final class BundleToken {}

private final class OverrideBundle: Bundle {
    private let overrides: [String: Any]

    init(inner: Bundle, overrides: [String: Any]) {
        self.overrides = overrides
        super.init(for: type(of: BundleToken.self))
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var infoDictionary: [String: Any]? {
        overrides
    }
}
