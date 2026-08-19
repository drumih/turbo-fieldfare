import Foundation
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@Suite struct GenericErrorBannerPolicyTests {
    @Test func nilErrorIsHidden() {
        #expect(!GenericErrorBannerPolicy.shouldShow(error: nil, loadState: .notLoaded))
    }

    @Test func cancellationIsHidden() {
        #expect(!GenericErrorBannerPolicy.shouldShow(
            error: .cancelled, loadState: .notLoaded))
    }

    @Test func generationErrorIsShown() {
        #expect(GenericErrorBannerPolicy.shouldShow(
            error: .unknown("decode failed"), loadState: .notLoaded))
    }

    @Test func loadFailureIsOwnedByTheLoadPresentation() {
        let error = AppInferenceError.modelLoadFailed("broken")
        #expect(!GenericErrorBannerPolicy.shouldShow(
            error: error, loadState: .failed(error)))
    }
}
