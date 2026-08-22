import Testing
@testable import TurboFieldfareAppCore

/// Phase D item 1. The section was last in the inspector and shown
/// unconditionally in `public/`, while root put it second and hid it until a
/// text model existed. Neither did what the empty state needs, so the rule now
/// lives somewhere it can be checked without launching the app.
@Suite struct VisionSectionVisibilityTests {
    private func shows(modelInstalled: Bool,
                       packInstalled: Bool,
                       operating: Bool = false,
                       state: AppModelInstallState = .idle,
                       runtimeEnabled: Bool = true,
                       runtimeSupported: Bool = true) -> Bool {
        VisionSectionVisibility.shows(
            visionRuntimeEnabled: runtimeEnabled,
            visionRuntimeSupported: runtimeSupported,
            isModelInstalled: modelInstalled,
            isVisionPackInstalled: packInstalled,
            isCompanionOperationInProgress: operating,
            installState: state)
    }

    /// The screen where someone decides whether this app suits them. Hiding the
    /// section here meant image support was undiscoverable until after a
    /// 14.62 GB download.
    @Test func theEmptyStateAdvertisesImageSupport() {
        #expect(shows(modelInstalled: false, packInstalled: false))
    }

    @Test func aModelWithoutThePackOffersToInstallIt() {
        #expect(shows(modelInstalled: true, packInstalled: false))
    }

    /// Installed and healthy is the one case with nothing to say, so the section
    /// collapses rather than occupying the inspector permanently.
    @Test func anInstalledHealthyPackCollapsesTheSection() {
        #expect(!shows(modelInstalled: true, packInstalled: true))
    }

    @Test func workInProgressKeepsTheSectionOpen() {
        #expect(shows(modelInstalled: true, packInstalled: true, operating: true))
    }

    @Test func aFailureKeepsTheSectionOpenSoItCanBeRead() {
        #expect(shows(modelInstalled: true, packInstalled: true,
                      state: .failed("pack verification failed")))
        #expect(shows(modelInstalled: true, packInstalled: true,
                      state: .recoverable("download interrupted")))
    }

    /// The one hard off switch: nothing about the pack is shown when the runtime
    /// cannot use it at all.
    @Test func aDisabledVisionRuntimeHidesEveryCase() {
        #expect(!shows(modelInstalled: false, packInstalled: false, runtimeEnabled: false))
        #expect(!shows(modelInstalled: true, packInstalled: false, runtimeEnabled: false))
        #expect(!shows(modelInstalled: true, packInstalled: true,
                       operating: true, runtimeEnabled: false))
    }

    @Test func unsupportedHardwareKeepsTheExplanationVisible() {
        #expect(shows(modelInstalled: false, packInstalled: false,
                      runtimeSupported: false))
        #expect(shows(modelInstalled: true, packInstalled: true,
                      runtimeSupported: false))
    }
}
