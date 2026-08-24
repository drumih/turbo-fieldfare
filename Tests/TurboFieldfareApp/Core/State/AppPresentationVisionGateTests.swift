import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The banner offers exactly the model actions the model will actually perform.
///
/// `canLoadModel`, `canReloadModel` and `canUnloadModel` all refuse while a
/// companion operation is in flight, but the presentation snapshot carried no
/// vision state — so this resolved to "Installed · Not loaded" with a `.load`
/// action, and `ModelActionBanner` rendered it as a fully enabled button whose
/// click did nothing for the whole 1.5 GB transfer.
@Suite struct AppPresentationVisionGateTests {
    private static let modelDirectory = URL(
        fileURLWithPath: "/tmp/TurboFieldfare/gemma4.gturbo", isDirectory: true)

    private func snapshot(
        loadState: AppModelLoadState = .notLoaded,
        hasStaleRuntime: Bool = false,
        companionOperationInProgress: Bool
    ) -> AppPresentationSnapshot {
        AppPresentationSnapshot(
            requiresInstallation: false,
            installState: .idle,
            installReadiness: .checking,
            loadState: loadState,
            hasStaleRuntime: hasStaleRuntime,
            isRunning: false,
            isGenerationCancellationPending: false,
            generationPhase: .idle,
            isVisionCompanionOperationInProgress: companionOperationInProgress)
    }

    @Test func anUnloadedModelOffersLoadWhenNoCompanionOperationRuns() {
        let state = AppPresentationState.resolve(
            snapshot(companionOperationInProgress: false))
        #expect(state.primaryAction == .load)
        #expect(state.conversationAction == .load)
    }

    /// The case the banner got wrong: unloaded — which every companion
    /// operation requires — with a transfer running.
    @Test func noModelActionIsOfferedWhileACompanionOperationRuns() {
        let state = AppPresentationState.resolve(
            snapshot(companionOperationInProgress: true))
        #expect(state.primaryAction == nil,
                "the banner would render an enabled button for \(state.label)")
        #expect(state.conversationAction == nil)
        #expect(state.showsActivity)
    }

    /// Not just the unloaded case: a stale runtime otherwise offers Reload, and
    /// `canReloadModel` refuses for the same reason.
    @Test func astaleRuntimeOffersNoReloadWhileACompanionOperationRuns() {
        let ready = AppModelLoadState.ready(
            modelDirectory: Self.modelDirectory, loadSeconds: 1)

        let during = AppPresentationState.resolve(
            snapshot(loadState: ready,
                     hasStaleRuntime: true,
                     companionOperationInProgress: true))
        #expect(during.primaryAction == nil)

        let idle = AppPresentationState.resolve(
            snapshot(loadState: ready,
                     hasStaleRuntime: true,
                     companionOperationInProgress: false))
        #expect(idle.primaryAction == .reload,
                "without a companion operation the stale-runtime prompt has to survive")
    }

    /// Every action the banner surfaces has to be one the model would run.
    @Test func everySurfacedActionIsOneTheModelWouldPerform() {
        let states: [AppModelLoadState] = [
            .notLoaded,
            .ready(modelDirectory: Self.modelDirectory, loadSeconds: 1),
        ]
        for loadState in states {
            let state = AppPresentationState.resolve(
                snapshot(loadState: loadState, companionOperationInProgress: true))
            #expect(state.primaryAction == nil && state.secondaryAction == nil,
                    "\(loadState) offered \(String(describing: state.primaryAction)) during a companion operation")
        }
    }
}
