import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppPresentationStateTests {
    @Test func staleReadyPrecedesLastRun() {
        var snapshot = Self.installedSnapshot(loadState: .ready(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"), loadSeconds: 1))
        snapshot.hasStaleRuntime = true
        snapshot.lastStopReason = .maxTokens
        let state = AppPresentationState.resolve(snapshot)
        #expect(state.label == "Reload required")
        #expect(state.primaryAction == .reload)
    }

    @Test func loadFailureIsVisibleAndRetryable() {
        let state = AppPresentationState.resolve(Self.installedSnapshot(
            loadState: .failed(.modelLoadFailed("synthetic"))))
        #expect(state.label == "Model load failed")
        #expect(state.detail == "Model load failed: synthetic")
        #expect(state.severity == .error)
        #expect(state.primaryAction == .retryLoad)
    }

    @Test func readinessFailureIsStoppedNotActive() {
        var snapshot = Self.installedSnapshot(loadState: .notLoaded)
        snapshot.requiresInstallation = true
        snapshot.installReadiness = .failed("disk probe failed")
        let state = AppPresentationState.resolve(snapshot)
        #expect(state.label == "Storage check failed")
        #expect(state.detail == "disk probe failed")
        #expect(!state.showsActivity)
    }

    @Test func activeLifecycleStatesHaveExpectedPresentation() {
        var snapshot = Self.installedSnapshot(loadState: .loading(.tokenizer))
        var state = AppPresentationState.resolve(snapshot)
        #expect(state.showsActivity)

        snapshot.loadState = .unloading
        state = AppPresentationState.resolve(snapshot)
        #expect(state.label == "Unloading model")
    }

    @Test func installedAndReadyStatesHaveExpectedLabels() {
        var snapshot = Self.installedSnapshot(loadState: .notLoaded)
        var state = AppPresentationState.resolve(snapshot)
        #expect(state.label == "Installed · Not loaded")
        #expect(state.primaryAction == .load)

        snapshot.loadState = .ready(modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
                                    loadSeconds: 1)
        state = AppPresentationState.resolve(snapshot)
        #expect(state.label == "Ready")
    }

    @Test func conversationActionOnlyExposesIdleRecovery() {
        var snapshot = Self.installedSnapshot(loadState: .notLoaded)
        #expect(AppPresentationState.resolve(snapshot).conversationAction == .load)

        snapshot.loadState = .failed(.modelLoadFailed("synthetic"))
        #expect(AppPresentationState.resolve(snapshot).conversationAction == .retryLoad)

        snapshot.loadState = .ready(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            loadSeconds: 1)
        snapshot.hasStaleRuntime = true
        #expect(AppPresentationState.resolve(snapshot).conversationAction == .reload)

        snapshot.hasStaleRuntime = false
        #expect(AppPresentationState.resolve(snapshot).conversationAction == nil)

        snapshot.loadState = .loading(.tokenizer)
        #expect(AppPresentationState.resolve(snapshot).conversationAction == nil)

        snapshot.loadState = .unloading
        #expect(AppPresentationState.resolve(snapshot).conversationAction == nil)
    }

    @Test func lifecyclePriorityTable() {
        let ready = AppModelLoadState.ready(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"), loadSeconds: 1)
        var cases: [(AppPresentationSnapshot, String, Bool)] = []

        var snapshot = Self.installedSnapshot(loadState: .notLoaded)
        snapshot.requiresInstallation = true
        snapshot.installState = .checking
        cases.append((snapshot, "Checking installation", true))
        snapshot.installState = .cancelling
        cases.append((snapshot, "Cancelling installation", true))
        snapshot.installState = .cancelled
        cases.append((snapshot, "Download paused", false))
        snapshot.installState = .failed("network")
        cases.append((snapshot, "Installation failed", false))

        snapshot = Self.installedSnapshot(loadState: .cancelling)
        cases.append((snapshot, "Cancelling load", true))
        snapshot.loadState = .unloading
        cases.append((snapshot, "Unloading model", true))

        snapshot = Self.installedSnapshot(loadState: ready)
        snapshot.isRunning = true
        snapshot.generationPhase = .compressing
        cases.append((snapshot, "Compressing history", true))
        snapshot.generationPhase = .prefill
        cases.append((snapshot, "Prefill", false))
        snapshot.generationPhase = .decode
        cases.append((snapshot, "Generating", false))
        snapshot.isGenerationCancellationPending = true
        cases.append((snapshot, "Stopping", true))

        for (input, label, activity) in cases {
            let state = AppPresentationState.resolve(input)
            #expect(state.label == label)
            #expect(state.showsActivity == activity)
        }
    }

    @Test func prefillProgressUsesCompactFractionLabel() {
        let ready = AppModelLoadState.ready(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            loadSeconds: 1)
        var snapshot = Self.installedSnapshot(loadState: ready)
        snapshot.isRunning = true
        snapshot.generationPhase = .prefill
        snapshot.livePrefillDone = 128
        snapshot.livePrefillTotal = 514

        let state = AppPresentationState.resolve(snapshot)

        #expect(state.label == "Prefill (128/514)")
    }

    @Test func primaryHeaderActionCoversEveryModelRecoveryState() {
        var snapshot = Self.installedSnapshot(loadState: .notLoaded)
        #expect(AppPresentationState.resolve(snapshot).primaryAction == .load)

        snapshot.loadState = .loading(.tokenizer)
        #expect(AppPresentationState.resolve(snapshot).primaryAction == .cancelLoad)

        snapshot.loadState = .failed(.modelLoadFailed("synthetic"))
        #expect(AppPresentationState.resolve(snapshot).primaryAction == .retryLoad)

        snapshot.loadState = .ready(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            loadSeconds: 1)
        snapshot.hasStaleRuntime = true
        #expect(AppPresentationState.resolve(snapshot).primaryAction == .reload)

        snapshot.hasStaleRuntime = false
        snapshot.isRunning = true
        snapshot.generationPhase = .decode
        #expect(AppPresentationState.resolve(snapshot).primaryAction == nil)

        snapshot = Self.installedSnapshot(loadState: .notLoaded)
        snapshot.requiresInstallation = true
        snapshot.installReadiness = .ready(AppModelInstallRequirement(
            requiredBytes: 1,
            availableBytes: 2))
        #expect(AppPresentationState.resolve(snapshot).primaryAction == .install)

        snapshot.installState = .copyingPayload(
            reusedBytes: 0,
            downloadedThisRunBytes: 1,
            totalBytes: 2)
        #expect(AppPresentationState.resolve(snapshot).primaryAction == .cancelInstall)

        snapshot.installState = .cancelling
        #expect(AppPresentationState.resolve(snapshot).primaryAction == nil)
    }

    @Test func readyModelOnlyOffersUnloadAsASecondaryAction() {
        var snapshot = Self.installedSnapshot(loadState: .ready(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            loadSeconds: 1))

        var state = AppPresentationState.resolve(snapshot)
        #expect(state.primaryAction == nil)
        #expect(state.secondaryAction == .unload)

        snapshot.lastStopReason = .eos
        state = AppPresentationState.resolve(snapshot)
        #expect(state.label == "Done · eos")
        #expect(state.secondaryAction == .unload)
    }

    private static func installedSnapshot(loadState: AppModelLoadState) -> AppPresentationSnapshot {
        AppPresentationSnapshot(requiresInstallation: false,
                                installState: .idle,
                                installReadiness: .checking,
                                loadState: loadState,
                                hasStaleRuntime: false,
                                isRunning: false,
                                isGenerationCancellationPending: false,
                                generationPhase: .idle)
    }
}
