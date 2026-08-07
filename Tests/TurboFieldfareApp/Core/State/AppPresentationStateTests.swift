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
    }

    @Test func loadFailureIsVisibleAndRetryable() {
        let state = AppPresentationState.resolve(Self.installedSnapshot(
            loadState: .failed(.modelLoadFailed("synthetic"))))
        #expect(state.label == "Model load failed")
        #expect(state.detail == "Model load failed: synthetic")
        #expect(state.severity == .error)
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

    @Test func lastStopReasonUsesHumanLabelNotRawWireValue() {
        let ready = AppModelLoadState.ready(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            loadSeconds: 1)
        var snapshot = Self.installedSnapshot(loadState: ready)
        snapshot.lastStopReason = .endOfTurn
        let state = AppPresentationState.resolve(snapshot)
        // Normal end-of-turn is just "Done" — not "Done · finished".
        #expect(state.label == "Done")
        #expect(!state.label.contains("endOfTurn"))
        #expect(!state.label.contains("finished"))

        snapshot.lastStopReason = .maxTokens
        let lengthLimited = AppPresentationState.resolve(snapshot)
        #expect(lengthLimited.label == "Done · length limit")
        #expect(!lengthLimited.label.contains("maxTokens"))
    }

    @Test func insufficientSpaceUsesFormattedShortfall() {
        var snapshot = Self.installedSnapshot(loadState: .notLoaded)
        snapshot.requiresInstallation = true
        snapshot.installReadiness = .insufficientSpace(
            AppModelInstallRequirement(
                requiredBytes: 2_000_000_000,
                availableBytes: 500_000_000))
        let state = AppPresentationState.resolve(snapshot)
        #expect(state.label == "Not enough storage")
        let detail = state.detail ?? ""
        #expect(detail.contains("more required"))
        // Must not dump a raw integer byte count like "1500000000 bytes more required".
        #expect(!detail.contains("1500000000"))
        #expect(!detail.hasPrefix(String(1_500_000_000)))
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
