import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelSwitchGuardTests {
    private let installedDirectory = URL(fileURLWithPath: "/tmp/model.gturbo")

    private func makeEntry(repoID: String) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: repoID,
            revision: "main",
            trustTier: .custom,
            recordedIndexSHA256: "abc",
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    private func installedStates(_ repoIDs: [String]) -> ModelInstallStates {
        var states = ModelInstallStates()
        for repoID in repoIDs {
            states.setState(.installed(modelDirectory: installedDirectory), for: repoID)
        }
        return states
    }

    @Test func allowsSwitchToAnInstalledModel() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/two"),
            currentRepoID: "owner/one",
            loadState: .ready(modelDirectory: installedDirectory, loadSeconds: 1),
            isGenerating: false,
            installStates: installedStates(["owner/one", "owner/two"]))
        #expect(verdict == .allowed)
    }

    @Test func reportsAlreadyLoaded() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/one"),
            currentRepoID: "owner/one",
            loadState: .ready(modelDirectory: installedDirectory, loadSeconds: 1),
            isGenerating: false,
            installStates: installedStates(["owner/one"]))
        #expect(verdict == .alreadyLoaded)
    }

    @Test func blocksSwitchWhileGenerating() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/two"),
            currentRepoID: "owner/one",
            loadState: .ready(modelDirectory: installedDirectory, loadSeconds: 1),
            isGenerating: true,
            installStates: installedStates(["owner/one", "owner/two"]))
        #expect(verdict == .blockedByGeneration)
    }

    @Test func blocksSwitchToAnUninstalledModel() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/two"),
            currentRepoID: "owner/one",
            loadState: .ready(modelDirectory: installedDirectory, loadSeconds: 1),
            isGenerating: false,
            installStates: installedStates(["owner/one"]))
        #expect(verdict == .notInstalled)
    }

    @Test func blocksSwitchWhileAlreadyLoadingOrUnloading() {
        for busyState in [AppModelLoadState.loading(.tokenizer), .unloading, .cancelling] {
            let verdict = ModelSwitchGuard.evaluate(
                target: makeEntry(repoID: "owner/two"),
                currentRepoID: "owner/one",
                loadState: busyState,
                isGenerating: false,
                installStates: installedStates(["owner/one", "owner/two"]))
            #expect(verdict == .busy)
        }
    }

    @Test func allowsSwitchWhenNothingIsLoaded() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/two"),
            currentRepoID: nil,
            loadState: .notLoaded,
            isGenerating: false,
            installStates: installedStates(["owner/two"]))
        #expect(verdict == .allowed)
    }

    /// Selecting the same model again after a failed load must be allowed —
    /// otherwise a load failure would leave the user unable to retry without
    /// first switching to some other model.
    @Test func allowsReloadOfTheSameModelAfterAFailedLoad() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/one"),
            currentRepoID: "owner/one",
            loadState: .failed(.modelLoadFailed("weights missing")),
            isGenerating: false,
            installStates: installedStates(["owner/one"]))
        #expect(verdict == .allowed)
    }
}
