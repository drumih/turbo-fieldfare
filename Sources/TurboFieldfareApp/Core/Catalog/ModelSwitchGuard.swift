import Foundation

public enum ModelSwitchVerdict: Equatable, Sendable {
    case allowed
    case alreadyLoaded
    case blockedByGeneration
    case notInstalled
    case busy
}

/// Decides whether a model switch may proceed.
///
/// Pure so the rules are testable without a runner: `AppModel` performs the
/// unload and load, this only says whether it may.
public enum ModelSwitchGuard {
    public static func evaluate(target: ModelCatalogEntry,
                                currentRepoID: String?,
                                loadState: AppModelLoadState,
                                isGenerating: Bool,
                                installStates: ModelInstallStates) -> ModelSwitchVerdict {
        if isGenerating { return .blockedByGeneration }
        // `isLoading` also covers cancelling and unloading despite its name.
        if loadState.isLoading { return .busy }
        // Only a *ready* same-model selection is a no-op. After a failed load,
        // reselecting the same model has to be allowed or the user would have
        // no way to retry without switching away and back.
        if target.repoID == currentRepoID, loadState.isReady { return .alreadyLoaded }
        guard installStates.installedRepoIDs.contains(target.repoID) else {
            return .notInstalled
        }
        return .allowed
    }
}
