import Foundation

/// Serialises installs.
///
/// Two concurrent repacks would contend for the same SSD bandwidth and, worse,
/// each would size its free-space reserve as if it were alone — so both could
/// pass the readiness check and then run the volume dry.
public actor ModelInstallQueue {
    private(set) public var activeRepoID: String?
    private var pending: [String] = []

    public init() {}

    public var pendingRepoIDs: [String] { pending }

    /// Returns true when the caller may start immediately.
    @discardableResult
    public func enqueue(repoID: String) -> Bool {
        if activeRepoID == repoID || pending.contains(repoID) { return false }
        guard activeRepoID != nil else {
            activeRepoID = repoID
            return true
        }
        pending.append(repoID)
        return false
    }

    /// Clears the active slot and promotes the next repository, if any.
    @discardableResult
    public func finishActive() -> String? {
        activeRepoID = pending.isEmpty ? nil : pending.removeFirst()
        return activeRepoID
    }

    public func cancel(repoID: String) {
        if activeRepoID == repoID {
            _ = finishActive()
            return
        }
        pending.removeAll { $0 == repoID }
    }
}

/// Install state for every known model, so the picker can render each row's
/// progress without the single-model assumption baked into `AppModel` today.
public struct ModelInstallStates: Equatable, Sendable {
    private var states: [String: AppModelInstallState] = [:]

    public init() {}

    public func state(for repoID: String) -> AppModelInstallState {
        states[repoID] ?? .idle
    }

    public mutating func setState(_ state: AppModelInstallState, for repoID: String) {
        states[repoID] = state
    }

    public var installedRepoIDs: Set<String> {
        Set(states.compactMap { key, value in
            if case .installed = value { return key }
            return nil
        })
    }
}
