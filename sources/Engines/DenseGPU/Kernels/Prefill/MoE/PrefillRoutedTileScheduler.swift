enum PrefillRoutedTileSchedulerDrainReason: Sendable, Equatable {
    case pendingTileHasNoAssignedSlots
    case avoidingSlotPlanUnavailable
    case maxPendingDepthReached
}

enum PrefillRoutedTileSchedulerDecision: Sendable, Equatable {
    case issueWithoutPending
    case prefetchNext(avoidingSlots: [Int])
    case drainBeforeIssue(reason: PrefillRoutedTileSchedulerDrainReason)
}

struct PrefillRoutedTileSchedulerInput: Sendable, Equatable {
    let hasPendingTile: Bool
    let pendingDepth: Int
    let pendingAssignedSlots: [Int]
    let avoidingSlotPlanAvailable: Bool

    init(
        hasPendingTile: Bool,
        pendingDepth: Int? = nil,
        pendingAssignedSlots: [Int],
        avoidingSlotPlanAvailable: Bool
    ) {
        self.hasPendingTile = hasPendingTile
        self.pendingDepth = max(0, pendingDepth ?? (hasPendingTile ? 1 : 0))
        self.pendingAssignedSlots = pendingAssignedSlots
        self.avoidingSlotPlanAvailable = avoidingSlotPlanAvailable
    }
}

struct PrefillRoutedTileSchedulerConfig: Sendable, Equatable {
    let maxPendingDepth: Int
    let tileExperts: Int

    init(maxPendingDepth: Int = 1, tileExperts: Int = 8) {
        self.maxPendingDepth = max(1, maxPendingDepth)
        self.tileExperts = max(1, min(16, tileExperts))
    }

    func fitsSlotBudget(slotCount: Int, reservedHits: Int = 0) -> Bool {
        guard slotCount > 0, reservedHits >= 0 else { return false }
        return (maxPendingDepth + 1) * tileExperts + reservedHits <= slotCount
    }

    /// Shrinks the I/O tile when a deliberately small streamed-expert cache
    /// cannot hold the default tile plus its one-tile lookahead. This keeps
    /// bounded-cache runtimes streaming instead of requiring more resident
    /// expert slots merely to enter the batched prefill path.
    func fitting(slotCount: Int, reservedHits: Int = 0) -> Self? {
        guard slotCount > reservedHits, reservedHits >= 0 else { return nil }
        let availablePerTile = (slotCount - reservedHits) / (maxPendingDepth + 1)
        guard availablePerTile > 0 else { return nil }
        return Self(
            maxPendingDepth: maxPendingDepth,
            tileExperts: min(tileExperts, availablePerTile))
    }
}

struct PrefillRoutedTileScheduler: Sendable, Equatable {
    let config: PrefillRoutedTileSchedulerConfig

    init(config: PrefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()) {
        self.config = config
    }

    func decide(_ input: PrefillRoutedTileSchedulerInput) -> PrefillRoutedTileSchedulerDecision {
        guard input.hasPendingTile else {
            return .issueWithoutPending
        }
        guard input.pendingDepth <= config.maxPendingDepth else {
            return .drainBeforeIssue(reason: .maxPendingDepthReached)
        }
        guard !input.pendingAssignedSlots.isEmpty else {
            return .drainBeforeIssue(reason: .pendingTileHasNoAssignedSlots)
        }
        guard input.avoidingSlotPlanAvailable else {
            return .drainBeforeIssue(reason: .avoidingSlotPlanUnavailable)
        }
        return .prefetchNext(avoidingSlots: input.pendingAssignedSlots)
    }
}
