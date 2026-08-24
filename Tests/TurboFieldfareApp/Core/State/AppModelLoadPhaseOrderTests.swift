import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// Load phases are emitted in order by the runtime but applied through
/// independently created main-actor tasks, and ordering between those is not
/// guaranteed. A late `.loading` overwriting `.ready` leaves the UI showing a
/// phase the runtime has already finished, with Run disabled.
@Suite struct AppModelLoadPhaseOrderTests {
    @MainActor
    @Test func aLatePhaseNeverOverwritesANewerOne() async throws {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path

        model.applyLoadState(.loading(.validatingDirectory), generation: 0, sequence: 1)
        model.applyLoadState(.loading(.tokenizer), generation: 0, sequence: 2)
        // Arrives out of order, which is exactly what used to happen.
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 1),
                             generation: 0, sequence: 4)
        model.applyLoadState(.loading(.preparingRunner), generation: 0, sequence: 3)

        #expect(model.loadState.isReady,
                "a stale phase overwrote the ready state")
    }

    /// In-order phases must still all be applied.
    @MainActor
    @Test func phasesInOrderAreAllApplied() async throws {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path

        model.applyLoadState(.loading(.validatingDirectory), generation: 0, sequence: 1)
        #expect(model.loadState == .loading(.validatingDirectory))
        model.applyLoadState(.loading(.verifyingWeights), generation: 0, sequence: 2)
        #expect(model.loadState == .loading(.verifyingWeights))
        model.applyLoadState(.loading(.preparingRunner), generation: 0, sequence: 3)
        #expect(model.loadState == .loading(.preparingRunner))
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 1),
                             generation: 0, sequence: 4)
        #expect(model.loadState.isReady)
    }

    /// The same race against the other outcome, which the ordering guard did
    /// not cover: `.failed` is raised by the model itself at sequence 0, so it
    /// never advanced the counter, and a phase emitted just before the throw
    /// but delivered just after it passed the check and put the UI back into a
    /// load. With no task left to cancel and a state that is neither
    /// `.notLoaded` nor `.failed`, both Cancel and Load are disabled — a
    /// spinner with no way out.
    @MainActor
    @Test func aLatePhaseNeverReopensAFailedLoad() async throws {
        // A real install, because `canLoadModel` is gated on `isModelInstalled`:
        // pointed at a bare temporary directory this asserted a Load button that
        // could never be enabled for a reason that has nothing to do with phase
        // ordering.
        let directory = try makeCompleteModelInstall("load-phase-order-late-phase")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())

        model.applyLoadState(.loading(.validatingDirectory), generation: 0, sequence: 1)
        model.applyLoadState(.failed(.modelLoadFailed("bad companion pack")),
                             generation: 0)
        model.applyLoadState(.loading(.mappingImageTower), generation: 0, sequence: 2)

        #expect(model.loadState.isFailed,
                "a phase emitted before the failure reopened the load")
        #expect(model.canLoadModel,
                "the load could be neither cancelled nor restarted")
    }

    /// The seal lasts exactly one load: the next `beginLoad` resets the counter,
    /// so sealing must not make a later load ignore its own phases.
    @MainActor
    @Test func aFailedLoadDoesNotSealTheNextOne() async throws {
        let client = MockLifecycleInferenceClient()
        let directory = try makeCompleteModelInstall("load-phase-order-after-failure")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: client)

        model.applyLoadState(.failed(.modelLoadFailed("first attempt")), generation: 0)
        #expect(model.loadState.isFailed)

        model.loadModel()
        let deadline = Date().addingTimeInterval(5)
        while !model.loadState.isReady, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(model.loadState.isReady, "the next load's phases were sealed out")
    }

    /// A real load must still end ready, with the sequence reset for the next.
    @MainActor
    @Test func aRealLoadEndsReadyAndTheNextOneStartsClean() async throws {
        let client = MockLifecycleInferenceClient()
        let directory = try makeCompleteModelInstall("load-phase-order")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: client)

        for _ in 0..<3 {
            model.loadModel()
            let deadline = Date().addingTimeInterval(5)
            while !model.loadState.isReady, Date() < deadline {
                try? await Task.sleep(nanoseconds: 5_000_000)
            }
            #expect(model.loadState.isReady)
            model.loadState = .notLoaded
        }
    }
}
