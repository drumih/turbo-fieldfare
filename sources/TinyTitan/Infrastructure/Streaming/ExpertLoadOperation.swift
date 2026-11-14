import Foundation
import TinyTitanCore

public enum ExpertLoadOperationState: Sendable, Equatable {
    case submitted
    case inFlight
    case completed
    case failed
}

/// One routed-expert storage batch whose submission is independent from its
/// completion. The operation is deliberately backend-neutral: pread and Metal
/// I/O publish the same state and error contract to the decode scheduler.
///
/// unchecked-invariant: all mutable state and continuations are guarded by
/// `condition`; terminal transition happens exactly once.
public final class ExpertLoadOperation: @unchecked Sendable {
    private let condition = NSCondition()
    private var currentState: ExpertLoadOperationState = .submitted
    private var failure: (any Error)?
    private var continuations: [CheckedContinuation<Void, any Error>] = []
    private var firstStagePublished = false
    private var firstStageContinuations: [CheckedContinuation<Void, any Error>] = []

    public let submittedNanos: UInt64
    public let completionToken: ExpertIOCompletionToken?
    /// Present only for the Metal-I/O staging route. The dependent compute
    /// command owns the copy into the expert cache and must release it after
    /// that command completes.
    let metalStagingTransfer: MetalExpertStagingTransfer?
    /// A staged Metal load is not resident until the event-gated GPU blit has
    /// completed. Pread writes cache slots directly and therefore remains
    /// false.
    let requiresGPUFinalization: Bool
    /// True when the read is split and `publishFirstStage` will fire between
    /// the front (gate+up) and tail (down) fetches. Whole-region backends and
    /// miss-free plans never publish, and a caller that plans work against the
    /// first stage must check this rather than assume the split exists.
    let publishesFirstStage: Bool
    private var startedAtNanos: UInt64 = 0
    private var completedAtNanos: UInt64 = 0

    private let eventCoordinator: ExpertIOEventCoordinator?
    private let backendSignalsEvent: Bool

    init(
        completionToken: ExpertIOCompletionToken? = nil,
        eventCoordinator: ExpertIOEventCoordinator? = nil,
        backendSignalsEvent: Bool = false,
        metalStagingTransfer: MetalExpertStagingTransfer? = nil,
        requiresGPUFinalization: Bool = false,
        publishesFirstStage: Bool = false,
        submittedNanos: UInt64 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    ) {
        precondition(
            !requiresGPUFinalization || metalStagingTransfer != nil,
            "a staged Metal load requires a staging transfer")
        self.completionToken = completionToken
        self.eventCoordinator = eventCoordinator
        self.backendSignalsEvent = backendSignalsEvent
        self.metalStagingTransfer = metalStagingTransfer
        self.requiresGPUFinalization = requiresGPUFinalization
        self.publishesFirstStage = publishesFirstStage
        self.submittedNanos = submittedNanos
    }

    func releaseStagingTransfer() { metalStagingTransfer?.release() }

    public var state: ExpertLoadOperationState {
        condition.withLock { currentState }
    }

    public var submissionToStartNanos: UInt64 {
        condition.withLock {
            startedAtNanos >= submittedNanos ? startedAtNanos - submittedNanos : 0
        }
    }

    public var loadNanos: UInt64 {
        condition.withLock {
            completedAtNanos >= startedAtNanos ? completedAtNanos - startedAtNanos : 0
        }
    }

    public var startedNanos: UInt64 { condition.withLock { startedAtNanos } }
    public var completedNanos: UInt64 { condition.withLock { completedAtNanos } }

    /// Blocking compatibility wait for tests and the v4.1 fallback path.
    public func wait() throws {
        condition.lock()
        while currentState == .submitted || currentState == .inFlight {
            condition.wait()
        }
        let error = failure
        condition.unlock()
        if let error { throw error }
    }

    /// Suspension-only completion for schedulers. No worker thread is occupied
    /// while the storage service owns the request.
    public func completion() async throws {
        try await withCheckedThrowingContinuation { continuation in
            condition.lock()
            switch currentState {
            case .completed:
                condition.unlock()
                continuation.resume()
            case .failed:
                let error =
                    failure
                    ?? ModelError.internalInconsistency(
                        detail: "the expert load failed without recording an error")
                condition.unlock()
                continuation.resume(throwing: error)
            case .submitted, .inFlight:
                continuations.append(continuation)
                condition.unlock()
            }
        }
    }

    /// Signals that the split read's front range (gate+up) has landed in the
    /// cache slots. Whole-region operations never publish; their stage waiters
    /// are released by `finish` instead. Idempotent.
    func publishFirstStage() {
        condition.lock()
        guard !firstStagePublished else {
            condition.unlock()
            return
        }
        firstStagePublished = true
        let waiting = firstStageContinuations
        firstStageContinuations.removeAll(keepingCapacity: false)
        condition.unlock()
        for continuation in waiting {
            continuation.resume()
        }
    }

    /// Resumes once the operation's first stage is resident: at
    /// `publishFirstStage` for split reads, or at completion for whole-region
    /// operations and failures.
    func firstStageCompletion() async throws {
        try await withCheckedThrowingContinuation { continuation in
            condition.lock()
            if firstStagePublished {
                condition.unlock()
                continuation.resume()
                return
            }
            switch currentState {
            case .completed:
                condition.unlock()
                continuation.resume()
            case .failed:
                let error =
                    failure
                    ?? ModelError.internalInconsistency(
                        detail: "the expert load failed without recording an error")
                condition.unlock()
                continuation.resume(throwing: error)
            case .submitted, .inFlight:
                firstStageContinuations.append(continuation)
                condition.unlock()
            }
        }
    }

    func markInFlight() {
        condition.withLock {
            precondition(
                currentState == .submitted,
                "expert load operation started more than once")
            currentState = .inFlight
            startedAtNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        }
    }

    func finish(_ result: Result<Void, any Error>) {
        condition.lock()
        precondition(
            currentState == .submitted || currentState == .inFlight,
            "expert load operation completed more than once")
        completedAtNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        switch result {
        case .success:
            currentState = .completed
        case .failure(let error):
            currentState = .failed
            failure = error
        }
        let waiting = continuations
        continuations.removeAll(keepingCapacity: false)
        let waitingForStage = firstStageContinuations
        firstStageContinuations.removeAll(keepingCapacity: false)
        condition.broadcast()
        condition.unlock()

        if let completionToken, let eventCoordinator {
            if backendSignalsEvent {
                eventCoordinator.recordBackendSignal(completionToken)
            } else {
                switch result {
                case .success:
                    eventCoordinator.publish(completionToken, succeeded: true)
                case .failure:
                    // Failure still advances the timeline so a pre-submitted GPU
                    // command cannot deadlock. Event-aware kernels see status=2
                    // and avoid dereferencing the incomplete expert slots.
                    eventCoordinator.publish(completionToken, succeeded: false)
                }
            }
        }

        for continuation in waiting {
            switch result {
            case .success:
                continuation.resume()
            case .failure(let error):
                continuation.resume(throwing: error)
            }
        }
        for continuation in waitingForStage {
            switch result {
            case .success:
                continuation.resume()
            case .failure(let error):
                continuation.resume(throwing: error)
            }
        }
    }
}

enum ExpertIOPriority: Sendable {
    case demand
    case speculative
}

/// Persistent bounded submission service. The C pread backend owns the actual
/// fixed reader threads; this service gives queued demand batches precedence
/// over queued speculative batches.
/// unchecked-invariant: both queues are read and mutated only under `condition`.
final class ExpertIOScheduler: @unchecked Sendable {
    static let shared = ExpertIOScheduler(workerCount: 4)

    private let condition = NSCondition()
    private var demandQueue: [@Sendable () -> Void] = []
    private var speculativeQueue: [@Sendable () -> Void] = []

    init(workerCount: Int) {
        precondition(workerCount > 0)
        for index in 0..<workerCount {
            DispatchQueue(label: "TinyTitan.expert-io.\(index)", qos: .userInitiated)
                .async { [weak self] in
                    self?.runWorker()
                }
        }
    }

    func submit(_ work: @escaping @Sendable () -> Void) {
        submit(priority: .demand, work)
    }

    func submit(priority: ExpertIOPriority, _ work: @escaping @Sendable () -> Void) {
        condition.lock()
        switch priority {
        case .demand:
            demandQueue.append(work)
        case .speculative:
            speculativeQueue.append(work)
        }
        condition.signal()
        condition.unlock()
    }

    private func runWorker() {
        while true {
            condition.lock()
            while demandQueue.isEmpty && speculativeQueue.isEmpty {
                condition.wait()
            }
            let work =
                demandQueue.isEmpty
                ? speculativeQueue.removeFirst() : demandQueue.removeFirst()
            condition.unlock()
            work()
        }
    }
}
