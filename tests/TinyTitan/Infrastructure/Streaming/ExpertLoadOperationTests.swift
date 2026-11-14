import Foundation
import Metal
import Testing

@testable import TinyTitan
import TinyTitanCore

@Suite struct ExpertLoadOperationTests {
    @Test func operationPublishesStateAndResumesAsyncWaiter() async throws {
        let operation = ExpertLoadOperation()
        #expect(operation.state == .submitted)

        let waiter = Task { try await operation.completion() }
        operation.markInFlight()
        #expect(operation.state == .inFlight)
        operation.finish(.success(()))

        try await waiter.value
        #expect(operation.state == .completed)
        #expect(operation.completedNanos >= operation.startedNanos)
    }

    @Test func operationPropagatesFailureToBlockingAndAsyncWaiters() async {
        struct ExpectedFailure: Error {}
        let operation = ExpertLoadOperation()
        operation.markInFlight()
        operation.finish(.failure(ExpectedFailure()))

        await #expect(throws: ExpectedFailure.self) {
            try await operation.completion()
        }
        #expect(throws: ExpectedFailure.self) { try operation.wait() }
        #expect(operation.state == .failed)
    }

    // MARK: - First-stage (split-read) signal

    /// The split reader publishes when the gate+up ranges of every miss have
    /// landed; a waiter must resume at that point, not at full completion.
    @Test func firstStageWaiterResumesAtPublicationNotCompletion() async throws {
        let operation = ExpertLoadOperation()
        operation.markInFlight()
        let waiter = Task { try await operation.firstStageCompletion() }
        operation.publishFirstStage()
        // A resumed waiter returns promptly; an un-resumed one would hang
        // until the finish below, and this expectation runs before it.
        try await waiter.value
        operation.finish(.success(()))
    }

    /// Completion releases a stage waiter too — an operation without a split
    /// (whole-region reads) never publishes, and the runner's stage await
    /// must not hang on it.
    @Test func firstStageWaiterReleasedByCompletion() async throws {
        let operation = ExpertLoadOperation()
        operation.markInFlight()
        let waiter = Task { try await operation.firstStageCompletion() }
        operation.finish(.success(()))
        try await waiter.value
    }

    /// A failed read releases the stage waiter with the failure: the runner's
    /// phase-1 fixup must not encode against slots whose bytes never landed.
    @Test func firstStageWaiterPropagatesFailure() async {
        struct ExpectedFailure: Error {}
        let operation = ExpertLoadOperation()
        operation.markInFlight()
        let waiter = Task { try await operation.firstStageCompletion() }
        operation.finish(.failure(ExpectedFailure()))
        await #expect(throws: ExpectedFailure.self) { try await waiter.value }
    }

    @Test func firstStagePublicationIsIdempotent() async throws {
        let operation = ExpertLoadOperation()
        operation.markInFlight()
        operation.publishFirstStage()
        operation.publishFirstStage()
        try await operation.firstStageCompletion()
        operation.finish(.success(()))
    }

    @Test func sharedEventDoesNotAdvancePastOutOfOrderBatch() throws {
        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let firstToken = try coordinator.reserve()
        let secondToken = try coordinator.reserve()
        let first = ExpertLoadOperation(
            completionToken: firstToken,
            eventCoordinator: coordinator)
        let second = ExpertLoadOperation(
            completionToken: secondToken,
            eventCoordinator: coordinator)

        second.finish(.success(()))
        #expect(
            secondToken.status.contents().advanced(by: secondToken.statusOffset)
                .load(as: UInt32.self) == 1)
        #expect(secondToken.event.signaledValue == 0)

        first.finish(.success(()))
        #expect(firstToken.event.signaledValue == secondToken.value)
        #expect(firstToken.value != secondToken.value)
    }

    @Test func timelineModeDecidesWhetherLayersBlockEachOther() async throws {
        let context = try MetalContext()

        // Shared mode is today's wiring: every layer gets the same object, so
        // one unfinished read gates every later release in the model.
        let shared = try #require(
            ExpertIOEventCoordinator.makeTimelines(
                device: context.device, numLayers: 4, mode: .shared))
        #expect(shared.count == 4)
        for timeline in shared {
            #expect(timeline === shared[0])
        }

        // Per-layer mode gives each layer its own prefix, so a completed read
        // releases while an unrelated layer is still loading.
        let perLayer = try #require(
            ExpertIOEventCoordinator.makeTimelines(
                device: context.device, numLayers: 4, mode: .perLayer))
        #expect(perLayer.count == 4)
        var identities: [ObjectIdentifier] = []
        for timeline in perLayer {
            #expect(!identities.contains(ObjectIdentifier(timeline)))
            identities.append(ObjectIdentifier(timeline))
        }

        let blockedToken = try perLayer[0].reserve()
        let readyToken = try perLayer[1].reserve()
        let blockedOperation = ExpertLoadOperation(
            completionToken: blockedToken,
            eventCoordinator: perLayer[0])
        let readyOperation = ExpertLoadOperation(
            completionToken: readyToken,
            eventCoordinator: perLayer[1])

        readyOperation.finish(.success(()))
        #expect(readyToken.event.signaledValue == readyToken.value)
        #expect(blockedToken.event.signaledValue == 0)
        blockedOperation.finish(.success(()))
        #expect(blockedToken.event.signaledValue == blockedToken.value)
    }

    @Test func failedBatchSignalsTerminalStatusInsteadOfDeadlocking() throws {
        struct ExpectedFailure: Error {}
        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let token = try coordinator.reserve()
        let operation = ExpertLoadOperation(
            completionToken: token,
            eventCoordinator: coordinator)

        operation.finish(.failure(ExpectedFailure()))

        #expect(
            token.status.contents().advanced(by: token.statusOffset)
                .load(as: UInt32.self) == 2)
        #expect(token.event.signaledValue == token.value)
        #expect(throws: ExpectedFailure.self) { try operation.wait() }
    }
}
