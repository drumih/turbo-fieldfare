import Darwin
import Foundation
import Metal
import Synchronization
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareCLICore

/// Synthetic files and controlled workers exercise real stage ownership and
/// admission without a model or GPU commands. Event timeouts bound failures.
@Suite struct NextLayerExpertPrefetcherTests {
    enum Failure: Error, Equatable { case partialRead, foreground }

    private func streamer(_ url: URL) throws -> PreadExpertStreamer {
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try PreadExpertStreamer(
            layout: PreadExpertStreamerTests.makeLayout(path: url.path),
            device: device, slotCount: 4)
    }

    @Test func wrongPendingReadJoinsBeforeTheStageIsReused() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try streamer(url)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let prefetch = try NextLayerExpertPrefetcher(
            streamer: streamer, environment: [:],
            readEarly: { streamer, expert, destination in
                if expert == 3 {
                    entered.signal()
                    guard release.wait(timeout: .now() + 5) == .success else { throw Failure.partialRead }
                }
                try streamer.readEarly(expert: expert, into: destination)
            })
        try prefetch.beginForward()
        try prefetch.issue(layer: 1, experts: [3], streamer: streamer)
        #expect(entered.wait(timeout: .now() + 5) == .success)
        let observed = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        let failure = Mutex<(any Error)?>(nil)
        DispatchQueue.global().async {
            do {
                observed.signal()
                try prefetch.issue(layer: 2, experts: [2], streamer: streamer)
            } catch { failure.withLock { $0 = error } }
            finished.signal()
        }
        #expect(observed.wait(timeout: .now() + 5) == .success)
        #expect(finished.wait(timeout: .now() + 0.05) == .timedOut)
        release.signal()
        #expect(finished.wait(timeout: .now() + 5) == .success)
        #expect(failure.withLock { $0 == nil })
        try prefetch.finishRequest()
        #expect(streamer.residentExperts().isEmpty)
    }

    @Test func pendingMatchSwapsBeforeBindingAndJoinsBeforeAdmission() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try streamer(url)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let prefetch = try NextLayerExpertPrefetcher(
            streamer: streamer, environment: [:],
            readEarly: { streamer, expert, destination in
                entered.signal()
                guard release.wait(timeout: .now() + 5) == .success else { throw Failure.partialRead }
                try streamer.readEarly(expert: expert, into: destination)
            })
        try prefetch.issue(layer: 1, experts: [3], streamer: streamer)
        #expect(entered.wait(timeout: .now() + 5) == .success)
        let plan = streamer.planExpertsCached(experts: [3, 1])
        let old = streamer.expertCachePlanBuffers(plan)[0].buffer
        let skipped = prefetch.adopt(into: RoutedExpertFetchPlan(layer: 1, cachePlan: plan), streamer: streamer)
        #expect(skipped == [0])
        #expect(streamer.expertCachePlanBuffers(plan)[0].buffer !== old)
        let beforeAdmission = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        let failure = Mutex<(any Error)?>(nil)
        DispatchQueue.global().async {
            do {
                _ = try streamer.executeExpertCachePlan(plan, skippingMisses: skipped, beforeAdmitting: {
                    beforeAdmission.signal()
                    try prefetch.waitForAdopted()
                })
            } catch { failure.withLock { $0 = error } }
            finished.signal()
        }
        #expect(beforeAdmission.wait(timeout: .now() + 5) == .success)
        #expect(finished.wait(timeout: .now()) == .timedOut)
        release.signal()
        #expect(finished.wait(timeout: .now() + 5) == .success)
        #expect(failure.withLock { $0 == nil })
        let bytes = PreadExpertStreamerTests.bytes(
            of: streamer.expertCachePlanBuffers(plan)[0].buffer, offset: 0,
            count: PreadExpertStreamerTests.expertStride)
        #expect(bytes.allSatisfy { $0 == PreadExpertStreamerTests.tagByte(3) })
        #expect(streamer.residentExperts() == [1, 3])
        #expect(prefetch.stats.reads == 1)
        #expect(prefetch.stats.used == 1)
        try prefetch.finishRequest()
    }

    @Test func completedMatchAdmitsTheStagedBytes() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try streamer(url)
        let prefetch = try NextLayerExpertPrefetcher(streamer: streamer, environment: [:])
        try prefetch.issue(layer: 1, experts: [3], streamer: streamer)
        try prefetch.waitForPending()
        let plan = streamer.planExpertsCached(experts: [3])
        let skipped = prefetch.adopt(into: RoutedExpertFetchPlan(layer: 1, cachePlan: plan), streamer: streamer)
        _ = try streamer.executeExpertCachePlan(plan, skippingMisses: skipped,
                                               beforeAdmitting: { try prefetch.waitForAdopted() })
        try prefetch.finishRequest()
        #expect(streamer.residentExperts() == [3])
        let bytes = PreadExpertStreamerTests.bytes(
            of: streamer.expertCachePlanBuffers(plan)[0].buffer, offset: 0,
            count: PreadExpertStreamerTests.expertStride)
        #expect(bytes.allSatisfy { $0 == PreadExpertStreamerTests.tagByte(3) })
    }

    @Test func pendingMatchFinishesDuringDemandBeforeAdmissionJoin() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try streamer(url)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let prefetch = try NextLayerExpertPrefetcher(
            streamer: streamer, environment: [:],
            readEarly: { streamer, expert, destination in
                entered.signal()
                guard release.wait(timeout: .now() + 5) == .success else { throw Failure.partialRead }
                try streamer.readEarly(expert: expert, into: destination)
            })
        try prefetch.issue(layer: 1, experts: [3], streamer: streamer)
        #expect(entered.wait(timeout: .now() + 5) == .success)
        let plan = streamer.planExpertsCached(experts: [3, 1])
        let old = streamer.expertCachePlanBuffers(plan)[0].buffer
        let skipped = prefetch.adopt(
            into: RoutedExpertFetchPlan(layer: 1, cachePlan: plan), streamer: streamer)
        #expect(skipped == [0])
        #expect(streamer.expertCachePlanBuffers(plan)[0].buffer !== old)
        release.signal()
        // Complete the worker without clearing adoption metadata, representing
        // an early read hidden by foreground demand work before admission.
        try prefetch.waitForPending()
        var joinedBeforeAdmission = false
        let blobs = try streamer.executeExpertCachePlan(
            plan, skippingMisses: skipped, beforeAdmitting: {
                try prefetch.waitForAdopted()
                joinedBeforeAdmission = true
            })
        #expect(joinedBeforeAdmission)
        let bytes = PreadExpertStreamerTests.bytes(
            of: blobs[0].buffer, offset: 0, count: PreadExpertStreamerTests.expertStride)
        #expect(bytes.allSatisfy { $0 == PreadExpertStreamerTests.tagByte(3) })
        #expect(streamer.residentExperts() == [1, 3])
        #expect(prefetch.stats.reads == 1)
        #expect(prefetch.stats.used == 1)
        try prefetch.finishRequest()
    }

    @Test func finalPartialReadFailureIsDrainedAndSurfacedOnce() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try streamer(url)
        let prefetch = try NextLayerExpertPrefetcher(
            streamer: streamer, environment: [:],
            readEarly: { _, _, destination in
                memset(destination, 0xAA, 32)
                throw Failure.partialRead
            })
        try prefetch.issue(layer: 1, experts: [3], streamer: streamer)
        #expect(throws: Failure.self) { try prefetch.finishRequest() }
        // The writer was joined even on error; no stale failure on reuse.
        try prefetch.finishRequest()
        #expect(streamer.residentExperts().isEmpty)
    }

    @Test func foregroundFailureAfterAdoptionJoinsLateFailingWriter() async throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try streamer(url)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let prefetch = try NextLayerExpertPrefetcher(
            streamer: streamer, environment: [:],
            readEarly: { _, _, destination in
                entered.signal()
                guard release.wait(timeout: .now() + 5) == .success else { throw Failure.partialRead }
                memset(destination, 0xAA, 32)
                throw Failure.partialRead
            })
        try prefetch.issue(layer: 1, experts: [3], streamer: streamer)
        let workerEntered = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            DispatchQueue.global().async {
                continuation.resume(returning: entered.wait(timeout: .now() + 5) == .success)
            }
        }
        #expect(workerEntered)
        do {
            _ = try await withFinishedRequest(operation: { () -> Int in
                let plan = streamer.planExpertsCached(experts: [3])
                #expect(prefetch.adopt(
                    into: RoutedExpertFetchPlan(layer: 1, cachePlan: plan), streamer: streamer) == [0])
                throw Failure.foreground
            }, finish: {
                release.signal()
                try prefetch.finishRequest()
            })
            Issue.record("foreground failure must propagate")
        } catch {
            let combined = try #require(error as? RequestCompletionFailure)
            #expect(combined.primary as? Failure == .foreground)
            #expect(combined.drain as? Failure == .partialRead)
        }
        // Completion joined the adopted writer and cleared the failed stage.
        try prefetch.finishRequest()
        #expect(streamer.residentExperts().isEmpty)
    }

    @Test(arguments: [false, true])
    func realDemandFailureJoinsAdoptedWriterBeforeRecoveryAndSlotReuse(lateReadFails: Bool) throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let originalFile = try Data(contentsOf: url)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let streamer = try PreadExpertStreamer(
            layout: PreadExpertStreamerTests.makeLayout(path: url.path), device: device, slotCount: 2)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let prefetch = try NextLayerExpertPrefetcher(
            streamer: streamer, environment: [:], readEarly: { streamer, expert, destination in
                try streamer.readEarly(expert: expert, into: destination)
                entered.signal()
                guard release.wait(timeout: .now() + 10) == .success else { throw Failure.partialRead }
                // A real staged read with a controlled final write, still targeting
                // the buffer adopted into the failed demand plan's slot.
                memset(destination, Int32(PreadExpertStreamerTests.tagByte(expert)),
                       PreadExpertStreamerTests.expertStride)
                if lateReadFails { throw Failure.partialRead }
            })
        try prefetch.issue(layer: 1, experts: [3], streamer: streamer)
        #expect(entered.wait(timeout: .now() + 5) == .success)
        let plan = streamer.planExpertsCached(experts: [3, 1])
        let skipped = prefetch.adopt(
            into: RoutedExpertFetchPlan(layer: 1, cachePlan: plan), streamer: streamer)
        #expect(skipped == [0])
        // Expert 1 is now beyond EOF, forcing the actual demand reader to fail.
        #expect(truncate(url.path, off_t(PreadExpertStreamerTests.streamOffset)
                         + off_t(PreadExpertStreamerTests.expertStride)) == 0)
        let joining = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        let recovery = Mutex((reset: false, error: Optional<any Error>.none))
        DispatchQueue.global().async {
            do {
                _ = try streamer.executeExpertCachePlan(plan, skippingMisses: skipped, beforeAdmitting: {
                    joining.signal()
                    try prefetch.waitForAdopted()
                })
                Issue.record("truncated foreground demand must fail")
            } catch {
                recovery.withLock { $0.error = error }
                // The same empty-lineage recovery used after a failed app turn.
                do {
                    try MultimodalConversationKVRecovery.restoreAfterFailure(
                        positionBefore: 0, generationError: error,
                        rewind: { _ in Issue.record("empty lineage must reset") },
                        reset: { recovery.withLock { $0.reset = true } })
                } catch { Issue.record("unexpected recovery error: \(error)") }
            }
            finished.signal()
        }
        #expect(joining.wait(timeout: .now() + 5) == .success)
        let finishedEarly = finished.wait(timeout: .now()) == .success
        #expect(!finishedEarly)
        #expect(!recovery.withLock { $0.reset })
        try originalFile.write(to: url)
        release.signal()
        if !finishedEarly { #expect(finished.wait(timeout: .now() + 5) == .success) }
        #expect(recovery.withLock { $0.reset && $0.error != nil })
        let error = try #require(recovery.withLock { $0.error })
        #expect(requestPrimaryError(error) is StreamerError)
        if lateReadFails {
            #expect((error as? RequestCompletionFailure)?.drain as? Failure == .partialRead)
            #expect(throws: Failure.self) { try prefetch.finishRequest() }
        } else {
            try prefetch.finishRequest()
        }
        // The explicit completion consumed any late failure before reuse.
        try prefetch.finishRequest()
        #expect(streamer.residentExperts().isEmpty)
        // Both physical slots are reused after recovery; the old writer can no
        // longer overwrite bytes from the retry's authoritative demand reads.
        let retry = streamer.planExpertsCached(experts: [0, 1])
        #expect(Set(retry.assignedSlots) == Set(plan.assignedSlots))
        let blobs = try streamer.executeExpertCachePlan(retry)
        for (index, blob) in blobs.enumerated() {
            let bytes = PreadExpertStreamerTests.bytes(
                of: blob.buffer, offset: 0, count: PreadExpertStreamerTests.expertStride)
            #expect(bytes.allSatisfy { $0 == PreadExpertStreamerTests.tagByte(retry.experts[index]) })
        }
        #expect(streamer.residentExperts() == [0, 1])
    }

}
