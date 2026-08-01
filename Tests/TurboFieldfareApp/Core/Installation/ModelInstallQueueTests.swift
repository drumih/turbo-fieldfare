import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelInstallQueueTests {
    @Test func firstEnqueuedRepositoryBecomesActive() async {
        let queue = ModelInstallQueue()
        let started = await queue.enqueue(repoID: "owner/one")
        #expect(started)
        #expect(await queue.activeRepoID == "owner/one")
    }

    @Test func secondEnqueuedRepositoryWaits() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        let started = await queue.enqueue(repoID: "owner/two")
        #expect(started == false)
        #expect(await queue.activeRepoID == "owner/one")
        #expect(await queue.pendingRepoIDs == ["owner/two"])
    }

    @Test func finishingActivePromotesTheNextRepository() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        _ = await queue.enqueue(repoID: "owner/two")
        let promoted = await queue.finishActive()
        #expect(promoted == "owner/two")
        #expect(await queue.activeRepoID == "owner/two")
        #expect(await queue.pendingRepoIDs.isEmpty)
    }

    @Test func enqueueingTheSameRepositoryTwiceIsIgnored() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        _ = await queue.enqueue(repoID: "owner/two")
        _ = await queue.enqueue(repoID: "owner/two")
        #expect(await queue.pendingRepoIDs == ["owner/two"])
    }

    @Test func cancellingAPendingRepositoryRemovesItWithoutTouchingActive() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        _ = await queue.enqueue(repoID: "owner/two")
        await queue.cancel(repoID: "owner/two")
        #expect(await queue.activeRepoID == "owner/one")
        #expect(await queue.pendingRepoIDs.isEmpty)
    }

    @Test func cancellingTheActiveRepositoryPromotesTheNext() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        _ = await queue.enqueue(repoID: "owner/two")
        await queue.cancel(repoID: "owner/one")
        #expect(await queue.activeRepoID == "owner/two")
    }

    @Test func statesAreTrackedPerRepository() {
        var states = ModelInstallStates()
        #expect(states.state(for: "owner/one") == .idle)
        states.setState(.installed(modelDirectory: URL(fileURLWithPath: "/tmp/a")),
                        for: "owner/one")
        states.setState(.checking, for: "owner/two")
        #expect(states.state(for: "owner/two") == .checking)
        #expect(states.installedRepoIDs == ["owner/one"])
    }
}
