import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// The primitive that keeps a stored continuation off actor state (ledger
/// AUD-143). Each of these is the race that the check-then-append order had: a
/// wake that lands before the continuation exists, and a cancellation that lands
/// after one.
@Suite("Suspension slot")
struct SuspensionSlotTests {
    /// A waiter registered but not yet suspended must not sleep on a wake that
    /// already happened. Nothing else can release it, so the only proof is a
    /// deadline: the child that would hang is cancelled when the timer wins.
    @Test func aWakeThatArrivesBeforeTheContinuationStillReleasesTheWaiter() async {
        let slot = SuspensionSlot()
        slot.signal(.wake)

        let releasedFirst = await withTaskGroup(of: Bool.self) { group -> Bool in
            group.addTask {
                await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                    slot.handOver(continuation)
                }
                return true
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(2))
                return false
            }
            let woke = await group.next() ?? false
            group.cancelAll()
            return woke
        }
        #expect(releasedFirst, "handOver suspended on a wake nobody holds")
        #expect(slot.received == .wake)
    }

    @Test func aWakeAfterTheHandOverResumesTheSuspension() async {
        let slot = SuspensionSlot()
        let waiter = Task {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                slot.handOver(continuation)
            }
            return slot.received
        }
        // Long enough for the child to be suspended, not just scheduled.
        try? await Task.sleep(for: .milliseconds(50))
        #expect(slot.received == nil, "the caller was released before it was woken")
        #expect(slot.signal(.wake))
        #expect(await waiter.value == .wake)
    }

    /// The resume-exactly-once property: the first reason recorded owns the
    /// resume, so a wake racing a cancellation cannot resume the same caller
    /// twice — which is the double free the old shape produced in release builds.
    @Test func theFirstReasonWinsAndLaterOnesAreIgnored() async {
        let slot = SuspensionSlot()
        #expect(slot.signal(.wake))
        #expect(!slot.signal(.cancel), "a second reason resumed the same caller again")
        #expect(slot.received == .wake)

        let other = SuspensionSlot()
        #expect(other.signal(.cancel))
        #expect(!other.signal(.wake))
        #expect(other.received == .cancel)
    }

    /// A cancellation is delivered to a caller that has already suspended, and
    /// the caller reads it as `.cancel` rather than as a grant.
    @Test func aCancellationReachesASuspendedCallerAsACancel() async {
        let slot = SuspensionSlot()
        let waiter = Task {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                slot.handOver(continuation)
            }
            return slot.received
        }
        try? await Task.sleep(for: .milliseconds(50))
        #expect(slot.signal(.cancel))
        #expect(await waiter.value == .cancel)
    }
}
