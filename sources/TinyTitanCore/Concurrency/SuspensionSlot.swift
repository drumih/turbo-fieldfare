import Foundation

/// One caller's suspension point, usable across the boundary that a stored
/// `CheckedContinuation` cannot cross.
///
/// A continuation only exists inside `withCheckedContinuation`'s closure, and the
/// runtime does not run that closure on the calling actor's executor — it takes the
/// non-isolated, non-sending form. Anything the closure reaches is therefore
/// reached off the actor. That is not theoretical: the shape with an actor's own
/// array appended to from inside the closure, built by `swiftc -O` outside this
/// repository and with no test framework in it, returned clean 0 of 6 runs — four
/// `SIGSEGV` faulting in the append's own buffer handling, one `SIGBUS`, one hang.
/// Inside the repository the same mutation aborts the release test bundle with
/// signal 6 from the concurrency runtime's isolation check. See ledger AUD-143.
///
/// So the closure may touch only this slot, which is lock-guarded and correct on
/// whatever executor runs it, while the *queue* it belongs to stays actor state.
///
/// The second half of the design is the ordering. The caller registers itself on
/// the actor in the same synchronous step as the condition check that decided it
/// must wait, and hands the continuation over afterwards. A wake that lands in
/// between is recorded on the slot, and `handOver` resumes at once instead of
/// suspending on a waiter nobody holds any more. That closes the lost-wakeup
/// window the check-then-append order had, which is what let a queued model
/// switch be overtaken by new work for the resident model.
///
/// unchecked-invariant: `continuation` and `reason` are read and written only under
/// `lock`, and the first `signal(_:)` to record a reason owns the resume — later
/// ones are ignored — so whichever path claims the slot is the only one that can
/// resume the continuation it holds.
package final class SuspensionSlot: @unchecked Sendable {
    package enum Reason: Sendable {
        /// The condition the caller waited for is satisfied; it may proceed.
        case wake
        /// The caller was cancelled and removed from the queue; it must not proceed.
        case cancel
    }

    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var reason: Reason?

    package init() {}

    /// Hand the suspension over. Called from `withCheckedContinuation`'s closure,
    /// so it must be safe on any executor, and it is: it touches only this slot.
    package func handOver(_ continuation: CheckedContinuation<Void, Never>) {
        let resumeNow: Bool = lock.withLock {
            guard reason == nil else {
                self.continuation = continuation
                return true
            }
            self.continuation = continuation
            return false
        }
        if resumeNow { continuation.resume() }
    }

    /// Record why this waiter is being released, and resume it if it is already
    /// suspended. Idempotent: the first reason recorded wins, so a wake racing a
    /// cancellation resumes the caller exactly once, on the reason that got there
    /// first. Returns whether this call is the one that recorded the reason — true
    /// even when the continuation has not arrived yet, because the recorded reason
    /// is what `handOver` will resume on.
    @discardableResult
    package func signal(_ reason: Reason) -> Bool {
        let outcome: (claimed: CheckedContinuation<Void, Never>?, recorded: Bool) =
            lock.withLock {
                guard self.reason == nil else { return (nil, false) }
                self.reason = reason
                let pending = continuation
                continuation = nil
                return (pending, true)
            }
        outcome.claimed?.resume()
        return outcome.recorded
    }

    /// The reason recorded for this slot, or nil while the caller is still meant
    /// to be waiting. Read after waking to decide whether to proceed.
    package var received: Reason? { lock.withLock { reason } }
}
