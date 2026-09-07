import Foundation

/// Lets the input thread cancel the load the command loop is running.
///
/// The two threads race by design: the app writes `.load` and then, if the user
/// gives up, `.cancel`, and the cancel can arrive before the loop has even
/// started the load. So a cancel that finds nothing running has to be
/// remembered and applied to the load when it begins.
///
/// The mistake was remembering it unconditionally. `.cancel` is also what stops
/// a *generation*, and that cancel arrives when no load is queued at all — it
/// then sat latched indefinitely and cancelled the next load the user asked
/// for, however much later. Pressing Stop on an answer and then reloading the
/// model reported "decode service returned cancelled for a load request", with
/// nothing having cancelled anything.
///
/// A cancel is therefore only held when a load is actually queued and waiting
/// to start. Anything else is a generation's cancel and must not touch loads.
final class ServiceLoadCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<Void, Error>?
    private var queuedLoads = 0
    private var cancelRequested = false

    /// The input thread has read a `.load` and put it on the queue. It has not
    /// started yet, so a cancel arriving now belongs to it.
    func enqueueLoad() {
        lock.lock(); defer { lock.unlock() }
        queuedLoads += 1
    }

    func begin(_ task: Task<Void, Error>) {
        lock.lock()
        let alreadyCancelled = cancelRequested
        cancelRequested = false
        self.task = task
        lock.unlock()
        // A cancel that raced ahead of the task still has to land on it.
        if alreadyCancelled { task.cancel() }
    }

    /// The queued load is over, whether it ran or never started.
    ///
    /// The count is cleared here rather than in `begin` because a load can fail
    /// before it starts — its runtime options may not parse — and a count left
    /// standing turns the next generation's `.cancel` into a held cancel, which
    /// is the latch this type exists to prevent.
    func finish() {
        lock.lock(); defer { lock.unlock() }
        task = nil
        if queuedLoads > 0 { queuedLoads -= 1 }
        if queuedLoads == 0 { cancelRequested = false }
    }

    func cancel() {
        lock.lock()
        let current = task
        // Only a load that exists, or one already on its way, may be cancelled.
        if current == nil, queuedLoads > 0 { cancelRequested = true }
        lock.unlock()
        current?.cancel()
    }

    /// Test-visible: whether a cancel is being held for a queued load.
    var isHoldingCancel: Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelRequested
    }
}
