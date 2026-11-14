import Foundation
import TinyTitanCore

/// Serializes forward steps through one `RealForwardRunner`.
///
/// The runner's scratch buffers, decode cursors and statistic counters are
/// owned exclusively -- `RealForwardRunner` documents that as its whole safety
/// argument -- so two sequences must never be inside a step at the same time,
/// even though each uses its own KV and GDN slot. A batched scheduler therefore
/// holds this gate around each `produce`/`prefillChunked` call: the slots are
/// independent, the step is not.
///
/// Actor-isolated, so acquisition and release are both `await`ed. Waiting is
/// FIFO and cancellable; a cancelled waiter is removed and released with a cancel
/// reason so it cannot be handed the gate later.
public actor ForwardStepGate {
    private var busy = false
    /// Slots for the callers queued behind `busy`, appended in the same actor step
    /// as the check that put them there. `SuspensionSlot` says why the continuation
    /// itself cannot live in actor state.
    private var waiters: [SuspensionSlot] = []

    public init() {}

    /// Wait until no step is in flight, then take the gate.
    public func acquire() async throws {
        try Task.checkCancellation()
        guard let slot = queueOrTake() else { return }
        await withTaskCancellationHandler {
            await withCheckedContinuation { slot.handOver($0) }
        } onCancel: {
            Task { await self.cancel(slot) }
        }
        if slot.received == .cancel { throw CancellationError() }
        if Task.isCancelled {
            // Handed the gate and cancelled in the same breath: it is held now, so
            // give it back before throwing rather than stranding the step.
            release()
            throw CancellationError()
        }
    }

    /// Take the gate or queue on a fresh slot, in one step that contains no
    /// suspension. Nil means the gate was taken and the caller may proceed.
    private func queueOrTake() -> SuspensionSlot? {
        if !busy {
            busy = true
            return nil
        }
        let slot = SuspensionSlot()
        waiters.append(slot)
        return slot
    }

    /// Release the gate, handing it to the oldest waiter if there is one.
    public func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().signal(.wake)
        }
    }

    private func cancel(_ slot: SuspensionSlot) {
        if let index = waiters.firstIndex(where: { $0 === slot }) {
            waiters.remove(at: index)
        }
        slot.signal(.cancel)
    }
}
