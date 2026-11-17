import Foundation
import Metal
import TinyTitanCore

/// A fixed raw-byte staging ring for v4.3 predictive routed-expert reads.
///
/// Slots are intentionally outside the authoritative expert cache. A completed
/// entry becomes a cache resident only when the exact router selects it and the
/// ordinary cache planner adopts its bytes. Failed or incorrect predictions
/// are simply discarded without changing cache mappings or slot generations.
/// unchecked-invariant: slot ownership and operation association are guarded
/// by `lock`, and backing buffers are retained while an operation can write.
final class ExpertPrefetchRing: @unchecked Sendable {
    private struct Slot {
        let buffer: MTLBuffer
        var layer = -1
        var expert = -1
        var operation: ExpertLoadOperation?
    }

    private let lock = NSLock()
    private var slots: [Slot]
    /// Reads to issue for a given target layer, from `PrefetchDepthProfile`.
    /// A layer absent from the table issues the ring's own capacity, which is
    /// what it did before the profile existed.
    private let depthByTargetLayer: [Int: Int]

    init(
        device: MTLDevice, expertStride: Int, slotCount: Int,
        depthByTargetLayer: [Int: Int]
    ) throws {
        guard expertStride > 0, slotCount > 0 else {
            throw ModelError.internalInconsistency(detail: "invalid prefetch ring geometry")
        }
        var allocated: [Slot] = []
        allocated.reserveCapacity(slotCount)
        for index in 0..<slotCount {
            guard
                let buffer = device.makeBuffer(
                    length: expertStride,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "decode.prefetch.\(index)"
            allocated.append(Slot(buffer: buffer))
        }
        slots = allocated
        self.depthByTargetLayer = depthByTargetLayer
    }

    /// Slots in the ring: one per read it can hold in flight.
    var capacity: Int { slots.count }

    /// The names one `begin` should read, in rank order: the residency and
    /// in-flight filters run *first*, then the layer's depth and the free slots
    /// cap what survives. Capping the ranking first was the withdrawn v4.3 ring
    /// -- most of its top ranks were already resident, so the filter starved the
    /// reads instead of aiming them (see the comment at the call site).
    static func staged(
        experts: [Int], resident: Set<Int>, active: Set<Int>, freeSlots: Int, depth: Int
    ) -> [Int] {
        var seen: Set<Int> = []
        let wanted = experts.filter {
            !resident.contains($0) && !active.contains($0) && seen.insert($0).inserted
        }
        return Array(wanted.prefix(min(freeSlots, depth)))
    }

    /// Begins missing next-layer reads in unused staging slots. Already queued
    /// predictions are deduplicated; finished entries are reclaimed only when
    /// they have not been selected by a later exact route.
    /// `currentLayer` is the layer whose demand reads have just been
    /// planned; slots for it and earlier layers are stale and reclaimed.
    /// Slots for later layers (a two-layer-ahead read still waiting for its
    /// layer) are kept, which is what lets more than one layer be in flight.
    /// Reads issued by every `begin`, for the runner's per-token stats.
    private(set) var issuedReads = 0
    /// Slot states observed at each `begin`, summed: why the ring had no
    /// free slot. `held` = completed but its layer has not been planned yet.
    private(set) var observedSubmitted = 0
    private(set) var observedInFlight = 0
    private(set) var observedHeld = 0
    private(set) var observedFree = 0
    private(set) var begins = 0
    /// Queue wait and load time of every speculative operation reclaimed or
    /// consumed, summed, with the count.
    private(set) var reclaimedOps = 0
    private(set) var reclaimedQueueNanos: UInt64 = 0
    private(set) var reclaimedLoadNanos: UInt64 = 0

    func begin(
        model: Model, layer: Int, experts: [Int], resident: Set<Int>,
        currentLayer: Int
    ) throws {
        lock.lock()
        reclaimTerminalSlotsUnlocked(through: currentLayer)
        begins &+= 1
        for slot in slots {
            if slot.expert < 0 {
                observedFree &+= 1
                continue
            }
            switch slot.operation?.state {
            case .submitted: observedSubmitted &+= 1
            case .inFlight: observedInFlight &+= 1
            default: observedHeld &+= 1
            }
        }
        let active = Set(
            slots.compactMap { slot in
                slot.layer == layer && slot.expert >= 0 ? slot.expert : nil
            })
        let free = slots.indices.filter { slots[$0].expert < 0 }
        let selectedExperts = Self.staged(
            experts: experts, resident: resident, active: active,
            freeSlots: free.count, depth: depthByTargetLayer[layer] ?? slots.count)
        guard !selectedExperts.isEmpty else {
            lock.unlock()
            return
        }
        let selectedSlots = Array(free.prefix(selectedExperts.count))
        let staged = zip(selectedSlots, selectedExperts).map { slot, expert in
            (slot: slot, expert: expert, buffer: slots[slot].buffer)
        }
        for item in staged {
            slots[item.slot].layer = layer
            slots[item.slot].expert = item.expert
            slots[item.slot].operation = nil
        }
        lock.unlock()

        do {
            let operation = try model.beginRoutedExpertPrefetch(
                layer: layer, experts: selectedExperts, into: staged.map(\.buffer))
            lock.lock()
            for item in staged { slots[item.slot].operation = operation }
            issuedReads &+= staged.count
            lock.unlock()
        } catch {
            lock.lock()
            for item in staged {
                slots[item.slot].layer = -1
                slots[item.slot].expert = -1
                slots[item.slot].operation = nil
            }
            lock.unlock()
            throw error
        }
    }

    /// Completed raw bytes for one exact route. In-flight reads are not
    /// awaited: a demand miss remains authoritative and may start immediately.
    /// Marks a token boundary: every finished read in the ring belongs to
    /// the token that just ended, so its slot is free. Without this, a
    /// wrong prediction for one of the last layers kept its slot until the
    /// next token's layer index caught up -- which for layer 47 is never --
    /// and the ring ran with ~0.2 free slots (measured: 1.78 of 2 held).
    func beginToken() {
        lock.lock()
        reclaimTerminalSlotsUnlocked(through: Int.max)
        lock.unlock()
    }

    /// One line for the stats footer.
    var summary: String {
        let b = Double(max(1, begins))
        let n = Double(max(1, reclaimedOps))
        return String(
            format: "prefetch_ring begins=%d slots=%d links_deep=%d free=%.2f submitted=%.2f "
                + "inflight=%.2f held=%.2f "
                + "spec_ops=%d spec_queue_ms=%.2f spec_load_ms=%.2f",
            begins, slots.count, depthByTargetLayer.count, Double(observedFree) / b,
            Double(observedSubmitted) / b, Double(observedInFlight) / b,
            Double(observedHeld) / b,
            reclaimedOps, Double(reclaimedQueueNanos) / n / 1e6,
            Double(reclaimedLoadNanos) / n / 1e6)
    }

    func readyBuffers(layer: Int, experts: [Int]) -> [Int: MTLBuffer] {
        let requested = Set(experts)
        lock.lock()
        defer { lock.unlock() }
        var result: [Int: MTLBuffer] = [:]
        for slot in slots where slot.layer == layer && requested.contains(slot.expert) {
            if slot.operation?.state == .completed {
                result[slot.expert] = slot.buffer
            }
        }
        return result
    }

    func consume(layer: Int, experts: Set<Int>) {
        lock.lock()
        defer { lock.unlock() }
        for index in slots.indices
        where slots[index].layer == layer
            && experts.contains(slots[index].expert)
        {
            if let op = slots[index].operation {
                reclaimedOps &+= 1
                reclaimedQueueNanos &+= op.submissionToStartNanos
                reclaimedLoadNanos &+= op.loadNanos
            }
            slots[index].layer = -1
            slots[index].expert = -1
            slots[index].operation = nil
        }
    }

    private func reclaimTerminalSlotsUnlocked(through passedLayer: Int) {
        for index in slots.indices where slots[index].layer <= passedLayer {
            switch slots[index].operation?.state {
            case .completed, .failed, .none:
                if let op = slots[index].operation {
                    reclaimedOps &+= 1
                    reclaimedQueueNanos &+= op.submissionToStartNanos
                    reclaimedLoadNanos &+= op.loadNanos
                }
                slots[index].layer = -1
                slots[index].expert = -1
                slots[index].operation = nil
            case .submitted, .inFlight:
                break
            }
        }
    }
}

extension ExpertPrefetchRing {
    /// Zero the reads-issued counter `RealForwardRunner.totalPrefetchIssued`
    /// derives from (`resetStatistics`). Slot table, in-flight operations and
    /// the other diagnostic counters keep their own lifecycle — this zeroes
    /// exactly the number statistics exposes, nothing else.
    func resetIssuedReads() {
        lock.lock()
        defer { lock.unlock() }
        issuedReads = 0
    }
}
