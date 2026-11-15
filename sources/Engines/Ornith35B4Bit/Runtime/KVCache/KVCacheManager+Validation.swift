import Foundation
import Metal

// Range, slot and view construction checks, and the residency advice.
//
// Split out of `KVCacheManager.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
extension KVCacheManager {

    func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(
            start + count <= maxContext,
            "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    func makeView(
        buffer: MTLBuffer, layer: Int, offset: Int,
        validTokenCount: Int
    ) -> KVView {
        KVView(
            buffer: buffer, offset: offset, stride: strides[layer],
            validTokenCount: validTokenCount, precision: precision,
            valueBytes: valueBytes[layer], groupSize: Self.quantizationGroupSize)
    }

    package static func rowLayout(
        elements: Int,
        precision: KVCachePrecision
    ) -> (stride: Int, valueBytes: Int) {
        if precision == .fp16 {
            return (elements * fp16Size, elements * fp16Size)
        }
        let packed = (elements * precision.rawValue + 7) / 8
        let alignedPacked = (packed + 1) & ~1
        let groups = (elements + quantizationGroupSize - 1) / quantizationGroupSize
        return (alignedPacked + groups * 2 * fp16Size, alignedPacked)
    }

    func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(
            count <= maxContext,
            "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    func validateSlot(_ slot: Int) {
        precondition(
            slot >= 0 && slot < slots,
            "slot \(slot) is out of range 0..<\(slots)")
    }

    func physicalSlot(layer: Int, position: Int) -> Int {
        precondition(capacityTokens[layer] > 0, "layer has no KV storage")
        return position % capacityTokens[layer]
    }

    func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(
            physicalStart + count <= capacity,
            "range \(start)..<\(start + count) wraps KV ring capacity \(capacity)")
    }

    func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
