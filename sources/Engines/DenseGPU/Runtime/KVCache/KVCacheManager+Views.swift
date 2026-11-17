import Foundation
import Metal

// The K/V view and range accessors the attention encoders read.
//
// Split out of `KVCacheManager.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
extension KVCacheManager {

    public func keyView(layer: Int) -> KVView {
        keyView(layer: layer, slot: 0, validTokenCount: positions[0])
    }

    public func keyView(layer: Int, validTokenCount: Int) -> KVView {
        keyView(layer: layer, slot: 0, validTokenCount: validTokenCount)
    }

    public func keyView(layer: Int, slot: Int, validTokenCount: Int) -> KVView {
        validateSlot(slot)
        validateValidTokenCount(validTokenCount)
        return makeView(
            buffer: kBuffers[layer], layer: layer,
            offset: regionBase(layer: layer, slot: slot) * strides[layer],
            validTokenCount: validTokenCount)
    }

    public func valueView(layer: Int) -> KVView {
        valueView(layer: layer, slot: 0, validTokenCount: positions[0])
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        keyView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        valueView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    public func valueView(layer: Int, validTokenCount: Int) -> KVView {
        valueView(layer: layer, slot: 0, validTokenCount: validTokenCount)
    }

    public func valueView(layer: Int, slot: Int, validTokenCount: Int) -> KVView {
        validateSlot(slot)
        validateValidTokenCount(validTokenCount)
        return makeView(
            buffer: vBuffers[layer], layer: layer,
            offset: regionBase(layer: layer, slot: slot) * strides[layer],
            validTokenCount: validTokenCount)
    }

    public func keyRangeView(
        layer: Int, start: Int, count: Int,
        slot: Int = 0
    ) -> KVView {
        let range = kRange(layer: layer, start: start, count: count, slot: slot)
        return makeView(
            buffer: range.buffer, layer: layer, offset: range.offset,
            validTokenCount: count)
    }

    public func valueRangeView(
        layer: Int, start: Int, count: Int,
        slot: Int = 0
    ) -> KVView {
        let range = vRange(layer: layer, start: start, count: count, slot: slot)
        return makeView(
            buffer: range.buffer, layer: layer, offset: range.offset,
            validTokenCount: count)
    }
}
