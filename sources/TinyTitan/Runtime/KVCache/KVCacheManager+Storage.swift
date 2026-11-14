import Foundation
import Metal
import TinyTitanCore

// Allocation, worst-case sizing and the K/V accessors the attention encoders
// and the runner use. Split out of `KVCacheManager.swift` (2026-10-21) under
// the 500-line-per-file rule; the declarations are unchanged.
extension KVCacheManager {

    /// Worst-case bytes `slots` sequences' K/V stores can occupy at
    /// `maxContext`, without allocating anything.
    ///
    /// Worst case means every slot grown to the advertised context: growth is
    /// lazy, so this is the ceiling a batched session must be able to hold, not
    /// what it maps at load. It mirrors the layout `init` builds, and a test
    /// compares the two against each other so a storage-format change cannot
    /// leave the formula quietly wrong.
    public static func worstCaseBytes(
        config: ArchConfig,
        maxContext: Int,
        precision: KVCachePrecision = .fp16,
        slots: Int = 1,
        fp16RingEnabled: Bool = false,
        slidingWindow: Int? = nil,
        maxPrefillChunkTokens: Int = 128,
        windowTokens: Int? = nil
    ) -> Int {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(slots > 0, "slots must be positive")
        var total = 0
        for layer in 0..<config.numLayers {
            let maskValue = config.fullAttentionLayerMask[layer]
            if maskValue == 2 { continue }
            let isFull = maskValue != 0
            let elements =
                isFull
                ? config.numFullKVHeads * config.fullHeadDim
                : config.numKVHeads * config.headDim
            let stride = rowLayout(elements: elements, precision: precision).stride
            let capacity: Int
            if isFull, let windowTokens {
                capacity = min(windowTokens, maxContext)
            } else if fp16RingEnabled && !isFull {
                capacity = min(
                    maxContext,
                    max(
                        1,
                        (slidingWindow ?? config.slidingWindow)
                            + maxPrefillChunkTokens))
            } else {
                capacity = maxContext
            }
            // One K buffer and one V buffer per layer.
            total += slots * capacity * stride * 2
        }
        return total
    }

    /// Bytes the host arenas occupy for `slots` sequences at `maxContext`.
    /// Zero when the tier is off. The arena is not Metal memory, but it is
    /// physical memory, so a bounded server counts it beside the window.
    public static func worstCaseHostArenaBytes(
        config: ArchConfig,
        maxContext: Int,
        precision: KVCachePrecision = .fp16,
        slots: Int = 1
    ) -> Int {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(slots > 0, "slots must be positive")
        var total = 0
        for layer in 0..<config.numLayers {
            let maskValue = config.fullAttentionLayerMask[layer]
            guard maskValue == 1 else { continue }
            let elements = config.numFullKVHeads * config.fullHeadDim
            let stride = rowLayout(elements: elements, precision: precision).stride
            total += slots * maxContext * stride * 2
        }
        return total
    }

    /// Grows linear layers so every one can hold `tokens`, copying what is
    /// already stored.
    ///
    /// Must be called before writing at a position beyond the current capacity.
    /// Ring-backed SWA layers are never grown -- their capacity is the window and
    /// is deliberate. Capacity doubles, so a conversation reaching the advertised
    /// 262144 limit pays five copies in total rather than one per token.
    ///
    /// Buffers are `storageModeShared`, so the copy is a plain `memcpy`; callers
    /// fetch buffers through the accessors at use time and never cache them
    /// across tokens, which is what makes swapping them safe here.
    public func reserve(tokens: Int) throws {
        try reserve(tokens: tokens, slot: 0)
    }

    /// Grows the stores so slot `slot` can hold `tokens`, copying every slot's
    /// live rows. All slots share one allocation per layer, so growth is
    /// all-or-nothing even though only one cursor motivated it.
    public func reserve(tokens: Int, slot: Int) throws {
        validateSlot(slot)
        let needed = min(max(tokens, 1), maxContext)
        for layer in 0..<kinds.count {
            guard kinds[layer] != .linear else { continue }
            // A streamed full-attention layer's VRAM window is fixed and its
            // arena already spans `maxContext`; there is nothing to grow.
            guard !streamsKVFromHost || !streamsLayer(layer) else { continue }
            if fp16RingEnabled && kinds[layer] == .swa { continue }
            let current = capacityTokens[layer]
            guard current < needed else { continue }
            var target = current
            while target < needed { target *= 2 }
            target = min(target, maxContext)

            let stride = strides[layer]
            let length = slots * target * stride
            guard let newK = device.makeBuffer(length: length, options: .storageModeShared),
                let newV = device.makeBuffer(length: length, options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            newK.label = "kv.K.layer\(layer)"
            newV.label = "kv.V.layer\(layer)"
            // Each slot's live rows move from its old region to its new one.
            // At one slot this is the original single `memcpy` from the base.
            for s in 0..<slots {
                let usedBytes = min(current, positions[s]) * stride
                guard usedBytes > 0 else { continue }
                memcpy(
                    newK.contents() + s * target * stride,
                    kBuffers[layer].contents() + s * current * stride, usedBytes)
                memcpy(
                    newV.contents() + s * target * stride,
                    vBuffers[layer].contents() + s * current * stride, usedBytes)
            }
            kBuffers[layer] = newK
            vBuffers[layer] = newV
            capacityTokens[layer] = target
        }
    }

    public func layerKind(_ layer: Int) -> LayerKind { kinds[layer] }

    /// Bytes per token for `layer` (K and V share the same stride).
    public func stride(layer: Int) -> Int { strides[layer] }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    public func ringCapacity(layer: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    /// Total bytes of the K buffer for `layer`, across all slots.
    public func bufferLength(layer: Int) -> Int {
        return slots * capacityTokens[layer] * strides[layer]
    }

    /// First token index of slot `slot`'s region within a layer buffer.
    func regionBase(layer: Int, slot: Int) -> Int {
        validateSlot(slot)
        return slot * capacityTokens[layer]
    }

    /// Write target for this layer's K projection at `position` in `slot`.
    public func kSlot(
        layer: Int, position: Int,
        slot: Int = 0
    ) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear, "linear layers have no KV slots")
        validateRange(start: position, count: 1)
        if streamsKVFromHost && streamsLayer(layer) {
            let row = ensureWindowRow(
                layer: layer, slot: slot, position: position, forWrite: true)
            return (
                kBuffers[layer],
                (slot * (windowTokens ?? 0) + row) * strides[layer]
            )
        }
        let token =
            regionBase(layer: layer, slot: slot)
            + physicalSlot(layer: layer, position: position)
        return (kBuffers[layer], token * strides[layer])
    }

    /// Write target for this layer's V projection at `position` in `slot`.
    /// Always distinct from `kSlot`: K runs per-head k_norm + RoPE while V is
    /// normed differently, so the two buffers cannot alias.
    public func vSlot(
        layer: Int, position: Int,
        slot: Int = 0
    ) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear, "linear layers have no KV slots")
        validateRange(start: position, count: 1)
        if streamsKVFromHost && streamsLayer(layer) {
            let row = ensureWindowRow(
                layer: layer, slot: slot, position: position, forWrite: false)
            return (
                vBuffers[layer],
                (slot * (windowTokens ?? 0) + row) * strides[layer]
            )
        }
        let token =
            regionBase(layer: layer, slot: slot)
            + physicalSlot(layer: layer, position: position)
        return (vBuffers[layer], token * strides[layer])
    }

    public func kRange(
        layer: Int, start: Int, count: Int,
        slot: Int = 0
    ) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        if streamsKVFromHost && streamsLayer(layer) {
            if let staged = stagedRange(
                layer: layer, slot: slot, start: start, count: count,
                buffers: prefillStagingK)
            {
                return staged
            }
            precondition(
                count == 1,
                "a streamed layer has no contiguous window range; "
                    + "write a prefill chunk through the staging buffers")
            let row = ensureWindowRow(
                layer: layer, slot: slot, position: start,
                forWrite: count == 1)
            return (
                kBuffers[layer],
                (slot * (windowTokens ?? 0) + row) * strides[layer],
                strides[layer]
            )
        }
        let token =
            regionBase(layer: layer, slot: slot)
            + physicalSlot(layer: layer, position: start)
        return (kBuffers[layer], token * strides[layer], strides[layer])
    }

    public func vRange(
        layer: Int, start: Int, count: Int,
        slot: Int = 0
    ) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        if streamsKVFromHost && streamsLayer(layer) {
            if let staged = stagedRange(
                layer: layer, slot: slot, start: start, count: count,
                buffers: prefillStagingV)
            {
                return staged
            }
            let row = ensureWindowRow(
                layer: layer, slot: slot, position: start,
                forWrite: false)
            return (
                vBuffers[layer],
                (slot * (windowTokens ?? 0) + row) * strides[layer],
                strides[layer]
            )
        }
        let token =
            regionBase(layer: layer, slot: slot)
            + physicalSlot(layer: layer, position: start)
        return (vBuffers[layer], token * strides[layer], strides[layer])
    }

    /// Advance slot 0's cursor once the current token's K/V are written across
    /// all layers.
    public func advance() { advance(slot: 0, by: 1) }

    public func advance(by count: Int) { advance(slot: 0, by: count) }

    public func advance(slot: Int, by count: Int) {
        validateSlot(slot)
        precondition(count >= 0, "advance count must be non-negative")
        precondition(
            positions[slot] + count <= maxContext,
            "advance would exceed maxContext")
        if count > 0 {
            commitStreamedRows(slot: slot, from: positions[slot], count: count)
        }
        positions[slot] += count
    }

    /// Slot `slot`'s logical cursor.
    public func position(slot: Int) -> Int {
        validateSlot(slot)
        return positions[slot]
    }

    /// Rewind the logical cursor without copying KV. Speculative rows are
    /// append-only and become unreachable immediately; a later pass
    /// overwrites them.
    /// Ring-backed draft KV is safe because MTP verification never rewinds by
    /// more than its two-token proposal depth.
    func rewind(to newPosition: Int) throws {
        try rewind(slot: 0, to: newPosition)
    }

    func rewind(slot: Int, to newPosition: Int) throws {
        validateSlot(slot)
        guard newPosition >= 0, newPosition <= positions[slot] else {
            throw InferenceStateSnapshotError.invalidPosition(newPosition)
        }
        positions[slot] = newPosition
    }

    /// Drop all cached positions and return physical pages to the OS.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. `MADV_DONTNEED` on the page-aligned span
    /// releases resident memory between turns; pages fault back in on next write.
    public func reset() {
        for slot in 0..<slots { positions[slot] = 0 }
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            advise(kBuffers[layer], pageSize: pageSize, seen: &advised)
            advise(vBuffers[layer], pageSize: pageSize, seen: &advised)
        }
        resetStreamedArenas()
    }

    /// Reset one slot's cursor without touching the others.
    ///
    /// The rows are neither zeroed nor returned to the OS: a recycled batch slot
    /// overwrites them before they can become visible again (attention reads
    /// only `[0, validTokenCount)`), and the whole-store `reset()` is what
    /// releases pages between generations.
    public func reset(slot: Int) {
        validateSlot(slot)
        positions[slot] = 0
        resetStreamedWindow(slot: slot)
    }

}
