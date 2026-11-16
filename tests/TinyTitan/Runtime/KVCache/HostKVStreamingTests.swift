import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// Acceptance tests for the host-KV tier: a contiguous cache and a streamed
/// cache with a VRAM window must agree, the window must be the Metal cost, and
/// a step's copies must be bounded by its selection rather than its context.
@Suite struct HostKVStreamingTests {
    private let config = ArchConfig.qwen36_35B_A3B
    private let layer = 3

    private func makeManager(
        context: MetalContext,
        config: ArchConfig? = nil,
        maxContext: Int,
        window: Int?,
        precision: KVCachePrecision = .fp16
    ) throws -> KVCacheManager {
        let config = config ?? self.config
        return try KVCacheManager(
            device: context.device,
            config: config,
            maxContext: maxContext,
            precision: precision,
            slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: 128,
            windowTokens: window)
    }

    /// A four-layer toy geometry for the acceptance runs that need 4K/128K/1M
    /// contexts: two full layers and two linear ones, 32-byte rows at 32 dims.
    /// 512 dims is used by the parity test because it keeps the serial window
    /// kernel, which is the same reduction order as the serial contiguous
    /// kernel -- the 256-dim production shape takes the simd path instead.
    private func makeConfig(headDim: Int) -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 128,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: 1_024,
            slidingWindow: 1_024,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 10_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 4,
            numExperts: 8,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 1, 2, 1],
            hiddenActivation: "silu",
            attentionScale: 0.125,
            linearAttention: .none)
    }

    private func stride(_ kv: KVCacheManager) -> Int {
        kv.stride(layer: layer)
    }

    private func fillRow(
        _ kv: KVCacheManager, position: Int, seed: Int, slot: Int = 0
    ) {
        let bytes = stride(kv)
        let k = kv.kSlot(layer: layer, position: position, slot: slot)
        let v = kv.vSlot(layer: layer, position: position, slot: slot)
        writeHalfRow(
            to: k.buffer, offset: k.offset, bytes: bytes,
            seed: seed, position: position, inverted: false)
        writeHalfRow(
            to: v.buffer, offset: v.offset, bytes: bytes,
            seed: seed, position: position, inverted: true)
    }

    /// Valid finite FP16 data, not arbitrary bytes: `0x7C01` is a NaN, and a
    /// random byte pattern would make a wrong row look like a kernel bug.
    private func writeHalfRow(
        to buffer: MTLBuffer, offset: Int, bytes: Int,
        seed: Int, position: Int, inverted: Bool
    ) {
        let count = bytes / MemoryLayout<Float16>.stride
        var values = [Float16](repeating: 0, count: count)
        for i in 0..<count {
            let phase =
                Float(seed &* 31 &+ i &* 7 &+ position &* 3) * 0.001
            values[i] = Float16(
                (inverted ? cos(phase) : sin(phase)) * 0.25)
        }
        copyBytes(values, to: buffer.contents().advanced(by: offset))
    }

    private func copyBytes<T>(_ values: [T], to destination: UnsafeMutableRawPointer) {
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(destination, base, raw.count)
            }
        }
    }

    private func readRow(
        _ kv: KVCacheManager, position: Int, slot: Int = 0, key: Bool = true
    ) -> [UInt8] {
        if kv.streamsLayer(layer) {
            return kv.hostRowBytes(
                layer: layer, slot: slot, position: position, key: key)
        }
        let bytes = stride(kv)
        let slotRange =
            key
            ? kv.kSlot(layer: layer, position: position, slot: slot)
            : kv.vSlot(layer: layer, position: position, slot: slot)
        let p = slotRange.buffer.contents().advanced(by: slotRange.offset)
            .assumingMemoryBound(to: UInt8.self)
        return (0..<bytes).map { p[$0] }
    }

    private func makeVector(
        _ context: MetalContext, count: Int, seed: Float
    ) throws -> MTLBuffer {
        let elements = count * MemoryLayout<Float16>.stride
        guard
            let buffer = context.device.makeBuffer(
                length: elements, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("host-kv test vector") }
        let p = buffer.contents().bindMemory(
            to: Float16.self, capacity: count)
        for i in 0..<count {
            p[i] = Float16(sin(Float(i) * 0.13 + seed) * 0.5)
        }
        return buffer
    }

    private func encode(
        _ attention: Attention,
        context: MetalContext,
        kv: KVCacheManager,
        q: MTLBuffer,
        out: MTLBuffer,
        seqLen: Int,
        selection: KVWindowSelection? = nil,
        keepMask: MTLBuffer? = nil
    ) throws {
        let key = kv.keyView(layer: layer, validTokenCount: seqLen)
        let value = kv.valueView(layer: layer, validTokenCount: seqLen)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try attention.encodeFull(
            commandBuffer: commandBuffer,
            q: q,
            k: key.buffer, kOffset: key.offset,
            v: value.buffer, vOffset: value.offset,
            out: out,
            headDim: UInt32(kv.config.fullHeadDim),
            numQHeads: UInt32(kv.config.numHeads),
            numKVHeads: UInt32(kv.config.numFullKVHeads),
            seqLen: UInt32(seqLen),
            scale: Float(kv.config.attentionScale),
            kvFormat: key,
            keepMask: keepMask,
            windowSelection: selection)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
    }

    private func readOutput(_ out: MTLBuffer, count: Int) -> [Float] {
        let p = out.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }

    /// 1. Short-context parity: the window path and the contiguous path see the
    /// same rows, in the same order, and produce the same attention. With no
    /// eviction the allocation order is the logical order, so this is exact,
    /// not a tolerance.
    @Test func shortContextParity() throws {
        let context = try MetalContext()
        let wide = makeConfig(headDim: 512)
        let maxContext = 4_096
        let contiguous = try makeManager(
            context: context, config: wide, maxContext: maxContext, window: nil)
        let streamed = try makeManager(
            context: context, config: wide, maxContext: maxContext,
            window: maxContext)
        #expect(streamed.hostArenaBytes > 0)
        #expect(streamed.windowBytes > 0)
        for position in 0..<maxContext {
            fillRow(contiguous, position: position, seed: 11)
            fillRow(streamed, position: position, seed: 11)
            contiguous.advance()
            streamed.advance()
        }
        for position in [0, 1_024, 2_048, 4_095] {
            #expect(
                readRow(contiguous, position: position)
                    == readRow(streamed, position: position),
                "row \(position) must match across the tier")
        }

        let elements = wide.numHeads * wide.fullHeadDim
        let q = try makeVector(context, count: elements, seed: 1.7)
        let outA = try makeVector(context, count: elements, seed: 0)
        let outB = try makeVector(context, count: elements, seed: 0)
        let attention = try Attention(context: context)
        try encode(
            attention, context: context, kv: contiguous, q: q, out: outA,
            seqLen: maxContext)
        let selection = try streamed.prepareDecodeWindow(
            layer: layer, slot: 0, position: maxContext - 1,
            indices: nil, count: maxContext)
        try encode(
            attention, context: context, kv: streamed, q: q, out: outB,
            seqLen: maxContext, selection: selection)
        let want = readOutput(outA, count: elements)
        let got = readOutput(outB, count: elements)
        for i in 0..<elements {
            #expect(
                abs(want[i] - got[i]) < 1e-6,
                "output \(i): \(got[i]) vs \(want[i])")
        }
    }

    /// 2. Window boundary: a selection that keeps crossing a 32-row window
    /// matches contiguous attention over the same mask, and the newest row is
    /// written back to the arena before it can be evicted.
    @Test func windowBoundaryMatchesContiguousSelection() throws {
        let context = try MetalContext()
        let toy = makeConfig(headDim: 32)
        let maxContext = 131_072
        let window = 64
        let contiguous = try makeManager(
            context: context, config: toy, maxContext: maxContext, window: nil)
        // Full-attention layers grow lazily; a 128K run must reserve its
        // prefix or `physicalSlot` wraps into the initial 8K rows.
        try contiguous.reserve(tokens: maxContext)
        #expect(contiguous.capacity(layer: layer) >= maxContext)
        let streamed = try makeManager(
            context: context, config: toy, maxContext: maxContext, window: window)
        let attention = try Attention(context: context)
        let elements = toy.numHeads * toy.fullHeadDim
        let q = try makeVector(context, count: elements, seed: 3.1)
        let outA = try makeVector(context, count: elements, seed: 0)
        let outB = try makeVector(context, count: elements, seed: 0)
        let checked: Set<Int> = [65, 4_096, 65_536, maxContext - 1]

        func positions(at position: Int) -> [UInt32] {
            var selected = Set<UInt32>()
            for p in Swift.stride(from: 0, through: position, by: 8_192) {
                selected.insert(UInt32(p))
            }
            for p in max(0, position - 7)...position {
                selected.insert(UInt32(p))
            }
            selected.insert(0)
            selected.insert(UInt32(position))
            return selected.sorted()
        }

        for position in 0..<maxContext {
            fillRow(contiguous, position: position, seed: 5)
            fillRow(streamed, position: position, seed: 5)
            contiguous.advance()
            streamed.advance()
            guard checked.contains(position) else { continue }
            let keys = positions(at: position)
            var mask = [UInt8](repeating: 0, count: position + 1)
            for key in keys { mask[Int(key)] = 1 }
            let maskBuffer = try #require(
                context.device.makeBuffer(
                    length: mask.count, options: .storageModeShared))
            copyBytes(mask, to: maskBuffer.contents())
            try encode(
                attention, context: context, kv: contiguous, q: q, out: outA,
                seqLen: position + 1, keepMask: maskBuffer)
            let indexBuffer = try #require(
                context.device.makeBuffer(
                    length: keys.count * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared))
            copyBytes(keys, to: indexBuffer.contents())
            let selection = try streamed.prepareDecodeWindow(
                layer: layer, slot: 0, position: position,
                indices: indexBuffer, count: keys.count)
            try encode(
                attention, context: context, kv: streamed, q: q, out: outB,
                seqLen: position + 1, selection: selection)
            let want = readOutput(outA, count: elements)
            let got = readOutput(outB, count: elements)
            for i in 0..<elements {
                #expect(
                    abs(want[i] - got[i]) < 2e-3,
                    "position \(position) output \(i): \(got[i]) vs \(want[i])")
            }
        }
        // The newest row must have reached the arena even though the window
        // was recycled thousands of times on the way there.
        for position in [0, 64, 4_096, 65_536, maxContext - 1] {
            #expect(
                readRow(streamed, position: position)
                    == readRow(contiguous, position: position),
                "arena row \(position) must match after crossing the window")
        }
    }

    /// 3. Memory: the Metal allocation is the window; the arena is the
    /// difference. At 1M positions the toy's two full layers map only 64 rows
    /// per layer in VRAM, while the arena spans the whole logical cache.
    @Test func memoryIsTheWindowNotTheFullKV() throws {
        let context = try MetalContext()
        let toy = makeConfig(headDim: 32)
        let maxContext = 1_048_576
        let window = 64
        let contiguous = try makeManager(
            context: context, config: toy, maxContext: maxContext, window: nil)
        let streamed = try makeManager(
            context: context, config: toy, maxContext: maxContext, window: window)
        let rowBytes = stride(contiguous)
        let fullLayers = (0..<toy.numLayers)
            .filter { contiguous.layerKind($0) == .full }.count
        #expect(fullLayers == 2)
        let contiguousKVMetal =
            (0..<toy.numLayers)
            .filter { contiguous.layerKind($0) != .linear }
            .reduce(0) { $0 + contiguous.bufferLength(layer: $1) * 2 }
        // The contiguous manager grows lazily, so its current Metal cost is
        // the initial capacity rather than the advertised context.
        #expect(
            contiguousKVMetal
                == fullLayers * KVCacheManager.initialCapacityTokens * rowBytes * 2)
        #expect(
            streamed.windowBytes == fullLayers * window * rowBytes * 2,
            "the streamed Metal cost is the window")
        #expect(
            streamed.hostArenaBytes == fullLayers * maxContext * rowBytes * 2,
            "the arena is the full logical cache")
        #expect(streamed.hostArenaBytes > streamed.windowBytes * 1_000)
        #expect(streamed.bufferLength(layer: layer) == window * rowBytes)
    }

    /// 4. Speed is selection-bounded: the bytes a step copies are a function
    /// of its selection, and a repeated selection is already resident. Both
    /// hold at a context far larger than the window.
    @Test func copiesStaySelectionBounded() throws {
        let context = try MetalContext()
        let maxContext = 4_096
        let window = 128
        let streamed = try makeManager(
            context: context, maxContext: maxContext, window: window)
        for position in 0..<2_048 {
            fillRow(streamed, position: position, seed: 9)
            streamed.advance()
        }
        var keys: [UInt32] = (0..<16).map { UInt32($0 * 127 % 2_048) }
        keys.append(2_047)
        let indexBuffer = try #require(
            context.device.makeBuffer(
                length: keys.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared))
        copyBytes(keys, to: indexBuffer.contents())
        _ = try streamed.prepareDecodeWindow(
            layer: layer, slot: 0, position: 2_047,
            indices: indexBuffer, count: keys.count)
        let firstCopies = streamed.windowCopyBytes
        #expect(
            firstCopies <= keys.count * stride(streamed) * 2,
            "the first gather copies exactly its selection")
        streamed.windowCopyBytes = 0
        _ = try streamed.prepareDecodeWindow(
            layer: layer, slot: 0, position: 2_047,
            indices: indexBuffer, count: keys.count)
        #expect(
            streamed.windowCopyBytes == 0,
            "a repeated selection is already in the window")
        // A disjoint selection still costs one row copy per key, not one per
        // visible position.
        var other: [UInt32] = (0..<16).map { UInt32(1_000 + $0 * 61) }
        other.append(2_047)
        let otherBuffer = try #require(
            context.device.makeBuffer(
                length: other.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared))
        copyBytes(other, to: otherBuffer.contents())
        streamed.windowCopyBytes = 0
        _ = try streamed.prepareDecodeWindow(
            layer: layer, slot: 0, position: 2_047,
            indices: otherBuffer, count: other.count)
        #expect(
            streamed.windowCopyBytes <= other.count * stride(streamed) * 2,
            "a disjoint selection copies its keys, not the context")
    }

    /// 5. Prefill: a chunk's KV writes go to staging, the selected union is
    /// gathered into the window, and the tiled kernel reads through the row
    /// indirection. The attention must equal the contiguous kernel's result
    /// over the same selection, and the chunk's rows must reach the arena.
    @Test func prefillWindowMatchesContiguousSelection() throws {
        let context = try MetalContext()
        let maxContext = 128
        let window = 32
        let start = 56
        let tokens = 8
        let visible = start + tokens
        let indexWidth = 8
        let contiguous = try makeManager(
            context: context, maxContext: maxContext, window: nil)
        let streamed = try makeManager(
            context: context, maxContext: maxContext, window: window)
        for position in 0..<start {
            fillRow(contiguous, position: position, seed: 13)
            fillRow(streamed, position: position, seed: 13)
            contiguous.advance()
            streamed.advance()
        }
        // Current chunk: contiguous writes straight intto its cache; streamed
        // writes into staging, exactly as `copyPrefillKVToCache` does.
        streamed.beginPrefillWrite(
            layer: layer, slot: 0, startPosition: start, tokenCount: tokens)
        for position in start..<visible {
            let bytes = stride(contiguous)
            let ck = contiguous.kRange(
                layer: layer, start: position, count: 1)
            let cv = contiguous.vRange(
                layer: layer, start: position, count: 1)
            let sk = streamed.kRange(layer: layer, start: position, count: 1)
            let sv = streamed.vRange(layer: layer, start: position, count: 1)
            writeHalfRow(
                to: ck.buffer, offset: ck.offset, bytes: bytes,
                seed: 13, position: position, inverted: false)
            writeHalfRow(
                to: sk.buffer, offset: sk.offset, bytes: bytes,
                seed: 13, position: position, inverted: false)
            writeHalfRow(
                to: cv.buffer, offset: cv.offset, bytes: bytes,
                seed: 13, position: position, inverted: true)
            writeHalfRow(
                to: sv.buffer, offset: sv.offset, bytes: bytes,
                seed: 13, position: position, inverted: true)
        }

        // Every row selects eight keys, including its own position and the
        // first block, so the union crosses the window and both the staged and
        // the archived rows are exercised.
        var mask = [UInt8](repeating: 0, count: tokens * visible)
        var indices = [UInt32](repeating: 0, count: tokens * indexWidth)
        var counts = [UInt32](repeating: 0, count: tokens)
        for row in 0..<tokens {
            let own = start + row
            let selected = [0, 1, 2, 3, 31, 47, own - 1, own]
            counts[row] = UInt32(selected.count)
            for (i, key) in selected.enumerated() {
                let clamped = min(max(key, 0), own)
                indices[row * indexWidth + i] = UInt32(clamped)
                mask[row * visible + clamped] = 1
            }
        }
        let maskBuffer = try #require(
            context.device.makeBuffer(
                length: mask.count, options: .storageModeShared))
        let indexBuffer = try #require(
            context.device.makeBuffer(
                length: indices.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared))
        let countBuffer = try #require(
            context.device.makeBuffer(
                length: counts.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared))
        copyBytes(mask, to: maskBuffer.contents())
        copyBytes(indices, to: indexBuffer.contents())
        copyBytes(counts, to: countBuffer.contents())
        let selection = QSASelection(
            mask: maskBuffer, maskStride: visible,
            indices: indexBuffer, indexStride: indexWidth,
            counts: countBuffer)

        let qElements = tokens * config.numHeads * config.fullHeadDim
        let outElements = qElements
        let q = try makeVector(context, count: qElements, seed: 7.3)
        let outA = try makeVector(context, count: outElements, seed: 0)
        let outB = try makeVector(context, count: outElements, seed: 0)
        let prefill = try PrefillAttention(context: context)
        func params() -> PrefillAttentionParams {
            PrefillAttentionParams(
                startPosition: UInt32(start),
                queryCount: UInt32(tokens),
                headDim: UInt32(config.fullHeadDim),
                numQHeads: UInt32(config.numHeads),
                numKVHeads: UInt32(config.numFullKVHeads),
                kvValidCount: UInt32(visible),
                slidingWindow: UInt32(visible),
                kvTokenStrideElements: UInt32(config.numFullKVHeads * config.fullHeadDim),
                qTokenStrideElements: UInt32(config.numHeads * config.fullHeadDim),
                oTokenStrideElements: UInt32(config.numHeads * config.fullHeadDim),
                scale: Float(config.attentionScale),
                kvBits: 16,
                kvTokenStrideBytes: UInt32(stride(contiguous)),
                kvValueBytes: UInt32(stride(contiguous)))
        }

        let ck = contiguous.keyView(layer: layer, validTokenCount: visible)
        let cv = contiguous.valueView(layer: layer, validTokenCount: visible)
        let streamedKey = streamed.keyView(layer: layer, validTokenCount: visible)
        let streamedValue = streamed.valueView(layer: layer, validTokenCount: visible)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try prefill.encodeCausal(
            commandBuffer: commandBuffer,
            q: q, k: ck.buffer, kOffset: ck.offset,
            v: cv.buffer, vOffset: cv.offset,
            out: outA,
            params: params(),
            keepMask: selection.mask, keepStride: selection.maskStride,
            keepIndices: selection.indices,
            keepIndexStride: selection.indexStride,
            keepCounts: selection.counts,
            path: .causalTiled)
        let prepared = try streamed.preparePrefillWindow(
            layer: layer, slot: 0, startPosition: start,
            tokenCount: tokens, selection: selection,
            commandBuffer: commandBuffer)
        try prefill.encodeCausal(
            commandBuffer: commandBuffer,
            q: q, k: streamedKey.buffer, kOffset: streamedKey.offset,
            v: streamedValue.buffer, vOffset: streamedValue.offset,
            out: outB,
            params: params(),
            keepMask: prepared.mask, keepStride: 0,
            keepIndices: prepared.indices,
            keepIndexStride: prepared.indexStride,
            keepCounts: prepared.counts,
            keepRows: prepared.rows,
            path: .causalTiled)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        let want = readOutput(outA, count: outElements)
        let got = readOutput(outB, count: outElements)
        for i in 0..<outElements {
            #expect(
                abs(want[i] - got[i]) < 2e-3,
                "prefill output \(i): \(got[i]) vs \(want[i])")
        }
        streamed.advance(slot: 0, by: tokens)
        for position in start..<visible {
            #expect(
                readRow(streamed, position: position)
                    == readRow(contiguous, position: position),
                "prefill arena row \(position) must match after staging commit")
        }
    }

    /// The snapshot tier must read the arena, not the window: a prefix longer
    /// than the window has to round-trip even though most of its rows are not
    /// resident at snapshot time.
    @Test func streamedSnapshotRoundTripsThroughArena() throws {
        let context = try MetalContext()
        let maxContext = 128
        let window = 16
        let tokens = 100
        let source = try makeManager(
            context: context, maxContext: maxContext, window: window)
        for position in 0..<tokens {
            fillRow(source, position: position, seed: 21)
            source.advance()
        }
        let lengths = try source.snapshotSegmentLengths(at: tokens)
        var payload = Data()
        try source.appendSnapshotPayload(to: &payload, segmentLengths: lengths)
        #expect(payload.count == 10 * 2 * tokens * stride(source))

        let receiver = try makeManager(
            context: context, maxContext: maxContext, window: window)
        var offset = 0
        try payload.withUnsafeBytes { bytes in
            try receiver.restoreSnapshot(
                position: tokens, segmentLengths: lengths,
                bytes: bytes, offset: &offset)
        }
        #expect(receiver.position == tokens)
        #expect(offset == payload.count)
        for position in [0, 1, 15, 63, tokens - 1] {
            for key in [true, false] {
                #expect(
                    receiver.hostRowBytes(
                        layer: layer, slot: 0, position: position, key: key)
                        == source.hostRowBytes(
                            layer: layer, slot: 0, position: position, key: key),
                    "restored row \(position) key=\(key)")
            }
        }
    }

}
