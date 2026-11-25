import Foundation
import Metal
import TinyTitanCore

// Provenance: copied from sources/TinyTitan/Inkling2GGUF/Inkling2GGUFDecoder.swift
// (TT-121 phase 2) under the variant-isolation rule; the graph is unchanged and
// now reads a verified `.ssdai` install instead of mapping the GGUF.

/// Native decoder for the Inkling 2 Q4 install, ported from llama.cpp's
/// `src/models/inkling2.cpp` graph. One command buffer runs a chunk of
/// consecutive tokens — one at decode, up to `maxChunk` at prefill — through
/// the 48 layer visits (depth mixes, loop controllers, short convolutions,
/// relative attention, the sigmoid router and the routed experts) with no CPU
/// round trip; the next token is the GPU argmax of the chunk's last position.
package final class InklingQ4Decoder {
    typealias Operand = GGUFGemvIds.Operand

    let manifest: InklingQ4Manifest
    let weights: InklingQ4Weights
    package let state: InklingQ4State
    private let context: MetalContext
    let ops: InklingQ4Ops
    /// Tokens of the chunk being encoded.
    private(set) var tokens = 1
    /// Sequence slot whose KV and convolution caches are active for this chunk.
    private(set) var activeSlot = 0
    /// CPU time spent encoding, and GPU time of, the last chunk's command buffer.
    private(set) var lastEncodeSeconds: Double = 0
    private(set) var lastGPUSeconds: Double = 0
    private var freePersistent: [Int] = []
    private var transientCursor = 0
    /// The routed experts' slots when they stream; nil when they are mapped.
    package let expertCache: InklingQ4ExpertCache?
    /// The command buffer being encoded. A chunk is one buffer unless experts
    /// stream, when each layer visit's router ends one (`resolveRouting`).
    private var segment: (buffer: MTLCommandBuffer, encoder: MTLComputeCommandEncoder)?
    /// Committed command buffers not yet waited for, oldest first.
    private var inFlight: [MTLCommandBuffer] = []
    /// When experts stream, each decode visit also runs the next visit's
    /// router on its own MoE input and the cache reads that prediction in the
    /// background while the GPU works. Off, a visit loads only what its own
    /// router chose. Either way the logits are the same bits.
    package var lookaheadPrefetch = true
    /// When experts stream, a decode token is one command buffer whose
    /// visits wait on a shared event while the CPU resolves their slots
    /// (`InklingQ4EventRouting`). Off, each router ends a command buffer.
    /// Either way the logits are the same bits.
    package var eventRouting = true
    /// Diagnostic: with event routing and a routing log, also run the
    /// routers of the visits 2 ... `traceDepth` ahead on each decode visit's
    /// MoE input and log their choices. 0 or 1 adds nothing.
    package var traceDepth = 0
    /// Scratch for those routers: ids, logits and weights per depth.
    lazy var deepScratch: (ids: MTLBuffer, logits: MTLBuffer, weights: MTLBuffer)? = {
        let n = 8
        let device = context.device
        guard let ids = device.makeBuffer(length: n * manifest.expertsPerToken * 4),
            let logits = device.makeBuffer(length: n * manifest.expertCount * 4),
            let weights = device.makeBuffer(length: n * manifest.expertsPerToken * 4)
        else { return nil }
        return (ids, logits, weights)
    }()
    let routing: InklingQ4EventRouting?
    private var segmentGPUSeconds: Double = 0
    private var segmentWaitSeconds: Double = 0

    package init(
        directory: URL, maxContext: Int = 4_096, maxChunk: Int = 256,
        slots: Int = 1, experts residency: InklingQ4ExpertResidency = .mapped,
        context: MetalContext? = nil
    ) throws {
        let context = try context ?? MetalContext()
        self.context = context
        let install = try InklingQ4Install(directory: directory)
        manifest = try InklingQ4Manifest(install: install)
        let supported = InklingQ4Ops.supportedTypes
        for layer in manifest.layers {
            let types = [layer.gateUpType, layer.downType] + Array(layer.projectionTypes.values)
            if let unsupported = types.first(where: { !supported.contains($0) }) {
                throw GGUFDequantError.unsupportedType(
                    "\(unsupported.name) in layer \(layer.index)"
                )
            }
        }
        guard maxChunk >= 1, maxChunk <= maxContext else {
            throw MetalError.invalidState("chunk \(maxChunk) for context \(maxContext)")
        }
        switch residency {
        case .mapped:
            expertCache = nil
        case .slots(let count):
            expertCache = try InklingQ4ExpertCache(
                install: install, manifest: manifest, slots: count, device: context.device)
        }
        weights = try InklingQ4Weights(
            install: install, manifest: manifest, device: context.device, cache: expertCache)
        guard weights.tokenEmbedding.type == .q8_0,
            supported.contains(weights.output.type)
        else {
            throw GGUFDequantError.unsupportedType(
                "embedding \(weights.tokenEmbedding.type.name) / head \(weights.output.type.name)")
        }
        state = try InklingQ4State(
            manifest: manifest, device: context.device, maxContext: maxContext,
            maxChunk: maxChunk, slots: slots)
        ops = try InklingQ4Ops(context: context)
        if let expertCache, !expertCache.isIdentity {
            routing = try InklingQ4EventRouting(
                device: context.device, cache: expertCache, state: state,
                expertsPerToken: manifest.expertsPerToken)
        } else {
            routing = nil
        }
    }

    /// Feeds `token` at `position` into `slot` and returns the greedy next
    /// token. With `returnLogits`, also returns the softcapped logits of the
    /// unpadded vocabulary.
    package func step(
        token: Int, position: Int, slot: Int = 0, returnLogits: Bool = false
    ) throws -> (next: Int, logits: [Float]?) {
        try prefill([token], position: position, slot: slot, returnLogits: returnLogits)
    }

    /// Feeds `prompt` at positions `position ..<` into `slot`, in chunks of up
    /// to `maxChunk` tokens, and returns the greedy token after its last one
    /// (with its softcapped logits when asked).
    package func prefill(
        _ prompt: [Int], position: Int, slot: Int = 0, returnLogits: Bool = false
    ) throws -> (next: Int, logits: [Float]?) {
        guard !prompt.isEmpty, position + prompt.count <= state.maxContext else {
            throw MetalError.invalidState(
                "\(prompt.count) tokens at \(position) for context \(state.maxContext)")
        }
        guard slot >= 0, slot < state.slots else {
            throw MetalError.invalidState("slot \(slot) out of range 0..<\(state.slots)")
        }
        var result: (next: Int, logits: [Float]?) = (0, nil)
        var offset = 0
        while offset < prompt.count {
            let chunk = Array(prompt[offset..<min(prompt.count, offset + state.maxChunk)])
            let last = offset + chunk.count == prompt.count
            result = try run(
                chunk, position: position + offset, slot: slot, returnLogits: returnLogits && last)
            offset += chunk.count
        }
        return result
    }

    /// Greedy generation: prefill `prompt`, then decode until `count` tokens.
    func generate(prompt: [Int], count: Int) throws -> [Int] {
        state.reset()
        var next = try prefill(prompt, position: 0).next
        var produced: [Int] = []
        for offset in 0..<count {
            produced.append(next)
            if offset + 1 < count {
                next = try step(token: next, position: prompt.count + offset).next
            }
        }
        return produced
    }

    private func run(_ chunk: [Int], position: Int, slot: Int, returnLogits: Bool) throws
        -> (next: Int, logits: [Float]?)
    {
        activeSlot = slot
        tokens = chunk.count
        let tokenSlots = state.tokens.contents().bindMemory(to: Int32.self, capacity: chunk.count)
        for (index, token) in chunk.enumerated() { tokenSlots[index] = Int32(token) }
        state.step.contents().storeBytes(of: UInt32(position), as: UInt32.self)
        let encodeStarted = Date()
        segmentGPUSeconds = 0
        segmentWaitSeconds = 0
        do {
            try encodeChunk(try beginSegment())
            try endSegment()
        } catch {
            // A handler may have failed first and left the GPU running on
            // stale slots; its error is the one that explains the result.
            throw routing?.takeFailure() ?? error
        }
        if let failure = routing?.takeFailure() { throw failure }
        lastEncodeSeconds = Date().timeIntervalSince(encodeStarted) - segmentWaitSeconds
        lastGPUSeconds = segmentGPUSeconds
        let next = Int(state.nextToken.contents().load(as: Int32.self))
        guard returnLogits else { return (next, nil) }
        let cap = manifest.finalLogitSoftcap
        let raw = state.logits.contents().bindMemory(to: Float.self, capacity: manifest.vocabulary)
        let logits = (0..<manifest.vocabulary).map { index in
            cap > 0 ? cap * tanh(raw[index] / cap) : raw[index]
        }
        return (next, logits)
    }

    // MARK: - Pool slabs

    var hidden: Int { manifest.hidden }

    /// A slab for a value that may outlive its layer visit (a visit's output).
    func persistentSlab() throws -> Int {
        guard let slab = freePersistent.popLast() else {
            throw MetalError.invalidState("no free persistent slab")
        }
        return slab
    }

    /// `count` consecutive slabs for one visit's temporaries.
    func rows(_ count: Int) throws -> Int {
        guard transientCursor + count <= state.transientSlabs else {
            throw MetalError.invalidState("transient slabs exhausted at \(transientCursor)")
        }
        defer { transientCursor += count }
        return state.persistentSlabs + transientCursor
    }

    /// Starts a visit: temporaries are released and every persistent slab
    /// outside `live` returns to the free list.
    private func beginVisit(live: Set<Int>) {
        transientCursor = 0
        freePersistent = (0..<state.persistentSlabs).filter { !live.contains($0) }.reversed()
    }

    func row(_ index: Int) -> Operand { .init(state.pool, offset: index * state.slabFloats * 4) }

    func scratch(_ buffer: MTLBuffer) -> Operand { .init(buffer) }

    func rmsInto(
        _ x: Int, _ weight: InklingQ4Tensor, into out: Int? = nil,
        _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let target = try out ?? rows(1)
        try ops.rmsNorm(
            row(x), weight: weight.operand, out: row(target), width: hidden, count: tokens,
            epsilon: manifest.rmsEpsilon, encoder: e)
        return target
    }

    func mix(
        _ states: [Int], query: Operand, count: Int,
        norm: (weight: Operand, epsilon: Float)? = nil, _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let out = try rows(count)
        try ops.depthMix(
            pool: .init(state.pool), rows: states, queries: query, queryCount: count,
            out: row(out), width: hidden, tokens: tokens, slab: state.slabFloats, norm: norm,
            encoder: e)
        return out
    }

    // MARK: - Graph

    // MARK: - Segments

    /// Starts a command buffer and its encoder, and returns the encoder.
    func beginSegment() throws -> MTLComputeCommandEncoder {
        guard let buffer = context.queue.makeCommandBuffer(),
            let encoder = buffer.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { throw MetalError.commandEncoderFailed }
        segment = (buffer, encoder)
        return encoder
    }

    /// Ends and commits the current command buffer without waiting: the GPU
    /// starts on it while the CPU encodes what follows.
    func commitSegment() {
        guard let (buffer, encoder) = segment else { return }
        segment = nil
        encoder.endEncoding()
        buffer.commit()
        inFlight.append(buffer)
    }

    /// Commits the current command buffer and waits for every one in flight.
    func endSegment() throws {
        commitSegment()
        let started = Date()
        if let routing, let last = inFlight.last { routing.drive(last) }
        inFlight.last?.waitUntilCompleted()
        segmentWaitSeconds += Date().timeIntervalSince(started)
        let finished = inFlight
        inFlight = []
        for buffer in finished {
            if let error = buffer.error { throw error }
            segmentGPUSeconds += buffer.gpuEndTime - buffer.gpuStartTime
        }
    }

    /// At a decode visit with streamed experts and `eventRouting` on: ends
    /// the current encoder at the router and returns the encoder the experts
    /// go to, in the same command buffer, behind the CPU's slot resolution.
    /// Nil when the visit must end its command buffer instead.
    func encodeEventResolve(
        layer: Int, predicts: Int?, deep: (ids: MTLBuffer, layers: [Int])? = nil
    ) throws -> MTLComputeCommandEncoder? {
        guard tokens == 1, eventRouting, let routing, let current = segment else { return nil }
        let next = try routing.encodeResolve(
            buffer: current.buffer, encoder: current.encoder, layer: layer, predicts: predicts,
            deep: deep)
        segment = (current.buffer, next)
        return next
    }

    /// The encoder dispatches go to now (a visit may have started a new one).
    var currentEncoder: MTLComputeCommandEncoder {
        get throws {
            guard let segment else { throw MetalError.invalidState("no command buffer open") }
            return segment.encoder
        }
    }

    private func encodeChunk(_ first: MTLComputeCommandEncoder) throws {
        var e = first
        let m = manifest
        beginVisit(live: [])
        let embedded = try rows(1)
        try ops.embed(
            weights.tokenEmbedding, token: scratch(state.tokens), out: row(embedded),
            tokens: tokens, encoder: e)
        var cur = try rmsInto(embedded, weights.tokenNorm, into: try persistentSlab(), e)
        var blocks: [Int] = []
        var preludeBlocks: [Int] = []
        var preludeCur = cur
        var bank: [Int] = []
        var bankLast = cur
        let core = m.loopPasses > 1 ? m.loopEnd - m.loopStart : 0

        for visit in 0..<m.unrolledLayers {
            e = try currentEncoder
            let stored = m.storedLayer(forUnrolled: visit)
            let layer = weights.layers[stored]
            // Each pass of the looped core restarts from the prelude state and
            // reads the previous pass's blocks through messages.
            if core > 0, stored == m.loopStart, visit > m.loopStart {
                bank = Array(blocks[(preludeBlocks.count + 1)...])
                bankLast = cur
                blocks = preludeBlocks
                cur = preludeCur
            } else if core > 0, stored == m.loopStart {
                preludeBlocks = blocks
                preludeCur = cur
            }
            beginVisit(live: Set(blocks + preludeBlocks + bank + [cur, preludeCur, bankLast]))
            var messages: [Int?] = [nil, nil, nil, nil]
            if core > 0, stored >= m.loopStart, stored < m.loopEnd,
                (visit - m.loopStart) / core > 0, let attnQuery = weights.loopResQuery,
                let kvQuery = weights.loopKVResQuery
            {
                let offset = 2 * (stored - m.loopStart) * hidden * 4
                let attn = Operand(
                    attnQuery.operand.buffer, offset: attnQuery.operand.offset + offset)
                let kv = Operand(kvQuery.operand.buffer, offset: kvQuery.operand.offset + offset)
                var mAttn = 0
                var mKV = 0
                try ops.concurrent(e) {
                    mAttn = try mix(bank + [bankLast], query: attn, count: 2, e)
                    mKV = try mix(bank + [bankLast], query: kv, count: 2, e)
                }
                messages = [mAttn, mKV, mKV + 1, mAttn + 1]
            }
            cur = try encodeVisit(
                visit: visit, layer: layer, cur: cur, blocks: &blocks, messages: messages, e)
        }

        // Only the chunk's last position needs logits: the final mix reads the
        // slabs through a pool view offset to that token.
        e = try currentEncoder
        beginVisit(live: Set(blocks + [cur]))
        let mixedSlab = try rows(1)
        try ops.depthMix(
            pool: .init(state.pool, offset: (tokens - 1) * hidden * 4), rows: blocks + [cur],
            queries: weights.outputResQuery.operand, queryCount: 1, out: row(mixedSlab),
            width: hidden, tokens: 1, slab: state.slabFloats,
            norm: (weights.outputNorm.operand, m.rmsEpsilon), encoder: e)
        if !ops.skipped.contains("head") {
            try ops.gemv(weights.output, x: row(mixedSlab), y: scratch(state.logits), encoder: e)
        }
        try ops.argmax(
            logits: scratch(state.logits), count: m.vocabulary,
            partials: scratch(state.argmaxPartials), groupCount: InklingQ4State.argmaxGroups,
            token: scratch(state.nextToken), encoder: e)
    }
}
