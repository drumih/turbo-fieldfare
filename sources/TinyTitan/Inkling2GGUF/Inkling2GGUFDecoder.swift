import Foundation
import Metal
import TinyTitanCore

/// Native decoder for Inkling2 GGUFs, ported from llama.cpp's
/// `src/models/inkling2.cpp` graph. One command buffer runs a chunk of
/// consecutive tokens — one at decode, up to `maxChunk` at prefill — through
/// the 48 layer visits (depth mixes, loop controllers, short convolutions,
/// relative attention, the sigmoid router and the routed experts) with no CPU
/// round trip; the next token is the GPU argmax of the chunk's last position.
final class Inkling2GGUFDecoder {
    typealias Operand = GGUFGemvIds.Operand

    let manifest: Inkling2GGUFManifest
    let weights: Inkling2GGUFWeights
    let state: Inkling2GGUFState
    private let context: MetalContext
    let ops: Inkling2GGUFOps
    /// Tokens of the chunk being encoded.
    private(set) var tokens = 1
    /// CPU time spent encoding, and GPU time of, the last chunk's command buffer.
    private(set) var lastEncodeSeconds: Double = 0
    private(set) var lastGPUSeconds: Double = 0
    private var freePersistent: [Int] = []
    private var transientCursor = 0

    init(
        url: URL, maxContext: Int = 4_096, maxChunk: Int = 256, context: MetalContext? = nil
    ) throws {
        let context = try context ?? MetalContext()
        self.context = context
        manifest = try Inkling2GGUFManifest(url: url)
        let supported = Inkling2GGUFOps.supportedTypes
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
        weights = try Inkling2GGUFWeights(manifest: manifest, device: context.device)
        guard [.q8_0, .iq4XS].contains(weights.tokenEmbedding.type),
            supported.contains(weights.output.type)
        else {
            throw GGUFDequantError.unsupportedType(
                "embedding \(weights.tokenEmbedding.type.name) / head \(weights.output.type.name)")
        }
        state = try Inkling2GGUFState(
            manifest: manifest, device: context.device, maxContext: maxContext,
            maxChunk: maxChunk)
        ops = try Inkling2GGUFOps(context: context)
    }

    /// Feeds `token` at `position` and returns the greedy next token. With
    /// `returnLogits`, also returns the softcapped logits of the unpadded vocabulary.
    func step(token: Int, position: Int, returnLogits: Bool = false) throws
        -> (next: Int, logits: [Float]?)
    {
        try prefill([token], position: position, returnLogits: returnLogits)
    }

    /// Feeds `prompt` at positions `position ..<`, in chunks of up to
    /// `maxChunk` tokens, and returns the greedy token after its last one
    /// (with its softcapped logits when asked).
    func prefill(_ prompt: [Int], position: Int, returnLogits: Bool = false) throws
        -> (next: Int, logits: [Float]?)
    {
        guard !prompt.isEmpty, position + prompt.count <= state.maxContext else {
            throw MetalError.invalidState(
                "\(prompt.count) tokens at \(position) for context \(state.maxContext)")
        }
        var result: (next: Int, logits: [Float]?) = (0, nil)
        var offset = 0
        while offset < prompt.count {
            let chunk = Array(prompt[offset..<min(prompt.count, offset + state.maxChunk)])
            let last = offset + chunk.count == prompt.count
            result = try run(chunk, position: position + offset, returnLogits: returnLogits && last)
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

    private func run(_ chunk: [Int], position: Int, returnLogits: Bool) throws
        -> (next: Int, logits: [Float]?)
    {
        tokens = chunk.count
        let tokenSlots = state.tokens.contents().bindMemory(to: Int32.self, capacity: chunk.count)
        for (index, token) in chunk.enumerated() { tokenSlots[index] = Int32(token) }
        state.step.contents().storeBytes(of: UInt32(position), as: UInt32.self)
        guard let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { throw MetalError.commandEncoderFailed }
        let encodeStarted = Date()
        try encodeChunk(encoder)
        encoder.endEncoding()
        lastEncodeSeconds = Date().timeIntervalSince(encodeStarted)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        lastGPUSeconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
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
        _ x: Int, _ weight: Inkling2GGUFTensor, into out: Int? = nil,
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

    private func encodeChunk(_ e: MTLComputeCommandEncoder) throws {
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
            partials: scratch(state.argmaxPartials), groupCount: Inkling2GGUFState.argmaxGroups,
            token: scratch(state.nextToken), encoder: e)
    }
}
