import Foundation
import Metal
import TinyTitanCore

/// Native single-token decoder for Inkling2 GGUFs, ported from llama.cpp's
/// `src/models/inkling2.cpp` graph. A token's whole forward — 48 layer visits,
/// depth mixes, loop controllers, short convolutions, relative attention, the
/// sigmoid router and the routed experts — is one command buffer with no CPU
/// round trip; the next token is the GPU argmax.
final class Inkling2GGUFDecoder {
    typealias Operand = GGUFGemvIds.Operand

    let manifest: Inkling2GGUFManifest
    let weights: Inkling2GGUFWeights
    let state: Inkling2GGUFState
    private let context: MetalContext
    let ops: Inkling2GGUFOps
    var rowCursor = 0
    /// CPU time spent encoding, and GPU time of, the last token's command buffer.
    private(set) var lastEncodeSeconds: Double = 0
    private(set) var lastGPUSeconds: Double = 0
    /// What each pool row of the last token holds, for diagnostics.
    private(set) var rowLabels: [String] = []
    var labelPrefix = ""

    init(url: URL, maxContext: Int = 4_096, context: MetalContext? = nil) throws {
        let context = try context ?? MetalContext()
        self.context = context
        manifest = try Inkling2GGUFManifest(url: url)
        let supported: Set<GGUFType> = [.q8_0, .nvfp4, .f32, .bf16]
        for layer in manifest.layers {
            let types = [layer.gateUpType, layer.downType] + Array(layer.projectionTypes.values)
            if let unsupported = types.first(where: { !supported.contains($0) }) {
                throw GGUFDequantError.unsupportedType(
                    "\(unsupported.name) in layer \(layer.index) (this decoder runs NVFP4/Q8_0 files)"
                )
            }
        }
        weights = try Inkling2GGUFWeights(manifest: manifest, device: context.device)
        state = try Inkling2GGUFState(
            manifest: manifest, device: context.device, maxContext: maxContext)
        ops = try Inkling2GGUFOps(context: context)
    }

    /// Feeds `token` at `position` and returns the greedy next token. With
    /// `returnLogits`, also returns the softcapped logits of the unpadded vocabulary.
    func step(token: Int, position: Int, returnLogits: Bool = false) throws
        -> (next: Int, logits: [Float]?)
    {
        guard position < state.maxContext else {
            throw MetalError.invalidState("position \(position) past context \(state.maxContext)")
        }
        state.token.contents().storeBytes(of: Int32(token), as: Int32.self)
        state.step.contents().storeBytes(of: UInt32(position), as: UInt32.self)
        guard let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { throw MetalError.commandEncoderFailed }
        let encodeStarted = Date()
        try encodeToken(encoder)
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

    /// Greedy generation: prefill `prompt` token by token, then `count` tokens.
    func generate(prompt: [Int], count: Int) throws -> [Int] {
        state.reset()
        var next = 0
        for (position, token) in prompt.enumerated() {
            next = try step(token: token, position: position).next
        }
        var produced: [Int] = []
        for offset in 0..<count {
            produced.append(next)
            if offset + 1 < count {
                next = try step(token: next, position: prompt.count + offset).next
            }
        }
        return produced
    }

    // MARK: - Graph

    var hidden: Int { manifest.hidden }

    func rows(_ count: Int) throws -> Int {
        guard rowCursor + count <= state.poolRows else {
            throw MetalError.invalidState("hidden pool exhausted at \(rowCursor) rows")
        }
        defer { rowCursor += count }
        rowLabels.append(contentsOf: (0..<count).map { "\(labelPrefix)#\($0)" })
        return rowCursor
    }

    /// L2 norm of every pool row the last token wrote, with its label.
    func rowNorms() -> [(label: String, norm: Float)] {
        rowSummaries().map { ($0.label, $0.norm) }
    }

    /// Label, L2 norm, sum and first three values of every pool row the last
    /// token wrote: the same summary llama.cpp's eval callback prints.
    func rowSummaries() -> [(label: String, norm: Float, sum: Float, first: [Float])] {
        let base = state.pool.contents().bindMemory(to: Float.self, capacity: rowCursor * hidden)
        return (0..<rowCursor).map { index in
            var squares: Float = 0
            var sum: Float = 0
            for i in 0..<hidden {
                let value = base[index * hidden + i]
                squares += value * value
                sum += value
            }
            let first = (0..<3).map { base[index * hidden + $0] }
            return (rowLabels[index], squares.squareRoot(), sum, first)
        }
    }

    func row(_ index: Int) -> Operand { .init(state.pool, offset: index * hidden * 4) }

    func scratch(_ buffer: MTLBuffer) -> Operand { .init(buffer) }

    func rmsInto(_ x: Int, _ weight: Inkling2GGUFTensor, _ e: MTLComputeCommandEncoder)
        throws -> Int
    {
        let out = try rows(1)
        try ops.rmsNorm(
            row(x), weight: weight.operand, out: row(out), width: hidden,
            epsilon: manifest.rmsEpsilon, encoder: e)
        return out
    }

    func mix(
        _ states: [Int], query: Operand, count: Int,
        norm: (weight: Operand, epsilon: Float)? = nil, _ e: MTLComputeCommandEncoder
    ) throws -> Int {
        let out = try rows(count)
        try ops.depthMix(
            pool: state.pool, rows: states, queries: query, queryCount: count, out: row(out),
            width: hidden, norm: norm, encoder: e)
        return out
    }

    private func encodeToken(_ e: MTLComputeCommandEncoder) throws {
        rowCursor = 0
        rowLabels = []
        labelPrefix = "embed"
        let m = manifest
        let embedded = try rows(1)
        try ops.embed(
            weights.tokenEmbedding, token: scratch(state.token), out: row(embedded), encoder: e)
        var cur = try rmsInto(embedded, weights.tokenNorm, e)
        var blocks: [Int] = []
        var preludeBlocks: [Int] = []
        var preludeCur = cur
        var bank: [Int] = []
        var bankLast = cur
        let core = m.loopPasses > 1 ? m.loopEnd - m.loopStart : 0

        for visit in 0..<m.unrolledLayers {
            let stored = m.storedLayer(forUnrolled: visit)
            let layer = weights.layers[stored]
            var messages: [Int?] = [nil, nil, nil, nil]
            if core > 0, stored >= m.loopStart, stored < m.loopEnd {
                let pass = (visit - m.loopStart) / core
                if stored == m.loopStart {
                    if pass == 0 {
                        preludeBlocks = blocks
                        preludeCur = cur
                    } else {
                        bank = Array(blocks[(preludeBlocks.count + 1)...])
                        bankLast = cur
                        blocks = preludeBlocks
                        cur = preludeCur
                    }
                }
                if pass > 0, let attnQuery = weights.loopResQuery,
                    let kvQuery = weights.loopKVResQuery
                {
                    let offset = 2 * (stored - m.loopStart) * hidden * 4
                    let attn = Operand(
                        attnQuery.operand.buffer, offset: attnQuery.operand.offset + offset)
                    let kv = Operand(
                        kvQuery.operand.buffer, offset: kvQuery.operand.offset + offset)
                    var mAttn = 0
                    var mKV = 0
                    try ops.concurrent(e) {
                        mAttn = try mix(bank + [bankLast], query: attn, count: 2, e)
                        mKV = try mix(bank + [bankLast], query: kv, count: 2, e)
                    }
                    messages = [mAttn, mKV, mKV + 1, mAttn + 1]
                }
            }
            labelPrefix = "v\(visit)"
            cur = try encodeVisit(
                visit: visit, layer: layer, cur: cur, blocks: &blocks, messages: messages, e)
        }

        labelPrefix = "final"
        let mixed = try mix(blocks + [cur], query: weights.outputResQuery.operand, count: 1, e)
        let normed = try rmsInto(mixed, weights.outputNorm, e)
        if !ops.skipped.contains("head") {
            try ops.gemv(weights.output, x: row(normed), y: scratch(state.logits), encoder: e)
        }
        try ops.argmax(
            logits: scratch(state.logits), count: m.vocabulary,
            partials: scratch(state.argmaxPartials), groupCount: Inkling2GGUFState.argmaxGroups,
            token: scratch(state.nextToken), encoder: e)
    }
}
