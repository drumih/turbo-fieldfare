import Foundation

/// The text-only 24-layer loop. This is a single-token prefill reference: it
/// builds the prelude, repeats the 6..17 core three times, and runs the coda.
/// KV/conv caches for continuation tokens are not implemented here yet.
public struct Inkling2TextModel: Sendable {
    public let config: Inkling2Config
    private let tensors: Inkling2DenseTensors
    private let layers: [Inkling2DecoderLayer]
    private let layerIndices: [Int]
    private let messageQueries: [[[Float]]]
    private let finalResidual: Inkling2AttentionResidual
    private let finalNorm: [Float]
    private let embeddingNorm: [Float]

    public init(
        directory: URL, config: Inkling2Config, store: Inkling2ExpertStore
    ) throws {
        self.config = config
        let tensors = try Inkling2DenseTensors(directory: directory)
        self.tensors = tensors
        self.embeddingNorm = try tensors.embeddingNorm()
        self.finalNorm = try tensors.finalNorm()
        var layers: [Inkling2DecoderLayer] = []
        let local = Set(config.text.localLayers)
        for index in 0..<config.text.layerCount {
            let attention: any Inkling2AttentionBlock
            if local.contains(index) {
                attention = try Inkling2SlidingAttention(
                    tensors: tensors, layer: index,
                    heads: config.text.attentionHeads,
                    kvHeads: config.text.keyValueHeads,
                    headDim: config.text.headDim,
                    relativeDim: config.text.relativeDimension,
                    extent: config.text.relativeExtent,
                    sliding: config.text.slidingWindow,
                    hidden: config.text.hiddenSize,
                    kernelSize: config.text.convolutionKernel)
            } else {
                attention = try Inkling2GlobalAttention(
                    tensors: tensors, layer: index,
                    heads: config.text.attentionHeads,
                    kvHeads: config.text.keyValueHeads,
                    headDim: config.text.headDim,
                    relativeDim: config.text.relativeDimension,
                    extent: config.text.relativeExtent,
                    sliding: 0,
                    hidden: config.text.hiddenSize,
                    kernelSize: config.text.convolutionKernel)
            }
            layers.append(
                try Inkling2DecoderLayer(
                    tensors: tensors, store: store, layer: index,
                    hidden: config.text.hiddenSize,
                    kernelSize: config.text.convolutionKernel,
                    isBlockBoundary: index % config.text.attentionResidualBlock == 0,
                    isSliding: local.contains(index),
                    attention: attention))
        }
        self.layers = layers
        self.layerIndices =
            Array(0..<config.text.loop.coreStart)
            + Array(config.text.loop.coreStart..<config.text.loop.coreEnd)
            .flatMap { Array(repeating: $0, count: config.text.loop.passes) }
            + Array(config.text.loop.coreEnd..<config.text.layerCount)
        self.messageQueries = try Self.loadMessageQueries(
            tensors: tensors, loop: config.text.loop, hidden: config.text.hiddenSize)
        self.finalResidual = try Inkling2AttentionResidual(
            tensors: tensors, tensorName: "model.llm.attn_res.output_query",
            queries: 1, hidden: config.text.hiddenSize)
    }

    private static func loadMessageQueries(
        tensors: Inkling2DenseTensors, loop: Inkling2Config.Loop, hidden: Int
    ) throws -> [[[Float]]] {
        let count = loop.coreEnd - loop.coreStart
        let attnName = "model.llm.middle_cycle_cross_pass_queries"
        let kvName = "model.llm.middle_cycle_cross_pass_kv_queries"
        guard tensors.hasTensor(attnName), tensors.hasTensor(kvName) else {
            return Array(
                repeating: Array(
                    repeating: [Float](repeating: 0, count: hidden), count: 4),
                count: count)
        }
        let attn = try tensors.matrix(attnName)
        let kv = try tensors.matrix(kvName)
        var result: [[[Float]]] = []
        for index in 0..<count {
            let a0 = Array(attn[(index * 2 * hidden)..<((index * 2 + 1) * hidden)])
            let a1 = Array(attn[((index * 2 + 1) * hidden)..<((index * 2 + 2) * hidden)])
            let k0 = Array(kv[(index * 2 * hidden)..<((index * 2 + 1) * hidden)])
            let k1 = Array(kv[((index * 2 + 1) * hidden)..<((index * 2 + 2) * hidden)])
            result.append([a0, k0, k1, a1])
        }
        return result
    }

    public func forward(token: Int) throws -> [Float] {
        let hidden = config.text.hiddenSize
        var x = Inkling2Ops.rmsNorm(
            try tensors.embeddingRow(token), weight: embeddingNorm)
        var blocks: [[Float]] = []
        var bank: [[Float]]?
        var prelude = x
        var preludeBlocks: [[Float]] = []
        for (visit, index) in layerIndices.enumerated() {
            if index == config.text.loop.coreStart {
                if visit == index {
                    prelude = x
                    preludeBlocks = blocks
                } else {
                    var states = Array(blocks[(preludeBlocks.count + 1)...])
                    states.append(x)
                    bank = states.map {
                        Inkling2Ops.rmsNorm($0, weight: [Float](repeating: 1, count: hidden))
                    }
                    x = prelude
                    blocks = preludeBlocks
                }
            }
            var messages: [Float]?
            if let bank, index < config.text.loop.coreEnd {
                let query = messageQueries[index - config.text.loop.coreStart]
                var message = [Float]()
                for row in query {
                    var scores: [Float] = []
                    for state in bank {
                        var dot: Float = 0
                        for dimension in 0..<hidden { dot += row[dimension] * state[dimension] }
                        scores.append(dot / Float(hidden).squareRoot())
                    }
                    let probabilities = Inkling2AttentionMath.softmax(scores)
                    var combined = [Float](repeating: 0, count: hidden)
                    for (stateIndex, probability) in probabilities.enumerated() {
                        for dimension in 0..<hidden {
                            combined[dimension] += probability * bank[stateIndex][dimension]
                        }
                    }
                    message.append(contentsOf: combined)
                }
                messages = message
            }
            x = try layers[index].forward(x, blocks: &blocks, messages: messages)
        }
        let final = finalResidual.forward(blocks, partial: x)
        let normalized = Inkling2Ops.rmsNorm(final, weight: finalNorm)
        return try tensors.logits(normalized)
    }
}
