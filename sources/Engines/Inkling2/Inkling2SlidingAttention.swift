import Foundation

/// The sliding-window attention block. It is the 20 local layers' path; the
/// four global layers need the dynamic convolution and are not handled here.
public struct Inkling2SlidingAttention: Sendable {
    public typealias State = Inkling2AttentionState

    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let relativeDim: Int
    public let extent: Int
    public let sliding: Int
    public let hidden: Int

    private let q: [Float]
    private let k: [Float]
    private let v: [Float]
    private let r: [Float]
    private let gate: [Float]
    private let sink: [Float]
    private let out: [Float]
    private let qScale: [Float]
    private let relScale: [Float]
    private let relProjection: [Float]
    private let kConv: [Float]
    private let vConv: [Float]
    private let kernelSize: Int
    private let gpu: Inkling2GPUExpert?
    private let gpuWeights: [String: Data]
    private let logScalingAlpha: Float?
    private let logScalingNFloor: Int?

    public init(
        tensors: Inkling2DenseTensors, layer: Int,
        heads: Int, kvHeads: Int, headDim: Int,
        relativeDim: Int, extent: Int, sliding: Int,
        hidden: Int, kernelSize: Int, gpu: Inkling2GPUExpert? = nil,
        logScalingAlpha: Float? = nil, logScalingNFloor: Int? = nil
    ) throws {
        let prefix = "model.llm.layers.\(layer).attn."
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.relativeDim = relativeDim
        self.extent = extent
        self.sliding = sliding
        self.hidden = hidden
        self.kernelSize = kernelSize
        self.gpu = gpu
        self.logScalingAlpha = logScalingAlpha
        self.logScalingNFloor = logScalingNFloor
        var gpuWeights: [String: Data] = [:]
        if gpu != nil {
            for tensor in [
                "attn.wq_du.weight", "attn.wk_dv.weight", "attn.wv_dv.weight",
                "attn.wr_du.weight", "attn.gate_attn.weight",
                "attn.attn_sink_proj.weight", "attn.wo_ud.weight",
            ] {
                gpuWeights[tensor] = try tensors.tensorBytes(
                    "model.llm.layers.\(layer).\(tensor)")
            }
        }
        self.gpuWeights = gpuWeights
        self.q = try tensors.layerMatrix(layer, "attn.wq_du.weight")
        self.k = try tensors.layerMatrix(layer, "attn.wk_dv.weight")
        self.v = try tensors.layerMatrix(layer, "attn.wv_dv.weight")
        self.r = try tensors.layerMatrix(layer, "attn.wr_du.weight")
        self.gate = try tensors.layerMatrix(layer, "attn.gate_attn.weight")
        self.sink = try tensors.layerMatrix(layer, "attn.attn_sink_proj.weight")
        self.out = try tensors.layerMatrix(layer, "attn.wo_ud.weight")
        self.qScale = try tensors.layerMatrix(layer, "attn.q_l2_clip.scale")
        self.relScale = try tensors.layerMatrix(layer, "attn.rel_l2_clip.scale")
        self.relProjection = try tensors.layerMatrix(layer, "attn.rel_logits_proj.proj")
        self.kConv = try tensors.layerMatrix(layer, "attn.k_sconv.weight")
        self.vConv = try tensors.layerMatrix(layer, "attn.v_sconv.weight")
        _ = prefix
    }

    // lint:allow-long The projection, convolution and attention stages share
    // the q/k/v/r buffers; splitting them would pass five arrays between helpers.
    public func forward(
        _ x: [Float], tokenCount: Int, state: State? = nil
    ) throws -> (values: [Float], state: State) {
        try forward(query: x, key: x, value: x, tokenCount: tokenCount, state: state)
    }

    // lint:allow-long The projection, convolution and attention stages share
    // the q/k/v/r buffers; splitting them would pass five arrays between helpers.
    public func forward(
        query: [Float], key: [Float], value: [Float],
        tokenCount: Int, state: State? = nil
    ) throws -> (values: [Float], state: State) {
        precondition(query.count == tokenCount * hidden)
        precondition(key.count == tokenCount * hidden)
        precondition(value.count == tokenCount * hidden)
        let qWidth = heads * headDim
        let kvWidth = kvHeads * headDim
        let rWidth = heads * relativeDim
        var qAll = [Float](repeating: 0, count: tokenCount * qWidth)
        var kAll = [Float](repeating: 0, count: tokenCount * kvWidth)
        var vAll = [Float](repeating: 0, count: tokenCount * kvWidth)
        var rAll = [Float](repeating: 0, count: tokenCount * rWidth)
        var gateAll = [Float](repeating: 0, count: tokenCount * qWidth)
        var sinkAll = [Float](repeating: 0, count: tokenCount * heads)
        for token in 0..<tokenCount {
            let queryInput = Array(query[(token * hidden)..<((token + 1) * hidden)])
            let keyInput = Array(key[(token * hidden)..<((token + 1) * hidden)])
            let valueInput = Array(value[(token * hidden)..<((token + 1) * hidden)])
            let qToken = try project(
                "attn.wq_du.weight", q, rows: qWidth, columns: hidden, queryInput)
            let kToken = try project(
                "attn.wk_dv.weight", k, rows: kvWidth, columns: hidden, keyInput)
            let vToken = try project(
                "attn.wv_dv.weight", v, rows: kvWidth, columns: hidden, valueInput)
            let rToken = try project(
                "attn.wr_du.weight", r, rows: rWidth, columns: hidden, queryInput)
            let gateToken = try project(
                "attn.gate_attn.weight", gate, rows: qWidth, columns: hidden, queryInput)
            let sinkToken = try project(
                "attn.attn_sink_proj.weight", sink, rows: heads, columns: hidden, queryInput)
            for head in 0..<heads {
                let qRange =
                    (token * qWidth + head * headDim)..<(token * qWidth + (head + 1) * headDim)
                let scale = Array(qScale[head * headDim..<(head + 1) * headDim])
                qAll.replaceSubrange(
                    qRange,
                    with: Inkling2AttentionMath.l2Clip(
                        Array(qToken[head * headDim..<(head + 1) * headDim]),
                        threshold: 22.627_417, scale: scale))
                let rStart = token * rWidth + head * relativeDim
                let rEnd = token * rWidth + (head + 1) * relativeDim
                let rRange = rStart..<rEnd
                let rScaleHead = Array(relScale[head * relativeDim..<(head + 1) * relativeDim])
                rAll.replaceSubrange(
                    rRange,
                    with: Inkling2AttentionMath.l2Clip(
                        Array(rToken[head * relativeDim..<(head + 1) * relativeDim]),
                        threshold: 8, scale: rScaleHead))
                for index in 0..<headDim {
                    gateAll[token * qWidth + head * headDim + index] =
                        gateToken[head * headDim + index]
                }
                sinkAll[token * heads + head] = sinkToken[head]
            }
            kAll.replaceSubrange(
                (token * kvWidth)..<((token + 1) * kvWidth), with: kToken)
            vAll.replaceSubrange(
                (token * kvWidth)..<((token + 1) * kvWidth), with: vToken)
        }
        // Chunked short convolution over the whole KV width.
        let priorK = state?.k ?? [Float](repeating: 0, count: kvWidth * (kernelSize - 1))
        let priorV = state?.v ?? [Float](repeating: 0, count: kvWidth * (kernelSize - 1))
        let kConvResult = Inkling2AttentionMath.shortConvolution(
            kAll, length: tokenCount, channels: kvWidth, kernel: kConv,
            kernelSize: kernelSize, state: priorK)
        let vConvResult = Inkling2AttentionMath.shortConvolution(
            vAll, length: tokenCount, channels: kvWidth, kernel: vConv,
            kernelSize: kernelSize, state: priorV)
        kAll = kConvResult.values
        vAll = vConvResult.values
        let pastKeys = state?.keys ?? []
        let pastValues = state?.values ?? []
        let pastCount = pastKeys.count / kvWidth
        if sliding == 0, let logScalingAlpha, let logScalingNFloor,
            logScalingNFloor > 0
        {
            for token in 0..<tokenCount {
                let position = Float(pastCount + token + 1)
                let tau =
                    1 + logScalingAlpha
                    * log(
                        max(position / Float(logScalingNFloor), 1))
                for index in 0..<qWidth {
                    qAll[token * qWidth + index] *= tau
                }
                for index in 0..<rWidth {
                    rAll[token * rWidth + index] *= tau
                }
            }
        }
        // L2 clip K per KV head.
        for token in 0..<tokenCount {
            for head in 0..<kvHeads {
                let range =
                    (token * kvWidth + head * headDim)..<(token * kvWidth + (head + 1) * headDim)
                kAll.replaceSubrange(
                    range,
                    with: Inkling2AttentionMath.l2Clip(
                        Array(kAll[range]), threshold: 22.627_417))
            }
        }
        // No-gamma value RMS norm per KV head.
        for token in 0..<tokenCount {
            for head in 0..<kvHeads {
                let range =
                    (token * kvWidth + head * headDim)..<(token * kvWidth + (head + 1) * headDim)
                let values = Array(vAll[range])
                let norm = (values.reduce(Float(0)) { $0 + $1 * $1 } / Float(headDim) + 1e-6)
                    .squareRoot()
                for index in range { vAll[index] = values[index - range.lowerBound] / norm }
            }
        }
        let allK = pastKeys + kAll
        let allV = pastValues + vAll
        var output = [Float](repeating: 0, count: tokenCount * hidden)
        if let gpu {
            let headsOutput = try gpu.attention(
                q: qAll, k: allK, v: allV, r: rAll, gate: gateAll, sink: sinkAll,
                relative: relProjection, tokens: tokenCount, heads: heads,
                kvHeads: kvHeads, dim: headDim, relativeDim: relativeDim,
                extent: extent, sliding: sliding, pastKeys: pastCount)
            for query in 0..<tokenCount {
                let headOutput = Array(
                    headsOutput[(query * qWidth)..<((query + 1) * qWidth)])
                let projected = try project(
                    "attn.wo_ud.weight", out, rows: hidden, columns: qWidth, headOutput)
                output.replaceSubrange(
                    (query * hidden)..<((query + 1) * hidden), with: projected)
            }
            return (
                output,
                State(
                    k: kConvResult.state, v: vConvResult.state,
                    keys: allK, values: allV, position: pastCount + tokenCount)
            )
        }
        for query in 0..<tokenCount {
            var headOutput = [Float](repeating: 0, count: qWidth)
            for head in 0..<heads {
                let qVec = Array(
                    qAll[
                        (query * qWidth + head * headDim)..<(query * qWidth + (head + 1) * headDim)]
                )
                let rVec = Array(
                    rAll[
                        (query * rWidth + head * relativeDim)..<(query * rWidth + (head + 1)
                            * relativeDim)
                    ])
                let absolute = pastCount + query
                let firstKey = sliding > 0 ? max(0, absolute - sliding + 1) : 0
                var logits: [Float] = []
                for key in firstKey...absolute {
                    let distance = absolute - key
                    guard distance < extent else { continue }
                    var score: Float = 0
                    let kvHead = head / (heads / kvHeads)
                    for dimension in 0..<headDim {
                        score +=
                            qVec[dimension] * allK[key * kvWidth + kvHead * headDim + dimension]
                    }
                    score /= Float(headDim).squareRoot()
                    for dimension in 0..<relativeDim {
                        score += rVec[dimension] * relProjection[dimension * extent + distance]
                    }
                    logits.append(score)
                }
                logits.append(sinkAll[query * heads + head])
                let probabilities = Inkling2AttentionMath.softmax(logits)
                var headResult = [Float](repeating: 0, count: headDim)
                for (offset, key) in (firstKey...absolute).enumerated() {
                    let probability = probabilities[offset]
                    let kvHead = head / (heads / kvHeads)
                    for dimension in 0..<headDim {
                        headResult[dimension] +=
                            probability * allV[key * kvWidth + kvHead * headDim + dimension]
                    }
                }
                for dimension in 0..<headDim {
                    let g = 1 / (1 + exp(-gateAll[query * qWidth + head * headDim + dimension]))
                    headOutput[head * headDim + dimension] = headResult[dimension] * g
                }
            }
            let projected = try project(
                "attn.wo_ud.weight", out, rows: hidden, columns: qWidth, headOutput)
            output.replaceSubrange(
                (query * hidden)..<((query + 1) * hidden), with: projected)
        }
        return (
            output,
            State(
                k: kConvResult.state, v: vConvResult.state,
                keys: allK, values: allV, position: pastCount + tokenCount)
        )
    }

    private func project(
        _ name: String, _ weight: [Float], rows: Int, columns: Int, _ input: [Float]
    ) throws -> [Float] {
        if let gpu, let data = gpuWeights[name] {
            return try gpu.gemvBF16(
                weights: data, x: input, rows: rows, columns: columns)
        }
        precondition(weight.count == rows * columns)
        precondition(input.count == columns)
        var result = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            var dot: Float = 0
            let base = row * columns
            for column in 0..<columns {
                dot += weight[base + column] * input[column]
            }
            result[row] = dot
        }
        return result
    }
}

extension Inkling2SlidingAttention: Inkling2AttentionBlock {
    public func forward(
        query: [Float], key: [Float], value: [Float],
        state: inout Inkling2AttentionState?
    ) throws -> [Float] {
        let result = try forward(
            query: query, key: key, value: value, tokenCount: 1, state: state)
        state = result.state
        return result.values
    }
}
