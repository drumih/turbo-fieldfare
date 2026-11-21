import Foundation

/// The sliding-window attention block. It is the 20 local layers' path; the
/// four global layers need the dynamic convolution and are not handled here.
public struct Inkling2SlidingAttention: Sendable {
    public struct State: Sendable {
        public var k: [Float]
        public var v: [Float]
    }

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

    public init(
        tensors: Inkling2DenseTensors, layer: Int,
        heads: Int, kvHeads: Int, headDim: Int,
        relativeDim: Int, extent: Int, sliding: Int,
        hidden: Int, kernelSize: Int
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
        precondition(x.count == tokenCount * hidden)
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
            let input = Array(x[(token * hidden)..<((token + 1) * hidden)])
            let qToken = project(q, rows: qWidth, columns: hidden, input)
            let kToken = project(k, rows: kvWidth, columns: hidden, input)
            let vToken = project(v, rows: kvWidth, columns: hidden, input)
            let rToken = project(r, rows: rWidth, columns: hidden, input)
            let gateToken = project(gate, rows: qWidth, columns: hidden, input)
            let sinkToken = project(sink, rows: heads, columns: hidden, input)
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
        var output = [Float](repeating: 0, count: tokenCount * hidden)
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
                let firstKey = max(0, query - sliding + 1)
                var logits: [Float] = []
                for key in firstKey...query {
                    let distance = query - key
                    guard distance < extent else { continue }
                    var score: Float = 0
                    let kvHead = head % kvHeads
                    for dimension in 0..<headDim {
                        score +=
                            qVec[dimension] * kAll[key * kvWidth + kvHead * headDim + dimension]
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
                for (offset, key) in (firstKey...query).enumerated() {
                    let probability = probabilities[offset]
                    let kvHead = head % kvHeads
                    for dimension in 0..<headDim {
                        headResult[dimension] +=
                            probability * vAll[key * kvWidth + kvHead * headDim + dimension]
                    }
                }
                for dimension in 0..<headDim {
                    let g = 1 / (1 + exp(-gateAll[query * qWidth + head * headDim + dimension]))
                    headOutput[head * headDim + dimension] = headResult[dimension] * g
                }
            }
            let projected = project(out, rows: hidden, columns: qWidth, headOutput)
            output.replaceSubrange(
                (query * hidden)..<((query + 1) * hidden), with: projected)
        }
        return (output, State(k: kConvResult.state, v: vConvResult.state))
    }

    private func project(
        _ weight: [Float], rows: Int, columns: Int, _ input: [Float]
    ) -> [Float] {
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
    public func forward(_ x: [Float]) throws -> [Float] {
        try forward(x, tokenCount: 1).values
    }
}
