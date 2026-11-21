import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2DecoderLayerTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func layerZeroInternals() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let tensors = try Inkling2DenseTensors(directory: directory)
        let x = Inkling2Ops.rmsNorm(
            try tensors.embeddingRow(200_028), weight: try tensors.embeddingNorm())
        let attention = try Inkling2SlidingAttention(
            tensors: tensors, layer: 0, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 1024,
            hidden: 2304, kernelSize: 4)
        let attnNorm = try tensors.layerMatrix(0, "attn_norm.weight")
        let attnInput = Inkling2Ops.rmsNorm(x, weight: attnNorm)
        func project(_ weight: [Float], rows: Int, columns: Int, _ input: [Float]) -> [Float] {
            (0..<rows).map { row in
                var dot: Float = 0
                for column in 0..<columns { dot += weight[row * columns + column] * input[column] }
                return dot
            }
        }
        let qDebug = project(
            try tensors.layerMatrix(0, "attn.wq_du.weight"), rows: 3072,
            columns: 2304, attnInput)
        let kDebug = project(
            try tensors.layerMatrix(0, "attn.wk_dv.weight"), rows: 1024,
            columns: 2304, attnInput)
        let vDebug = project(
            try tensors.layerMatrix(0, "attn.wv_dv.weight"), rows: 1024,
            columns: 2304, attnInput)
        print("qraw first=\(qDebug.prefix(5))")
        print("kraw first=\(kDebug.prefix(5))")
        print("vraw first=\(vDebug.prefix(5))")
        let kConvWeights = try tensors.layerMatrix(0, "attn.k_sconv.weight")
        let kDebugConv = Inkling2AttentionMath.shortConvolution(
            kDebug, length: 1, channels: 1024, kernel: kConvWeights,
            kernelSize: 4, state: nil
        ).values
        print("kconv first=\(kDebugConv.prefix(5))")
        let rDebug = project(
            try tensors.layerMatrix(0, "attn.wr_du.weight"), rows: 384,
            columns: 2304, attnInput)
        let relScale = try tensors.layerMatrix(0, "attn.rel_l2_clip.scale")
        let rClipDebug = Inkling2AttentionMath.l2Clip(
            Array(rDebug.prefix(16)), threshold: 8,
            scale: Array(relScale.prefix(16)))
        let relProj = try tensors.layerMatrix(0, "attn.rel_logits_proj.proj")
        var relBias: Float = 0
        for dimension in 0..<16 { relBias += rClipDebug[dimension] * relProj[dimension * 1024] }
        print("rclip first=\(rClipDebug.prefix(5)) relbias=\(relBias)")
        let qScale = try tensors.layerMatrix(0, "attn.q_l2_clip.scale")
        let qClipDebug = Inkling2AttentionMath.l2Clip(
            Array(qDebug.prefix(128)), threshold: 22.627_417,
            scale: Array(qScale.prefix(128)))
        print("qclip first=\(qClipDebug.prefix(5))")
        let sinkDebug = project(
            try tensors.layerMatrix(0, "attn.attn_sink_proj.weight"), rows: 24,
            columns: 2304, attnInput)
        let gateDebug = project(
            try tensors.layerMatrix(0, "attn.gate_attn.weight"), rows: 3072,
            columns: 2304, attnInput)
        print("sink first=\(sinkDebug.prefix(5))")
        let vConvWeights = try tensors.layerMatrix(0, "attn.v_sconv.weight")
        let vDebugConv = Inkling2AttentionMath.shortConvolution(
            vDebug, length: 1, channels: 1024, kernel: vConvWeights,
            kernelSize: 4, state: nil
        ).values
        let vNormRaw = Array(vDebugConv.prefix(128))
        let vNorm = (vNormRaw.reduce(Float(0)) { $0 + $1 * $1 } / 128 + 1e-6).squareRoot()
        let vHead = vNormRaw.map { $0 / vNorm }
        var score: Float = 0
        for d in 0..<128 { score += qClipDebug[d] * kDebugConv[d] }
        score /= sqrtf(128)
        let logits: [Float] = [score + relBias, sinkDebug[0]]
        let probs = Inkling2AttentionMath.softmax(logits)
        print("score=\(score) probs=\(probs) vnorm0=\(vHead[0])")
        let gate0 = 1 / (1 + exp(-gateDebug[0]))
        print("head0out=\(probs[0] * vHead[0] * gate0)")
        print("gate first=\(gateDebug.prefix(5))")
        let attn = try attention.forward(attnInput, tokenCount: 1).values
        print("attn first=\(attn.prefix(5))")
        let attnConvWeights = try tensors.layerMatrix(0, "attn_sconv.weight")
        let attnConv = Inkling2AttentionMath.shortConvolution(
            attn, length: 1, channels: 2304, kernel: attnConvWeights,
            kernelSize: 4, state: nil
        ).values
        print("attnConv first=\(attnConv.prefix(5))")
        let mlpInput = try Inkling2AttentionResidual(
            tensors: tensors, tensorName: "model.llm.attn_res.mlp_queries.0",
            queries: 1, hidden: 2304)
        let hidden = mlpInput.forward([x], partial: attnConv)
        print("hidden first=\(hidden.prefix(5))")
        let mlpNorm = try tensors.layerMatrix(0, "mlp_norm.weight")
        let normed = Inkling2Ops.rmsNorm(hidden, weight: mlpNorm)
        print("normed first=\(normed.prefix(5))")
        let store = try Inkling2ExpertStore(directory: directory)
        let mlp = try Inkling2MLP(
            tensors: tensors, store: store, layer: 0, gpu: try Inkling2GPUExpert())
        let mlpOut = try mlp.forward(normed)
        print("mlp first=\(mlpOut.prefix(5))")
        let mlpConvWeights = try tensors.layerMatrix(0, "mlp_sconv.weight")
        let mlpConv = Inkling2AttentionMath.shortConvolution(
            mlpOut, length: 1, channels: 2304, kernel: mlpConvWeights,
            kernelSize: 4, state: nil
        ).values
        let final = zip(attnConv, mlpConv).map { $0 + $1 }
        print("final first=\(final.prefix(5))")
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func layerSixChangesItsInput() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let tensors = try Inkling2DenseTensors(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let attention = try Inkling2SlidingAttention(
            tensors: tensors, layer: 6, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 1024,
            hidden: 2304, kernelSize: 4)
        let layer = try Inkling2DecoderLayer(
            tensors: tensors, store: store, layer: 6, hidden: 2304,
            kernelSize: 4, isBlockBoundary: true, isSliding: true,
            attention: attention, gpu: try Inkling2GPUExpert())
        let x = try tensors.embeddingRow(0)
        var blocks: [[Float]] = [x]
        var cache = Inkling2DecoderCache()
        let output = try layer.forward(x, blocks: &blocks, cache: &cache)
        print("layer6 input=\(x.prefix(5)) output=\(output.prefix(5))")
        #expect(output != x)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realGlobalBlockFiveRuns() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let tensors = try Inkling2DenseTensors(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let attention = try Inkling2GlobalAttention(
            tensors: tensors, layer: 5, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 0,
            hidden: 2304, kernelSize: 4)
        let layer = try Inkling2DecoderLayer(
            tensors: tensors, store: store, layer: 5, hidden: 2304,
            kernelSize: 4, isBlockBoundary: false,
            isSliding: false, attention: attention)
        var blocks: [[Float]] = [try tensors.embeddingRow(0)]
        var cache = Inkling2DecoderCache()
        let output = try layer.forward(
            tensors.embeddingRow(0), blocks: &blocks, cache: &cache)
        #expect(output.count == 2304)
        #expect(output.allSatisfy { $0.isFinite })
        #expect(blocks.count == 1)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realLayerZeroBlockRuns() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let tensors = try Inkling2DenseTensors(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let attention = try Inkling2SlidingAttention(
            tensors: tensors, layer: 0, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 1024,
            hidden: 2304, kernelSize: 4)
        let layer = try Inkling2DecoderLayer(
            tensors: tensors, store: store, layer: 0, hidden: 2304,
            kernelSize: 4, isBlockBoundary: true,
            isSliding: true, attention: attention)
        let embedNorm = try tensors.embeddingNorm()
        let x = Inkling2Ops.rmsNorm(
            try tensors.embeddingRow(200_028), weight: embedNorm)
        print("embed first=\(x.prefix(5))")
        var blocks: [[Float]] = []
        var cache = Inkling2DecoderCache()
        let output = try layer.forward(x, blocks: &blocks, cache: &cache)
        print("layer0 first=\(output.prefix(5))")
        #expect(output.count == 2304)
        #expect(output.allSatisfy { $0.isFinite })
        #expect(blocks.count == 1)
    }
}
