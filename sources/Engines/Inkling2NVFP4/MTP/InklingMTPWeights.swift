import Foundation
import Metal
import TinyTitanCore
import TinyTitanFormat

/// Configuration of Inkling 2's bundled DSpark MTP draft head (`mtp_config` in
/// `tokenizer/config.json`, TT-129).
package struct InklingMTPConfig: Sendable, Equatable {
    package let hiddenSize: Int
    package let numLayers: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let headDim: Int
    package let intermediateSize: Int
    package let localLayerIDs: Set<Int>
    package let slidingWindow: Int
    package let blockSize: Int
    package let markovRank: Int
    package let targetLayerIDs: [Int]
    package let rmsEpsilon: Float
    package let ropeTheta: Float

    package var numSpeculativeTokens: Int { blockSize - 1 }
    package var qWidth: Int { numHeads * headDim }
    package var kvWidth: Int { numKVHeads * headDim }

    package static func load(from directory: URL) -> InklingMTPConfig? {
        let url = directory.appendingPathComponent("tokenizer/config.json")
        guard let data = try? BoundedMetadataRead.read(fileAt: url, maxBytes: 4 << 20),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let mtp = root["mtp_config"] as? [String: Any]
        else { return nil }
        guard let hidden = mtp["hidden_size"] as? Int,
            let layers = mtp["num_hidden_layers"] as? Int,
            let heads = mtp["num_attention_heads"] as? Int,
            let kvHeads = mtp["num_key_value_heads"] as? Int,
            let headDim = mtp["head_dim"] as? Int,
            let inter = mtp["intermediate_size"] as? Int,
            let blockSize = mtp["block_size"] as? Int,
            let markovRank = mtp["markov_rank"] as? Int,
            let targetIDs = mtp["target_layer_ids"] as? [Int],
            blockSize >= 2, layers >= 1, !targetIDs.isEmpty
        else { return nil }
        let localIDs = Set(mtp["local_layer_ids"] as? [Int] ?? [])
        let sliding = mtp["sliding_window_size"] as? Int ?? 1024
        let eps = Float((mtp["rms_norm_eps"] as? Double) ?? 1e-6)
        let theta = Float((mtp["rope_theta"] as? Double) ?? 1_000_000.0)
        return InklingMTPConfig(
            hiddenSize: hidden, numLayers: layers, numHeads: heads, numKVHeads: kvHeads,
            headDim: headDim, intermediateSize: inter, localLayerIDs: localIDs,
            slidingWindow: sliding, blockSize: blockSize, markovRank: markovRank,
            targetLayerIDs: targetIDs, rmsEpsilon: eps, ropeTheta: theta)
    }
}

/// One dense transformer layer of the DSpark draft backbone.
struct InklingMTPLayerWeights {
    let index: Int
    let isLocal: Bool
    let dflareNorm: InklingQ4Tensor
    let inputNorm: InklingQ4Tensor
    let postAttnNorm: InklingQ4Tensor
    let qProj: InklingQ4Tensor
    let kProj: InklingQ4Tensor
    let vProj: InklingQ4Tensor
    let oProj: InklingQ4Tensor
    let kProjCtx: InklingQ4Tensor
    let vProjCtx: InklingQ4Tensor
    let qNorm: InklingQ4Tensor
    let kNorm: InklingQ4Tensor
    let kNormCtx: InklingQ4Tensor
    let gateProj: InklingQ4Tensor
    let upProj: InklingQ4Tensor
    let downProj: InklingQ4Tensor
}

/// All 54 `mtp.*` BF16 tensors bound directly from the already-mapped
/// `model_weights.bin` buffer (zero extra file I/O, TT-129).
final class InklingMTPWeights {
    let config: InklingMTPConfig
    let layers: [InklingMTPLayerWeights]
    let finalNorm: InklingQ4Tensor
    let maskEmbed: InklingQ4Tensor
    /// Row-wise softmax of `mtp.fuse_weight` (`[numLayers + 1, targetLayerIDs.count]`).
    let fuseAlpha: MTLBuffer
    let markovDflareNorm: InklingQ4Tensor
    let markovW1: InklingQ4Tensor
    let rnnProjWeight: InklingQ4Tensor
    let rnnProjBias: InklingQ4Tensor
    let hiddenOutWeight: InklingQ4Tensor
    let hiddenOutBias: InklingQ4Tensor

    init?(
        install: InklingQ4Install, weights: InklingQ4Weights, device: MTLDevice
    ) throws {
        guard let config = InklingMTPConfig.load(from: install.directory) else { return nil }
        let entries = install.checkpointResident.entries
        guard entries["mtp.fuse_weight"] != nil else { return nil }
        self.config = config
        let buf = weights.residentBuffer
        let length = weights.residentLength

        func tensor(_ name: String) throws -> InklingQ4Tensor {
            guard let entry = entries[name] else {
                throw InklingQ4Manifest.Error.missingTensor(name)
            }
            guard entry.fileOffset + entry.sizeBytes <= UInt64(length) else {
                throw InklingQ4Manifest.Error.invalidGeometry("\(name) lies past resident file")
            }
            let dims = [entry.shape.0, entry.shape.1, entry.shape.2, entry.shape.3]
                .filter { $0 > 0 }.map(Int.init)
            let columns = dims.last ?? 1
            let rows = dims.dropLast().reduce(1, *)
            return InklingQ4Tensor(
                operand: .init(buf, offset: Int(entry.fileOffset)),
                type: try InklingQ4Manifest.type(indexCode: entry.dtype, tensor: name),
                rows: rows, columns: columns, byteCount: Int(entry.sizeBytes))
        }

        // Compute row-wise softmax of `mtp.fuse_weight` [numLayers + 1, taps] in FP32.
        let fuseEntry = try tensor("mtp.fuse_weight")
        let rowCount = config.numLayers + 1
        let tapCount = config.targetLayerIDs.count
        guard fuseEntry.rows == rowCount, fuseEntry.columns == tapCount else {
            throw InklingQ4Manifest.Error.invalidGeometry(
                "mtp.fuse_weight shape (\(fuseEntry.rows), \(fuseEntry.columns))")
        }
        let fusePtr = (buf.contents() + fuseEntry.operand.offset)
            .bindMemory(to: UInt16.self, capacity: rowCount * tapCount)
        var alphaHost = [Float](repeating: 0, count: rowCount * tapCount)
        for r in 0..<rowCount {
            var vals = (0..<tapCount).map { c -> Float in
                Float(bitPattern: UInt32(fusePtr[r * tapCount + c]) << 16)
            }
            let peak = vals.max() ?? 0
            vals = vals.map { exp($0 - peak) }
            let sum = vals.reduce(0, +)
            for c in 0..<tapCount { alphaHost[r * tapCount + c] = vals[c] / sum }
        }
        guard
            let alphaBuf = device.makeBuffer(
                bytes: &alphaHost, length: max(alphaHost.count * 4, 16),
                options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("Inkling MTP fuseAlpha") }
        self.fuseAlpha = alphaBuf

        var builtLayers: [InklingMTPLayerWeights] = []
        for i in 0..<config.numLayers {
            let p = "mtp.layers.\(i)."
            builtLayers.append(
                InklingMTPLayerWeights(
                    index: i,
                    isLocal: config.localLayerIDs.contains(i),
                    dflareNorm: try tensor("\(p)dflare_norm.weight"),
                    inputNorm: try tensor("\(p)input_layernorm.weight"),
                    postAttnNorm: try tensor("\(p)post_attention_layernorm.weight"),
                    qProj: try tensor("\(p)self_attn.q_proj.weight"),
                    kProj: try tensor("\(p)self_attn.k_proj.weight"),
                    vProj: try tensor("\(p)self_attn.v_proj.weight"),
                    oProj: try tensor("\(p)self_attn.o_proj.weight"),
                    kProjCtx: try tensor("\(p)self_attn.k_proj_ctx.weight"),
                    vProjCtx: try tensor("\(p)self_attn.v_proj_ctx.weight"),
                    qNorm: try tensor("\(p)self_attn.q_norm.weight"),
                    kNorm: try tensor("\(p)self_attn.k_norm.weight"),
                    kNormCtx: try tensor("\(p)self_attn.k_norm_ctx.weight"),
                    gateProj: try tensor("\(p)mlp.gate_proj.weight"),
                    upProj: try tensor("\(p)mlp.up_proj.weight"),
                    downProj: try tensor("\(p)mlp.down_proj.weight")))
        }
        self.layers = builtLayers
        self.finalNorm = try tensor("mtp.norm.weight")
        self.maskEmbed = try tensor("mtp.mask_embed")
        self.markovDflareNorm = try tensor("mtp.markov_head.dflare_norm.weight")
        self.markovW1 = try tensor("mtp.markov_head.markov_w1.weight")
        self.rnnProjWeight = try tensor("mtp.markov_head.rnn_proj.weight")
        self.rnnProjBias = try tensor("mtp.markov_head.rnn_proj.bias")
        self.hiddenOutWeight = try tensor("mtp.markov_head.hidden_out.weight")
        self.hiddenOutBias = try tensor("mtp.markov_head.hidden_out.bias")
    }
}
