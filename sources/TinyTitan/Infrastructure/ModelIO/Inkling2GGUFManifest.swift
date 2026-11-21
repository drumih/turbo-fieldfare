import Foundation

/// Geometry and tensor directory of an Inkling2 GGUF, read and validated
/// without touching weights. llama.cpp's `inkling2` tensor names
/// (`src/models/inkling2.cpp`, `load_arch_tensors`) are the contract.
///
/// The file stores `storedLayers` (24) layers but declares `block_count` as the
/// *unrolled* depth (48): the looped core `loopStart..<loopEnd` (6..<18) runs
/// `loopPasses` (3) times, and the per-layer arrays (`head_count`,
/// `sliding_window_pattern`, ...) are indexed by unrolled position.
///
/// Quantization is per tensor, not per file: Unsloth's `UD-Q2_K_XL` stores the
/// routed experts as IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S (gate+up) and Q2_0 /
/// NVFP4 / MXFP4 (down), and `UD-Q4_K_XL` as NVFP4 and Q8_0 — neither is
/// mostly Q2_K or Q4_K. `Layer` records each layer's types so a decoder picks
/// kernels per tensor.
struct Inkling2GGUFManifest: Sendable {
    enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case wrongArchitecture(String?)
        case missingTensor(String)
        case missingMetadata(String)
        case invalidGeometry(String)
        case wrongShape(String, expected: [UInt64], got: [UInt64])

        var description: String {
            switch self {
            case .wrongArchitecture(let arch):
                return "GGUF architecture is \(arch ?? "missing"), expected inkling2"
            case .missingTensor(let name):
                return "Inkling2 GGUF is missing tensor \(name)"
            case .missingMetadata(let key):
                return "Inkling2 GGUF is missing metadata inkling2.\(key)"
            case .invalidGeometry(let detail):
                return "Inkling2 GGUF geometry is invalid: \(detail)"
            case .wrongShape(let name, let expected, let got):
                return "Inkling2 tensor \(name) has shape \(got), expected \(expected)"
            }
        }
    }

    /// One stored layer: its attention kind and the quantization of each
    /// tensor the decoder dispatches a kernel for.
    struct Layer: Sendable, Equatable {
        let index: Int
        let isSliding: Bool
        let heads: Int
        let kvHeads: Int
        let expertFeedForward: Int
        let gateUpType: GGUFType
        let downType: GGUFType
        /// Per-expert output multipliers (`ffn_*_exps.scale`), present where
        /// the source checkpoint carried an NVFP4 global scale.
        let hasExpertScales: Bool
        let projectionTypes: [String: GGUFType]
    }

    let model: GGUFModel
    let unrolledLayers: Int
    let storedLayers: Int
    let hidden: Int
    let headDimension: Int
    let expertCount: Int
    let expertsPerToken: Int
    let loopStart: Int
    let loopEnd: Int
    let loopPasses: Int
    let loopGateRank: Int
    let attentionResidualBlock: Int
    let slidingWindow: Int
    let relativeLength: Int
    let relativeExtent: Int
    let convolutionTaps: Int
    let paddedVocabulary: Int
    let vocabulary: Int
    let rmsEpsilon: Float
    let qkClip: Float
    let relativeClip: Float
    let logScaleFloor: Int
    let logScaleAlpha: Float
    let expertWeightsScale: Float
    let situBeta: Float
    let situLinearBeta: Float
    let finalLogitSoftcap: Float
    let layers: [Layer]

    static let projections = [
        "attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "attn_rel",
    ]

    init(url: URL) throws {
        let model = try GGUFModelReader.read(modelAt: url)
        guard model.architecture == "inkling2" else {
            throw Error.wrongArchitecture(model.architecture)
        }
        func int(_ key: String) throws -> Int {
            guard let value = model.architectureValue(key)?.integer else {
                throw Error.missingMetadata(key)
            }
            return Int(value)
        }
        func float(_ key: String) throws -> Float {
            guard let value = model.architectureValue(key)?.double else {
                throw Error.missingMetadata(key)
            }
            return Float(value)
        }
        func perLayer(_ key: String, count: Int) throws -> [GGUFValue] {
            if let array = model.architectureValue(key)?.array, array.count == count {
                return array
            }
            if let one = model.architectureValue(key) { return Array(repeating: one, count: count) }
            throw Error.missingMetadata(key)
        }

        self.model = model
        unrolledLayers = try int("block_count")
        hidden = try int("embedding_length")
        headDimension = try int("attention.key_length")
        expertCount = try int("expert_count")
        expertsPerToken = try int("expert_used_count")
        loopStart = try int("loop.core_start")
        loopEnd = try int("loop.core_end")
        loopPasses = try int("num_loops")
        loopGateRank = try int("loop.gate_rank")
        attentionResidualBlock = try int("attn_res.block_size")
        slidingWindow = try int("attention.sliding_window")
        relativeLength = try int("attention.relative_length")
        relativeExtent = try int("attention.relative_extent")
        convolutionTaps = try int("shortconv.l_cache")
        paddedVocabulary = try int("vocab_size")
        vocabulary = try int("vocab_size_unpadded")
        logScaleFloor = try int("attention.log_scale_floor")
        rmsEpsilon = try float("attention.layer_norm_rms_epsilon")
        qkClip = try float("attention.qk_clip")
        relativeClip = try float("attention.relative_clip")
        logScaleAlpha = try float("attention.log_scale_alpha")
        expertWeightsScale = try float("expert_weights_scale")
        situBeta = try float("activation.situ_beta")
        situLinearBeta = try float("activation.situ_linear_beta")
        finalLogitSoftcap = try float("final_logit_softcapping")

        let extra = (loopPasses - 1) * (loopEnd - loopStart)
        storedLayers = unrolledLayers - extra
        guard loopPasses >= 1, loopStart > 0, loopStart < loopEnd, loopEnd <= storedLayers,
            hidden % 256 == 0, vocabulary <= paddedVocabulary, convolutionTaps >= 2
        else {
            throw Error.invalidGeometry(
                "loop \(loopStart)..<\(loopEnd) x\(loopPasses), \(unrolledLayers) unrolled layers")
        }
        let heads = try perLayer("attention.head_count", count: unrolledLayers)
        let kvHeads = try perLayer("attention.head_count_kv", count: unrolledLayers)
        let sliding = try perLayer("attention.sliding_window_pattern", count: unrolledLayers)
        let feedForward = try perLayer("expert_feed_forward_length", count: unrolledLayers)

        var layers: [Layer] = []
        for index in 0..<storedLayers {
            // Per-layer arrays are indexed by unrolled position; a layer after
            // the looped core first runs `core * (passes - 1)` positions later.
            let at = index < loopEnd ? index : index + extra
            guard let headCount = heads[at].integer, let kvCount = kvHeads[at].integer,
                let isSliding = sliding[at].boolean, let ff = feedForward[at].integer
            else { throw Error.invalidGeometry("layer \(index) metadata is not numeric") }
            layers.append(
                try Self.layer(
                    model, index: index, isSliding: isSliding, heads: Int(headCount),
                    kvHeads: Int(kvCount), feedForward: Int(ff), hidden: hidden,
                    headDimension: headDimension, experts: expertCount))
        }
        self.layers = layers
        try validateShared()
    }

    func tensor(_ name: String) throws -> GGUFTensor {
        guard let location = model.tensor(named: name) else { throw Error.missingTensor(name) }
        return location.tensor
    }

    /// The stored layer an unrolled position runs (llama.cpp `loop_layer`).
    func storedLayer(forUnrolled position: Int) -> Int {
        Self.storedLayer(
            forUnrolled: position, loopStart: loopStart, loopEnd: loopEnd, passes: loopPasses)
    }

    static func storedLayer(
        forUnrolled position: Int, loopStart: Int, loopEnd: Int, passes: Int
    ) -> Int {
        guard position >= loopStart else { return position }
        let core = loopEnd - loopStart
        let looped = loopStart + core * passes
        if position < looped { return loopStart + (position - loopStart) % core }
        return position - core * (passes - 1)
    }

    private static func layer(
        _ model: GGUFModel, index: Int, isSliding: Bool, heads: Int, kvHeads: Int,
        feedForward: Int, hidden: Int, headDimension: Int, experts: Int
    ) throws -> Layer {
        let prefix = "blk.\(index)."
        func tensor(_ suffix: String) throws -> GGUFTensor {
            guard let location = model.tensor(named: prefix + suffix) else {
                throw Error.missingTensor(prefix + suffix)
            }
            return location.tensor
        }
        func shape(_ suffix: String, _ expected: [Int]) throws -> GGUFTensor {
            let found = try tensor(suffix)
            let want = expected.map(UInt64.init)
            guard found.dimensions == want else {
                throw Error.wrongShape(prefix + suffix, expected: want, got: found.dimensions)
            }
            return found
        }
        var types: [String: GGUFType] = [:]
        let attention = headDimension * heads
        let kv = headDimension * kvHeads
        types["attn_q"] = try shape("attn_q.weight", [hidden, attention]).type
        types["attn_k"] = try shape("attn_k.weight", [hidden, kv]).type
        types["attn_v"] = try shape("attn_v.weight", [hidden, kv]).type
        types["attn_output"] = try shape("attn_output.weight", [attention, hidden]).type
        types["attn_gate"] = try shape("attn_gate.weight", [hidden, attention]).type
        types["attn_rel"] = try tensor("attn_rel.weight").type
        for suffix in [
            "attn_q_clip.weight", "attn_sink_proj.weight", "attn_rel_bias.weight",
            "attn_rel_clip.weight", "attn_k_conv.weight", "attn_v_conv.weight",
            "attn_post_conv.weight", "ffn_post_conv.weight", "ffn_res_query.weight",
            "ffn_norm.weight", "ffn_gate_inp.weight",
        ] {
            _ = try tensor(suffix)
        }
        if isSliding {
            _ = try tensor("attn_norm.weight")
        } else {
            for suffix in [
                "attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight",
                "attn_k_conv_proj.weight", "attn_v_conv_proj.weight",
            ] {
                _ = try tensor(suffix)
            }
        }
        if index > 0 { _ = try tensor("attn_res_query.weight") }
        let gateUp = try shape("ffn_gate_up_exps.weight", [hidden, 2 * feedForward, experts])
        let down = try shape("ffn_down_exps.weight", [feedForward, hidden, experts])
        let scaled = model.tensor(named: prefix + "ffn_gate_up_exps.scale") != nil
        guard scaled == (model.tensor(named: prefix + "ffn_down_exps.scale") != nil) else {
            throw Error.missingTensor(prefix + "ffn_*_exps.scale (only one of the pair)")
        }
        return Layer(
            index: index, isSliding: isSliding, heads: heads, kvHeads: kvHeads,
            expertFeedForward: feedForward, gateUpType: gateUp.type, downType: down.type,
            hasExpertScales: scaled, projectionTypes: types)
    }

    private func validateShared() throws {
        func shape(_ name: String, _ expected: [Int]) throws {
            let found = try tensor(name)
            let want = expected.map(UInt64.init)
            guard found.dimensions == want else {
                throw Error.wrongShape(name, expected: want, got: found.dimensions)
            }
        }
        try shape("token_embd.weight", [hidden, paddedVocabulary])
        try shape("output.weight", [hidden, paddedVocabulary])
        try shape("token_embd_norm.weight", [hidden])
        try shape("output_norm.weight", [hidden])
        try shape("output_res_query.weight", [hidden])
        if loopPasses > 1 {
            let queries = 2 * (loopEnd - loopStart)
            try shape("loop_res_query.weight", [hidden, queries])
            try shape("loop_kv_res_query.weight", [hidden, queries])
            for layer in loopStart..<loopEnd {
                for module in ["attn", "k", "v", "mlp"] {
                    try shape(
                        "blk.\(layer).loop_\(module)_down.weight", [2 * hidden, loopGateRank])
                    try shape("blk.\(layer).loop_\(module)_gain.weight", [loopGateRank, hidden])
                    try shape("blk.\(layer).loop_\(module)_forget.weight", [loopGateRank, hidden])
                }
            }
        }
    }
}
