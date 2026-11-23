import Foundation

/// The `ArchInfo` of the original Inkling 2 Mini checkpoint, read from its
/// `config.json` `text_config` (TT-121). The constants are the same ones the
/// GGUF carries as `inkling2.*` metadata, under the checkpoint's own names;
/// `Inkling2Arch` turns either set into the manifest's shape.
enum Inkling2CheckpointArch {
    /// The `config.json` `model_type` of an Inkling 2 checkpoint.
    static let modelType = "inkling2_loop"

    /// - Parameter tiedEmbeddings: whether the checkpoint has no separate
    ///   unembedding, which the tensor list says and the config does not.
    static func archInfo(config data: Data, tiedEmbeddings: Bool) throws -> ArchInfo {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            root["model_type"] as? String == modelType,
            let text = root["text_config"] as? [String: Any]
        else {
            throw RepackError.configurationInvalid(
                detail: "config.json is not an Inkling 2 checkpoint (model_type \(modelType))")
        }
        func number(_ key: String, in object: [String: Any] = text) throws -> Double {
            guard let value = object[key] as? NSNumber else {
                throw RepackError.configurationInvalid(
                    detail: "Inkling 2 config.json text_config lacks \(key)")
            }
            return value.doubleValue
        }
        func int(_ key: String, in object: [String: Any] = text) throws -> Int {
            let value = try number(key, in: object)
            guard value == value.rounded(), value >= 0 else {
                throw RepackError.configurationInvalid(
                    detail: "Inkling 2 config.json \(key) is not a count: \(value)")
            }
            return Int(value)
        }
        guard let loop = text["loop_config"] as? [String: Any] else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 config.json text_config lacks loop_config")
        }
        guard loop["is_varloop"] as? Bool != true else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 variable-length loops are not an install this format describes")
        }
        guard try int("n_shared_experts") == 0 else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 config.json declares shared experts; the install has none")
        }
        guard let local = text["local_layer_ids"] as? [NSNumber] else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 config.json text_config lacks local_layer_ids")
        }
        return try Inkling2Arch.archInfo(
            Inkling2Arch.Parameters(
                hiddenSize: try int("hidden_size"),
                intermediateSize: try int("intermediate_size"),
                expertIntermediateSize: try int("intermediate_size"),
                heads: try int("num_attention_heads"),
                kvHeads: try int("num_key_value_heads"),
                headDim: try int("head_dim"),
                paddedVocabulary: try int("vocab_size"),
                vocabulary: try int("unpadded_vocab_size"),
                slidingWindow: try int("sliding_window_size"),
                finalLogitSoftcap: try number("final_logit_softcapping"),
                storedLayers: try int("num_hidden_layers"),
                experts: try int("n_routed_experts"),
                expertsPerToken: try int("num_experts_per_tok"),
                localLayers: local.map(\.intValue),
                tiedEmbeddings: tiedEmbeddings,
                loopStart: try int("core_start", in: loop),
                loopEnd: try int("core_end", in: loop),
                loopPasses: try int("num_passes", in: loop),
                loopGateRank: try number("gate_rank", in: loop),
                attentionResidualBlock: try number("n_layers_per_attn_res_block"),
                relativeLength: try number("d_rel"),
                relativeExtent: try number("rel_extent"),
                relativeClip: try number("rel_l2_clip_threshold"),
                convolutionTaps: try number("sconv_kernel_size"),
                rmsEpsilon: try number("rms_norm_eps"),
                qkClip: try number("qk_l2_clip_threshold"),
                logScaleFloor: try number("log_scaling_n_floor"),
                logScaleAlpha: try number("log_scaling_alpha"),
                expertWeightsScale: try number("route_scale"),
                situBeta: try number("mlp_gate_softcap"),
                situLinearBeta: try number("mlp_up_softcap"),
                contextLength: try number("model_max_length"),
                quantGroupSize: 16))
    }
}
