import Foundation
import TinyTitanCore

/// The text-only architecture contract of Inkling 2 Mini, read from the
/// checkpoint's `config.json`. Vision and MTP are recorded but never entered by
/// the text engine.
public struct Inkling2Config: Sendable, Equatable {
    public struct Loop: Sendable, Equatable {
        public let coreStart: Int
        public let coreEnd: Int
        public let passes: Int
    }

    public struct Text: Sendable, Equatable {
        public let hiddenSize: Int
        public let layerCount: Int
        public let vocabularySize: Int
        public let attentionHeads: Int
        public let keyValueHeads: Int
        public let headDim: Int
        public let experts: Int
        public let expertsPerToken: Int
        public let intermediateSize: Int
        public let contextTokens: Int
        public let slidingWindow: Int
        public let localLayers: [Int]
        public let relativeExtent: Int
        public let relativeDimension: Int
        public let convolutionKernel: Int
        public let attentionResidualBlock: Int
        public let loop: Loop
    }

    public struct Vision: Sendable, Equatable {
        public let layerCount: Int
        public let hiddenSize: Int
        public let patchSize: Int
    }

    public struct MTP: Sendable, Equatable {
        public let layerCount: Int
        public let hiddenSize: Int
        public let blockSize: Int
    }

    public let text: Text
    public let vision: Vision?
    public let mtp: MTP?

    public static func load(directory: URL) throws -> Inkling2Config {
        let data = try BoundedMetadataRead.read(
            fileAt: directory.appendingPathComponent("config.json"),
            maxBytes: 4 * 1024 * 1024)
        return try parse(data)
    }

    public static func parse(_ data: Data) throws -> Inkling2Config {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = root["text_config"] as? [String: Any]
        else {
            throw Error.invalid("config.json has no text_config")
        }
        guard let loopObject = text["loop_config"] as? [String: Any],
            let coreStart = loopObject["core_start"] as? Int,
            let coreEnd = loopObject["core_end"] as? Int,
            let passes = loopObject["num_passes"] as? Int
        else {
            throw Error.missing("loop_config")
        }
        let localLayers = text["local_layer_ids"] as? [Int] ?? []
        let textConfig = Text(
            hiddenSize: try require(text, "hidden_size"),
            layerCount: try require(text, "num_hidden_layers"),
            vocabularySize: try require(text, "vocab_size"),
            attentionHeads: try require(text, "num_attention_heads"),
            keyValueHeads: try require(text, "num_key_value_heads"),
            headDim: try require(text, "head_dim"),
            experts: try require(text, "n_routed_experts"),
            expertsPerToken: try require(text, "num_experts_per_tok"),
            intermediateSize: try require(text, "intermediate_size"),
            contextTokens: try require(text, "model_max_length"),
            slidingWindow: try require(text, "sliding_window_size"),
            localLayers: localLayers,
            relativeExtent: try require(text, "rel_extent"),
            relativeDimension: try require(text, "d_rel"),
            convolutionKernel: try require(text, "sconv_kernel_size"),
            attentionResidualBlock: try require(text, "n_layers_per_attn_res_block"),
            loop: Loop(coreStart: coreStart, coreEnd: coreEnd, passes: passes))
        guard textConfig.layerCount == 24,
            textConfig.experts == 128,
            textConfig.expertsPerToken == 8,
            textConfig.loop.coreStart == 6,
            textConfig.loop.coreEnd == 18,
            textConfig.loop.passes == 3,
            localLayers.count == 20
        else {
            throw Error.invalid("text shape does not match Inkling 2 Mini")
        }
        var vision: Vision?
        if let object = root["vision_config"] as? [String: Any] {
            vision = Vision(
                layerCount: try require(object, "num_hidden_layers"),
                hiddenSize: try require(object, "hidden_size"),
                patchSize: try require(object, "native_patch_size"))
        }
        var mtp: MTP?
        if let object = root["mtp_config"] as? [String: Any] {
            mtp = MTP(
                layerCount: try require(object, "num_hidden_layers"),
                hiddenSize: try require(object, "hidden_size"),
                blockSize: try require(object, "block_size"))
        }
        return Inkling2Config(text: textConfig, vision: vision, mtp: mtp)
    }

    private static func require(_ object: [String: Any], _ key: String) throws -> Int {
        guard let value = object[key] as? Int else { throw Error.missing(key) }
        return value
    }

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case missing(String)
        case invalid(String)

        public var description: String {
            switch self {
            case .missing(let detail): "missing Inkling config \(detail)"
            case .invalid(let detail): "invalid Inkling config: \(detail)"
            }
        }
    }
}
