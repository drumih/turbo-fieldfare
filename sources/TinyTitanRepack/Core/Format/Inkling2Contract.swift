import Foundation
import TinyTitanCore

/// The Inkling 2 Mini checkpoint contract, read before any weight is touched.
///
/// The checkpoint is one file with mixed precisions: routed experts in layers
/// 0-19 are NVFP4, layers 20-23 are MXFP8, and the rest is BF16/FP32. This type
/// proves that split from `config.json` and `hf_quant_config.json`; it does not
/// decode weights and it makes no runtime-support claim.
public struct Inkling2Contract: Sendable, Equatable {
    public enum Precision: String, Sendable, Equatable {
        case nvfp4
        case mxfp8
        case bf16
        case fp32
    }

    public struct Layer: Sendable, Equatable {
        public let layer: Int
        public let w13: Precision
        public let w2: Precision
    }

    public let layers: Int
    public let routedExperts: Int
    public let expertsPerToken: Int
    public let contextTokens: Int
    public let perLayer: [Layer]
    public let groupCounts: [String: Int]

    public static func load(directory: URL) throws -> Inkling2Contract {
        let config = try BoundedMetadataRead.read(
            fileAt: directory.appendingPathComponent("config.json"),
            maxBytes: 4 * 1024 * 1024)
        let quant = try BoundedMetadataRead.read(
            fileAt: directory.appendingPathComponent("hf_quant_config.json"),
            maxBytes: 16 * 1024 * 1024)
        return try parse(config: config, quant: quant)
    }

    public static func parse(config: Data, quant: Data) throws -> Inkling2Contract {
        let configObject = try object(config, name: "config.json")
        let quantObject = try object(quant, name: "hf_quant_config.json")
        guard let text = configObject["text_config"] as? [String: Any] else {
            throw Error.missing("text_config")
        }
        guard let layers = text["num_hidden_layers"] as? Int, layers == 24 else {
            throw Error.invalid(
                "expected 24 text layers, got \(text["num_hidden_layers"] ?? "nil")")
        }
        guard
            let quantization = configObject["quantization_config"] as? [String: Any],
            quantization["quant_method"] as? String == "modelopt"
        else {
            throw Error.invalid("config.json does not declare ModelOpt quantization")
        }
        guard let model = quantObject["model"] as? [String: Any],
            let groups = model["parameter_groups"] as? [[String: Any]],
            !groups.isEmpty
        else {
            throw Error.missing("model.parameter_groups")
        }

        var formatByTensor: [String: Precision] = [:]
        var groupCounts: [String: Int] = [:]
        for group in groups {
            let precision = try precision(of: group)
            guard let tensors = group["tensors"] as? [String], !tensors.isEmpty else {
                throw Error.invalid("a \(precision.rawValue) group names no tensors")
            }
            groupCounts[precision.rawValue, default: 0] += tensors.count
            for tensor in tensors {
                if formatByTensor[tensor] != nil {
                    throw Error.invalid("\(tensor) is claimed by two groups")
                }
                formatByTensor[tensor] = precision
            }
        }

        var perLayer: [Layer] = []
        perLayer.reserveCapacity(layers)
        for layer in 0..<layers {
            let expected: Precision = layer < 20 ? .nvfp4 : .mxfp8
            func tensor(_ suffix: String) throws -> Precision {
                let name = "model.llm.layers.\(layer).mlp.experts.\(suffix)"
                guard let found = formatByTensor[name] else {
                    throw Error.missing(name)
                }
                guard found == expected else {
                    throw Error.invalid(
                        "layer \(layer) \(suffix) is \(found.rawValue), expected \(expected.rawValue)"
                    )
                }
                return found
            }
            perLayer.append(
                Layer(layer: layer, w13: try tensor("w13_weight"), w2: try tensor("w2_weight")))
        }

        return Inkling2Contract(
            layers: layers,
            routedExperts: text["n_routed_experts"] as? Int ?? 0,
            expertsPerToken: text["num_experts_per_tok"] as? Int ?? 0,
            contextTokens: text["model_max_length"] as? Int ?? 0,
            perLayer: perLayer,
            groupCounts: groupCounts)
    }

    private static func precision(of group: [String: Any]) throws -> Precision {
        switch group["compute_dtype"] as? String {
        case "nvfp4": return .nvfp4
        case "mxfp8": return .mxfp8
        case "bfloat16": return .bf16
        case "float32": return .fp32
        case let other:
            throw Error.invalid("unknown compute_dtype \(other ?? "nil")")
        }
    }

    private static func object(_ data: Data, name: String) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Error.invalid("\(name) is not a JSON object")
        }
        return object
    }

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case missing(String)
        case invalid(String)

        public var description: String {
            switch self {
            case .missing(let detail): "missing \(detail)"
            case .invalid(let detail): "invalid Inkling 2 contract: \(detail)"
            }
        }
    }
}
