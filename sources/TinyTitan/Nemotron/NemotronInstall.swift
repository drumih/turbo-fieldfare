import Foundation
import TinyTitanCore

/// One tensor in a Nemotron install's resident payload.
public struct NemotronTensorRef: Sendable {
    public let name: String
    /// 0 = affine-packed U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8
    /// Logical (dequantized) shape, trailing zeros dropped.
    public let shape: [Int]
    public let fileOffset: Int
    public let sizeBytes: Int
    public let scaleOffset: Int
    public let scaleSize: Int
    public let biasOffset: Int
    public let biasSize: Int
    /// 4 or 8 for a packed tensor, nil for BF16/FP32.
    public let bits: Int?

    public var rows: Int { shape.first ?? 0 }
    public var columns: Int { shape.count > 1 ? shape[1] : 1 }
}

/// One role inside a packed expert blob, relative to the expert's start.
public struct NemotronExpertTensor: Sendable {
    public let offset: Int
    public let size: Int
    /// "U32" or "BF16".
    public let dtype: String
    public let shape: [Int]
    public let bits: Int?
}

public struct NemotronLayerExperts: Sendable {
    public let layer: Int
    public let count: Int
    public let stride: Int
    public let file: String
    /// The layout every expert in this layer shares, keyed by role
    /// (`up`, `down`, `up_scales`, ...). The writer emits identical blobs.
    public let roles: [String: NemotronExpertTensor]
}

/// The full text geometry of a Nemotron 3.5 Bolt install.
public struct NemotronGeometry: Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let vocabSize: Int
    public let layerKinds: [NemotronLayerKind]
    public let attentionLayers: [Int]
    public let attentionHeads: Int
    public let attentionKVHeads: Int
    public let attentionHeadDim: Int
    public let attentionScale: Float
    public let mambaIntermediate: Int
    public let mambaHeads: Int
    public let mambaHeadDim: Int
    public let ssmStateSize: Int
    public let mambaGroups: Int
    public let convKernel: Int
    public let moeLatentSize: Int
    public let moeSharedIntermediate: Int
    public let expertCounts: [Int]
    public let moeIntermediateSizes: [Int]
    public let topKExperts: Int
    public let routerGroupCount: Int
    public let routerTopKGroup: Int
    public let routedScalingFactor: Float
    public let routerNormTopK: Bool
    public let normEpsilon: Float
    public let mambaEpsilon: Float
    public let timeStepMin: Float
    public let mambaConvBias: Bool
    public let eosTokens: [Int]

    public var maxExperts: Int { expertCounts.max() ?? 0 }
    public var maxMoeIntermediate: Int { moeIntermediateSizes.max() ?? 0 }
}

public enum NemotronLayerKind: UInt8, Sendable {
    case mamba = 0
    case moe = 1
    case attention = 2
}

public enum NemotronInstallError: Error, CustomStringConvertible {
    case manifest(String)
    case layout(String)
    case resident(String)
    case missingTensor(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .manifest(let detail): return "Nemotron manifest: \(detail)"
        case .layout(let detail): return "Nemotron expert layout: \(detail)"
        case .resident(let detail): return "Nemotron resident index: \(detail)"
        case .missingTensor(let name): return "Nemotron install is missing \(name)"
        case .unsupported(let detail): return "Nemotron install is not supported: \(detail)"
        }
    }
}

/// A read-only mapping of one payload file.

/// A Nemotron 3.5 Bolt `.ssdai` install, read directly: the manifest's text
/// geometry, the resident index, and the packed-expert layout.
///
/// The generic `Model.load` path is Qwen-shaped (it validates a nine-role
/// expert bundle and a 2,816-wide expert tile); this family is deliberately a
/// separate reader until its graph is a first-class engine. The format is the
/// same install, and `TinyTitanRepack --verify-install` is what proves the
/// bytes.
public final class NemotronInstall {
    public let directory: URL
    public let geometry: NemotronGeometry
    public let resident: [String: NemotronTensorRef]
    public let experts: [Int: NemotronLayerExperts]

    let weights: NemotronMappedFile
    let layerFiles: [Int: NemotronMappedFile]

    /// The mappings, for a matvec backend that wraps them in `MTLBuffer`s.
    var weightsMapping: (base: UnsafeRawPointer, size: Int) {
        (weights.base, weights.size)
    }

    func layerMapping(_ layer: Int) -> (base: UnsafeRawPointer, size: Int)? {
        guard let file = layerFiles[layer] else { return nil }
        return (file.base, file.size)
    }

    public init(directory: URL) throws {
        self.directory = directory
        let manifestData = try BoundedMetadataRead.read(
            fileAt: directory.appendingPathComponent("manifest.json"),
            maxBytes: 64 * 1024 * 1024)
        guard
            let manifest = try JSONSerialization.jsonObject(with: manifestData)
                as? [String: Any]
        else {
            throw NemotronInstallError.manifest("not a JSON object")
        }
        guard let arch = manifest["arch"] as? [String: Any],
            let family = arch["family"] as? String, family == "nemotron_h"
        else {
            throw NemotronInstallError.unsupported("arch.family is not nemotron_h")
        }
        guard let detail = arch["nemotronH"] as? [String: Any] else {
            throw NemotronInstallError.manifest("arch.nemotronH is missing")
        }

        func integer(_ object: [String: Any], _ key: String) throws -> Int {
            if let value = object[key] as? Int { return value }
            if let value = object[key] as? NSNumber { return value.intValue }
            throw NemotronInstallError.manifest("\(key) is missing")
        }
        func double(_ object: [String: Any], _ key: String) throws -> Double {
            if let value = object[key] as? NSNumber { return value.doubleValue }
            throw NemotronInstallError.manifest("\(key) is missing")
        }
        func strings(_ object: [String: Any], _ key: String) throws -> [String] {
            guard let value = object[key] as? [String] else {
                throw NemotronInstallError.manifest("\(key) is missing")
            }
            return value
        }
        func ints(_ object: [String: Any], _ key: String) throws -> [Int] {
            guard let raw = object[key] as? [Any] else {
                throw NemotronInstallError.manifest("\(key) is missing")
            }
            return try raw.map {
                if let value = $0 as? Int { return value }
                if let value = $0 as? NSNumber { return value.intValue }
                throw NemotronInstallError.manifest("\(key) has a non-integer entry")
            }
        }

        let kinds = try strings(detail, "layerKinds").map { name -> NemotronLayerKind in
            switch name {
            case "mamba": return .mamba
            case "moe": return .moe
            case "attention": return .attention
            default:
                return .moe
            }
        }
        let counts = try ints(detail, "expertCounts")
        let widths = try ints(detail, "moeIntermediateSizes")
        let numLayers = try integer(arch, "numLayers")
        guard kinds.count == numLayers, counts.count == numLayers, widths.count == numLayers
        else {
            throw NemotronInstallError.manifest(
                "layerKinds/expertCounts/moeIntermediateSizes are not \(numLayers) entries")
        }
        let attentionLayers = try ints(detail, "attentionLayers")
        let headDim = try integer(arch, "headDim")
        var eos: [Int] = [2, 11]
        let generationURL = directory.appendingPathComponent("tokenizer/generation_config.json")
        if let data = try? BoundedMetadataRead.read(
            fileAt: generationURL, maxBytes: 1024 * 1024),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let many = object["eos_token_id"] as? [Int] {
                eos = many
            } else if let one = object["eos_token_id"] as? Int {
                eos = [one]
            }
        }
        geometry = NemotronGeometry(
            hiddenSize: try integer(arch, "hiddenSize"),
            numLayers: numLayers,
            vocabSize: try integer(arch, "vocabSize"),
            layerKinds: kinds,
            attentionLayers: attentionLayers,
            attentionHeads: try integer(arch, "numHeads"),
            attentionKVHeads: try integer(arch, "numKVHeads"),
            attentionHeadDim: headDim,
            attentionScale: Float(try double(arch, "attentionScale")),
            mambaIntermediate: try integer(detail, "mambaIntermediate"),
            mambaHeads: try integer(detail, "mambaNumHeads"),
            mambaHeadDim: try integer(detail, "mambaHeadDim"),
            ssmStateSize: try integer(detail, "ssmStateSize"),
            mambaGroups: try integer(detail, "mambaGroups"),
            convKernel: try integer(detail, "mambaConvKernel"),
            moeLatentSize: try integer(detail, "moeLatentSize"),
            moeSharedIntermediate: try integer(detail, "moeSharedIntermediate"),
            expertCounts: counts,
            moeIntermediateSizes: widths,
            topKExperts: try integer(arch, "topKExperts"),
            routerGroupCount: try integer(detail, "routerGroupCount"),
            routerTopKGroup: try integer(detail, "routerTopKGroup"),
            routedScalingFactor: Float(try double(detail, "routedScalingFactor")),
            routerNormTopK: (detail["moeNormTopK"] as? Bool) ?? true,
            normEpsilon: Float(try double(detail, "normEpsilon")),
            mambaEpsilon: Float(try double(detail, "mambaEpsilon")),
            timeStepMin: Float(try double(detail, "timeStepMin")),
            mambaConvBias: (detail["mambaConvBias"] as? Bool) ?? true,
            eosTokens: eos)

        weights = try NemotronMappedFile(
            url: directory.appendingPathComponent("model_weights.bin"))
        resident = try Self.readResidentIndex(weights)

        var layerFiles: [Int: NemotronMappedFile] = [:]
        var experts: [Int: NemotronLayerExperts] = [:]
        let layoutURL = directory.appendingPathComponent("packed_experts/layout.json")
        let layoutData = try BoundedMetadataRead.read(
            fileAt: layoutURL, maxBytes: 64 * 1024 * 1024)
        guard let layoutRoot = try JSONSerialization.jsonObject(with: layoutData) as? [String: Any],
            let layers = layoutRoot["layers"] as? [[String: Any]]
        else {
            throw NemotronInstallError.layout("layout.json is not an object with layers")
        }
        for layer in layers {
            guard let index = layer["layer"] as? Int,
                let file = layer["file"] as? String,
                let expertList = layer["experts"] as? [[String: Any]]
            else {
                throw NemotronInstallError.layout("a layer entry is incomplete")
            }
            let count = (layer["expertsPerLayer"] as? Int) ?? expertList.count
            guard count == expertList.count else {
                throw NemotronInstallError.layout(
                    "layer \(index) declares \(count) experts but lists \(expertList.count)")
            }
            guard count > 0 else {
                experts[index] = NemotronLayerExperts(
                    layer: index, count: 0, stride: 0, file: file, roles: [:])
                continue
            }
            guard let first = expertList.first,
                let blob = first["tensors"] as? [String: Any],
                let stride = first["size"] as? Int
            else {
                throw NemotronInstallError.layout("layer \(index) expert 0 is incomplete")
            }
            var roles: [String: NemotronExpertTensor] = [:]
            for (name, value) in blob {
                guard let entry = value as? [String: Any],
                    let offset = entry["offset"] as? Int,
                    let size = entry["size"] as? Int,
                    let dtype = entry["dtype"] as? String,
                    let shape = entry["shape"] as? [Int]
                else {
                    throw NemotronInstallError.layout(
                        "layer \(index) role \(name) is incomplete")
                }
                roles[name] = NemotronExpertTensor(
                    offset: offset, size: size, dtype: dtype,
                    shape: shape, bits: entry["bits"] as? Int)
            }
            for expert in expertList {
                guard let expertStride = expert["size"] as? Int,
                    expertStride == stride,
                    let tensors = expert["tensors"] as? [String: Any],
                    tensors.count == roles.count
                else {
                    throw NemotronInstallError.layout(
                        "layer \(index) has an expert with a different blob shape")
                }
            }
            experts[index] = NemotronLayerExperts(
                layer: index, count: count, stride: stride, file: file, roles: roles)
            let url = directory.appendingPathComponent("packed_experts/\(file)")
            layerFiles[index] = try NemotronMappedFile(url: url)
        }
        self.layerFiles = layerFiles
        self.experts = experts

        try validateGeometry()
    }

    /// Structural cross-checks between the manifest and the payload. Cheap and
    /// specific: a mismatch here is a repack bug, not a bad request.
    func validateGeometry() throws {
        let g = geometry
        guard
            try tensor("language_model.backbone.embeddings.weight").shape
                == [g.vocabSize, g.hiddenSize]
        else {
            throw NemotronInstallError.resident("embedding shape disagrees with the manifest")
        }
        for layer in 0..<g.numLayers {
            let norm = try tensor("language_model.backbone.layers.\(layer).norm.weight")
            guard norm.shape == [g.hiddenSize], norm.dtype == 1 else {
                throw NemotronInstallError.resident("layer \(layer) norm is not BF16 [hidden]")
            }
            switch g.layerKinds[layer] {
            case .mamba:
                let inProj = try tensor(
                    "language_model.backbone.layers.\(layer).mixer.in_proj.weight")
                let expected =
                    2 * g.mambaIntermediate + 2 * g.mambaGroups * g.ssmStateSize
                    + g.mambaHeads
                guard inProj.shape == [expected, g.hiddenSize], inProj.bits != nil else {
                    throw NemotronInstallError.resident(
                        "layer \(layer) in_proj is not the Mamba-2 projection")
                }
                let outProj = try tensor(
                    "language_model.backbone.layers.\(layer).mixer.out_proj.weight")
                guard outProj.shape == [g.hiddenSize, g.mambaIntermediate] else {
                    throw NemotronInstallError.resident("layer \(layer) out_proj shape")
                }
            case .moe:
                let experts = try self.expertsFor(layer: layer)
                guard experts.count == g.expertCounts[layer] else {
                    throw NemotronInstallError.resident("layer \(layer) expert count")
                }
                guard let up = experts.roles["up"], let down = experts.roles["down"],
                    up.shape == [g.moeIntermediateSizes[layer], g.moeLatentSize],
                    down.shape == [g.moeLatentSize, g.moeIntermediateSizes[layer]]
                else {
                    throw NemotronInstallError.resident(
                        "layer \(layer) expert roles do not match the latent MoE")
                }
                let gate = try tensor(
                    "language_model.backbone.layers.\(layer).mixer.gate.weight")
                guard gate.shape == [g.expertCounts[layer], g.hiddenSize] else {
                    throw NemotronInstallError.resident("layer \(layer) router shape")
                }
            case .attention:
                let q = try tensor(
                    "language_model.backbone.layers.\(layer).mixer.q_proj.weight")
                guard q.shape == [g.attentionHeads * g.attentionHeadDim, g.hiddenSize] else {
                    throw NemotronInstallError.resident("layer \(layer) q_proj shape")
                }
            }
        }
    }

    public func tensor(_ name: String) throws -> NemotronTensorRef {
        guard let ref = resident[name] else {
            throw NemotronInstallError.missingTensor(name)
        }
        return ref
    }

    public func pointer(of ref: NemotronTensorRef) -> UnsafeRawPointer {
        weights.pointer(offset: ref.fileOffset)
    }

    public func pointer(of ref: NemotronTensorRef, scale: Bool) -> UnsafeRawPointer {
        weights.pointer(offset: scale ? ref.scaleOffset : ref.biasOffset)
    }

    public func expertsFor(layer: Int) throws -> NemotronLayerExperts {
        guard let entry = experts[layer] else {
            throw NemotronInstallError.layout("layer \(layer) has no layout entry")
        }
        return entry
    }

    /// The base pointer of one expert's blob in its layer file.
    public func expertPointer(layer: Int, expert: Int) throws -> UnsafeRawPointer {
        guard let file = layerFiles[layer], let entry = experts[layer] else {
            throw NemotronInstallError.layout("layer \(layer) has no packed file")
        }
        return file.pointer(offset: expert * entry.stride)
    }

    /// Every role pointer inside one expert blob.
    public func expertTensorPointer(
        layer: Int, expert: Int, role: String
    ) throws -> UnsafeRawPointer {
        guard let entry = experts[layer], let tensor = entry.roles[role] else {
            throw NemotronInstallError.layout("layer \(layer) has no role \(role)")
        }
        return try expertPointer(layer: layer, expert: expert)
            .advanced(by: tensor.offset)
    }

    /// Read a BF16 or FP32 vector into floats. Used for norms and the Mamba
    /// scalars, which are small enough to keep as decoded values.
    public func floatVector(_ name: String) throws -> [Float] {
        let ref = try tensor(name)
        let count = ref.shape.reduce(1, *)
        let pointer = pointer(of: ref)
        if ref.dtype == 3 {
            let raw = pointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: raw, count: count))
        }
        guard ref.dtype == 1 else {
            throw NemotronInstallError.unsupported("\(name) is not BF16/FP32")
        }
        let raw = pointer.assumingMemoryBound(to: UInt16.self)
        var out = [Float](repeating: 0, count: count)
        for index in 0..<count {
            out[index] = NemotronMath.bfloat16(raw[index])
        }
        return out
    }

}

/// Small shared math helpers that are not worth a C call.
