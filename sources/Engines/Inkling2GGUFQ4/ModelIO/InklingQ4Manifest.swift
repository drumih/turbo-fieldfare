import Foundation
import TinyTitanCore
import TinyTitanFormat

/// Geometry and tensor directory of an Inkling 2 Q4 install, read from its
/// manifest (`arch` plus the variant's own `variantGeometry`), the resident
/// index and the packed-expert layout. It carries the same facts the GGUF
/// route's `Inkling2GGUFManifest` read from GGUF metadata, so the copied graph
/// runs unchanged; the repacker wrote them from that same metadata.
struct InklingQ4Manifest: Sendable {
    enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case missingTensor(String)
        case missingGeometry(String)
        case invalidGeometry(String)
        case unsupportedEncoding(String)

        var description: String {
            switch self {
            case .missingTensor(let name): "the install is missing tensor \(name)"
            case .missingGeometry(let key): "the manifest's variantGeometry has no \(key)"
            case .invalidGeometry(let detail): "Inkling 2 geometry is invalid: \(detail)"
            case .unsupportedEncoding(let detail): "unsupported encoding: \(detail)"
            }
        }
    }

    /// One stored layer: its attention kind and the encoding of each tensor
    /// the decoder dispatches a kernel for.
    struct Layer: Sendable, Equatable {
        let index: Int
        let isSliding: Bool
        let heads: Int
        let kvHeads: Int
        let expertFeedForward: Int
        let gateUpType: GGUFType
        let downType: GGUFType
        let hasExpertScales: Bool
        let projectionTypes: [String: GGUFType]
    }

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
    let contextLength: Int
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

    init(install: InklingQ4Install) throws {
        let arch = install.manifest.arch
        guard let geometry = arch.variantGeometry else { throw Error.missingGeometry("(all)") }
        func scalar(_ key: String) throws -> Double {
            guard let value = geometry.scalars[key] else { throw Error.missingGeometry(key) }
            return value
        }
        func int(_ key: String) throws -> Int {
            let value = try scalar(key)
            guard value == value.rounded(), abs(value) < 1e15 else {
                throw Error.invalidGeometry("\(key) = \(value) is not an integer")
            }
            return Int(value)
        }
        guard let local = geometry.lists["localLayers"] else {
            throw Error.missingGeometry("localLayers")
        }
        storedLayers = arch.numLayers
        hidden = arch.hiddenSize
        headDimension = arch.headDim
        expertCount = arch.numExperts
        expertsPerToken = arch.topKExperts
        slidingWindow = arch.slidingWindow
        finalLogitSoftcap = Float(arch.finalLogitSoftcap)
        unrolledLayers = try int("unrolledLayers")
        loopStart = try int("loopStart")
        loopEnd = try int("loopEnd")
        loopPasses = try int("loopPasses")
        loopGateRank = try int("loopGateRank")
        attentionResidualBlock = try int("attentionResidualBlock")
        relativeLength = try int("relativeLength")
        relativeExtent = try int("relativeExtent")
        convolutionTaps = try int("convolutionTaps")
        paddedVocabulary = try int("paddedVocabulary")
        vocabulary = try int("vocabulary")
        contextLength = try int("contextLength")
        logScaleFloor = try int("logScaleFloor")
        rmsEpsilon = Float(try scalar("rmsEpsilon"))
        qkClip = Float(try scalar("qkClip"))
        relativeClip = Float(try scalar("relativeClip"))
        logScaleAlpha = Float(try scalar("logScaleAlpha"))
        expertWeightsScale = Float(try scalar("expertWeightsScale"))
        situBeta = Float(try scalar("situBeta"))
        situLinearBeta = Float(try scalar("situLinearBeta"))

        guard loopPasses >= 1, loopStart > 0, loopStart < loopEnd, loopEnd <= storedLayers,
            unrolledLayers == storedLayers + (loopPasses - 1) * (loopEnd - loopStart),
            hidden % 256 == 0, vocabulary <= paddedVocabulary,
            paddedVocabulary == arch.vocabSize, convolutionTaps >= 2,
            local.allSatisfy({ (0..<arch.numLayers).contains($0) })
        else {
            throw Error.invalidGeometry(
                "loop \(loopStart)..<\(loopEnd) x\(loopPasses), \(unrolledLayers) unrolled of "
                    + "\(storedLayers) stored layers")
        }
        let slidingLayers = Set(local)
        var built: [Layer] = []
        for index in 0..<arch.numLayers {
            built.append(
                try Self.layer(
                    install, index: index, isSliding: slidingLayers.contains(index),
                    heads: arch.numHeads, kvHeads: arch.numKVHeads,
                    feedForward: arch.moeIntermediateSize, hidden: arch.hiddenSize,
                    headDimension: arch.headDim))
        }
        layers = built
    }

    /// The stored layer an unrolled position runs (llama.cpp `loop_layer`).
    func storedLayer(forUnrolled position: Int) -> Int {
        guard position >= loopStart else { return position }
        let core = loopEnd - loopStart
        let looped = loopStart + core * loopPasses
        if position < looped { return loopStart + (position - loopStart) % core }
        return position - core * (loopPasses - 1)
    }

    /// The ggml type an index dtype code stands for (the repacker's mapping).
    static func type(indexCode code: UInt8, tensor name: String) throws -> GGUFType {
        switch code {
        case 1: return .bf16
        case 2: return .f16
        case 3: return .f32
        case 4: return .q8_0
        case 5: return .nvfp4
        default: throw Error.unsupportedEncoding("\(name): index dtype \(code)")
        }
    }

    /// The ggml type a packed-expert sub-tensor's dtype names.
    static func type(layoutDType dtype: String, tensor name: String) throws -> GGUFType {
        switch dtype {
        case "GGUF_NVFP4": return .nvfp4
        case "GGUF_Q8_0": return .q8_0
        default: throw Error.unsupportedEncoding("\(name): expert dtype \(dtype)")
        }
    }

    private static func layer(
        _ install: InklingQ4Install, index: Int, isSliding: Bool, heads: Int, kvHeads: Int,
        feedForward: Int, hidden: Int, headDimension: Int
    ) throws -> Layer {
        let prefix = "blk.\(index)."
        func entry(_ suffix: String) throws -> ResidentIndexEntry {
            guard let found = install.resident.entries[prefix + suffix] else {
                throw Error.missingTensor(prefix + suffix)
            }
            return found
        }
        func typed(_ suffix: String, rows: Int, columns: Int) throws -> GGUFType {
            let found = try entry(suffix)
            let (foundRows, foundColumns) = InklingQ4Weights.matrix(found.shape)
            guard foundRows == rows, foundColumns == columns else {
                throw Error.invalidGeometry(
                    "\(prefix + suffix) is \(foundRows) x \(foundColumns), expected "
                        + "\(rows) x \(columns)")
            }
            return try type(indexCode: found.dtype, tensor: prefix + suffix)
        }
        let attention = headDimension * heads
        let kv = headDimension * kvHeads
        var types: [String: GGUFType] = [:]
        types["attn_q"] = try typed("attn_q.weight", rows: attention, columns: hidden)
        types["attn_k"] = try typed("attn_k.weight", rows: kv, columns: hidden)
        types["attn_v"] = try typed("attn_v.weight", rows: kv, columns: hidden)
        types["attn_output"] = try typed("attn_output.weight", rows: hidden, columns: attention)
        types["attn_gate"] = try typed("attn_gate.weight", rows: attention, columns: hidden)
        types["attn_rel"] = try type(
            indexCode: try entry("attn_rel.weight").dtype, tensor: prefix + "attn_rel.weight")
        guard let experts = install.layout.layers.first(where: { $0.layer == index }),
            let first = experts.experts.first,
            let gateUp = first.tensors["gate_up"], let down = first.tensors["down"]
        else { throw Error.missingTensor(prefix + "packed experts") }
        guard gateUp.shape == [UInt32(2 * feedForward), UInt32(hidden)],
            down.shape == [UInt32(hidden), UInt32(feedForward)]
        else {
            throw Error.invalidGeometry(
                "layer \(index) experts are \(gateUp.shape) / \(down.shape)")
        }
        let scaled = first.tensors["gate_up_scale"] != nil
        guard scaled == (first.tensors["down_scale"] != nil) else {
            throw Error.missingTensor(prefix + "expert scale (only one of the pair)")
        }
        return Layer(
            index: index, isSliding: isSliding, heads: heads, kvHeads: kvHeads,
            expertFeedForward: feedForward,
            gateUpType: try type(layoutDType: gateUp.dtype, tensor: prefix + "gate_up"),
            downType: try type(layoutDType: down.dtype, tensor: prefix + "down"),
            hasExpertScales: scaled, projectionTypes: types)
    }
}
