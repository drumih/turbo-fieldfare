import Foundation

/// One dequantized routed expert. The expert is read as a byte range from the
/// checkpoint and expanded only for the duration of its GEMM; the full 20 GB
/// tensor is never resident.
public struct Inkling2ExpertWeights: Sendable {
    public let layer: Int
    public let expert: Int
    public let w13: [Float]
    public let w2: [Float]
    public let hidden: Int
    public let intermediate: Int
}

/// Reads one routed expert at a time from the main safetensors file.
public struct Inkling2ExpertStore: Sendable {
    private let file: Inkling2Safetensors
    private let experts: Int

    public init(directory: URL) throws {
        self.file = try Inkling2Safetensors(
            url: directory.appendingPathComponent("model-00001-of-00001.safetensors"))
        guard
            let name = file.tensors.keys.first(where: { $0.hasSuffix(".w13_weight") }),
            let expertTensor = file.tensor(name)
        else {
            throw Inkling2Safetensors.Error.missing("a routed-expert w13_weight")
        }
        self.experts = expertTensor.shape.first ?? 0
    }

    public func load(layer: Int, expert: Int) throws -> Inkling2ExpertWeights {
        guard layer >= 0, layer < 24 else {
            throw Inkling2Safetensors.Error.invalid("layer \(layer) outside 0..<24")
        }
        guard expert >= 0, expert < experts else {
            throw Inkling2Safetensors.Error.invalid("expert \(expert) outside 0..<\(experts)")
        }
        let prefix = "model.llm.layers.\(layer).mlp.experts"
        let hidden = try rows("\(prefix).w13_weight")
        if layer < 20 {
            return try loadNVFP4(layer: layer, expert: expert, prefix: prefix, hidden: hidden)
        }
        return try loadMXFP8(layer: layer, expert: expert, prefix: prefix, hidden: hidden)
    }

    private func loadNVFP4(
        layer: Int, expert: Int, prefix: String, hidden: Int
    ) throws -> Inkling2ExpertWeights {
        let w13 = try slice("\(prefix).w13_weight", expert: expert, experts: experts)
        let w13Scale = try slice("\(prefix).w13_weight_scale", expert: expert, experts: experts)
        let w2 = try slice("\(prefix).w2_weight", expert: expert, experts: experts)
        let w2Scale = try slice("\(prefix).w2_weight_scale", expert: expert, experts: experts)
        let w13Shape = try shape("\(prefix).w13_weight")
        let w2Shape = try shape("\(prefix).w2_weight")
        let w13Columns = w13Shape[2] * 2
        let w2Columns = w2Shape[2] * 2
        let intermediate = w2Columns
        let w13Outer =
            try float("\(prefix).w13_weight_amax", index: expert)
            / float("\(prefix).w13_weight_outer_scale_denominator", index: 0)
        let w2Outer =
            try float("\(prefix).w2_weight_amax", index: expert)
            / float("\(prefix).w2_weight_outer_scale_denominator", index: 0)
        let w13Values = try Inkling2Quantization.dequantizeNVFP4(
            packed: w13, scales: w13Scale, outerScale: w13Outer,
            rows: hidden, columns: w13Columns)
        let w2Values = try Inkling2Quantization.dequantizeNVFP4(
            packed: w2, scales: w2Scale, outerScale: w2Outer,
            rows: hidden, columns: w2Columns)
        return Inkling2ExpertWeights(
            layer: layer, expert: expert, w13: w13Values, w2: w2Values,
            hidden: hidden, intermediate: intermediate)
    }

    private func loadMXFP8(
        layer: Int, expert: Int, prefix: String, hidden: Int
    ) throws -> Inkling2ExpertWeights {
        let w13 = try slice("\(prefix).w13_weight", expert: expert, experts: experts)
        let w13Scale = try slice("\(prefix).w13_weight_scale", expert: expert, experts: experts)
        let w2 = try slice("\(prefix).w2_weight", expert: expert, experts: experts)
        let w2Scale = try slice("\(prefix).w2_weight_scale", expert: expert, experts: experts)
        let w13Shape = try shape("\(prefix).w13_weight")
        let w2Shape = try shape("\(prefix).w2_weight")
        let intermediate = w2Shape[2]
        let w13Values = try Inkling2Quantization.dequantizeMXFP8(
            weights: w13, scales: w13Scale,
            rows: hidden, columns: w13Shape[2])
        let w2Values = try Inkling2Quantization.dequantizeMXFP8(
            weights: w2, scales: w2Scale,
            rows: hidden, columns: w2Shape[2])
        return Inkling2ExpertWeights(
            layer: layer, expert: expert, w13: w13Values, w2: w2Values,
            hidden: hidden, intermediate: intermediate)
    }

    private func rows(_ name: String) throws -> Int {
        let shape = try shape(name)
        guard shape.count == 3 else {
            throw Inkling2Safetensors.Error.invalid("\(name) is not 3-D")
        }
        return shape[1]
    }

    private func shape(_ name: String) throws -> [Int] {
        guard let tensor = file.tensor(name) else {
            throw Inkling2Safetensors.Error.missing(name)
        }
        return tensor.shape
    }

    private func slice(_ name: String, expert: Int, experts: Int) throws -> Data {
        guard let tensor = file.tensor(name), tensor.shape.count == 3 else {
            throw Inkling2Safetensors.Error.missing(name)
        }
        let stride = tensor.byteCount / experts
        return try file.data(for: name, offset: expert * stride, count: stride)
    }

    private func float(_ name: String, index: Int) throws -> Float {
        guard let tensor = file.tensor(name), tensor.dtype == .f32 else {
            throw Inkling2Safetensors.Error.missing(name)
        }
        let data = try file.data(for: name, offset: index * 4, count: 4)
        return data.withUnsafeBytes { raw in
            Float(bitPattern: raw.loadUnaligned(as: UInt32.self).littleEndian)
        }
    }
}
