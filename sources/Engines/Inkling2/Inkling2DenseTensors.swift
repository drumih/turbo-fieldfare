import Foundation

/// Row-level access to the dense (non-expert) side of the checkpoint.
public struct Inkling2DenseTensors: Sendable {
    private let file: Inkling2Safetensors
    public let vocabularySize: Int
    public let hiddenSize: Int

    public init(directory: URL) throws {
        self.file = try Inkling2Safetensors(
            url: directory.appendingPathComponent("model-00001-of-00001.safetensors"))
        guard let embedding = file.tensor("model.llm.embed.weight") else {
            throw Inkling2Safetensors.Error.missing("model.llm.embed.weight")
        }
        guard embedding.shape.count == 2 else {
            throw Inkling2Safetensors.Error.invalid("embedding is not 2-D")
        }
        self.vocabularySize = embedding.shape[0]
        self.hiddenSize = embedding.shape[1]
    }

    public func embeddingRow(_ token: Int) throws -> [Float] {
        guard token >= 0, token < vocabularySize else {
            throw Inkling2Safetensors.Error.invalid("token \(token) outside vocabulary")
        }
        guard let tensor = file.tensor("model.llm.embed.weight") else {
            throw Inkling2Safetensors.Error.missing("model.llm.embed.weight")
        }
        let rowBytes = hiddenSize * tensor.dtype.byteWidth
        let data = try file.data(
            for: "model.llm.embed.weight",
            offset: token * rowBytes,
            count: rowBytes)
        return Self.floats(data, dtype: tensor.dtype)
    }

    public func embeddingNorm() throws -> [Float] {
        try file.floatValues("model.llm.embed_norm.weight")
    }

    public func finalNorm() throws -> [Float] {
        try file.floatValues("model.llm.norm.weight")
    }

    public func layerNorm(_ layer: Int, _ suffix: String) throws -> [Float] {
        try file.floatValues("model.llm.layers.\(layer).\(suffix)")
    }

    public func layerMatrix(_ layer: Int, _ suffix: String) throws -> [Float] {
        try file.floatValues("model.llm.layers.\(layer).\(suffix)")
    }

    public func layerMatrixShape(_ layer: Int, _ suffix: String) throws -> [Int] {
        guard let tensor = file.tensor("model.llm.layers.\(layer).\(suffix)") else {
            throw Inkling2Safetensors.Error.missing("model.llm.layers.\(layer).\(suffix)")
        }
        return tensor.shape
    }

    private static func floats(_ data: Data, dtype: Inkling2Safetensors.DType) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(data.count / dtype.byteWidth)
        data.withUnsafeBytes { raw in
            if dtype == .f32 {
                for offset in stride(from: 0, to: raw.count, by: 4) {
                    values.append(
                        Float(
                            bitPattern: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
                }
            } else {
                for offset in stride(from: 0, to: raw.count, by: 2) {
                    let bits = raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self)
                    values.append(Float(bitPattern: UInt32(bits) << 16))
                }
            }
        }
        return values
    }
}

/// The small numerical primitives the text graph needs before attention/MoE.
public enum Inkling2Ops {
    public static func rmsNorm(
        _ values: [Float], weight: [Float], epsilon: Float = 1e-6
    ) -> [Float] {
        precondition(values.count == weight.count)
        var sum: Float = 0
        for value in values { sum += value * value }
        let inverse = 1 / (sum / Float(values.count) + epsilon).squareRoot()
        return zip(values, weight).map { $0 * inverse * $1 }
    }
}
