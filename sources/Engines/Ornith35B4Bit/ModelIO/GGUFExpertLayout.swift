import Foundation

/// A byte range of one shard that holds one matrix of one routed expert.
package struct GGUFTensorRange: Sendable, Equatable {
    package let shardIndex: Int
    /// Absolute offset in the shard file, not relative to the data section.
    package let fileOffset: UInt64
    package let byteCount: UInt64
    package let type: GGUFType
}

/// The three matrices of one routed expert.
package struct GGUFExpertWeights: Sendable, Equatable {
    package let layer: Int
    package let expert: Int
    package let gate: GGUFTensorRange
    package let up: GGUFTensorRange
    package let down: GGUFTensorRange

    package var totalBytes: UInt64 { gate.byteCount + up.byteCount + down.byteCount }
}

package enum GGUFExpertLayoutError: Error, Equatable, CustomStringConvertible {
    case missingTensor(String)
    case expertCountMismatch(tensor: String, found: Int, expected: Int)
    case expertOutOfRange(layer: Int, expert: Int, count: Int)
    case unknownTensorShape(String)
    case sliceNotBlockAligned(name: String, elements: UInt64, block: UInt64)
    case byteCountNotATile(name: String, byteCount: UInt64, slice: UInt64, count: Int)

    package var description: String {
        switch self {
        case .missingTensor(let name):
            return "GGUF model has no tensor \(name)"
        case .expertCountMismatch(let tensor, let found, let expected):
            return
                "\(tensor) carries \(found) experts, the architecture declares \(expected)"
        case .expertOutOfRange(let layer, let expert, let count):
            return "layer \(layer) has \(count) experts; expert \(expert) is out of range"
        case .unknownTensorShape(let name):
            return "\(name) is not a three-dimensional expert tensor"
        case .sliceNotBlockAligned(let name, let elements, let block):
            return "\(name) has \(elements) elements per expert, not a multiple of \(block)"
        case .byteCountNotATile(let name, let byteCount, let slice, let count):
            return
                "\(name) is \(byteCount) bytes, which is not \(count) slices of \(slice)"
        }
    }
}

/// Per-expert addressing inside `ffn_{gate,up,down}_exps.weight`.
///
/// GGUF stores the routed experts of a layer as one tensor with the expert as
/// its third extent, `[in, out, experts]`, in row-major order. Because every
/// expert slice must itself be a whole number of blocks, the experts are laid
/// out end to end with no padding, and an expert's offset is exactly
/// `expert * sliceBytes`. This type proves that rather than assuming it — the
/// `byteCount == experts * sliceBytes` check is what would catch a writer that
/// padded between experts, which would move every expert after the first.
///
/// It is the addressing half of the weight source: whichever path the engine
/// ends up taking (dequantising on fetch, or kernels that read the block types),
/// an expert is fetched by asking for these ranges.
package enum GGUFExpertLayout {
    package static let tensorSuffixes = (
        gate: "ffn_gate_exps.weight", up: "ffn_up_exps.weight", down: "ffn_down_exps.weight"
    )

    package static func expertsPerLayer(in model: GGUFModel) throws -> Int {
        guard let count = model.architectureValue("expert_count")?.integer, count > 0 else {
            throw GGUFExpertLayoutError.missingTensor("architecture expert_count")
        }
        return Int(count)
    }

    package static func weights(
        layer: Int,
        expert: Int,
        in model: GGUFModel
    ) throws -> GGUFExpertWeights {
        let count = try expertsPerLayer(in: model)
        guard expert >= 0, expert < count else {
            throw GGUFExpertLayoutError.expertOutOfRange(layer: layer, expert: expert, count: count)
        }
        return GGUFExpertWeights(
            layer: layer,
            expert: expert,
            gate: try range(layer: layer, suffix: tensorSuffixes.gate, expert: expert, in: model),
            up: try range(layer: layer, suffix: tensorSuffixes.up, expert: expert, in: model),
            down: try range(layer: layer, suffix: tensorSuffixes.down, expert: expert, in: model))
    }

    /// All expert tensors of a layer, with the slice size each declares.
    package static func slices(
        layer: Int,
        in model: GGUFModel
    ) throws -> (gate: UInt64, up: UInt64, down: UInt64) {
        (
            try sliceBytes(layer: layer, suffix: tensorSuffixes.gate, in: model),
            try sliceBytes(layer: layer, suffix: tensorSuffixes.up, in: model),
            try sliceBytes(layer: layer, suffix: tensorSuffixes.down, in: model)
        )
    }

    private static func range(
        layer: Int, suffix: String, expert: Int, in model: GGUFModel
    ) throws -> GGUFTensorRange {
        let name = tensorName(layer: layer, suffix: suffix)
        guard let location = model.tensor(named: name) else {
            throw GGUFExpertLayoutError.missingTensor(name)
        }
        let slice = try sliceBytes(location: location, name: name, in: model)
        let offset = location.tensor.offset + UInt64(expert) * slice
        let shard = model.shards[location.shardIndex]
        return GGUFTensorRange(
            shardIndex: location.shardIndex,
            fileOffset: shard.dataOffset + offset,
            byteCount: slice,
            type: location.tensor.type)
    }

    private static func sliceBytes(
        layer: Int, suffix: String, in model: GGUFModel
    ) throws -> UInt64 {
        let name = tensorName(layer: layer, suffix: suffix)
        guard let location = model.tensor(named: name) else {
            throw GGUFExpertLayoutError.missingTensor(name)
        }
        return try sliceBytes(location: location, name: name, in: model)
    }

    private static func sliceBytes(
        location: GGUFTensorLocation, name: String, in model: GGUFModel
    ) throws -> UInt64 {
        let tensor = location.tensor
        guard tensor.dimensions.count == 3 else {
            throw GGUFExpertLayoutError.unknownTensorShape(name)
        }
        let expected = try expertsPerLayer(in: model)
        let found = Int(tensor.dimensions[2])
        guard found == expected else {
            throw GGUFExpertLayoutError.expertCountMismatch(
                tensor: name, found: found, expected: expected)
        }
        let perExpert = tensor.dimensions[0].multipliedReportingOverflow(by: tensor.dimensions[1])
        guard !perExpert.overflow else {
            throw GGUFExpertLayoutError.unknownTensorShape(name)
        }
        guard perExpert.partialValue % tensor.type.blockSize == 0 else {
            throw GGUFExpertLayoutError.sliceNotBlockAligned(
                name: name, elements: perExpert.partialValue, block: tensor.type.blockSize)
        }
        let slice = (perExpert.partialValue / tensor.type.blockSize)
            .multipliedReportingOverflow(by: tensor.type.blockBytes)
        guard !slice.overflow else {
            throw GGUFExpertLayoutError.unknownTensorShape(name)
        }
        let total = UInt64(found).multipliedReportingOverflow(by: slice.partialValue)
        guard !total.overflow, total.partialValue == tensor.byteCount else {
            throw GGUFExpertLayoutError.byteCountNotATile(
                name: name,
                byteCount: tensor.byteCount,
                slice: slice.partialValue,
                count: found)
        }
        return slice.partialValue
    }

    private static func tensorName(layer: Int, suffix: String) -> String {
        "blk.\(layer).\(suffix)"
    }
}
