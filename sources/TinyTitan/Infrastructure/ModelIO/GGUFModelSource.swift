import Foundation

package enum GGUFModelSourceError: Error, Equatable, CustomStringConvertible {
    case missingPLE
    case noResidentTensors

    package var description: String {
        switch self {
        case .missingPLE:
            return "the checkpoint has no per_layer_token_embd.weight"
        case .noResidentTensors:
            return "the checkpoint has no resident tensors"
        }
    }
}

/// The checkpoint described as the engine's loader needs it: the architecture
/// the file actually declares, the tensors that live in the resident arena, the
/// PLE table that does not, and per-expert access.
///
/// This is the data layer between `GGUFModel` (files and blocks) and a `Model`
/// (Metal buffers and kernels). It deliberately allocates nothing large: the
/// architecture is cross-checked against the runtime preset rather than
/// trusted, and the resident set is a name list the arena builder consumes.
package struct GGUFModelSource: Sendable {
    package let model: GGUFModel
    package let arch: ArchConfig
    /// Resident tensors in the role map's terms, excluding the PLE table and
    /// the routed experts. This is what `GGUFResidentArena` decodes to bf16.
    package let residentNames: [String]
    package let pleTable: GGUFTensorLocation
    package let layerCount: Int
    package let expertsPerLayer: Int

    package static func open(
        directoryURL: URL, arch preset: ArchConfig = .qwen38FlashNext
    ) throws -> GGUFModelSource {
        let model = try GGUFModelReader.read(modelAt: directoryURL)
        // The file is authoritative for geometry; a checkpoint that is not the
        // architecture the runtime executes is refused here, by field.
        try GGUFManifestBuilder.crossCheck(model, preset: preset)
        _ = try GGUFResidentRoleMap.validate(model: model, arch: preset)

        var residents: [String] = []
        residents.reserveCapacity(model.tensors.count)
        var ple: GGUFTensorLocation?
        for (name, location) in model.tensors.sorted(by: { $0.key < $1.key }) {
            if name.hasSuffix("_exps.weight") { continue }
            if name == "per_layer_token_embd.weight" {
                ple = location
                continue
            }
            residents.append(name)
        }
        guard let pleTable = ple else { throw GGUFModelSourceError.missingPLE }
        guard !residents.isEmpty else { throw GGUFModelSourceError.noResidentTensors }
        return GGUFModelSource(
            model: model,
            arch: preset,
            residentNames: residents,
            pleTable: pleTable,
            layerCount: preset.numLayers,
            expertsPerLayer: try GGUFExpertLayout.expertsPerLayer(in: model))
    }

    /// The three matrices of one routed expert, addressed in the mmap'd shards.
    package func expert(layer: Int, expert: Int) throws -> GGUFExpertWeights {
        try GGUFExpertLayout.weights(layer: layer, expert: expert, in: model)
    }

    /// The PLE table's byte range for the GPU gather, as an absolute file
    /// offset plus the row stride a lookup multiplies by.
    package var pleTableRange:
        (shardIndex: Int, fileOffset: UInt64, byteCount: UInt64, rowBytes: UInt64)
    {
        let shard = model.shards[pleTable.shardIndex]
        return (
            pleTable.shardIndex,
            shard.dataOffset + pleTable.tensor.offset,
            pleTable.tensor.byteCount,
            pleTable.tensor.dimensions[0] / 32 * 18
        )
    }
}
