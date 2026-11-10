import Foundation
import Metal

package enum GGUFANEPrefillError: Error, Equatable, CustomStringConvertible {
    case notFullAttention(Int)
    case missingMetadata(String)
    case missingTensor(String)
    case writeFailed(String, Int32)

    package var description: String {
        switch self {
        case .notFullAttention(let layer):
            return "layer \(layer) is not a full-attention layer; ANE prefill covers only those"
        case .missingMetadata(let key):
            return "GGUF metadata has no \(key)"
        case .missingTensor(let name):
            return "GGUF has no tensor \(name)"
        case .writeFailed(let path, let code):
            return "cannot write \(path): errno \(code)"
        }
    }
}

package struct GGUFANEPrefillGeometry: Codable, Sendable, Equatable {
    package let hidden: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let headDim: Int
    package let rotaryDim: Int
    package let theta: Double
    package let chunk: Int
}

/// Emits the fp16 weights an ANE prefill sidecar needs for one full-attention
/// layer, read from the GGUF as it lies and dequantised only into this derived
/// source.
///
/// This is the same contract `tools/export_ane_prefill.py` takes from an
/// `.ssdai` install - `wq`, `wk`, `wv`, `wo`, `q_norm`, `k_norm`, each
/// `[out, in]` in fp16 (GGUF stores the transpose) - but the tensors come from
/// the checkpoint's Q6_K blocks, so the sidecar can be built without converting
/// the model. Only the 12 full-attention layers are eligible; the GDN layers
/// and the experts stay on the GPU, exactly as the qwen3.8 sidecar does.
package enum GGUFANEPrefillSource {
    /// `chunk` is the production prefill chunk; the sidecar's shapes are fixed
    /// to it.
    package static let chunk = 4096

    package static func geometry(of model: GGUFModel) throws -> GGUFANEPrefillGeometry {
        func integer(_ key: String) throws -> Int {
            guard let value = model.architectureValue(key)?.integer else {
                throw GGUFANEPrefillError.missingMetadata(key)
            }
            return Int(value)
        }
        guard let theta = model.architectureValue("rope.freq_base")?.double else {
            throw GGUFANEPrefillError.missingMetadata("rope.freq_base")
        }
        return GGUFANEPrefillGeometry(
            hidden: try integer("embedding_length"),
            numHeads: try integer("attention.head_count"),
            numKVHeads: try integer("attention.head_count_kv"),
            headDim: try integer("attention.key_length"),
            rotaryDim: try integer("rope.dimension_count"),
            theta: theta,
            chunk: chunk)
    }

    /// Writes `layer_<L>/` with the six fp16 tensors and the geometry.
    /// Returns each tensor's element count for the caller's report.
    @discardableResult
    package static func write(
        layer: Int, from model: GGUFModel, context: MetalContext, to directory: URL
    ) throws -> [String: Int] {
        let arch = ArchConfig.qwen38FlashNext
        guard arch.fullAttentionLayerMask.indices.contains(layer),
            arch.fullAttentionLayerMask[layer] == 1
        else {
            throw GGUFANEPrefillError.notFullAttention(layer)
        }
        let geometry = try geometry(of: model)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        // GGUF dims are [in, out] with dim0 fastest, so a tensor's memory is
        // already [out][in] - the exporter's layout. These counts are the
        // expected sizes, and the arena reports the same shape back.
        let projections: [(name: String, tensor: String, rows: Int, columns: Int)] = [
            ("wq", "attn_q.weight", 2 * geometry.numHeads * geometry.headDim, geometry.hidden),
            ("wk", "attn_k.weight", geometry.numKVHeads * geometry.headDim, geometry.hidden),
            ("wv", "attn_v.weight", geometry.numKVHeads * geometry.headDim, geometry.hidden),
            ("wo", "attn_output.weight", geometry.hidden, geometry.numHeads * geometry.headDim),
        ]
        var names: [String] = []
        for projection in projections { names.append("blk.\(layer).\(projection.tensor)") }
        names.append("blk.\(layer).attn_q_norm.weight")
        names.append("blk.\(layer).attn_k_norm.weight")
        // One GPU decode of all six tensors, the same kernels the decoder uses.
        let arena = try GGUFResidentArena.build(
            names: names, from: model, context: context)

        var written: [String: Int] = [:]
        for projection in projections {
            let full = "blk.\(layer).\(projection.tensor)"
            let entry = try ArenaReader.entry(arena, name: full)
            guard Int(entry.rows) == projection.rows,
                Int(entry.columns) == projection.columns
            else {
                throw GGUFANEPrefillError.missingTensor(full)
            }
            let half = ArenaReader.halves(
                arena, entry: entry, count: projection.rows * projection.columns)
            try writeHalf(half, to: directory.appendingPathComponent("\(projection.name).f16"))
            written[projection.name] = half.count
        }
        for name in ["q_norm", "k_norm"] {
            let tensor = name == "q_norm" ? "attn_q_norm.weight" : "attn_k_norm.weight"
            let full = "blk.\(layer).\(tensor)"
            let entry = try ArenaReader.entry(arena, name: full)
            let half = ArenaReader.halves(
                arena, entry: entry, count: Int(entry.rows * entry.columns))
            try writeHalf(half, to: directory.appendingPathComponent("\(name).f16"))
            written[name] = half.count
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(geometry).write(to: directory.appendingPathComponent("geometry.json"))
        return written
    }

    private enum ArenaReader {
        static func entry(_ arena: GGUFResidentArena, name: String) throws
            -> GGUFResidentArenaEntry
        {
            guard let entry = arena.entries[name] else {
                throw GGUFANEPrefillError.missingTensor(name)
            }
            return entry
        }

        /// The arena stores bf16; the exporter wants fp16. Both are converted
        /// from the same float, so the fp16 is the nearest representable value
        /// of the decoded weight.
        static func halves(
            _ arena: GGUFResidentArena, entry: GGUFResidentArenaEntry, count: Int
        ) -> [UInt16] {
            let source = arena.buffer.contents()
                .advanced(by: Int(entry.offset))
                .assumingMemoryBound(to: UInt16.self)
            var out = [UInt16](repeating: 0, count: count)
            for index in 0..<count {
                out[index] =
                    Float16(
                        Float(bitPattern: UInt32(source[index]) << 16)
                    ).bitPattern
            }
            return out
        }
    }

    private static func writeHalf(_ values: [UInt16], to url: URL) throws {
        var bytes = [UInt8]()
        bytes.reserveCapacity(values.count * 2)
        for value in values {
            bytes.append(UInt8(truncatingIfNeeded: value))
            bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        guard FileManager.default.createFile(atPath: url.path, contents: Data(bytes)) else {
            throw GGUFANEPrefillError.writeFailed(url.path, errno)
        }
    }
}
