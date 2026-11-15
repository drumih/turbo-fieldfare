import Foundation

/// Where one tensor lives: which shard of a split set, and the tensor itself.
package struct GGUFTensorLocation: Sendable {
    package let shardIndex: Int
    package let tensor: GGUFTensor
}

/// A complete GGUF model, with a split set's tensors merged into one namespace.
///
/// Discovery is by filename, not by scanning every `.gguf` in the directory: a
/// model directory here also holds `mmproj-*.gguf` (the vision projector) and
/// `mtp-*.gguf` (a speculative-decoding sidecar), and neither belongs to the
/// base model's tensor set. llama.cpp's convention is authoritative —
/// `<stem>-00001-of-00004.gguf` — and a file that does not follow it is read as
/// a single-shard model only when the directory holds exactly one `.gguf`.
package struct GGUFModel: Sendable {
    package let shards: [GGUFShard]
    package let tensors: [String: GGUFTensorLocation]
    /// The first shard's metadata: a GGUF split set puts every key there and
    /// leaves only `split.*` in the others.
    package let metadata: [String: GGUFValue]

    package var architecture: String? { metadata["general.architecture"]?.string }

    package var totalTensorBytes: UInt64 {
        tensors.values.reduce(UInt64(0)) { $0 + $1.tensor.byteCount }
    }

    package func tensor(named name: String) -> GGUFTensorLocation? { tensors[name] }

    /// Key with the architecture prefix, e.g. `qwen4exp.block_count`. Checking
    /// both spellings keeps this usable with the generic keys some writers emit.
    package func architectureValue(_ suffix: String) -> GGUFValue? {
        if let key = architecture, let value = metadata["\(key).\(suffix)"] { return value }
        return metadata[suffix]
    }

    package func tensorNames(withPrefix prefix: String) -> [String] {
        tensors.keys.filter { $0.hasPrefix(prefix) }.sorted()
    }
}

package enum GGUFModelReader {
    /// Read a model from a directory or from any shard of it.
    ///
    /// A directory must resolve unambiguously: one split set, or exactly one
    /// unrelated `.gguf`. Pointing at a shard directly is how a caller names a
    /// set whose siblings cannot be discovered by convention.
    package static func read(
        modelAt url: URL,
        maxDirectoryBytes: UInt64 = GGUFReader.defaultMaxDirectoryBytes
    ) throws -> GGUFModel {
        let entry = try entryShard(at: url)
        guard let split = splitName(entry.lastPathComponent) else {
            let shard = try GGUFReader.read(shardAt: entry, maxDirectoryBytes: maxDirectoryBytes)
            if let declared = shard.metadata["split.count"]?.integer, declared > 1 {
                throw GGUFError.splitMismatch(
                    detail:
                        "\(entry.lastPathComponent) declares split.count \(declared) but is not named as a shard"
                )
            }
            return try merge(shards: [shard], split: nil)
        }

        let directory = entry.deletingLastPathComponent()
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        var urls: [URL] = []
        urls.reserveCapacity(split.count)
        var missing: [String] = []
        for index in 1...split.count {
            let name = shardName(prefix: split.prefix, index: index, count: split.count)
            if names.contains(name) {
                urls.append(directory.appendingPathComponent(name))
            } else {
                missing.append(name)
            }
        }
        guard missing.isEmpty else {
            throw GGUFError.splitMismatch(
                detail: "missing shard(s) \(missing.joined(separator: ", "))")
        }
        let shards = try urls.map {
            try GGUFReader.read(shardAt: $0, maxDirectoryBytes: maxDirectoryBytes)
        }
        return try merge(shards: shards, split: split)
    }

    private static func merge(shards: [GGUFShard], split: SplitName?) throws -> GGUFModel {
        var tensors: [String: GGUFTensorLocation] = [:]
        var total = 0
        for (index, shard) in shards.enumerated() {
            if let declared = shard.metadata["split.count"]?.integer,
                declared != Int64(shards.count)
            {
                throw GGUFError.splitMismatch(
                    detail:
                        "\(shard.url.lastPathComponent) says split.count \(declared), but \(shards.count) shard(s) were read"
                )
            }
            if let number = shard.metadata["split.no"]?.integer, number != Int64(index) {
                throw GGUFError.splitMismatch(
                    detail:
                        "\(shard.url.lastPathComponent) says split.no \(number), expected \(index)"
                )
            }
            for tensor in shard.tensors {
                if let existing = tensors[tensor.name] {
                    throw GGUFError.splitMismatch(
                        detail:
                            "tensor \(tensor.name) is in both shard \(existing.shardIndex) and shard \(index)"
                    )
                }
                tensors[tensor.name] = GGUFTensorLocation(shardIndex: index, tensor: tensor)
                total += 1
            }
        }
        let metadata = shards[0].metadata
        if let declared = metadata["split.tensors.count"]?.integer, declared != Int64(total) {
            throw GGUFError.splitMismatch(
                detail: "split.tensors.count is \(declared), but \(total) tensor(s) were read")
        }
        if let split, let declared = metadata["split.count"]?.integer,
            declared != Int64(split.count)
        {
            throw GGUFError.splitMismatch(
                detail: "split.count is \(declared), but the filename says \(split.count)")
        }
        return GGUFModel(shards: shards, tensors: tensors, metadata: metadata)
    }

    private static func entryShard(at url: URL) throws -> URL {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw GGUFError.unreadable(path: url.path, errno: ENOENT)
        }
        if !isDirectory.boolValue { return url }

        let names = try FileManager.default.contentsOfDirectory(atPath: url.path)
        let ggufs = names.filter { $0.hasSuffix(".gguf") }.sorted()
        guard !ggufs.isEmpty else { throw GGUFError.noGGUFFiles(url.path) }
        let firstShards = ggufs.compactMap { splitName($0) }.filter { $0.index == 1 }
        if firstShards.count == 1 {
            guard let name = ggufs.first(where: { splitName($0)?.index == 1 }) else {
                throw GGUFError.ambiguousModelDirectory(url.path)
            }
            return url.appendingPathComponent(name)
        }
        guard ggufs.count == 1 else {
            throw GGUFError.ambiguousModelDirectory(url.path)
        }
        return url.appendingPathComponent(ggufs[0])
    }

    private struct SplitName {
        let prefix: String
        let index: Int
        let count: Int
    }

    /// `-00001-of-00004.gguf` is 20 characters: 15 of suffix plus 5 for `.gguf`.
    private static func splitName(_ name: String) -> SplitName? {
        guard name.hasSuffix(".gguf") else { return nil }
        let stem = String(name.dropLast(5))
        guard stem.count > 15 else { return nil }
        let suffix = Array(stem.suffix(15))
        guard suffix[0] == "-", suffix[6] == "-", suffix[7] == "o", suffix[8] == "f",
            suffix[9] == "-"
        else { return nil }
        let indexText = String(suffix[1..<6])
        let countText = String(suffix[10..<15])
        guard let index = Int(indexText), let count = Int(countText),
            index >= 1, count >= 1, index <= count
        else { return nil }
        let prefix = String(stem.dropLast(15))
        guard !prefix.isEmpty else { return nil }
        return SplitName(prefix: prefix, index: index, count: count)
    }

    private static func shardName(prefix: String, index: Int, count: Int) -> String {
        "\(prefix)-\(String(format: "%05d", index))-of-\(String(format: "%05d", count)).gguf"
    }
}
