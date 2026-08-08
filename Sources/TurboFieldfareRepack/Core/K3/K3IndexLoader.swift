import Foundation

/// Parses the K3 `model.safetensors.index.json`. Unlike the Gemma
/// `IndexLoader` there is no `config.json -> quantization` slot to read: the
/// K3 source checkpoint is BF16 trunk + compressed-tensors MXFP4 experts, and
/// the v2 quant profile is owned by `KimiK3FormatProfile` / the repack CLI,
/// not by source metadata.
enum K3IndexLoader {

    struct SourceMetadata {
        let indexPath: String
        let configPath: String
        let indexSha256Hex: String
        /// `tensor_name -> shard_filename`
        let weightMap: [String: String]
        /// Resolved set of shard files referenced by the index, in
        /// encounter order of the sorted tensor names (stable).
        let shardFilenames: [String]
    }

    static func load(snapshotDir: String) throws -> SourceMetadata {
        let indexPath  = (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json")
        let configPath = (snapshotDir as NSString).appendingPathComponent("config.json")

        let weightMap: [String: String]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let m = root["weight_map"] as? [String: String] else {
                throw RepackError.indexJsonInvalid(path: indexPath, detail: "no weight_map")
            }
            weightMap = m
        } catch let e as RepackError {
            throw e
        } catch {
            throw RepackError.indexJsonInvalid(path: indexPath, detail: "\(error)")
        }

        let indexSha = try Sha256Stream.hashFile(path: indexPath)

        var seen = Set<String>()
        var shards: [String] = []
        for k in weightMap.keys.sorted() {
            let shard = weightMap[k]!
            if !seen.contains(shard) { seen.insert(shard); shards.append(shard) }
        }

        return SourceMetadata(indexPath: indexPath, configPath: configPath,
                              indexSha256Hex: indexSha,
                              weightMap: weightMap,
                              shardFilenames: shards)
    }
}
