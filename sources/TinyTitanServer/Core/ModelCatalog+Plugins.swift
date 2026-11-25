import Foundation
import TinyTitanCore
import TinyTitanPluginAPI

/// Plugin-backed install probing and on-disk sizing, split out of
/// `ModelCatalog.swift` under the 500-line-per-file rule (TT-126).
extension ModelCatalog {
    static func probePluginInstall(
        _ directory: URL,
        resolved: ResolvedPlugin
    ) -> Result<Entry, ProbeFailure> {
        let factory = resolved.engineFactory()
        let metadata: EngineModelMetadata
        do {
            metadata = try factory.metadata(directory)
        } catch {
            return .failure(ProbeFailure(reason: "unreadable plugin metadata: \(error)"))
        }
        let hasTokenizer =
            GFTokenizer.tokenizerFolder(forModelDirectory: directory) != nil
            || factory.tokenizerFolder != nil
        guard hasTokenizer else {
            return .failure(
                ProbeFailure(reason: "no tokenizer/tokenizer.json; the install is incomplete"))
        }
        return .success(
            Entry(
                id: metadata.modelID,
                name: resolved.plugin.displayName,
                kind: .plugin(family: metadata.family, reasoning: metadata.reasoningControl),
                quant: metadata.weightBits,
                path: directory,
                sampling: metadata.sampling,
                contextLimit: nil,
                sizeBytes: sizeOnDisk(directory)))
    }

    /// Allocated bytes, so an APFS clone or a sparse file reports what it
    /// really occupies rather than its logical length.
    static func sizeOnDisk(_ directory: URL) -> Int64 {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .fileSizeKey, .isRegularFileKey]
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: keys)
        else { return 0 }
        var total: Int64 = 0
        for case let file as URL in walker {
            guard let values = try? file.resourceValues(forKeys: Set(keys)),
                values.isRegularFile == true
            else { continue }
            total += Int64(values.totalFileAllocatedSize ?? values.fileSize ?? 0)
        }
        return total
    }
}
