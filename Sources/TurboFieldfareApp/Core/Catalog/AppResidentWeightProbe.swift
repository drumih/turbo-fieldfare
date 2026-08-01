import Foundation

/// Measures how many bytes of a model stay resident in memory.
///
/// The whole design premise is that routed experts stream from SSD while
/// everything else — embeddings, attention, shared experts, router — is held
/// in memory. So resident bytes are the manifest's total minus the packed
/// expert files. This replaces the fixed estimate that the context picker used
/// before multi-model support, which would be wrong for any model whose split
/// differs from Gemma's.
public enum AppResidentWeightProbe {
    private static let expertFilePrefix = "packed_experts/"

    private struct ManifestShape: Decodable {
        struct FileEntry: Decodable {
            let size: UInt64
        }
        let files: [String: FileEntry]
    }

    public static func residentBytes(atModelDirectory modelDirectory: URL) -> UInt64? {
        let manifestURL = modelDirectory
            .appendingPathComponent("manifest.json", isDirectory: false)
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(ManifestShape.self, from: data)
        else { return nil }
        return manifest.files
            .filter { !$0.key.hasPrefix(expertFilePrefix) }
            .values
            .reduce(UInt64(0)) { $0 + $1.size }
    }

    /// Falls back to the compiled-in estimate when the manifest cannot be read.
    /// Reporting zero instead would make every context length appear to fit.
    public static func residentBytesOrEstimate(atModelDirectory modelDirectory: URL) -> UInt64 {
        residentBytes(atModelDirectory: modelDirectory)
            ?? AppMemoryBudget.residentWeightEstimateBytes
    }
}
