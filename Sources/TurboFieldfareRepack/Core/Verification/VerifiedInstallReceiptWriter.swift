import Foundation

enum VerifiedInstallReceiptWriter {
    static let fileName = "verified-install.json"

    static func encode(outputDir: String,
                              manifestSha256: String,
                              manifestSize: UInt64,
                              sourceRepoID: String?,
                              sourceRevision: String?,
                              toolVersion: String = "TurboFieldfareRepack",
                              files: [RepackAudit.OutputFile]) throws -> Data {
        var filesDict: [String: Any] = [:]
        for file in files {
            filesDict[file.relativePath] = [
                "size": file.size,
                "sha256": file.sha256
            ]
        }
        filesDict["manifest.json"] = [
            "size": manifestSize,
            "sha256": manifestSha256
        ]

        var receipt: [String: Any] = [
            "schemaVersion": 1,
            "manifestSha256": manifestSha256,
            "modelDirectoryPath": URL(fileURLWithPath: outputDir).standardizedFileURL.path,
            "verificationTimestamp": ISO8601DateFormatter().string(from: Date()),
            "toolVersion": toolVersion,
            "files": filesDict
        ]
        if let sourceRepoID {
            receipt["sourceRepoID"] = sourceRepoID
        }
        if let sourceRevision {
            receipt["sourceRevision"] = sourceRevision
        }
        return try JSONSerialization.data(withJSONObject: receipt,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
