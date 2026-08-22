import Darwin
import Foundation
import TurboFieldfareFormat

public struct VisionPackVerification: Sendable, Equatable {
    public let manifestSha256: String
    public let weightsBytes: UInt64
    public let sourceRevision: String
    public let sourceIndexSha256: String
    /// The text manifest this pack is bound to. Carried out so a successful
    /// verification can say *what it verified against*: two installs of the
    /// same revision are byte-identical, and without this the success line
    /// cannot distinguish which binding was checked.
    public let compatibleTextManifestSha256: String
}

public enum VisionPackVerifier {
    public static func verify(
        directory: URL,
        installedDirectory: URL? = nil,
        textModelDirectory: URL,
        verifyWeights: Bool = true,
        onHashProgress: ((UInt64, UInt64) throws -> Void)? = nil
    ) throws -> VisionPackVerification {
        let directory = directory.standardizedFileURL
        let installedDirectory = (installedDirectory ?? directory).standardizedFileURL
        let access = try GTurboDirectoryAccess(rootPath: directory.path)
        let expectedEntries = Set([
            GTurboVisionFormatV1.manifestFile,
            GTurboVisionFormatV1.weightsFile,
            GTurboVisionFormatV1.processorFile,
            GTurboVisionFormatV1.receiptFile,
        ])
        // Room for the four payload files plus every sidecar a round-trip can
        // leave beside them: `.DS_Store`, an AppleDouble `._x` per file, an
        // `.icloud` placeholder per file. The cap is what bounds the listing's
        // memory and its one fstat per entry, so it stays a small constant
        // instead of walking an arbitrarily large directory.
        let scanned = try access.relativeEntries(maxDepth: 0, maxEntries: 64)
        // Ignoring what the OS wrote beside the pack, exactly as the runtime
        // store does, so the installer and the loader agree on what a valid
        // pack directory is. A depth-0 scan yields basenames, which is what
        // isSidecarEntry matches on. Equality still rejects a missing, renamed
        // or genuinely unexpected file.
        let entries = Set(scanned.filter { !GTurboVisionFormatV1.isSidecarEntry($0) })
        guard entries == expectedEntries else {
            // Both directions, because a *missing* file used to be reported as
            // the surviving files being "unexpected", which sends the reader
            // looking for surplus entries that are not there.
            throw RepackError.configurationInvalid(
                detail: "vision companion directory does not match the v1 "
                    + "contract: " + GTurboVisionFormatV1.entryDifference(
                        expected: expectedEntries, actual: entries))
        }

        let manifestData = try access.readMetadata(
            GTurboVisionFormatV1.manifestFile,
            maxBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let receiptData = try access.readMetadata(
            GTurboVisionFormatV1.receiptFile,
            maxBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let manifest = try GTurboVisionManifestCodec.decode(manifestData)
        let receipt = try GTurboVisionReceiptCodec.decode(receiptData)

        let textManifestPath = textModelDirectory.standardizedFileURL
            .appendingPathComponent(GTurboVisionFormatV1.manifestFile).path
        let textManifestData = try Posix.readBoundedData(
            textManifestPath,
            maximumBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let textManifest = try GTurboManifestCodec.decode(textManifestData)
        guard let textSource = textManifest.sourceSnapshotHash,
              manifest.compatibleTextSourceSnapshotHash == textSource else {
            throw RepackError.installStateIncompatible(
                detail: "vision companion belongs to a different text checkpoint")
        }
        let textManifestSHA = hash(textManifestData)
        guard manifest.compatibleTextManifestSha256.lowercased() == textManifestSHA,
              receipt.compatibleTextManifestSha256.lowercased() == textManifestSHA else {
            throw RepackError.installStateIncompatible(
                detail: "vision companion text manifest binding changed")
        }

        let manifestSHA = hash(manifestData)
        let receiptDirectory = canonicalDirectoryPath(URL(
            fileURLWithPath: receipt.companionDirectoryPath,
            isDirectory: true))
        let expectedDirectory = canonicalDirectoryPath(installedDirectory)
        guard receipt.manifestSha256.lowercased() == manifestSHA else {
            throw RepackError.configurationInvalid(
                detail: "vision companion receipt manifest binding mismatch")
        }
        guard receiptDirectory == expectedDirectory else {
            throw RepackError.configurationInvalid(
                detail: "vision companion receipt directory binding mismatch")
        }
        guard receipt.sourceRepoID == manifest.modelID,
              receipt.sourceRevision == manifest.sourceRevision else {
            throw RepackError.configurationInvalid(
                detail: "vision companion receipt source binding mismatch")
        }

        let totalBytes = manifest.files.reduce(UInt64(0)) { total, entry in
            verifyWeights || entry.key != GTurboVisionFormatV1.weightsFile
                ? total + entry.value.size : total
        }
        var hashedBytes: UInt64 = 0
        for (relativePath, expected) in manifest.files {
            let actualSize = try access.fileSize(relativePath)
            guard actualSize == expected.size else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expected.size)")
            }
            guard receipt.files[relativePath] == expected else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) receipt mismatch")
            }
            if verifyWeights || relativePath != GTurboVisionFormatV1.weightsFile {
                let path = directory.appendingPathComponent(relativePath).path
                // Reported against the total the manifest declares, so the
                // caller can show a fraction rather than a bare spinner for the
                // minutes this takes over ~1.5 GB.
                let alreadyHashed = hashedBytes
                let actualSHA = try Sha256Stream.hashFile(
                    path: path, noCache: true, noFollow: true,
                    onProgress: onHashProgress.map { report in
                        { try report(alreadyHashed + $0, totalBytes) }
                    })
                hashedBytes = alreadyHashed + actualSize
                guard actualSHA == expected.sha256.lowercased() else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) hash mismatch")
                }
            }
        }
        guard let manifestReceipt = receipt.files[GTurboVisionFormatV1.manifestFile],
              manifestReceipt.size == UInt64(manifestData.count),
              manifestReceipt.sha256.lowercased() == manifestSHA else {
            throw RepackError.configurationInvalid(
                detail: "vision manifest receipt entry mismatch")
        }
        return VisionPackVerification(
            manifestSha256: manifestSHA,
            weightsBytes: manifest.files[GTurboVisionFormatV1.weightsFile]!.size,
            sourceRevision: manifest.sourceRevision,
            sourceIndexSha256: manifest.sourceIndexSha256,
            compatibleTextManifestSha256: textManifestSHA)
    }

    private static func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private static func canonicalDirectoryPath(_ url: URL) -> String {
        let standardized = url.standardizedFileURL
        return standardized.deletingLastPathComponent()
            .resolvingSymlinksInPath()
            .appendingPathComponent(standardized.lastPathComponent, isDirectory: true)
            .standardizedFileURL.path
    }
}
