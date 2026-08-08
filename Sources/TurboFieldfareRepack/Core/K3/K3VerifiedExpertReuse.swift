import Foundation
import TurboFieldfareFormat

/// A verified-install-backed source for copy-on-write reuse of K3's immutable
/// MXFP4 expert layer files. Validation is intentionally metadata-only: it
/// applies the same trust boundary as runtime `trusted-install`, then relies
/// on APFS clonefile to preserve the referenced bytes exactly.
struct K3VerifiedExpertReuse {
    private struct Receipt: Decodable {
        struct FileEntry: Decodable {
            let size: UInt64
            let sha256: String
        }

        let schemaVersion: Int
        let manifestSha256: String
        let modelDirectoryPath: String
        let sourceRepoID: String?
        let sourceRevision: String?
        let files: [String: FileEntry]
    }

    let sourceDirectory: String
    let expertFiles: [RepackAudit.OutputFile]
    let logicalBytes: UInt64

    static func load(sourceDirectory requestedSource: String,
                     plan: K3RepackPlan,
                     snapshot: K3Snapshot,
                     repoID: String) throws -> K3VerifiedExpertReuse {
        let sourceURL = URL(fileURLWithPath: requestedSource).standardizedFileURL
        let sourceDirectory = sourceURL.path
        guard try Posix.entryKind(sourceDirectory) == .directory else {
            throw RepackError.installPathUnsafe(
                path: sourceDirectory,
                detail: "expert reuse source is not a directory")
        }

        let manifestPath = (sourceDirectory as NSString)
            .appendingPathComponent("manifest.json")
        let manifestData = try Posix.readBoundedData(
            manifestPath, maximumBytes: 16 * 1024 * 1024)
        let manifest: GTurboManifestV2
        do {
            manifest = try GTurboManifestCodecV2.decode(manifestData)
        } catch {
            throw RepackError.installStateCorrupt(
                path: manifestPath, detail: "cannot decode v2 manifest: \(error)")
        }

        var manifestHasher = Sha256Stream()
        manifestData.withUnsafeBytes { manifestHasher.update($0) }
        let manifestSHA = manifestHasher.finalizeHexString()
        let receiptPath = (sourceDirectory as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        let receiptData = try Posix.readBoundedData(
            receiptPath, maximumBytes: 4 * 1024 * 1024)
        let receipt: Receipt
        do {
            receipt = try JSONDecoder().decode(Receipt.self, from: receiptData)
        } catch {
            throw RepackError.installStateCorrupt(
                path: receiptPath, detail: "cannot decode verified receipt: \(error)")
        }

        guard receipt.schemaVersion == 1,
              receipt.manifestSha256.lowercased() == manifestSHA.lowercased(),
              receipt.modelDirectoryPath == sourceDirectory else {
            throw RepackError.installStateCorrupt(
                path: receiptPath,
                detail: "receipt is not bound to the source manifest and directory")
        }
        var expectedReceiptFiles = Set(manifest.files.keys)
        expectedReceiptFiles.insert("manifest.json")
        guard Set(receipt.files.keys) == expectedReceiptFiles,
              let receiptManifest = receipt.files["manifest.json"],
              receiptManifest.size == UInt64(manifestData.count),
              receiptManifest.sha256.lowercased() == manifestSHA.lowercased() else {
            throw RepackError.installStateCorrupt(
                path: receiptPath, detail: "receipt file set or manifest entry mismatch")
        }
        for (relativePath, entry) in manifest.files {
            guard let recorded = receipt.files[relativePath],
                  recorded.size == entry.size,
                  recorded.sha256.lowercased() == entry.sha256.lowercased() else {
                throw RepackError.installStateCorrupt(
                    path: receiptPath,
                    detail: "receipt entry mismatch for \(relativePath)")
            }
        }

        guard manifest.modelID == repoID,
              receipt.sourceRepoID == repoID,
              receipt.sourceRevision == snapshot.resolvedCommit,
              manifest.sourceSnapshotHash == "sha256:" + snapshot.indexSha256Hex,
              manifest.arch == plan.manifestArch,
              manifest.flags == KimiK3FormatProfile.flags,
              manifest.numLayers == plan.arch.numLayers,
              manifest.expertsPerLayer == (plan.layers.first?.expertsPerLayer ?? 0),
              manifest.expertStride == (plan.layers.first?.expertStride ?? 0) else {
            throw RepackError.installStateIncompatible(
                detail: "expert reuse source does not match the pinned source and K3 plan")
        }

        let targetQuant = plan.trunkQuant == .int4
            ? KimiK3FormatProfile.quantInt4
            : KimiK3FormatProfile.quantInt8
        guard manifest.quant != targetQuant else {
            throw RepackError.installStateIncompatible(
                detail: "expert reuse source already uses the requested trunk quantization")
        }
        guard manifest.quant == KimiK3FormatProfile.quantInt4
                || manifest.quant == KimiK3FormatProfile.quantInt8 else {
            throw RepackError.installStateIncompatible(
                detail: "expert reuse source has an unsupported or mixed quant profile")
        }

        // The layout is a pure function of the architecture and expert plan.
        // Matching its expected bytes pins filenames, offsets and subtensors
        // without parsing another ~135 MB copy during a real upgrade.
        let expectedLayout = try K3GTurboJSON.encodeLayout(plan: plan)
        var layoutHasher = Sha256Stream()
        expectedLayout.withUnsafeBytes { layoutHasher.update($0) }
        let expectedLayoutSHA = layoutHasher.finalizeHexString()
        guard let layoutEntry = manifest.files["packed_experts/layout.json"],
              layoutEntry.size == UInt64(expectedLayout.count),
              layoutEntry.sha256.lowercased() == expectedLayoutSHA.lowercased() else {
            throw RepackError.installStateIncompatible(
                detail: "expert reuse source layout differs from the target plan")
        }

        let plannedRelativePaths = plan.layers.map {
            "packed_experts/" + ($0.path as NSString).lastPathComponent
        }
        let sourceRelativePaths = manifest.files.keys.filter {
            $0.hasPrefix("packed_experts/layer_") && $0.hasSuffix(".bin")
        }
        guard Set(sourceRelativePaths) == Set(plannedRelativePaths) else {
            throw RepackError.installStateIncompatible(
                detail: "expert reuse source layer-file set differs from the target plan")
        }

        var expertFiles: [RepackAudit.OutputFile] = []
        var logicalBytes: UInt64 = 0
        for (layer, relativePath) in zip(plan.layers, plannedRelativePaths) {
            guard let entry = manifest.files[relativePath], entry.size == layer.fileSize else {
                throw RepackError.installStateIncompatible(
                    detail: "expert reuse source size mismatch for \(relativePath)")
            }
            let path = (sourceDirectory as NSString).appendingPathComponent(relativePath)
            let descriptor = try Posix.openReadNoFollow(path)
            let actualSize = try Posix.fileSize(fd: descriptor, path: path)
            close(descriptor)
            guard actualSize == entry.size else {
                throw RepackError.installStateCorrupt(
                    path: path,
                    detail: "size \(actualSize) does not match verified entry \(entry.size)")
            }
            expertFiles.append(.init(relativePath: relativePath,
                                     size: entry.size,
                                     sha256: entry.sha256.lowercased()))
            logicalBytes += entry.size
        }
        expertFiles.sort { $0.relativePath < $1.relativePath }
        return K3VerifiedExpertReuse(sourceDirectory: sourceDirectory,
                                     expertFiles: expertFiles,
                                     logicalBytes: logicalBytes)
    }

    func cloneExperts(into plan: K3RepackPlan) throws {
        let destinationDirectory = (plan.resident.path as NSString)
            .deletingLastPathComponent
        let packedDirectory = (destinationDirectory as NSString)
            .appendingPathComponent("packed_experts")
        for file in expertFiles {
            try Task.checkCancellation()
            let source = (sourceDirectory as NSString)
                .appendingPathComponent(file.relativePath)
            let destination = (destinationDirectory as NSString)
                .appendingPathComponent(file.relativePath)
            try Posix.cloneFile(from: source, to: destination)
        }
        try Posix.fsyncDirectory(packedDirectory)
    }

    func recordFiles(in audit: RepackAudit, destinationRoot: String) throws {
        for file in expertFiles {
            let path = (destinationRoot as NSString)
                .appendingPathComponent(file.relativePath)
            let descriptor = try Posix.openReadNoFollow(path)
            let actualSize = try Posix.fileSize(fd: descriptor, path: path)
            close(descriptor)
            guard actualSize == file.size else {
                throw RepackError.installStateCorrupt(
                    path: path,
                    detail: "cloned size \(actualSize) does not match \(file.size)")
            }
        }
        audit.outputFiles.append(contentsOf: expertFiles)
    }
}
