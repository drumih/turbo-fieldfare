import Foundation
import TinyTitanFormat
import TinyTitanCore

public typealias ModelIntegrityPolicy = TinyTitanCore.ModelIntegrityPolicy

extension ModelIntegrityPolicy {
    /// Strongest policy that skips re-hashing the full payload: when the
    /// installer's `verified-install.json` receipt is present, the per-file
    /// digests were pinned at install time, so this trusts them rather than
    /// re-reading the weights.
    ///
    /// What the check actually is, because the difference matters when deciding
    /// whether to trust a directory: `validate` compares the receipt against the
    /// **manifest** — the file set, and each file's recorded size and digest, plus
    /// the manifest's own size and digest — so an edited or substituted manifest
    /// is caught, and the receipt is bound to the absolute install path so a moved
    /// directory is caught. It reads no files, so it cannot see a payload edited
    /// in place with its length preserved. Under this policy the only on-disk size
    /// checks are the loader's own, for `model_weights.bin`, `layout.json` and
    /// `packed_experts/*.bin`; tokenizer and sidecar files (including
    /// `ple_constants.json`) are not checked on disk at all.
    ///
    /// Without a receipt, fall back to a full SHA-256 of everything.
    public static func resolved(directoryURL: URL) -> ModelIntegrityPolicy {
        let receiptURL =
            directoryURL
            .appendingPathComponent(VerifiedInstallReceiptReader.fileName)
        return FileManager.default.fileExists(atPath: receiptURL.path)
            ? .sizeCheckTrustedReceipt
            : .fullSha256
    }
}

public struct VerifiedInstallReceipt: Codable, Equatable, Sendable {
    public struct FileEntry: Codable, Equatable, Sendable {
        public let size: UInt64
        public let sha256: String

        public init(size: UInt64, sha256: String) {
            self.size = size
            self.sha256 = sha256
        }
    }

    public let schemaVersion: Int
    public let manifestSha256: String
    public let modelDirectoryPath: String
    public let sourceRepoID: String?
    public let sourceRevision: String?
    public let verificationTimestamp: String
    public let toolVersion: String
    public let files: [String: FileEntry]

    public init(
        schemaVersion: Int = 1,
        manifestSha256: String,
        modelDirectoryPath: String,
        sourceRepoID: String? = nil,
        sourceRevision: String? = nil,
        verificationTimestamp: String,
        toolVersion: String,
        files: [String: FileEntry]
    ) {
        self.schemaVersion = schemaVersion
        self.manifestSha256 = manifestSha256
        self.modelDirectoryPath = modelDirectoryPath
        self.sourceRepoID = sourceRepoID
        self.sourceRevision = sourceRevision
        self.verificationTimestamp = verificationTimestamp
        self.toolVersion = toolVersion
        self.files = files
    }
}

public enum VerifiedInstallReceiptReader {
    public static let fileName = SSDAIInstallFileNames.verifiedInstallReceipt
    /// The receipt itself is small (one entry per install file), but it is read
    /// and validated beside the manifest at load, so it shares that document's
    /// bound rather than carrying a second, unrelated one that can drift below
    /// it. See `ManifestReader.defaultMaxBytes`.
    public static let defaultMaxBytes: UInt64 = ManifestReader.defaultMaxBytes

    public static func load(
        directoryURL: URL,
        maxBytes: UInt64 = defaultMaxBytes
    ) throws -> VerifiedInstallReceipt {
        let url = directoryURL.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName) is missing")
        }
        do {
            // K17, with the bound applied where it can still prevent the cost.
            // Reading first and checking the count afterwards verified the bytes
            // that were used, but `Data(contentsOf:)` had already materialized
            // the whole document — measured at +2,049 MB of process footprint for
            // a 2 GiB sparse file. One fd is held from open through fstat through
            // the read, so the size checked is the size read *and* an over-bound
            // document is refused before anything is allocated.
            let data = try BoundedMetadataRead.read(fileAt: url, maxBytes: maxBytes)
            return try JSONDecoder().decode(VerifiedInstallReceipt.self, from: data)
        } catch ModelError.metadataOverBound(let document, let bytes, let cap) {
            throw ModelError.trustedReceiptInvalid(
                detail: "\(document) size \(bytes) exceeds metadata cap \(cap)")
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)")
        }
    }

    public static func validate(
        _ receipt: VerifiedInstallReceipt,
        directoryURL: URL,
        manifest: Manifest,
        manifestSha256: String,
        manifestSize: UInt64
    ) throws {
        try validateManifestBinding(
            receipt,
            directoryURL: directoryURL,
            manifestSha256: manifestSha256)
        var expectedFiles = Set(manifest.files.keys)
        expectedFiles.insert("manifest.json")
        let receiptFiles = Set(receipt.files.keys)
        guard receiptFiles == expectedFiles else {
            let missing = expectedFiles.subtracting(receiptFiles).sorted()
            let extra = receiptFiles.subtracting(expectedFiles).sorted()
            throw ModelError.trustedReceiptInvalid(
                detail: "receipt file set mismatch missing=\(missing) extra=\(extra)")
        }
        guard let manifestReceiptEntry = receipt.files["manifest.json"] else {
            throw ModelError.trustedReceiptInvalid(detail: "receipt missing manifest.json")
        }
        guard manifestReceiptEntry.size == manifestSize else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json size mismatch")
        }
        guard manifestReceiptEntry.sha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json SHA mismatch")
        }

        for (rel, manifestEntry) in manifest.files {
            guard let receiptEntry = receipt.files[rel] else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt missing \(rel)")
            }
            guard receiptEntry.size == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt size mismatch for \(rel)")
            }
            guard receiptEntry.sha256.lowercased() == manifestEntry.sha256.lowercased() else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt SHA mismatch for \(rel)")
            }
        }
    }

    /// The integrity half of the binding: does this receipt describe *this*
    /// manifest? Separated from the path binding because the two failures mean
    /// different things — this one says the payload is not what was attested,
    /// which is unrecoverable without a reinstall.
    public static func validateManifestIntegrity(
        _ receipt: VerifiedInstallReceipt,
        manifestSha256: String
    ) throws {
        guard receipt.schemaVersion == 1 else {
            throw ModelError.trustedReceiptInvalid(
                detail: "unsupported schemaVersion \(receipt.schemaVersion)")
        }
        guard receipt.manifestSha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest SHA mismatch")
        }
    }

    /// The location half: was this receipt issued for this exact directory?
    ///
    /// A false result means the model was moved or renamed. The payload is
    /// intact — only the attestation needs re-issuing — so callers that can
    /// offer that should check this separately rather than treating it as
    /// corruption.
    public static func pathBindingMatches(
        _ receipt: VerifiedInstallReceipt,
        directoryURL: URL
    ) -> Bool {
        receipt.modelDirectoryPath == directoryURL.standardizedFileURL.path
    }

    public static func validateManifestBinding(
        _ receipt: VerifiedInstallReceipt,
        directoryURL: URL,
        manifestSha256: String
    ) throws {
        try validateManifestIntegrity(receipt, manifestSha256: manifestSha256)

        // A path mismatch is the expected outcome of moving or renaming an
        // installed model, not evidence of tampering — and it is a hard error
        // with no fallback, so the message has to carry the recovery. Without
        // it the only visible option is a multi-gigabyte reinstall.
        let actualPath = directoryURL.standardizedFileURL.path
        guard pathBindingMatches(receipt, directoryURL: directoryURL) else {
            throw ModelError.trustedReceiptInvalid(
                detail: """
                    model directory mismatch: the receipt was issued for \
                    \(receipt.modelDirectoryPath) but the model is now at \
                    \(actualPath). Moving or renaming an installed model \
                    invalidates its receipt. Re-issue it in place (re-hashes \
                    the payload, no re-download) with:
                      swift run -c release TinyTitanRepack --verify-install \
                    --input-ssdai \(actualPath)
                    """)
        }
    }
}
