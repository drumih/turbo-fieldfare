import Foundation
import TurboFieldfare

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    /// `descriptor` pins the checkpoint the installation must match. When it
    /// is nil the probe derives the expectation from the family the manifest
    /// itself declares, so a multi-model app validates whichever model is
    /// actually installed rather than whichever one is currently selected.
    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor? = nil
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        do {
            let family = try ManifestReader.peekFamily(directoryURL: directory)
            guard let baseline = ArchConfig.knownArchitectures[family] else {
                return .partial("unknown model family \(family.rawValue)")
            }
            let manifest = try ManifestReader.load(directoryURL: directory, expecting: baseline)
            // Validate the checkpoint against the descriptor for the family the
            // manifest itself declares, so the probe does not depend on which
            // model the app happens to have selected.
            guard let expected = descriptor
                    ?? AppModelInstallDescriptor.descriptor(for: family) else {
                return .partial("no descriptor for family \(family.rawValue)")
            }
            let expectedSource = "sha256:" + expected.sourceIndexSHA256
            guard manifest.sourceSnapshotHash == expectedSource else {
                return .partial("installed checkpoint does not match \(expected.displayName)")
            }
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard FileManager.default.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}
