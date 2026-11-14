import Foundation
import TinyTitanFormat

/// The generic install-manifest reader (7.0 spec §1.3): it reads `manifest.json`
/// and checks everything the *format* guarantees — magic, version, known flags,
/// page-aligned expert stride, the wire-level codec rules — without validating
/// against any model's architecture. A variant reads its manifest through here
/// and then validates the geometry against its own contract; the runtime's
/// model-specific cross-validation lives in `ManifestReader`, not here.
public enum SSDAIManifestReader {
    /// The metadata ceiling shared by the install readers: a manifest is tiny,
    /// so a corrupt or hostile file cannot allocate without limit.
    public static let defaultMaxBytes: UInt64 = 64 * 1024 * 1024

    /// Read and format-validate the manifest of an install directory. No model
    /// architecture is assumed or checked.
    public static func read(
        directoryURL: URL,
        maxBytes: UInt64 = defaultMaxBytes
    ) throws -> Manifest {
        let directory = try SSDAIModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let manifest: Manifest
        do {
            let wire = try SSDAIManifestCodec.decodeUnchecked(data)
            guard SSDAIFormatV1.isSupportedMagic(wire.magic) else {
                throw ModelError.notASSDAIDirectory
            }
            guard wire.versionMajor == SSDAIFormatV1.versionMajor,
                wire.versionMinor >= 0
            else {
                throw ModelError.unsupportedVersion(
                    major: wire.versionMajor,
                    minor: wire.versionMinor)
            }
            for key in wire.flags.keys where !SSDAIFormatV1.knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
            if wire.expertStride % SSDAIFormatV1.alignmentBytes != 0 {
                throw ModelError.expertStrideNotPageAligned(
                    stride: wire.expertStride,
                    pageSize: Int(SSDAIFormatV1.alignmentBytes))
            }
            try SSDAIManifestCodec.validate(wire)
            manifest = Manifest(wire: wire)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        return manifest
    }

    /// The raw family string the manifest declares, if any. A variant compares
    /// this against its own family; the generic layer does not interpret it.
    public static func declaredFamily(
        directoryURL: URL,
        maxBytes: UInt64 = defaultMaxBytes
    ) throws -> String? {
        let directory = try SSDAIModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        let wire = try SSDAIManifestCodec.decodeUnchecked(data)
        return wire.arch.family
    }
}
