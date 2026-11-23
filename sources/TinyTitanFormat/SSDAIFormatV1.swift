import Foundation

package enum SSDAIFormatV1 {
    /// The magic this format writes: `.ssdai`, renamed from `.ssdai` in 5.15.
    package static let magic = "SSDAI"

    /// The magic every install built before the rename carries.
    ///
    /// The rename moved a name, not a byte of payload: the manifest's magic is
    /// the only field that changed, and rewriting it would invalidate the
    /// hash-bound receipt (`verified-install.json` binds the manifest's digest
    /// and the directory path) for every existing install — 244 GB here alone.
    /// So reads accept both magics for one release and writes emit `magic`.
    package static let legacyMagic = "GTURBO"

    /// Whether a manifest's magic is one this runtime reads.
    package static func isSupportedMagic(_ value: String) -> Bool {
        value == magic || value == legacyMagic
    }

    package static let versionMajor = 1
    package static let versionMinor = 1

    /// The manifest `versionMinor` from which an architecture that has
    /// extension geometry (hyper-connections, the QSA indexer, the n-gram
    /// block) must *declare* it, rather than merely be checked when it does.
    ///
    /// 1.0 made every one of those fields optional so installs written before
    /// the families existed stayed loadable, and a reader that treats an absent
    /// field as agreement validates nothing: the guard only bites on a manifest
    /// that happens to carry the keys. 1.1 closes that by making the block
    /// mandatory for a family whose runtime contract has the geometry, while
    /// still reading 1.0 manifests — including the ones already installed, no
    /// receipt of which may be rewritten in place.
    package static let extensionGeometryMandatoryFromMinor = 1

    package static let alignmentBytes: UInt64 = 16_384
    package static let residentHeaderBytes = 24
    package static let residentEntryBytes = 72
    package static let residentIndexMaxBytes: UInt64 = 32 * 1024 * 1024
    /// Ceiling for `packed_experts/layout.json`, which is a different artifact
    /// from the resident index and scales with experts x layers rather than
    /// with resident tensors.
    ///
    /// 128 MiB. Qwen 3.6 is ~22 MB (40 x 256 = 10,240 entries) and
    /// Qwen3.8-Flash-Next is ~54 MB (48 x 512 = 24,576), so the old shared
    /// 32 MiB bound rejected a correct install. Kept separate so raising it
    /// does not also loosen the resident index, which stays small (a few
    /// hundred KB) for every family.
    package static let packedExpertsLayoutMaxBytes: UInt64 = 128 * 1024 * 1024

    package static let knownFlags: Set<String> = [
        "streamingPresent", "turboQuantKV", "aneSharedExpert",
    ]

    package enum DType: UInt8, Sendable {
        case u32 = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3
        /// Block encodings stored as their source wrote them
        /// (`SSDAIBlockEncodingV1`); the index checks their byte counts.
        case ggufQ8_0 = 4
        case ggufNVFP4 = 5
        case nvfp4 = 6
        case mxfp8 = 7
    }
}

package enum TinyTitanFormatError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalid(field: String, reason: String)
    case overflow(field: String)
    case truncated(field: String)

    package var description: String {
        switch self {
        case .invalid(let field, let reason): "\(field): \(reason)"
        case .overflow(let field): "\(field): arithmetic overflow"
        case .truncated(let field): "\(field): truncated"
        }
    }
}

@inline(__always)
package func ssdaiCheckedAdd(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else { throw TinyTitanFormatError.overflow(field: field) }
    return value
}

@inline(__always)
package func ssdaiCheckedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else { throw TinyTitanFormatError.overflow(field: field) }
    return value
}

package enum SSDAIPathValidator {
    package static func appleFilesystemKey(_ path: String) -> String {
        path.precomposedStringWithCanonicalMapping
            .lowercased(with: Locale(identifier: "en_US_POSIX"))
    }

    package static func validateRelativePath(_ path: String, field: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\0") else {
            throw TinyTitanFormatError.invalid(field: field, reason: "unsafe relative path")
        }
        let components = path.components(separatedBy: "/")
        guard components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw TinyTitanFormatError.invalid(field: field, reason: "non-canonical path")
        }
        let normalized = NSString.path(withComponents: components)
        guard normalized == path else {
            throw TinyTitanFormatError.invalid(field: field, reason: "non-normalized path")
        }
    }

    package static func validateBasename(_ name: String, field: String) throws {
        try validateRelativePath(name, field: field)
        guard !name.contains("/") else {
            throw TinyTitanFormatError.invalid(field: field, reason: "expected basename")
        }
    }
}

/// The two names an install directory reserves at its root.
///
/// `verified-install.json` is the trust anchor and it has three declarations:
/// the repacker writes it, the engine reads it, and the manifest validator
/// refuses either name as a payload file so a checkpoint cannot smuggle one in.
/// One string here rather than one per side, because a one-sided edit to the
/// receipt's name is a silent contract break.
public enum SSDAIInstallFileNames {
    public static let manifest = "manifest.json"
    public static let verifiedInstallReceipt = "verified-install.json"

    /// Names the manifest may never declare among its payload files.
    public static let reserved: Set<String> = [manifest, verifiedInstallReceipt]
}
