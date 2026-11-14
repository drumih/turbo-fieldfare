import Foundation

// Failure modes for the validation gates in `Model.load`.
//
// Split out of `ModelTypes.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion.
/// Failure modes for the validation gates in `Model.load`.
///
/// `package` rather than internal: `TinyTitanLib` sits above this target and is
/// the layer that turns these into the facade's typed errors, so it has to be
/// able to match on them. It stays out of any product.
package enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notASSDAIDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case unsupportedArchitecture(detail: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    /// A metadata document is larger than the bound its reader states, so it was
    /// refused **before** its bytes were allocated. The wording is the one
    /// the catalog and ANE prefill paths already use for the same refusal.
    case metadataOverBound(document: String, bytes: Int, cap: UInt64)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)
    case expertCacheUnplaceable(detail: String)
    /// A Metal command buffer reported `.error`; the GPU work it carried
    /// (decode layer, head, or routed-expert pass) did not complete.
    case commandBufferFailed(detail: String)
    /// A runtime invariant the code believes is impossible was violated
    /// (arch/kernel mismatch, pipeline state corruption). Thrown instead of
    /// trapping so generation fails loudly without crashing the process.
    case internalInconsistency(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model directory at \(p) is missing manifest.json"
        case .notASSDAIDirectory:
            return "manifest.json magic is neither \"SSDAI\" nor the legacy \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .unsupportedArchitecture(let detail):
            return "unsupported architecture: \(detail)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.ssdai is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .metadataOverBound(let document, let bytes, let cap):
            return "\(document) is \(bytes) bytes, over the \(cap)-byte metadata bound"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        case .expertCacheUnplaceable(let detail):
            return "expert cache cannot place requested experts: \(detail)"
        case .commandBufferFailed(let detail):
            return "Metal command buffer failed: \(detail)"
        case .internalInconsistency(let detail):
            return "internal inconsistency: \(detail)"
        }
    }
}
