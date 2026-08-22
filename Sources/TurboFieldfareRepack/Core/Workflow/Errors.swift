import Foundation

/// Who a refused lock is waiting on. The pack's use lease is *shared*, so a
/// blocked mutation is as likely to be waiting on a loaded model as on another
/// install, and saying "another installer" sent users hunting for a process
/// that does not exist.
public enum RepackBusyHolder: Sendable, Equatable {
    case installer
    case installerOrLoadedModel
}

public enum RepackError: Error, CustomStringConvertible {
    case fileOpenFailed(path: String, errno: Int32)
    case fileStatFailed(path: String, errno: Int32)
    case ftruncateFailed(path: String, errno: Int32)
    case pwriteShort(path: String, expected: Int, wrote: Int, errno: Int32)
    case preadShort(path: String, expected: Int, got: Int, errno: Int32)
    case mmapFailed(path: String, errno: Int32)
    case renameFailed(from: String, to: String, errno: Int32)
    case fsyncFailed(path: String, errno: Int32)
    case mkdirFailed(path: String, errno: Int32)
    case installBusy(path: String, holder: RepackBusyHolder = .installer)
    case installPathUnsafe(path: String, detail: String)
    case installStateMissing(path: String)
    case installStateCorrupt(path: String, detail: String)
    case installStateIncompatible(detail: String)
    case promotionUnsupported(path: String, errno: Int32)

    case safetensorsHeaderTooLarge(path: String, size: UInt64)
    case safetensorsHeaderInvalid(path: String, detail: String)
    case safetensorsUnknownDtype(path: String, dtype: String)
    case safetensorsTensorOutOfRange(path: String, name: String, end: UInt64, fileSize: UInt64)

    case indexJsonInvalid(path: String, detail: String)
    case configJsonInvalid(path: String, detail: String)
    case quantOverrideCountMismatch(expected: Int, actual: Int, sample: [String])

    case missingTensor(name: String)
    case unknownTensorPrefix(name: String)
    case missingScalesCompanion(name: String)
    case missingBiasesCompanion(name: String)
    case shapeMismatch(name: String, detail: String)
    case dtypeMismatch(name: String, detail: String)

    case sourceFingerprintRejected(path: String, sha256: String)

    case remoteProtocolInvalid(detail: String)
    case remoteHTTPStatus(url: String, status: Int)
    case remoteHTTPResponse(url: String, status: Int, retryAfter: String?)
    case remoteBodyTruncated(path: String, expected: UInt64, actual: UInt64)
    case remoteBodyExceeded(path: String, limit: UInt64, attempted: UInt64)
    case remoteRedirectRejected(url: String, detail: String)
    case remoteRetryDelayExceeded(value: String, capSeconds: UInt64)
    case remoteFileTooLarge(path: String, size: UInt64, cap: UInt64)
    case diskSpaceInsufficient(path: String, required: UInt64, available: UInt64)

    case scratchExceeded(requested: Int, limit: Int)
    case testHookStop(stage: String)
    case configurationInvalid(detail: String)

    public var description: String {
        switch self {
        case .fileOpenFailed(let p, let e):     return "open(\(p)) failed: errno \(e)"
        case .fileStatFailed(let p, let e):     return "fstat(\(p)) failed: errno \(e)"
        case .ftruncateFailed(let p, let e):    return "ftruncate(\(p)) failed: errno \(e)"
        case .pwriteShort(let p, let exp, let got, let e):
            return "pwrite(\(p)) short: expected \(exp), wrote \(got), errno \(e)"
        case .preadShort(let p, let exp, let got, let e):
            return "pread(\(p)) short: expected \(exp), got \(got), errno \(e)"
        case .mmapFailed(let p, let e):         return "mmap(\(p)) failed: errno \(e)"
        case .renameFailed(let a, let b, let e):return "rename(\(a) -> \(b)) failed: errno \(e)"
        case .fsyncFailed(let p, let e):        return "fsync(\(p)) failed: errno \(e)"
        case .mkdirFailed(let p, let e):        return "mkdir(\(p)) failed: errno \(e)"
        case .installBusy(let p, let holder):
            switch holder {
            case .installer:
                return "another installer holds \(p)"
            case .installerOrLoadedModel:
                return "\(p) is in use: another installer or a loaded model "
                    + "holds this pack; unload the model or wait for the "
                    + "install to finish"
            }
        case .installPathUnsafe(let p, let d):
            return "unsafe installer path \(p): \(d)"
        case .installStateMissing(let p):
            return "no resumable install state exists for \(p)"
        case .installStateCorrupt(let p, let d):
            return "install state at \(p) is corrupt: \(d)"
        case .installStateIncompatible(let d):
            return "install state is incompatible: \(d)"
        case .promotionUnsupported(let p, let e):
            return "atomic directory promotion is unsupported for \(p): errno \(e)"
        case .safetensorsHeaderTooLarge(let p, let s):
            return "safetensors header at \(p) size \(s) exceeds bound"
        case .safetensorsHeaderInvalid(let p, let d):
            return "safetensors header at \(p) invalid: \(d)"
        case .safetensorsUnknownDtype(let p, let d):
            return "safetensors at \(p) has unsupported dtype \(d)"
        case .safetensorsTensorOutOfRange(let p, let n, let end, let sz):
            return "safetensors \(p): tensor \(n) ends at \(end), file size \(sz)"
        case .indexJsonInvalid(let p, let d): return "index.json \(p) invalid: \(d)"
        case .configJsonInvalid(let p, let d): return "config.json \(p) invalid: \(d)"
        case .quantOverrideCountMismatch(let exp, let got, let sample):
            return "config.json quantization overrides: expected \(exp), got \(got); sample=\(sample.prefix(5))"
        case .missingTensor(let n): return "expected tensor missing: \(n)"
        case .unknownTensorPrefix(let n): return "unknown tensor prefix: \(n)"
        case .missingScalesCompanion(let n): return "quantized tensor \(n) missing .scales companion"
        case .missingBiasesCompanion(let n): return "quantized tensor \(n) missing .biases companion"
        case .shapeMismatch(let n, let d): return "shape mismatch for \(n): \(d)"
        case .dtypeMismatch(let n, let d): return "dtype mismatch for \(n): \(d)"
        case .sourceFingerprintRejected(let p, let s):
            return "source fingerprint \(s) for \(p) is not in known set"
        case .remoteProtocolInvalid(let d):
            return "remote protocol invalid: \(d)"
        case .remoteHTTPStatus(let u, let s):
            return "remote HTTP \(s): \(u)"
        case .remoteHTTPResponse(let u, let s, let retryAfter):
            if let retryAfter {
                return "remote HTTP \(s): \(u), Retry-After \(retryAfter)"
            }
            return "remote HTTP \(s): \(u)"
        case .remoteBodyTruncated(let p, let e, let a):
            return "remote body for \(p) was truncated: expected \(e), received \(a)"
        case .remoteBodyExceeded(let p, let l, let a):
            return "remote body for \(p) exceeded \(l) bytes at \(a) bytes"
        case .remoteRedirectRejected(let u, let d):
            return "remote redirect rejected for \(u): \(d)"
        case .remoteRetryDelayExceeded(let v, let c):
            return "remote Retry-After \(v) exceeds \(c)-second cap"
        case .remoteFileTooLarge(let p, let s, let c):
            return "remote file \(p) size \(s) exceeds cap \(c)"
        case .diskSpaceInsufficient(let p, let r, let a):
            return "insufficient disk space for \(p): required \(r), available \(a)"
        case .scratchExceeded(let r, let l):
            return "resident index size \(r) exceeds limit \(l)"
        case .testHookStop(let s): return "test hook stop at stage \(s)"
        case .configurationInvalid(let d): return "configuration invalid: \(d)"
        }
    }
}
