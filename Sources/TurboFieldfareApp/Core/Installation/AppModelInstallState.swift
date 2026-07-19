import Foundation

public enum AppModelInstallState: Equatable, Sendable {
    case idle
    case checking
    case downloadingMetadata
    case planning
    case reservingOutput
    case copyingPayload(doneBytes: UInt64, totalBytes: UInt64)
    case hashingOutput(String)
    case finalizing
    case cancelling
    case cancelled
    case installed(modelDirectory: URL)
    case failed(String)

    public var isInstalling: Bool {
        switch self {
        case .checking, .downloadingMetadata, .planning, .reservingOutput,
             .copyingPayload, .hashingOutput, .finalizing, .cancelling:
            return true
        case .idle, .cancelled, .installed, .failed:
            return false
        }
    }

    public var canCancel: Bool {
        switch self {
        case .checking, .downloadingMetadata, .planning, .reservingOutput,
             .copyingPayload, .hashingOutput, .finalizing:
            return true
        case .idle, .cancelling, .cancelled, .installed, .failed:
            return false
        }
    }
}

public enum AppModelInstallEvent: Equatable, Sendable {
    case checking
    case downloadingMetadata
    case planning
    case reservingOutput
    case copyingPayload(doneBytes: UInt64, totalBytes: UInt64)
    case hashingOutput(String)
    case finalizing
    case installed(URL)
}
