import Foundation
import TurboFieldfareRepackCore

public enum AppModelInstallState: Equatable, Sendable {
    case idle
    case checking
    case downloadingMetadata
    case planning
    case reservingOutput
    case copyingPayload(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    )
    case hashingOutput(String)
    case finalizing
    case cancelling
    case discarding
    case cancelled
    case recoverable(String)
    case installed(modelDirectory: URL)
    case failed(String)

    public var isInstalling: Bool {
        switch self {
        case .checking, .downloadingMetadata, .planning, .reservingOutput,
             .copyingPayload, .hashingOutput, .finalizing, .cancelling, .discarding:
            return true
        case .idle, .cancelled, .recoverable, .installed, .failed:
            return false
        }
    }

    public var canCancel: Bool {
        switch self {
        case .checking, .downloadingMetadata, .planning, .reservingOutput,
             .copyingPayload, .hashingOutput, .finalizing:
            return true
        case .idle, .cancelling, .discarding, .cancelled, .recoverable, .installed, .failed:
            return false
        }
    }
}

public enum AppModelInstallEvent: Equatable, Sendable {
    case checking
    case downloadingMetadata
    case planning
    case reservingOutput
    case copyingPayload(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    )
    case hashingOutput(String)
    case finalizing
    case installed(URL)
}
