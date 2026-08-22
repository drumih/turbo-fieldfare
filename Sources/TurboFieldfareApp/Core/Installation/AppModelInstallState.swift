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
    case activating
    case cancelling
    case discarding
    case cancelled
    case readyToActivate(URL)
    case recoverable(String)
    case installed(modelDirectory: URL)
    case failed(String)

    public var isInstalling: Bool {
        switch self {
        case .checking, .downloadingMetadata, .planning, .reservingOutput,
             .copyingPayload, .hashingOutput, .finalizing, .activating,
             .cancelling, .discarding:
            return true
        case .idle, .cancelled, .readyToActivate, .recoverable, .installed, .failed:
            return false
        }
    }

    public var canCancel: Bool {
        switch self {
        // `.activating` is cancellable because the only long phase in it is the
        // hash of the companion weights, which happens before anything is
        // renamed: abandoning it leaves the prepared pack exactly as it was.
        case .checking, .downloadingMetadata, .planning, .reservingOutput,
             .copyingPayload, .hashingOutput, .finalizing, .activating:
            return true
        case .idle, .cancelling, .discarding, .cancelled, .readyToActivate,
             .recoverable, .installed, .failed:
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
    case readyToActivate(URL)
    case installed(URL)
}
