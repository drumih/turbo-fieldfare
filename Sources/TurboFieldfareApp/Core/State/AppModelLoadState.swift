import Foundation

public enum AppModelLoadPhase: Equatable, Sendable {
    case validatingDirectory
    case tokenizer
    case verifyingWeights
    case preparingRunner

    public var label: String {
        switch self {
        case .validatingDirectory: return "Validating model directory"
        case .tokenizer: return "Loading tokenizer"
        case .verifyingWeights: return "Verifying weights"
        case .preparingRunner: return "Preparing runner"
        }
    }
}

public enum AppModelLoadState: Equatable, Sendable {
    case notLoaded
    case loading(AppModelLoadPhase)
    case cancelling
    case unloading
    case ready(modelDirectory: URL, loadSeconds: Double)
    case failed(AppInferenceError)

    public var isLoading: Bool {
        switch self {
        case .loading, .cancelling, .unloading: return true
        default: return false
        }
    }

    public var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    public var isFailed: Bool {
        if case .failed = self { return true }
        return false
    }
}

public enum AppGenerationPhase: String, Equatable, Sendable {
    case idle
    case prefill
    case decode
}
