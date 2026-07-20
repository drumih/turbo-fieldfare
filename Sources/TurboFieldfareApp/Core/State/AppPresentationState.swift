import Foundation

public enum AppPresentationSeverity: Equatable, Sendable {
    case neutral
    case active
    case success
    case warning
    case error
}

public enum AppModelAction: Equatable, Sendable {
    case install
    case cancelInstall
    case load
    case retryLoad
    case cancelLoad
    case reload
    case unload
}

public struct AppPresentationSnapshot: Equatable, Sendable {
    public var requiresInstallation: Bool
    public var installState: AppModelInstallState
    public var installReadiness: AppModelInstallReadiness
    public var loadState: AppModelLoadState
    public var hasStaleRuntime: Bool
    public var isRunning: Bool
    public var isGenerationCancellationPending: Bool
    public var generationPhase: AppGenerationPhase
    public var livePrefillDone: Int
    public var livePrefillTotal: Int
    public var lastStopReason: AppStopReason?

    public init(requiresInstallation: Bool,
                installState: AppModelInstallState,
                installReadiness: AppModelInstallReadiness,
                loadState: AppModelLoadState,
                hasStaleRuntime: Bool,
                isRunning: Bool,
                isGenerationCancellationPending: Bool,
                generationPhase: AppGenerationPhase,
                livePrefillDone: Int = 0,
                livePrefillTotal: Int = 0,
                lastStopReason: AppStopReason? = nil) {
        self.requiresInstallation = requiresInstallation
        self.installState = installState
        self.installReadiness = installReadiness
        self.loadState = loadState
        self.hasStaleRuntime = hasStaleRuntime
        self.isRunning = isRunning
        self.isGenerationCancellationPending = isGenerationCancellationPending
        self.generationPhase = generationPhase
        self.livePrefillDone = livePrefillDone
        self.livePrefillTotal = livePrefillTotal
        self.lastStopReason = lastStopReason
    }
}

public struct AppPresentationState: Equatable, Sendable {
    public var label: String
    public var detail: String?
    public var severity: AppPresentationSeverity
    public var showsActivity: Bool
    public var primaryAction: AppModelAction?
    public var secondaryAction: AppModelAction?

    public init(label: String,
                detail: String? = nil,
                severity: AppPresentationSeverity = .neutral,
                showsActivity: Bool = false,
                primaryAction: AppModelAction? = nil,
                secondaryAction: AppModelAction? = nil) {
        self.label = label
        self.detail = detail
        self.severity = severity
        self.showsActivity = showsActivity
        self.primaryAction = primaryAction
        self.secondaryAction = secondaryAction
    }

    public static func resolve(_ snapshot: AppPresentationSnapshot) -> Self {
        if snapshot.installState.isInstalling {
            if case .cancelling = snapshot.installState {
                return Self(label: "Cancelling installation",
                            severity: .active, showsActivity: true)
            }
            return Self(label: installLabel(snapshot.installState),
                        severity: .active, showsActivity: true,
                        primaryAction: snapshot.installState.canCancel ? .cancelInstall : nil)
        }

        if snapshot.requiresInstallation {
            if case .failed(let message) = snapshot.installState {
                return Self(label: "Installation failed", detail: message,
                            severity: .error, primaryAction: .install)
            }
            if case .cancelled = snapshot.installState {
                return Self(label: "Installation cancelled", severity: .warning,
                            primaryAction: .install)
            }
            if case .failed(let message) = snapshot.installReadiness {
                return Self(label: "Storage check failed", detail: message,
                            severity: .error)
            }
            if case .insufficientSpace(let requirement) = snapshot.installReadiness {
                return Self(label: "Not enough storage",
                            detail: "\(requirement.shortfallBytes) bytes more required",
                            severity: .warning)
            }
            if case .checking = snapshot.installReadiness {
                return Self(label: "Checking available space",
                            severity: .active, showsActivity: true)
            }
            if case .ready = snapshot.installReadiness {
                return Self(label: "Model required", primaryAction: .install)
            }
            return Self(label: "Model required")
        }

        switch snapshot.loadState {
        case .loading(let phase):
            return Self(label: phase.label, severity: .active, showsActivity: true,
                        primaryAction: .cancelLoad)
        case .cancelling:
            return Self(label: "Cancelling load", severity: .active, showsActivity: true)
        case .unloading:
            return Self(label: "Unloading model", severity: .active, showsActivity: true)
        case .failed(let error):
            return Self(label: "Model load failed", detail: error.userMessage,
                        severity: .error, primaryAction: .retryLoad)
        case .notLoaded, .ready:
            break
        }

        if snapshot.isRunning {
            if snapshot.isGenerationCancellationPending {
                return Self(label: "Stopping", severity: .active, showsActivity: true)
            }
            switch snapshot.generationPhase {
            case .prefill:
                let label = snapshot.livePrefillTotal > 0
                    ? "Prefill (\(snapshot.livePrefillDone)/\(snapshot.livePrefillTotal))"
                    : "Prefill"
                return Self(label: label, severity: .active)
            case .decode:
                return Self(label: "Generating", severity: .active)
            case .idle:
                return Self(label: "Starting generation", severity: .active,
                            showsActivity: true)
            }
        }

        if snapshot.hasStaleRuntime {
            return Self(label: "Reload required", severity: .warning,
                        primaryAction: .reload, secondaryAction: .unload)
        }

        if case .ready = snapshot.loadState {
            if let reason = snapshot.lastStopReason {
                return Self(label: "Done · \(reason.rawValue)", severity: .success,
                            secondaryAction: .unload)
            }
            return Self(label: "Ready", severity: .success, secondaryAction: .unload)
        }

        return Self(label: "Installed · Not loaded", primaryAction: .load)
    }

    private static func installLabel(_ state: AppModelInstallState) -> String {
        switch state {
        case .checking: return "Checking installation"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning installation"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading model"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing installation"
        case .cancelling: return "Cancelling installation"
        case .idle, .cancelled, .installed, .failed:
            return "Model required"
        }
    }
}
