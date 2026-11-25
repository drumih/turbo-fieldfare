import Foundation
import Metal

/// The `TINYTITAN_TENSOROPS` switch: whether the Apple10 tensor-ops prefill
/// pipeline may be built.
package enum TensorOpsMode: String, Equatable, Sendable {
    /// Build the pipeline only where the capability is present.
    case auto
    /// Never build it.
    case off
    /// Try to build it even without the capability, and fall back on failure.
    case on

    package static let environmentVariable = "TINYTITAN_TENSOROPS"

    /// The parsed mode plus the reason it is not the plain environment value,
    /// so a typo in an experiment is visible in the startup line instead of
    /// silently reading as `auto`.
    package struct Resolution: Equatable, Sendable {
        package let mode: TensorOpsMode
        package let note: String?
    }

    package static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Resolution {
        guard let raw = environment[environmentVariable] else {
            return Resolution(mode: .auto, note: nil)
        }
        switch raw {
        case "auto":
            return Resolution(mode: .auto, note: nil)
        case "off":
            return Resolution(mode: .off, note: nil)
        case "on":
            return Resolution(mode: .on, note: nil)
        default:
            return Resolution(
                mode: .auto,
                note: "\(environmentVariable)=\(raw) is not auto, off, or on; "
                    + "using auto")
        }
    }
}

/// What the policy decided before any pipeline is built.
package enum TensorOpsPlan: Equatable, Sendable {
    /// Build the pipeline and use it if construction succeeds.
    case attempt(reason: String)
    /// Do not build it; the scalar tiled path is the answer.
    case fallback(reason: String)

    package var attemptsPipeline: Bool {
        if case .attempt = self { return true }
        return false
    }

    package var reason: String {
        switch self {
        case .attempt(let reason), .fallback(let reason): reason
        }
    }
}

/// The single rule shared by the kernel builder and `tinytitan doctor`.
///
/// - `auto` uses tensor operations only when the capability is present.
/// - `off` never builds the pipeline.
/// - `on` attempts it regardless of the capability and falls back on failure.
package enum TensorOpsPolicy {
    package static func plan(
        capabilities: HardwareCapabilities,
        mode: TensorOpsMode,
        note: String? = nil
    ) -> TensorOpsPlan {
        let suffix = note.map { "; \($0)" } ?? ""
        switch mode {
        case .off:
            return .fallback(
                reason: "\(TensorOpsMode.environmentVariable)=off\(suffix)")
        case .on:
            return .attempt(
                reason: "\(TensorOpsMode.environmentVariable)=on (forced)\(suffix)")
        case .auto:
            guard capabilities.supportsApple10TensorOps else {
                return .fallback(
                    reason: "device does not support Apple10 MPP tensor operations"
                        + suffix)
            }
            return .attempt(reason: "Apple10 tensor operations available" + suffix)
        }
    }
}

/// Which path a built kernel actually took.
package enum TensorOpsRuntimePath: String, Equatable, Sendable {
    case tensorOps = "tensor-ops"
    case fallback
}

/// The final answer, after a pipeline build has succeeded or failed.
package struct TensorOpsDecision: Equatable, Sendable {
    package let mode: TensorOpsMode
    package let path: TensorOpsRuntimePath
    package let reason: String

    package init(
        mode: TensorOpsMode, path: TensorOpsRuntimePath, reason: String
    ) {
        self.mode = mode
        self.path = path
        self.reason = reason
    }

    /// One line for the startup log. The path is first so a reader can grep
    /// for `tensor-ops` or `fallback` without knowing the reason wording.
    package var logLine: String { logLine(subject: "prefill") }

    /// The same line naming a kernel family. K5 builds more than one
    /// tensor-ops pipeline (attention, projections, FFN) and each records its
    /// own decision, so the log has to say which one a fallback belongs to.
    package func logLine(subject: String) -> String {
        "TinyTitan tensor-ops \(subject): \(path.rawValue) "
            + "mode=\(mode.rawValue) (\(reason))"
    }
}
