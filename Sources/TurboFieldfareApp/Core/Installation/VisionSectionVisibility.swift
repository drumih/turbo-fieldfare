import Foundation

/// Whether the inspector shows its Image Support section.
///
/// This lived as a private computed property on the inspector view, where the
/// only way to check it was to launch the app and look. It decides four distinct
/// situations, so it is worth being able to test all four.
public enum VisionSectionVisibility {
    /// Shown while the section has something to say, hidden once it does not.
    ///
    /// An installed text model is deliberately *not* required. Requiring one hid
    /// image support behind a 14.62 GB download, so the screen where someone
    /// decides whether this app does what they need never mentioned that it
    /// handles images at all. The section still collapses once the pack is
    /// installed and healthy, so it costs space only while it is actionable.
    public static func shows(
        visionRuntimeEnabled: Bool,
        visionRuntimeSupported: Bool = true,
        isModelInstalled: Bool,
        isVisionPackInstalled: Bool,
        isCompanionOperationInProgress: Bool,
        installState: AppModelInstallState
    ) -> Bool {
        guard visionRuntimeEnabled else { return false }
        if !visionRuntimeSupported { return true }
        if !isModelInstalled { return true }
        if !isVisionPackInstalled { return true }
        if isCompanionOperationInProgress { return true }
        switch installState {
        case .failed, .recoverable: return true
        default: return false
        }
    }
}
